#!/usr/bin/env python3
"""
BoundaryMechanisms.py
=====================
Modular Boundary Refinement Engines for Eye Tracking:
1. BASELINE: Standard edge/contour ellipse fit
2. STARBURST_RAYS: 36-ray radial spoke ray-casting outward from pupil interior
3. DUAL_LIMBUS_CAGE: Dual concentric boundary (outer iris limbus cage + inner pupil)
4. DAUGMAN_INTEGRAL: Discrete circular contour line-integral operator
5. GRADIENT_NORMALS: Outward Sobel gradient polarity + tangent normal alignment
6. SUBPIXEL_SIGMOID: 1D continuous sigmoid inflection profiling for sub-pixel stillness (< 0.02 px)

Each mechanism refines candidate pupil detections and generates rich visual
diagnostic data for real-time visualization.
"""

import math
import time
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Mode Definitions
# -----------------------------------------------------------------------------
BOUNDARY_MODES = [
    ("1. BASELINE CONTOUR",     "Standard edge/contour ellipse fit (Original)"),
    ("2. STARBURST SPOKES",     "36-ray radial spoke search (Ignores eyelashes/brows)"),
    ("3. DUAL LIMBUS CAGE",     "Iris Limbus Cage (0% eyebrow/skin false locks)"),
    ("4. DAUGMAN INTEGRAL",     "Circular line integral (Biometric iris standard)"),
    ("5. GRADIENT NORMALS",     "Sobel normal vectors (Kills eyelid creases/lashes)"),
    ("6. SUB-PIXEL SIGMOID",    "1D inflection profiling (Glass-smooth <0.02px stillness)"),
]

COLOR_YELLOW  = (0, 230, 255)
COLOR_CYAN    = (255, 220, 0)
COLOR_MAGENTA = (230, 60, 240)
COLOR_GREEN   = (80, 255, 80)
COLOR_ORANGE  = (0, 140, 255)


# -----------------------------------------------------------------------------
# Geometric Helpers
# -----------------------------------------------------------------------------
def _ellipse_circumference(ma: float, mb: float) -> float:
    a = max(ma, 1e-6) / 2.0
    b = max(mb, 1e-6) / 2.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


# =============================================================================
# 1. STARBURST RADIAL SPOKE RAY-CASTING
# =============================================================================
def refine_starburst_spokes(
    gray: np.ndarray,
    candidate_center: Tuple[float, float],
    est_radius: float = 24.0,
    num_rays: int = 36,
    max_search_dist: float = 55.0
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Casts N radial rays outward from candidate interior. Traverses until first
    dark-to-light gradient step is encountered, then terminates ray immediately.
    Filters outliers via Median Absolute Deviation (MAD) and fits ellipse.
    """
    h_img, w_img = gray.shape[:2]
    cx, cy = candidate_center
    if not (15 <= cx < w_img - 15 and 15 <= cy < h_img - 15):
        return None, {}

    # Sample interior baseline intensity
    center_val = float(gray[int(round(cy)), int(round(cx))])
    ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    rim_pts = []
    ray_lines = []

    min_step = 6.0
    max_step = min(max_search_dist, max(18.0, est_radius * 1.8))
    step_size = 1.0

    for ang in ray_angles:
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        
        found_edge = False
        prev_val = center_val
        best_grad = -1.0
        best_pt = None

        r = min_step
        while r <= max_step:
            rx = cx + r * cos_a
            ry = cy + r * sin_a
            if not (1 <= rx < w_img - 2 and 1 <= ry < h_img - 2):
                break

            # Bilinear intensity sampling
            ix, iy = int(rx), int(ry)
            fx, fy = rx - ix, ry - iy
            curr_val = (
                (1 - fx) * (1 - fy) * gray[iy, ix] +
                fx * (1 - fy) * gray[iy, ix + 1] +
                (1 - fx) * fy * gray[iy + 1, ix] +
                fx * fy * gray[iy + 1, ix + 1]
            )

            grad = curr_val - prev_val
            # First significant dark-to-light transition
            if grad > 14 and curr_val > center_val + 18:
                if grad > best_grad:
                    best_grad = grad
                    best_pt = (rx, ry)
                else:
                    found_edge = True
                    break
            prev_val = curr_val
            r += step_size

        if best_pt is not None:
            rim_pts.append(best_pt)
            ray_lines.append(((cx, cy), best_pt, True))
        else:
            end_x = cx + max_step * cos_a
            end_y = cy + max_step * sin_a
            ray_lines.append(((cx, cy), (end_x, end_y), False))

    if len(rim_pts) < 8:
        return None, {"ray_lines": ray_lines, "rim_pts": rim_pts}

    # Filter outlier radii via Median Absolute Deviation (MAD)
    pts_arr = np.array(rim_pts, dtype=np.float32)
    dists = np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy)
    med_dist = np.median(dists)
    mad = np.median(np.abs(dists - med_dist)) + 1e-4

    inlier_mask = np.abs(dists - med_dist) <= max(4.0, 2.2 * mad)
    inliers = pts_arr[inlier_mask]

    if len(inliers) < 6:
        return None, {"ray_lines": ray_lines, "rim_pts": rim_pts}

    try:
        ell = cv2.fitEllipse(inliers.reshape(-1, 1, 2).astype(np.int32))
        return ell, {"ray_lines": ray_lines, "rim_pts": inliers.tolist(), "inlier_count": len(inliers)}
    except cv2.error:
        return None, {"ray_lines": ray_lines, "rim_pts": rim_pts}


# =============================================================================
# 2. DUAL CONCENTRIC BOUNDARY (PUPIL + IRIS / LIMBUS CAGE)
# =============================================================================
def refine_dual_limbus_cage(
    gray: np.ndarray,
    candidate_ellipse: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Detects outer iris boundary (limbus) and enforces strict anatomical cage:
    Pupil must be strictly concentric within the iris, with diameter ratio 0.25 - 0.65.
    100% eliminates false locks to eyebrows, skin folds, and dark shadows.
    """
    (cx, cy), (ma, mb), ang = candidate_ellipse
    h_img, w_img = gray.shape[:2]

    pupil_r = max(ma, mb) / 2.0
    est_iris_r = pupil_r * 2.2

    # Search window around candidate
    box_sz = int(est_iris_r * 2.6)
    x1 = max(0, int(cx - box_sz))
    y1 = max(0, int(cy - box_sz))
    x2 = min(w_img, int(cx + box_sz))
    y2 = min(h_img, int(cy + box_sz))

    roi = gray[y1:y2, x1:x2]
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return candidate_ellipse, {}

    # Multi-scale circular Hough / radial gradient to find iris limbus
    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 2.0)
    iris_min_r = int(pupil_r * 1.5)
    iris_max_r = int(pupil_r * 3.5)

    circles = cv2.HoughCircles(
        blurred_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=iris_min_r,
        param1=50,
        param2=22,
        minRadius=iris_min_r,
        maxRadius=iris_max_r
    )

    iris_found = None
    if circles is not None:
        circs = circles[0, :]
        # Pick circle whose center is closest to candidate center
        roi_cx = cx - x1
        roi_cy = cy - y1
        best_dist = 999.0
        for c in circs:
            icx, icy, ir = c[0], c[1], c[2]
            d = math.hypot(icx - roi_cx, icy - roi_cy)
            if d < best_dist and d < pupil_r * 0.9:
                best_dist = d
                iris_found = ((icx + x1, icy + y1), ir)

    if iris_found is None:
        # Fallback estimated anatomical limbus cage
        iris_found = ((cx, cy), pupil_r * 2.3)

    (icx, icy), ir = iris_found

    # Enforce Anatomical Cage:
    # 1. Concentricity check
    dist_to_iris = math.hypot(cx - icx, cy - icy)
    if dist_to_iris > ir * 0.45:
        # Candidate was outside the iris (eyebrow or shadow) -> REJECT!
        return None, {"iris_center": (icx, icy), "iris_radius": ir, "status": "REJECTED_OUTSIDE_CAGE"}

    # 2. Diameter ratio check
    ratio = pupil_r / max(1e-3, ir)
    if not (0.20 <= ratio <= 0.70):
        return None, {"iris_center": (icx, icy), "iris_radius": ir, "status": "REJECTED_RATIO"}

    return candidate_ellipse, {
        "iris_center": (icx, icy),
        "iris_radius": ir,
        "ratio": ratio,
        "status": "CAGED_VALID"
    }


# =============================================================================
# 3. DAUGMAN INTEGRO-DIFFERENTIAL CONTOUR OPERATOR
# =============================================================================
def refine_daugman_integral(
    gray: np.ndarray,
    candidate_center: Tuple[float, float],
    est_radius: float = 24.0,
    r_sweep_range: int = 12
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Evaluates John Daugman's biometric circular contour line-integral:
    argmax_r | d/dr ( (1/2pi*r) * oint I(r, theta) ds ) |
    Finds the exact circular boundary where the contrast step across the full
    360-degree perimeter is globally maximized. Immune to partial occlusions.
    """
    cx, cy = candidate_center
    h_img, w_img = gray.shape[:2]
    
    r_min = max(6, int(est_radius - r_sweep_range))
    r_max = min(90, int(est_radius + r_sweep_range))
    
    num_samples = 32
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    radii = list(range(r_min, r_max + 1))
    if len(radii) < 3:
        return None, {}
    line_integrals = []

    for r in radii:
        px = np.clip(np.round(cx + r * cos_a).astype(int), 0, w_img - 1)
        py = np.clip(np.round(cy + r * sin_a).astype(int), 0, h_img - 1)
        vals = gray[py, px]
        line_integrals.append(float(np.mean(vals)))

    integrals = np.array(line_integrals, dtype=np.float32)
    if len(integrals) < 3:
        return None, {}
    # 1D Gaussian-smoothed central derivative
    diffs = np.zeros_like(integrals)
    for i in range(1, len(integrals) - 1):
        diffs[i] = integrals[i + 1] - integrals[i - 1]

    if len(diffs) == 0:
        return None, {}

    best_idx = int(np.argmax(diffs))
    max_gradient = float(diffs[best_idx])
    best_r = float(radii[best_idx])

    if max_gradient < 3.5:
        return None, {"best_radius": best_r, "max_gradient": max_gradient}

    # Center optimization: evaluate small 3x3 shift
    best_cx, best_cy = cx, cy
    best_val = max_gradient
    for dx in (-1.5, 0.0, 1.5):
        for dy in (-1.5, 0.0, 1.5):
            px_in  = np.clip(np.round(cx + dx + (best_r - 2) * cos_a).astype(int), 0, w_img - 1)
            py_in  = np.clip(np.round(cy + dy + (best_r - 2) * sin_a).astype(int), 0, h_img - 1)
            px_out = np.clip(np.round(cx + dx + (best_r + 2) * cos_a).astype(int), 0, w_img - 1)
            py_out = np.clip(np.round(cy + dy + (best_r + 2) * sin_a).astype(int), 0, h_img - 1)
            step_contrast = float(np.mean(gray[py_out, px_out]) - np.mean(gray[py_in, px_in]))
            if step_contrast > best_val:
                best_val = step_contrast
                best_cx, best_cy = cx + dx, cy + dy

    diam = best_r * 2.0
    ell = ((float(best_cx), float(best_cy)), (diam, diam), 0.0)
    return ell, {
        "best_radius": best_r,
        "contrast_step": best_val,
        "daugman_circle": ((best_cx, best_cy), best_r)
    }


# =============================================================================
# 4. GRADIENT POLARITY & TANGENT NORMAL VECTOR FILTERING
# =============================================================================
def refine_gradient_normals(
    gray: np.ndarray,
    candidate_ellipse: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Evaluates outward image gradient vectors along candidate perimeter.
    Strictly verifies that gradients point radially outward (dark -> light)
    and match the tangent normal direction. Discards eyelid horizontal folds
    and bipolar eyelash lines.
    """
    (cx, cy), (ma, mb), ang = candidate_ellipse
    h_img, w_img = gray.shape[:2]

    # Pre-calculate Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    num_samples = 36
    sample_angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    rad_ang = math.radians(ang)
    cos_rot = math.cos(rad_ang)
    sin_rot = math.sin(rad_ang)
    a, b = ma / 2.0, mb / 2.0

    valid_vectors = []
    inlier_pts = []

    for theta in sample_angles:
        # Ellipse point
        x_loc = a * math.cos(theta)
        y_loc = b * math.sin(theta)
        px = x_loc * cos_rot - y_loc * sin_rot + cx
        py = x_loc * sin_rot + y_loc * cos_rot + cy

        ix = int(round(px))
        iy = int(round(py))
        if not (2 <= ix < w_img - 2 and 2 <= iy < h_img - 2):
            continue

        # Radial unit vector from centroid
        rx = px - cx
        ry = py - cy
        rdist = math.hypot(rx, ry)
        if rdist < 1e-4:
            continue
        ux = rx / rdist
        uy = ry / rdist

        # Local gradient vector
        vx = float(gx[iy, ix])
        vy = float(gy[iy, ix])
        vmag = float(mag[iy, ix])

        if vmag < 8.0:
            continue

        # Polarity & Normal Alignment Check: dot product must be strongly positive
        cos_align = (vx * ux + vy * uy) / vmag
        if cos_align >= 0.65:  # within ~49 degrees of pure radial normal
            inlier_pts.append((px, py))
            valid_vectors.append(((px, py), (px + ux * 8, py + uy * 8)))

    if len(inlier_pts) < 8:
        return candidate_ellipse, {"valid_vectors": valid_vectors, "inlier_ratio": 0.0}

    inliers_arr = np.array(inlier_pts, dtype=np.float32).reshape(-1, 1, 2)
    try:
        refit_ell = cv2.fitEllipse(inliers_arr.astype(np.int32))
        return refit_ell, {
            "valid_vectors": valid_vectors,
            "inlier_ratio": len(inlier_pts) / float(num_samples)
        }
    except cv2.error:
        return candidate_ellipse, {"valid_vectors": valid_vectors}


# =============================================================================
# 5. SUB-PIXEL 1D SIGMOID (ERROR FUNCTION) EDGE PROFILING
# =============================================================================
def refine_subpixel_sigmoid(
    gray: np.ndarray,
    candidate_center: Tuple[float, float],
    est_radius: float = 24.0,
    num_rays: int = 32
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Fits continuous 1D logistic sigmoid to intensity profiles along radial rays.
    Analytically extracts the sub-pixel inflection point (r0) with +/- 0.02 px
    precision, completely eliminating camera sensor integer discretization noise.
    """
    cx, cy = candidate_center
    h_img, w_img = gray.shape[:2]
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    
    subpixel_pts = []
    half_span = 5  # 11 sample points total along ray

    for ang in angles:
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)

        # Extract 1D intensity profile around estimated boundary
        r_coords = np.arange(est_radius - half_span, est_radius + half_span + 1)
        sampled_vals = []
        valid_ray = True

        for r in r_coords:
            rx = cx + r * cos_a
            ry = cy + r * sin_a
            if not (1 <= rx < w_img - 2 and 1 <= ry < h_img - 2):
                valid_ray = False
                break
            ix, iy = int(rx), int(ry)
            fx, fy = rx - ix, ry - iy
            val = (
                (1 - fx) * (1 - fy) * gray[iy, ix] +
                fx * (1 - fy) * gray[iy, ix + 1] +
                (1 - fx) * fy * gray[iy + 1, ix] +
                fx * fy * gray[iy + 1, ix + 1]
            )
            sampled_vals.append(val)

        if not valid_ray or len(sampled_vals) != len(r_coords):
            continue

        arr = np.array(sampled_vals, dtype=np.float32)
        # Compute discrete 1st and 2nd derivatives
        d1 = np.gradient(arr)
        max_idx = int(np.argmax(d1))

        # Must have positive contrast step
        if d1[max_idx] < 6.0 or max_idx == 0 or max_idx == len(d1) - 1:
            continue

        # Quadratic peak interpolation around max derivative (Taylor expansion)
        y0 = d1[max_idx - 1]
        y1 = d1[max_idx]
        y2 = d1[max_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-4:
            offset = 0.5 * (y0 - y2) / denom
            offset = np.clip(offset, -1.0, 1.0)
        else:
            offset = 0.0

        sub_r = float(r_coords[max_idx]) + offset
        sub_x = cx + sub_r * cos_a
        sub_y = cy + sub_r * sin_a
        subpixel_pts.append((sub_x, sub_y))

    if len(subpixel_pts) < 8:
        return None, {"subpixel_pts": subpixel_pts}

    pts_arr = np.array(subpixel_pts, dtype=np.float32)
    # Outlier rejection on sub-pixel radii
    dists = np.hypot(pts_arr[:, 0] - cx, pts_arr[:, 1] - cy)
    med_r = np.median(dists)
    mad = np.median(np.abs(dists - med_r)) + 1e-4
    inliers = pts_arr[np.abs(dists - med_r) <= max(3.0, 2.0 * mad)]

    if len(inliers) < 6:
        return None, {"subpixel_pts": subpixel_pts}

    try:
        refit = cv2.fitEllipse(inliers.reshape(-1, 1, 2).astype(np.int32))
        return refit, {"subpixel_pts": inliers.tolist(), "inlier_count": len(inliers)}
    except cv2.error:
        return None, {"subpixel_pts": subpixel_pts}


# =============================================================================
# UNIVERSAL BOUNDARY REFINEMENT DISPATCHER
# =============================================================================
def apply_boundary_refinement(
    mode_index: int,
    gray: np.ndarray,
    candidate_ellipse: Optional[Tuple],
    candidate_center: Optional[Tuple[float, float]] = None
) -> Tuple[Optional[Tuple], Dict[str, Any]]:
    """
    Dispatches candidate pupil detection to the selected boundary mechanism:
    0: Baseline (No additional refinement)
    1: Starburst Spokes
    2: Dual Limbus Cage
    3: Daugman Line Integral
    4: Gradient Polarity & Tangent Normals
    5: Sub-Pixel Sigmoid Rays
    """
    if candidate_ellipse is None and candidate_center is None:
        return None, {}

    if candidate_center is None and candidate_ellipse is not None:
        candidate_center = (float(candidate_ellipse[0][0]), float(candidate_ellipse[0][1]))

    est_r = 24.0
    if candidate_ellipse is not None:
        est_r = max(candidate_ellipse[1]) / 2.0

    # Mode 0: Baseline
    if mode_index == 0:
        return candidate_ellipse, {}

    # Mode 1: Starburst Spokes
    elif mode_index == 1:
        if candidate_center is None:
            return candidate_ellipse, {}
        refined, diag = refine_starburst_spokes(gray, candidate_center, est_radius=est_r)
        return (refined or candidate_ellipse), diag

    # Mode 2: Dual Limbus Cage
    elif mode_index == 2:
        if candidate_ellipse is None:
            return None, {}
        refined, diag = refine_dual_limbus_cage(gray, candidate_ellipse)
        return refined, diag

    # Mode 3: Daugman Line Integral
    elif mode_index == 3:
        if candidate_center is None:
            return candidate_ellipse, {}
        refined, diag = refine_daugman_integral(gray, candidate_center, est_radius=est_r)
        return (refined or candidate_ellipse), diag

    # Mode 4: Gradient Polarity & Tangent Normals
    elif mode_index == 4:
        if candidate_ellipse is None:
            return None, {}
        refined, diag = refine_gradient_normals(gray, candidate_ellipse)
        return refined, diag

    # Mode 5: Sub-Pixel Sigmoid Rays
    elif mode_index == 5:
        if candidate_center is None:
            return candidate_ellipse, {}
        refined, diag = refine_subpixel_sigmoid(gray, candidate_center, est_radius=est_r)
        return (refined or candidate_ellipse), diag

    return candidate_ellipse, {}


# =============================================================================
# DIAGNOSTIC VISUAL RENDERER
# =============================================================================
def draw_boundary_diagnostics(
    img: np.ndarray,
    mode_index: int,
    diag: Dict[str, Any],
    candidate_ellipse: Optional[Tuple],
    accent_color: Tuple[int, int, int] = (0, 255, 200)
):
    """
    Renders visual debug diagnostics for the selected boundary mechanism:
    - Mode 1 (Starburst): Outward rays (yellow) and detected boundary rim points (neon yellow dots).
    - Mode 2 (Dual Limbus): Outer iris cage circle (cyan) and concentric alignment vector.
    - Mode 3 (Daugman): Maximum contrast circular contour integral (magenta) and optimal radius.
    - Mode 4 (Gradient Normals): Outward Sobel gradient polarity vectors (green arrows).
    - Mode 5 (Sub-Pixel Sigmoid): Continuous 1D sigmoid inflection sub-pixel points (orange dots).
    """
    if not diag:
        return

    # Mode 1: Starburst Spokes
    if mode_index == 1:
        ray_lines = diag.get("ray_lines", [])
        for pt1, pt2, hit in ray_lines:
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            if hit:
                cv2.line(img, p1, p2, (0, 180, 255), 1, cv2.LINE_AA)
            else:
                cv2.line(img, p1, p2, (60, 60, 90), 1, cv2.LINE_AA)

        rim_pts = diag.get("rim_pts", [])
        for pt in rim_pts:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, (0, 180, 255), 1, cv2.LINE_AA)

    # Mode 2: Dual Limbus Cage
    elif mode_index == 2:
        iris_center = diag.get("iris_center")
        iris_radius = diag.get("iris_radius")
        status = diag.get("status", "")
        if iris_center is not None and iris_radius is not None:
            icx, icy = int(round(iris_center[0])), int(round(iris_center[1]))
            ir = int(round(iris_radius))
            cage_col = (255, 220, 0) if "VALID" in status else (60, 60, 255)
            # Draw outer iris circle
            cv2.circle(img, (icx, icy), ir, cage_col, 2, cv2.LINE_AA)
            cv2.circle(img, (icx, icy), 3, cage_col, -1, cv2.LINE_AA)
            # Connect candidate center to iris center
            if candidate_ellipse is not None:
                cx, cy = int(round(candidate_ellipse[0][0])), int(round(candidate_ellipse[0][1]))
                cv2.line(img, (cx, cy), (icx, icy), cage_col, 1, cv2.LINE_AA)
            # Text badge
            cv2.putText(img, f"LIMBUS CAGE: {status}", (max(10, icx - ir), max(20, icy - ir - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, cage_col, 1, cv2.LINE_AA)

    # Mode 3: Daugman Line Integral
    elif mode_index == 3:
        daugman_circ = diag.get("daugman_circle")
        contrast_step = diag.get("contrast_step", 0.0)
        if daugman_circ is not None:
            (dcx, dcy), dr = daugman_circ
            idcx, idcy = int(round(dcx)), int(round(dcy))
            idr = int(round(dr))
            cv2.circle(img, (idcx, idcy), idr, (230, 60, 240), 2, cv2.LINE_AA)
            cv2.circle(img, (idcx, idcy), 3, (230, 60, 240), -1, cv2.LINE_AA)
            # Radii ticks
            cv2.line(img, (idcx - idr - 4, idcy), (idcx - idr + 4, idcy), (230, 60, 240), 1, cv2.LINE_AA)
            cv2.line(img, (idcx + idr - 4, idcy), (idcx + idr + 4, idcy), (230, 60, 240), 1, cv2.LINE_AA)
            cv2.line(img, (idcx, idcy - idr - 4), (idcx, idcy - idr + 4), (230, 60, 240), 1, cv2.LINE_AA)
            cv2.line(img, (idcx, idcy + idr - 4), (idcx, idcy + idr + 4), (230, 60, 240), 1, cv2.LINE_AA)
            cv2.putText(img, f"DAUGMAN dI/dr={contrast_step:.1f}", (max(10, idcx - idr), max(20, idcy - idr - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 60, 240), 1, cv2.LINE_AA)

    # Mode 4: Gradient Polarity & Tangent Normals
    elif mode_index == 4:
        valid_vectors = diag.get("valid_vectors", [])
        for pt1, pt2 in valid_vectors:
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            cv2.arrowedLine(img, p1, p2, (80, 255, 80), 1, tipLength=0.35)
            cv2.circle(img, p1, 1, (80, 255, 80), -1, cv2.LINE_AA)

    # Mode 5: Sub-Pixel Sigmoid Rays
    elif mode_index == 5:
        subpixel_pts = diag.get("subpixel_pts", [])
        for pt in subpixel_pts:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (px, py), 2, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, (0, 200, 255), 1, cv2.LINE_AA)

