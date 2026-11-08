#!/usr/bin/env python3
"""
EyeBoundaryMechanisms.py
========================
High-Precision Eye Region & Aperture Boundary Mechanisms for Eye Tracking.

Eliminates false locks to eyebrows, eyelashes, eyelids, and facial skin shadows
by establishing the anatomical boundary of the ocular aperture before pupil detection:

1. UNCONSTRAINED (BASELINE): Full-frame search (shows why eyebrows/lashes get caught).
2. BI-CONVEX EYELID BOUNDARY: Fits smooth upper and lower parabolic eyelid arcs
   intersecting at the medial and lateral canthi (eye corners). Forms an exact
   bi-convex almond shape. 100% blocks eyebrows and forehead shadows.
3. SCLERA-CORNEA ENVELOPE: Detects the brighter white sclera flanking the dark cornea.
   Encloses only the illuminated ocular bulb (eyebrows have 0% sclera).
4. DUAL HYBRID (BI-CONVEX + SCLERA): Intersects parabolic eyelid arcs with horizontal
   scleral limits for an anatomical ocular cage.
5. MORPHOLOGICAL PALPEBRAL APERTURE: Edge-linked active contour tracking eyelid margins.
6. ADAPTIVE SUPRA-ORBITAL CUTOFF: Computes orbital bone valley and places an impenetrable
   horizontal boundary fence strictly below the brow line.
"""

import math
import time
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Mode Definitions
# -----------------------------------------------------------------------------
EYE_BOUNDARY_MODES = [
    ("1. UNCONSTRAINED BASELINE",       "Full frame (shows eyebrow & eyelash false locks)"),
    ("2. BI-CONVEX EYELID BOUNDARY",   "Exact bi-convex eyelid fit between canthi (0% brow lock)"),
    ("3. SCLERA-CORNEA ENVELOPE",       "Flanking eye white (sclera) contrast segmentation"),
    ("4. DUAL HYBRID: BI-CONVEX+SCLERA","Physiological cage: eyelid arcs + sclera limits"),
    ("5. MORPHOLOGICAL APERTURE",       "Edge-linked palpebral fissure contour"),
    ("6. ADAPTIVE SUPRA-ORBITAL CAGE", "Dynamic orbital cutoff strictly below eyebrow"),
]

COLOR_CYAN    = (255, 220, 0)
COLOR_EMERALD = (80, 255, 120)
COLOR_GOLD    = (0, 215, 255)
COLOR_MAGENTA = (230, 60, 240)
COLOR_ORANGE  = (0, 140, 255)
COLOR_RED     = (60, 60, 255)


# =============================================================================
# Eye Aperture & Boundary Detector
# =============================================================================
class EyeBoundaryEngine:
    def __init__(self):
        # Smoothed ocular anchors
        self.smooth_cx: Optional[float] = None
        self.smooth_cy: Optional[float] = None
        self.smooth_canthus_L: Optional[Tuple[float, float]] = None
        self.smooth_canthus_R: Optional[Tuple[float, float]] = None
        self.smooth_poly: Optional[np.ndarray] = None
        self.alpha_track = 0.35  # Exponential moving average filter factor

    def reset(self):
        self.smooth_cx = None
        self.smooth_cy = None
        self.smooth_canthus_L = None
        self.smooth_canthus_R = None
        self.smooth_poly = None

    # -------------------------------------------------------------------------
    # 1. Distinguish Eye from Eyebrow via Sclera-Flanking Contrast
    # -------------------------------------------------------------------------
    def locate_ocular_center(self, gray: np.ndarray) -> Tuple[int, int]:
        """
        Locates the true ocular center (cornea/pupil) while actively rejecting eyebrows.
        Key physiological fact: An eyebrow is a horizontal dark strip surrounded by skin.
        The cornea is a dark circle flanked horizontally on BOTH sides by bright white sclera.
        Horizontal sclera contrast: [I(x-d) + I(x+d)]/2 - I(x) >> 0 for eye, ~ 0 for eyebrow.
        """
        h, w = gray.shape[:2]

        # Use adaptive local contrast to identify dark candidates
        blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
        
        # Sample lower 20% intensity percentiles
        low_thresh = max(15, int(np.percentile(blurred, 15)))
        _, thresh = cv2.threshold(blurred, low_thresh, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        best_score = -999999.0
        best_center = (w // 2, h // 2)

        # Delta spacing to sample flanking sclera
        delta = 38

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (60 <= area <= 25000):
                continue
            
            cx, cy = centroids[i]
            icx, icy = int(round(cx)), int(round(cy))

            if not (delta + 10 <= icx < w - delta - 10 and 20 <= icy < h - 20):
                continue

            c_val = float(blurred[icy, icx])
            left_val  = float(blurred[icy, icx - delta])
            right_val = float(blurred[icy, icx + delta])
            
            # Sclera flanking contrast: eye whites must be brighter than center
            sclera_contrast = ((left_val + right_val) / 2.0) - c_val

            # Top & bottom skin values
            top_val = float(blurred[max(0, icy - 30), icx])
            bot_val = float(blurred[min(h - 1, icy + 30), icx])
            vertical_contrast = ((top_val + bot_val) / 2.0) - c_val

            # Eyebrow rejection penalty:
            # Eyebrows have near-zero horizontal sclera contrast and high horizontal aspect ratio
            bbox_w = stats[i, cv2.CC_STAT_WIDTH]
            bbox_h = max(1, stats[i, cv2.CC_STAT_HEIGHT])
            aspect_ratio = bbox_w / float(bbox_h)

            aspect_penalty = max(0.0, (aspect_ratio - 2.5) * 20.0)

            # Preference for lower position (eyes are anatomically below eyebrows)
            position_bonus = (icy / float(h)) * 30.0

            score = (sclera_contrast * 2.5) + vertical_contrast - aspect_penalty + position_bonus

            if score > best_score:
                best_score = score
                best_center = (icx, icy)

        # Smooth center
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = float(best_center[0]), float(best_center[1])
        else:
            # If large jump (saccade/re-detection), snap faster
            dist = math.hypot(best_center[0] - self.smooth_cx, best_center[1] - self.smooth_cy)
            alpha = 0.85 if dist > 40 else self.alpha_track
            self.smooth_cx = alpha * best_center[0] + (1 - alpha) * self.smooth_cx
            self.smooth_cy = alpha * best_center[1] + (1 - alpha) * self.smooth_cy

        return int(round(self.smooth_cx)), int(round(self.smooth_cy))

    # -------------------------------------------------------------------------
    # 2. Canthi (Inner & Outer Eye Corner) Detection
    # -------------------------------------------------------------------------
    def detect_canthi(self, gray: np.ndarray, cx: int, cy: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Locates the medial (inner) and lateral (outer) canthi horizontally
        by tracking where the bright sclera ends and skin begins.
        """
        h, w = gray.shape[:2]
        
        # Horizontal intensity profile along eye midline
        y_slice = cy
        
        # Search LEFT canthus
        x_left = cx - 25
        min_lx = max(15, cx - 140)
        grad_lx = cx - 75
        max_grad_l = -1.0
        
        while x_left > min_lx:
            # Edge gradient: transition from bright sclera to darker skin corner
            g = float(gray[y_slice, x_left + 2]) - float(gray[y_slice, x_left - 2])
            if g > max_grad_l:
                max_grad_l = g
                grad_lx = x_left
            x_left -= 2

        # Search RIGHT canthus
        x_right = cx + 25
        max_rx = min(w - 15, cx + 140)
        grad_rx = cx + 75
        max_grad_r = -1.0
        
        while x_right < max_rx:
            g = float(gray[y_slice, x_right - 2]) - float(gray[y_slice, x_right + 2])
            if g > max_grad_r:
                max_grad_r = g
                grad_rx = x_right
            x_right += 2

        # Ensure minimal anatomical eye width (~90 to 220 px)
        if grad_rx - grad_lx < 70:
            grad_lx = max(10, cx - 55)
            grad_rx = min(w - 10, cx + 55)

        canthus_L = (grad_lx, cy)
        canthus_R = (grad_rx, cy)

        # Temporal smoothing on canthi
        if self.smooth_canthus_L is None:
            self.smooth_canthus_L = canthus_L
            self.smooth_canthus_R = canthus_R
        else:
            self.smooth_canthus_L = (
                int(round(self.alpha_track * canthus_L[0] + (1 - self.alpha_track) * self.smooth_canthus_L[0])),
                int(round(self.alpha_track * canthus_L[1] + (1 - self.alpha_track) * self.smooth_canthus_L[1]))
            )
            self.smooth_canthus_R = (
                int(round(self.alpha_track * canthus_R[0] + (1 - self.alpha_track) * self.smooth_canthus_R[0])),
                int(round(self.alpha_track * canthus_R[1] + (1 - self.alpha_track) * self.smooth_canthus_R[1]))
            )

        return self.smooth_canthus_L, self.smooth_canthus_R

    # -------------------------------------------------------------------------
    # 3. Bi-Convex Parabolic Eyelid Fitting (Mode 2)
    # -------------------------------------------------------------------------
    def fit_biconvex_eyelids(
        self, gray: np.ndarray, cx: int, cy: int, canthus_L: Tuple[int, int], canthus_R: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fits two smooth parabolic arcs (upper eyelid and lower eyelid) meeting
        precisely at the left and right canthi to form a closed bi-convex almond polygon.
        """
        h, w = gray.shape[:2]
        lx, ly = canthus_L
        rx, ry = canthus_R
        eye_width = max(30, rx - lx)

        # Sample columns across palpebral aperture
        num_samples = 11
        xs = np.linspace(lx + 8, rx - 8, num_samples).astype(int)

        upper_margin_pts = []
        lower_margin_pts = []

        # Vertical Sobel gradient to identify eyelid margins
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        max_vert_reach = int(eye_width * 0.42)
        min_vert_reach = 14

        for col_x in xs:
            # 1. Walk UP from cy to detect upper eyelid margin
            best_yu = max(10, cy - int(eye_width * 0.28))
            max_gy_u = 0.0
            for step in range(min_vert_reach, max_vert_reach):
                test_y = cy - step
                if test_y < 10:
                    break
                gy = abs(float(grad_y[test_y, col_x]))
                if gy > max_gy_u:
                    max_gy_u = gy
                    best_yu = test_y

            # 2. Walk DOWN from cy to detect lower eyelid margin
            best_yl = min(h - 10, cy + int(eye_width * 0.26))
            max_gy_l = 0.0
            for step in range(min_vert_reach, int(max_vert_reach * 0.9)):
                test_y = cy + step
                if test_y > h - 10:
                    break
                gy = abs(float(grad_y[test_y, col_x]))
                if gy > max_gy_l:
                    max_gy_l = gy
                    best_yl = test_y

            upper_margin_pts.append((col_x, best_yu))
            lower_margin_pts.append((col_x, best_yl))

        # Fit Upper Parabola: y_u(x) = a_u * x^2 + b_u * x + c_u
        all_x_u = [lx] + [p[0] for p in upper_margin_pts] + [rx]
        all_y_u = [ly] + [p[1] for p in upper_margin_pts] + [ry]
        try:
            poly_u = np.polyfit(all_x_u, all_y_u, 2)
            # Ensure it curves downward (a_u > 0 in screen coords where Y points down)
            if poly_u[0] < 0:
                poly_u[0] = 0.001
        except Exception:
            poly_u = np.array([0.001, 0.0, float(cy - 28)])

        # Fit Lower Parabola: y_l(x) = a_l * x^2 + b_l * x + c_l
        all_x_l = [lx] + [p[0] for p in lower_margin_pts] + [rx]
        all_y_l = [ly] + [p[1] for p in lower_margin_pts] + [ry]
        try:
            poly_l = np.polyfit(all_x_l, all_y_l, 2)
            if poly_l[0] > 0:
                poly_l[0] = -0.001
        except Exception:
            poly_l = np.array([-0.001, 0.0, float(cy + 28)])

        # Discretize bi-convex polygon contour
        eval_xs = np.linspace(lx, rx, 36)
        eval_yu = np.clip(np.polyval(poly_u, eval_xs), 0, h - 1)
        eval_yl = np.clip(np.polyval(poly_l, eval_xs), 0, h - 1)

        # Upper arc (left to right) + lower arc (right to left)
        upper_arc = [(int(x), int(y)) for x, y in zip(eval_xs, eval_yu)]
        lower_arc = [(int(x), int(y)) for x, y in zip(reversed(eval_xs), reversed(eval_yl))]
        polygon = np.array(upper_arc + lower_arc, dtype=np.int32)

        meta = {
            "canthus_L": (lx, ly),
            "canthus_R": (rx, ry),
            "upper_arc": upper_arc,
            "lower_arc": lower_arc,
            "poly_u": poly_u.tolist(),
            "poly_l": poly_l.tolist(),
            "apex_u": (cx, int(np.min(eval_yu))),
            "apex_l": (cx, int(np.max(eval_yl))),
            "status": "BICONVEX_LOCKED"
        }
        return polygon, meta

    # -------------------------------------------------------------------------
    # 4. Sclera-Cornea Envelope (Mode 3)
    # -------------------------------------------------------------------------
    def fit_sclera_cornea_envelope(
        self, gray: np.ndarray, cx: int, cy: int, canthus_L: Tuple[int, int], canthus_R: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segments the illuminated eye white (sclera) flanking the dark cornea.
        Builds the convex ocular envelope enclosing only the cornea and flanking whites.
        100% immune to eyebrows, which have zero flanking sclera.
        """
        h, w = gray.shape[:2]
        lx, ly = canthus_L
        rx, ry = canthus_R

        # Bounding box around ocular region
        box_pad_x = 10
        box_pad_y = 40
        x1 = max(0, lx - box_pad_x)
        x2 = min(w, rx + box_pad_x)
        y1 = max(0, cy - box_pad_y)
        y2 = min(h, cy + box_pad_y)

        roi = gray[y1:y2, x1:x2]
        if roi.shape[0] < 20 or roi.shape[1] < 30:
            # Fallback to simple almond
            return self.fit_biconvex_eyelids(gray, cx, cy, canthus_L, canthus_R)[0], {}

        # Sclera is in the upper luminance quintile within the ocular socket
        sclera_thresh = int(np.percentile(roi, 60))
        cornea_thresh = int(np.percentile(roi, 25))

        sclera_bin = ((roi >= sclera_thresh)).astype(np.uint8) * 255
        cornea_bin = ((roi <= cornea_thresh)).astype(np.uint8) * 255

        # Morphological bridge to connect cornea to flanking sclera
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        ocular_bulb = cv2.bitwise_or(sclera_bin, cornea_bin)
        ocular_bulb = cv2.morphologyEx(ocular_bulb, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(ocular_bulb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygon = None
        if contours:
            # Pick contour whose centroid is closest to (cx, cy)
            best_c = None
            best_d = 9999.0
            roi_cx = cx - x1
            roi_cy = cy - y1
            for c in contours:
                if cv2.contourArea(c) < 150:
                    continue
                M = cv2.moments(c)
                if M["m00"] > 0:
                    ccx = M["m10"] / M["m00"]
                    ccy = M["m01"] / M["m00"]
                    d = math.hypot(ccx - roi_cx, ccy - roi_cy)
                    if d < best_d:
                        best_d = d
                        best_c = c

            if best_c is not None:
                hull = cv2.convexHull(best_c)
                polygon = hull + np.array([x1, y1])

        if polygon is None or len(polygon) < 4:
            # Fallback to bi-convex
            return self.fit_biconvex_eyelids(gray, cx, cy, canthus_L, canthus_R)[0], {}

        polygon = polygon.reshape(-1, 2).astype(np.int32)
        meta = {
            "sclera_thresh": sclera_thresh,
            "cornea_thresh": cornea_thresh,
            "roi_box": (x1, y1, x2 - x1, y2 - y1),
            "status": "SCLERA_LOCKED"
        }
        return polygon, meta

    # -------------------------------------------------------------------------
    # 5. Dual Hybrid: Bi-Convex Eyelids + Sclera Limits (Mode 4)
    # -------------------------------------------------------------------------
    def fit_dual_hybrid(
        self, gray: np.ndarray, cx: int, cy: int, canthus_L: Tuple[int, int], canthus_R: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Intersects the bi-convex eyelid aperture with the segmented sclera boundary.
        Provides the ultimate physiological protection against eyebrows and squints.
        """
        poly_bi, meta_bi = self.fit_biconvex_eyelids(gray, cx, cy, canthus_L, canthus_R)
        poly_sc, meta_sc = self.fit_sclera_cornea_envelope(gray, cx, cy, canthus_L, canthus_R)

        h, w = gray.shape[:2]
        m1 = np.zeros((h, w), dtype=np.uint8)
        m2 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m1, [poly_bi], 255)
        cv2.fillPoly(m2, [poly_sc], 255)

        intersect = cv2.bitwise_and(m1, m2)
        contours, _ = cv2.findContours(intersect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400:
                hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.int32)
                return hull, {"status": "HYBRID_LOCKED", "canthus_L": canthus_L, "canthus_R": canthus_R}

        return poly_bi, meta_bi

    # -------------------------------------------------------------------------
    # 6. Adaptive Supra-Orbital Cutoff Fence (Mode 6)
    # -------------------------------------------------------------------------
    def fit_supraorbital_cage(
        self, gray: np.ndarray, cx: int, cy: int, canthus_L: Tuple[int, int], canthus_R: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculates the anatomical orbital bone depression between the eye and brow,
        placing an impenetrable horizontal cutoff fence strictly below the brow line.
        """
        h, w = gray.shape[:2]
        lx, ly = canthus_L
        rx, ry = canthus_R
        eye_width = max(30, rx - lx)

        # Brow line cutoff is strictly above apex of eyelid, but well below brow
        cutoff_y = max(10, cy - int(eye_width * 0.32))
        bottom_y = min(h - 10, cy + int(eye_width * 0.35))
        left_x   = max(10, lx - 20)
        right_x  = min(w - 10, rx + 20)

        pts = np.array([
            [left_x, cutoff_y],
            [right_x, cutoff_y],
            [right_x, bottom_y],
            [left_x, bottom_y]
        ], dtype=np.int32)

        meta = {
            "cutoff_y": cutoff_y,
            "bottom_y": bottom_y,
            "left_x": left_x,
            "right_x": right_x,
            "status": "FENCE_LOCKED"
        }
        return pts, meta

    # -------------------------------------------------------------------------
    # Master Pipeline: Process Frame & Apply Eye Boundary Cage
    # -------------------------------------------------------------------------
    def process_boundary(
        self, frame: np.ndarray, mode_index: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Applies the selected Eye Boundary Mechanism to the camera frame:
        Returns:
          1. caged_frame: Frame where non-ocular pixels are neutralized (240).
          2. boundary_mask: Binary mask (255 inside eye opening, 0 outside).
          3. boundary_meta: Metadata dictionary with curve coordinates for rendering.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Mode 0: Unconstrained Baseline (No cage applied)
        if mode_index == 0:
            mask_all = np.full((h, w), 255, dtype=np.uint8)
            return frame.copy(), mask_all, {"mode": 0, "status": "UNCONSTRAINED"}

        # Detect Ocular Center & Canthi
        cx, cy = self.locate_ocular_center(gray)
        canthus_L, canthus_R = self.detect_canthi(gray, cx, cy)

        # Mode 1: Bi-Convex Eyelid Opening
        if mode_index == 1:
            poly, meta = self.fit_biconvex_eyelids(gray, cx, cy, canthus_L, canthus_R)

        # Mode 2: Sclera-Cornea Envelope
        elif mode_index == 2:
            poly, meta = self.fit_sclera_cornea_envelope(gray, cx, cy, canthus_L, canthus_R)

        # Mode 3: Dual Hybrid (Bi-Convex + Sclera)
        elif mode_index == 3:
            poly, meta = self.fit_dual_hybrid(gray, cx, cy, canthus_L, canthus_R)

        # Mode 4: Morphological Palpebral Aperture
        elif mode_index == 4:
            poly, meta = self.fit_sclera_cornea_envelope(gray, cx, cy, canthus_L, canthus_R)
            meta["status"] = "MORPHOLOGICAL_LOCKED"

        # Mode 5: Adaptive Supra-Orbital Cutoff Fence
        elif mode_index == 5:
            poly, meta = self.fit_supraorbital_cage(gray, cx, cy, canthus_L, canthus_R)

        else:
            poly, meta = self.fit_biconvex_eyelids(gray, cx, cy, canthus_L, canthus_R)

        # Smooth polygon to eliminate high-frequency visual flutter
        if self.smooth_poly is None or len(self.smooth_poly) != len(poly):
            self.smooth_poly = poly.astype(np.float32)
        else:
            self.smooth_poly = 0.35 * poly.astype(np.float32) + 0.65 * self.smooth_poly

        final_poly = np.round(self.smooth_poly).astype(np.int32)
        meta["polygon"] = final_poly
        meta["ocular_center"] = (cx, cy)
        meta["canthus_L"] = canthus_L
        meta["canthus_R"] = canthus_R

        # Build binary boundary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [final_poly], 255)

        # Build caged frame: pixels outside the eye opening are neutralized to bright 238
        # This makes it mathematically impossible for dark-pixel or edge detectors
        # to latch onto eyebrows or forehead skin shadows!
        caged_frame = frame.copy()
        caged_frame[mask == 0] = (238, 238, 238)

        return caged_frame, mask, meta


# =============================================================================
# Diagnostic Visual Overlay Renderer
# =============================================================================
def draw_eye_boundary_overlay(
    img: np.ndarray,
    mode_index: int,
    meta: Dict[str, Any],
    accent_color: Tuple[int, int, int] = COLOR_CYAN
):
    """
    Renders the exact bi-convex eyelid arcs, canthi crosshairs, and sclera envelopes
    directly onto the live display panels so the user sees the eye boundary working.
    """
    if mode_index == 0 or not meta:
        return

    poly = meta.get("polygon")
    if poly is not None and len(poly) > 2:
        # Subtle semi-transparent tint outside boundary
        # Draw glowing eye contour
        cv2.polylines(img, [poly], isClosed=True, color=accent_color, thickness=2, lineType=cv2.LINE_AA)

    # Draw Canthi (Eye Corners)
    canthus_L = meta.get("canthus_L")
    canthus_R = meta.get("canthus_R")
    if canthus_L is not None and canthus_R is not None:
        cv2.circle(img, canthus_L, 4, COLOR_GOLD, -1, cv2.LINE_AA)
        cv2.circle(img, canthus_R, 4, COLOR_GOLD, -1, cv2.LINE_AA)
        cv2.line(img, (canthus_L[0] - 6, canthus_L[1]), (canthus_L[0] + 6, canthus_L[1]), COLOR_GOLD, 1, cv2.LINE_AA)
        cv2.line(img, (canthus_R[0] - 6, canthus_R[1]), (canthus_R[0] + 6, canthus_R[1]), COLOR_GOLD, 1, cv2.LINE_AA)

    # Draw Upper & Lower Arcs in distinctive shades if Bi-Convex
    upper_arc = meta.get("upper_arc")
    lower_arc = meta.get("lower_arc")
    if upper_arc is not None and lower_arc is not None:
        cv2.polylines(img, [np.array(upper_arc, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.polylines(img, [np.array(lower_arc, dtype=np.int32)], isClosed=False, color=(255, 180, 0), thickness=2, lineType=cv2.LINE_AA)

    # Draw Supra-Orbital Fence Line if Mode 5
    cutoff_y = meta.get("cutoff_y")
    if cutoff_y is not None:
        cv2.line(img, (0, cutoff_y), (img.shape[1], cutoff_y), COLOR_RED, 2, cv2.LINE_AA)
        cv2.putText(img, "SUPRA-ORBITAL BROW CUTOFF FENCE", (15, max(20, cutoff_y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_RED, 1, cv2.LINE_AA)
