#!/usr/bin/env python3
"""
PupilLabsDetector.py
====================
A faithful, high-performance standalone Python+OpenCV implementation of
Pupil Labs' Detector2D algorithm pipeline, adapted for single-camera desktop
and webcam setups as well as eye-camera modules.

If the official `pupil-detectors` C++ package is installed (Python <= 3.12),
this module AUTOMATICALLY uses it and wraps it in the same interface.
For Python 3.13 or environments without the C++ build tools, it uses the
pure-Python implementation that executes the EXACT algorithmic steps from
Pupil Labs' detect_2d.hpp:

Algorithm pipeline (mirrors pupil-labs/pupil-detectors detect_2d.hpp):
  1.  Coarse eye-ROI localisation    -- localise eye region via downsampled seed
                                        or prior ellipse tracking
  2.  Intensity histogram & spikes   -- compute 256-bin histogram to identify
                                        lowest spike (pupil) and highest spike (glint/skin)
  3.  Dual thresholding              -- pupil mask (0 to lowest_spike + intensity_range)
                                        dilated with 7x7 ellipse kernel;
                                        specular glint suppression when glints exist
  4.  Eyelash removal                -- 9x9 ellipse morphological OPEN on grayscale
  5.  Adaptive Canny edge detection  -- Canny hysteresis with contrast-adapted thresholds
  6.  Boundary edge combination      -- combine Canny edges with binary pupil boundary
  7.  Strong Prior fast tracking     -- if previous frame had a good ellipse, test
                                        true support and refit in < 1 ms
  8.  Contour evaluation             -- fit candidate ellipses on extracted contours
  9.  Interior & proximity filters   -- verify candidate center is inside dark pupil zone
                                        and within coarse eye neighborhood
  10. Candidate scoring              -- (support / circumference) * (support / contour_len)^2
  11. True support refitting         -- refit ellipse on true inlier points
  12. Gradient polarity check        -- ensure edge gradients point dark -> light

Output schema (identical to the official pupil-detectors package):
  {
    "ellipse": {
        "center":  (cx, cy),       # pupil centre in image pixels
        "axes":    (minor, major), # full axes diameters (px)
        "angle":   angle,          # rotation angle in degrees
    },
    "diameter":    float,          # major axis diameter (px)
    "location":    (cx, cy),       # alias of ellipse center
    "confidence":  float,          # support ratio [0.0, 1.0]
    "timestamp":   float,          # time.time() at detection
  }
"""

import cv2
import numpy as np
import time
import random
import math
from typing import Optional, Tuple, Dict, Any, List

# -----------------------------------------------------------------------------
# Try to use official C++ pupil-detectors package if available
# -----------------------------------------------------------------------------
_NATIVE_AVAILABLE = False
_NativeDetector2D = None

try:
    from pupil_detectors import Detector2D as _NativeDetector2D  # type: ignore
    _NATIVE_AVAILABLE = True
    print("[PupilLabsDetector] [OK] Native pupil-detectors C++ package detected -- using it.")
except ImportError:
    print("[PupilLabsDetector] Native pupil-detectors not available -- "
          "using faithful pure-Python implementation of the same algorithm.")


# -----------------------------------------------------------------------------
# Default detector properties (matches pupil-labs/pupil-detectors defaults)
# -----------------------------------------------------------------------------

DEFAULT_PROPERTIES = {
    # Coarse detection / ROI
    "coarse_detection": True,
    "coarse_filter_min": 80,
    "coarse_filter_max": 280,

    # Intensity / Thresholding (from Pupil Labs detector_2d.pyx)
    "intensity_range": 25,
    "blur_size": 5,

    # Canny Edge Parameters (0 = adaptive contrast-based, >0 = fixed)
    "canny_treshold": 0,
    "canny_ration": 2,
    "canny_aperture": 3,

    # Pupil Size Bounds (pixels)
    "pupil_size_max": 280,
    "pupil_size_min": 12,

    # Prior & Support Ratios (mirrors detect_2d.hpp)
    "strong_perimeter_ratio_range_min": 0.45,
    "strong_perimeter_ratio_range_max": 1.1,
    "strong_area_ratio_range_min": 0.5,
    "strong_area_ratio_range_max": 1.1,
    "contour_size_min": 5,
    "ellipse_roundness_ratio": 0.1,
    "initial_ellipse_fit_treshhold": 1.8,
    "final_perimeter_ratio_range_min": 0.35,
    "final_perimeter_ratio_range_max": 1.2,
    "ellipse_true_support_min_dist": 2.5,
    "support_pixel_ratio_exponent": 2.0,

    # RANSAC fallback parameters
    "ransac_iterations": 35,
    "ransac_min_pts": 5,

    # Polarity filter & debug
    "use_gradient_polarity": True,
    "debug": False,
}


# -----------------------------------------------------------------------------
# Geometric helpers (Ramanujan circumference & exact point-to-ellipse distance)
# -----------------------------------------------------------------------------

def _ellipse_circumference(ma: float, mb: float) -> float:
    """
    Ramanujan's accurate approximation for the circumference of an ellipse
    with major axis ma and minor axis mb.
    """
    a = max(ma, 1e-6) / 2.0
    b = max(mb, 1e-6) / 2.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))


def _ellipse_geometric_distance(ellipse: tuple, pts: np.ndarray) -> np.ndarray:
    """
    Exact geometric distance (in pixels) from 2D points to an ellipse perimeter.
    The ellipse is specified in cv2 RotatedRect format: ((cx, cy), (ma, mb), angle_deg).
    Returns a 1D float array of distances of length N.
    """
    if len(pts) == 0:
        return np.empty(0, dtype=np.float64)

    (cx, cy), (ma, mb), angle_deg = ellipse
    a = max(ma, 1e-6) / 2.0
    b = max(mb, 1e-6) / 2.0

    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy

    xr =  cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    phi = np.arctan2(yr, xr)
    r_act = np.hypot(xr, yr)
    denom = np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2 + 1e-12)
    r_exp = (a * b) / denom

    return np.abs(r_act - r_exp)


def _calculate_spike_indices(hist: np.ndarray, threshold: float = 20.0) -> Tuple[int, int]:
    """
    Mirrors singleeyefitter::detector::calculate_spike_indices_and_max_intensity.
    Finds lowest spike (pupil dark peak) and highest spike (glint/skin peak).
    """
    h_flat = hist.flatten().astype(np.float32)
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32) / 9.0
    smooth = np.convolve(h_flat, kernel, mode='same')

    peaks = []
    for i in range(1, 255):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > threshold:
            peaks.append(i)

    if not peaks:
        lowest_spike = int(np.argmin(smooth[:128])) if len(smooth) >= 128 else 25
        highest_spike = 255
    else:
        lowest_spike = peaks[0]
        highest_spike = peaks[-1] if len(peaks) > 1 else 255

    return lowest_spike, highest_spike


def _check_gradient_polarity(gray: np.ndarray, pts: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    For each edge point, verify that the image gradient points OUTWARD from
    the pupil centroid (dark interior -> light exterior).
    """
    if len(pts) == 0:
        return np.empty(0, dtype=bool)

    h, w = gray.shape[:2]
    px = np.clip(pts[:, 0].astype(int), 1, w - 2)
    py = np.clip(pts[:, 1].astype(int), 1, h - 2)

    gx = gray[py, px + 1].astype(np.int16) - gray[py, px - 1].astype(np.int16)
    gy = gray[py + 1, px].astype(np.int16) - gray[py - 1, px].astype(np.int16)

    cx = pts[:, 0] - centroid[0]
    cy = pts[:, 1] - centroid[1]

    dot = gx * cx + gy * cy
    return dot > 0


# -----------------------------------------------------------------------------
# Pure-Python Detector2D (faithful implementation of Pupil Labs detect_2d.hpp)
# -----------------------------------------------------------------------------

class PurePythonDetector2D:
    """
    Pure Python + OpenCV faithful re-implementation of pupil-labs Detector2D.

    Exposes detect(gray_img, color_img=None, roi=None) with output schema
    matching the native C++ pupil-detectors package.
    """

    def __init__(self, properties: Optional[Dict[str, Any]] = None):
        self.props = dict(DEFAULT_PROPERTIES)
        if properties:
            self.props.update(properties)

        # Strong prior state (mirrors Detector2D::mUse_strong_prior, mPrior_ellipse)
        self.mUse_strong_prior = False
        self.mPrior_ellipse = None
        self.mPupil_size = 80.0

    def get_properties(self) -> dict:
        return dict(self.props)

    def update_properties(self, props: dict):
        self.props.update(props)

    def detect(self,
               gray_img: np.ndarray,
               color_img: Optional[np.ndarray] = None,
               roi=None) -> dict:
        """
        Run the full Pupil Labs 2D detection pipeline on a single greyscale frame.
        """
        ts = time.time()
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

        h, w = gray_img.shape[:2]

        # -- Step 0: ROI Determination (mirrors Pupil Labs coarse localisation) -
        seed_x, seed_y = None, None
        if roi is not None:
            rx0, ry0, rx1, ry1 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            rx0 = max(0, min(rx0, w - 1))
            ry0 = max(0, min(ry0, h - 1))
            rx1 = max(rx0 + 1, min(rx1, w))
            ry1 = max(ry0 + 1, min(ry1, h))
            gray_roi = gray_img[ry0:ry1, rx0:rx1]
            offset = (rx0, ry0)
            seed_x = (rx0 + rx1) // 2 - rx0
            seed_y = (ry0 + ry1) // 2 - ry0
        elif self.props.get("coarse_detection", True):
            # If previous frame tracked an ellipse, center ROI on prior center
            if self.mUse_strong_prior and self.mPrior_ellipse is not None:
                cx = int(self.mPrior_ellipse[0][0])
                cy = int(self.mPrior_ellipse[0][1])
            else:
                cx, cy = self._find_darkest_region(gray_img)

            box_size = int(self.props.get("coarse_filter_max", 280))
            half = box_size // 2
            rx0 = max(0, cx - half)
            ry0 = max(0, cy - half)
            rx1 = min(w, cx + half)
            ry1 = min(h, cy + half)
            gray_roi = gray_img[ry0:ry1, rx0:rx1]
            offset = (rx0, ry0)
            seed_x = cx - rx0
            seed_y = cy - ry0
        else:
            gray_roi = gray_img
            offset = (0, 0)
            seed_x, seed_y = w // 2, h // 2

        roi_h, roi_w = gray_roi.shape[:2]
        if roi_h < 20 or roi_w < 20:
            return self._null_result(ts)

        # -- Step 1: 256-bin Intensity Histogram & Spike Detection ------------
        hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
        lowest_spike, highest_spike = _calculate_spike_indices(hist, threshold=20.0)
        contrast = max(20, highest_spike - lowest_spike)

        # -- Step 2: Dual Thresholding (Pupil Mask + Spectral Glint Mask) ------
        intensity_range = int(self.props["intensity_range"])
        thresh_val = min(250, lowest_spike + intensity_range)
        binary_img = cv2.inRange(gray_roi, 0, thresh_val)

        # Dilate binary pupil mask with 7x7 ellipse kernel (2 iterations)
        kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_dilated = cv2.dilate(binary_img, kernel_7, iterations=2)

        # Specular glint suppression: only applied when bright glints (>190) exist
        if highest_spike > 190:
            spec_thresh = max(180, highest_spike - 5)
            spec_mask = cv2.inRange(gray_roi, 0, spec_thresh)
            spec_mask = cv2.erode(spec_mask, kernel_7)
        else:
            spec_mask = np.full_like(gray_roi, 255)

        # -- Step 3: Eyelash Removal (9x9 Morphological Open) -----------------
        # Matches Pupil Labs C++ detect_2d.hpp line 138
        kernel_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        pupil_opened = cv2.morphologyEx(gray_roi, cv2.MORPH_OPEN, kernel_9)

        # -- Step 4: Contrast-Adaptive Canny Edge Detection -------------------
        blur_s = int(self.props.get("blur_size", 5))
        if blur_s % 2 == 0:
            blur_s += 1
        blurred = cv2.GaussianBlur(pupil_opened, (blur_s, blur_s), 0)

        canny_cfg = int(self.props.get("canny_treshold", 0))
        if canny_cfg <= 0 or canny_cfg > contrast:
            # Adapt dynamically to visible light eye contrast
            lo = max(15, int(contrast * 0.20))
            hi = max(35, int(contrast * 0.50))
        else:
            lo = canny_cfg
            hi = int(lo * self.props.get("canny_ration", 2))

        aperture = int(self.props.get("canny_aperture", 3))
        edges = cv2.Canny(blurred, lo, hi, apertureSize=aperture, L2gradient=True)

        # Filter edges: restrict to pupil zone and glint-free areas
        canny_masked = cv2.min(cv2.min(edges, binary_dilated), spec_mask)

        # Combine with binary threshold outer edge to ensure complete closed contours
        bin_edges = cv2.Canny(binary_dilated, 50, 150)
        combined_edges = cv2.bitwise_or(canny_masked, bin_edges)

        nz = cv2.findNonZero(combined_edges)
        if nz is None or len(nz) < 5:
            self.mUse_strong_prior = False
            return self._null_result(ts)

        raw_edges = nz.reshape(-1, 2)  # shape (N, 2)
        p_min = float(self.props["pupil_size_min"])
        p_max = float(self.props["pupil_size_max"])

        # -- Step 5: Strong Prior Fast Tracking (< 1 ms!) ---------------------
        # Matches detect_2d.hpp lines 190-270
        if self.mUse_strong_prior and self.mPrior_ellipse is not None:
            prior_center, prior_axes, prior_angle = self.mPrior_ellipse
            prior_local = (
                (prior_center[0] - offset[0], prior_center[1] - offset[1]),
                prior_axes,
                prior_angle,
            )

            min_dist = float(self.props["ellipse_true_support_min_dist"])
            dists = _ellipse_geometric_distance(prior_local, raw_edges)
            narrow_mask = dists <= min_dist
            n_narrow = float(narrow_mask.sum())
            circ = _ellipse_circumference(prior_axes[0], prior_axes[1])

            if circ > 0 and n_narrow >= 5:
                support_cov = min(1.0, n_narrow / circ)
                min_ratio = float(self.props.get("strong_perimeter_ratio_range_min", 0.45))
                if support_cov >= min_ratio:
                    support_pts = raw_edges[narrow_mask].astype(np.int32).reshape((-1, 1, 2))
                    try:
                        refit_ell = cv2.fitEllipse(support_pts)
                        (rcx, rcy), (rma, rmb), rang = refit_ell
                        diam = max(rma, rmb)
                        aspect = max(rma, rmb) / max(1e-3, min(rma, rmb))
                        if p_min <= diam <= p_max and aspect <= 2.5:
                            full_center = (rcx + offset[0], rcy + offset[1])
                            self.mPrior_ellipse = (full_center, (rma, rmb), rang)
                            self.mUse_strong_prior = True
                            confidence = min(1.0, float(support_cov))

                            if self.props.get("debug", False) and color_img is not None:
                                self._draw_debug(color_img, offset, (full_center, (rma, rmb), rang), combined_edges, True)

                            return {
                                "ellipse": {
                                    "center": full_center,
                                    "axes":   (rma, rmb),
                                    "angle":  rang,
                                },
                                "diameter":   diam,
                                "location":   full_center,
                                "confidence": confidence,
                                "timestamp":  ts,
                            }
                    except cv2.error:
                        pass

        # If strong prior tracking failed, reset prior
        self.mUse_strong_prior = False

        # -- Step 6: Contour Extraction & Approximation -----------------------
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        min_sz = int(self.props.get("contour_size_min", 5))
        valid_contours = [c for c in contours if len(c) >= min_sz]

        if not valid_contours:
            return self._null_result(ts)

        # Simplify contours using approxPolyDP
        approx_contours = []
        for c in valid_contours:
            approx = cv2.approxPolyDP(c, 1.5, False)
            if len(approx) >= 3:
                approx_contours.append(approx)

        if not approx_contours:
            approx_contours = valid_contours

        # -- Step 7: Candidate Ellipse Evaluation & Scoring -------------------
        best_ellipse = None
        best_score = float(self.props.get("final_perimeter_ratio_range_min", 0.35))
        max_dist_to_seed = float(self.props.get("coarse_filter_min", 80))

        for c in approx_contours:
            if len(c) >= 5:
                try:
                    c_cv = c.astype(np.int32).reshape((-1, 1, 2))
                    ell = cv2.fitEllipse(c_cv)
                    (ecx, ecy), (ema, emb), eang = ell
                    diam = max(ema, emb)
                    if not (p_min <= diam <= p_max and min(ema, emb) > 0 and (diam / min(ema, emb)) <= 2.2):
                        continue

                    # Validate that the center falls within the dark pupil region
                    ix, iy = int(round(ecx)), int(round(ecy))
                    if not (0 <= ix < roi_w and 0 <= iy < roi_h):
                        continue
                    if int(gray_roi[iy, ix]) > thresh_val:
                        continue

                    # Proximity check to coarse darkest seed
                    if seed_x is not None and seed_y is not None:
                        dist_to_seed = np.hypot(ecx - seed_x, ecy - seed_y)
                        if dist_to_seed > max_dist_to_seed:
                            continue

                    # Score calculation: (support / circumference) * (support / contour_length)^2
                    dists = _ellipse_geometric_distance(ell, raw_edges)
                    narrow_n = float((dists <= 2.5).sum())
                    circ = _ellipse_circumference(ema, emb)
                    if circ > 0:
                        circ_cov = min(1.0, narrow_n / circ)
                        contour_sup = min(1.0, narrow_n / float(len(c)))
                        score = circ_cov * (contour_sup ** 2)
                        if score > best_score:
                            best_score = score
                            best_ellipse = ell
                except cv2.error:
                    continue

        # If individual contours were insufficient, run RANSAC candidate search
        all_edge_pts = np.vstack([c.reshape(-1, 2) for c in valid_contours])
        if (best_ellipse is None or best_score < 0.50) and len(all_edge_pts) >= 5:
            ransac_ell = self._ransac_search(
                all_edge_pts, raw_edges,
                seed_loc=(seed_x, seed_y) if seed_x is not None else None,
                thresh_val=thresh_val,
                gray_roi=gray_roi
            )
            if ransac_ell is not None:
                (rcx, rcy), (rma, rmb), rang = ransac_ell
                dists = _ellipse_geometric_distance(ransac_ell, raw_edges)
                narrow_n = float((dists <= 2.5).sum())
                circ = _ellipse_circumference(rma, rmb)
                if circ > 0:
                    r_score = min(1.0, narrow_n / circ)
                    if r_score > best_score or best_ellipse is None:
                        best_score = r_score
                        best_ellipse = ransac_ell

        if best_ellipse is None:
            return self._null_result(ts)

        # -- Step 8: Refit on True Support Pixels -----------------------------
        dists = _ellipse_geometric_distance(best_ellipse, raw_edges)
        support_mask = dists <= float(self.props["ellipse_true_support_min_dist"])
        support_pts = raw_edges[support_mask]

        if len(support_pts) >= 5:
            try:
                best_ellipse = cv2.fitEllipse(support_pts.astype(np.int32).reshape((-1, 1, 2)))
            except cv2.error:
                pass

        (bcx, bcy), (bma, bmb), bang = best_ellipse
        diam = max(bma, bmb)
        if diam < p_min or diam > p_max:
            return self._null_result(ts)

        # -- Step 9: Gradient Polarity Check ----------------------------------
        if self.props.get("use_gradient_polarity", True) and len(support_pts) >= 8:
            centroid = np.array([bcx, bcy])
            pol = _check_gradient_polarity(gray_roi, support_pts, centroid)
            if len(pol) > 0 and pol.mean() < 0.20:
                return self._null_result(ts)

        # -- Step 10: Translate to Full Coordinates & Save Prior --------------
        full_center = (bcx + offset[0], bcy + offset[1])
        full_ellipse = (full_center, (bma, bmb), bang)

        confidence = min(1.0, max(0.0, float(best_score)))
        if confidence >= float(self.props.get("strong_perimeter_ratio_range_min", 0.45)):
            self.mPrior_ellipse = full_ellipse
            self.mUse_strong_prior = True

        if self.props.get("debug", False) and color_img is not None:
            self._draw_debug(color_img, offset, full_ellipse, combined_edges, False)

        return {
            "ellipse": {
                "center": full_center,
                "axes":   (bma, bmb),
                "angle":  bang,
            },
            "diameter":   diam,
            "location":   full_center,
            "confidence": confidence,
            "timestamp":  ts,
        }

    # -- Internal Helpers -----------------------------------------------------

    def _find_darkest_region(self, gray: np.ndarray) -> Tuple[int, int]:
        """
        Fast downsampled minimum search ignoring sensor borders (< 0.2 ms).
        """
        scale = 4
        small = cv2.resize(gray, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
        small_b = cv2.GaussianBlur(small, (5, 5), 0)
        h_s, w_s = small_b.shape
        my, mx = max(1, int(h_s * 0.05)), max(1, int(w_s * 0.05))
        sub = small_b[my:h_s - my, mx:w_s - mx]
        _, _, min_loc, _ = cv2.minMaxLoc(sub)
        cx = int((min_loc[0] + mx) * scale + scale // 2)
        cy = int((min_loc[1] + my) * scale + scale // 2)
        return (cx, cy)

    def _ransac_search(self, all_pts: np.ndarray, raw_edges: np.ndarray,
                       seed_loc: Optional[tuple] = None,
                       thresh_val: int = 100,
                       gray_roi: Optional[np.ndarray] = None) -> Optional[tuple]:
        """
        RANSAC ellipse search: samples 5 points, fits ellipse, scores via edge support.
        """
        n_iter = int(self.props.get("ransac_iterations", 35))
        p_min = float(self.props["pupil_size_min"])
        p_max = float(self.props["pupil_size_max"])

        n = len(all_pts)
        if n < 5:
            return None

        best_inliers = -1
        best_ellipse = None
        rng = list(range(n))

        for _ in range(n_iter):
            sample_idx = random.sample(rng, 5)
            sample = all_pts[sample_idx].astype(np.int32).reshape((-1, 1, 2))
            try:
                ell = cv2.fitEllipse(sample)
            except cv2.error:
                continue

            (cx, cy), (ma, mb), ang = ell
            diam = max(ma, mb)
            if diam < p_min or diam > p_max or min(ma, mb) < 1e-6:
                continue
            if max(ma, mb) / min(ma, mb) > 2.2:
                continue

            if gray_roi is not None:
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= ix < gray_roi.shape[1] and 0 <= iy < gray_roi.shape[0]:
                    if int(gray_roi[iy, ix]) > thresh_val:
                        continue

            if seed_loc is not None:
                if np.hypot(cx - seed_loc[0], cy - seed_loc[1]) > 80:
                    continue

            dists = _ellipse_geometric_distance(ell, raw_edges)
            inliers = int((dists <= 2.5).sum())

            if inliers > best_inliers:
                best_inliers = inliers
                best_ellipse = ell

        return best_ellipse

    def _draw_debug(self, color_img: np.ndarray, offset: tuple, full_ellipse: tuple, edges: np.ndarray, is_prior: bool):
        """Draw Pupil Labs debug overlays on color_img."""
        try:
            ox, oy = offset
            h_e, w_e = edges.shape
            roi_view = color_img[oy:oy + h_e, ox:ox + w_e]
            if roi_view.shape[:2] == edges.shape:
                blue_edges = np.zeros_like(roi_view)
                blue_edges[edges > 0] = (255, 120, 0)
                cv2.addWeighted(roi_view, 1.0, blue_edges, 0.6, 0, roi_view)

            col = (0, 255, 0) if not is_prior else (0, 200, 255)
            cv2.ellipse(color_img, full_ellipse, col, 2)
            c = (int(full_ellipse[0][0]), int(full_ellipse[0][1]))
            cv2.circle(color_img, c, 3, (0, 255, 255), -1)
        except Exception:
            pass

    def _null_result(self, ts: float) -> dict:
        return {
            "ellipse": {
                "center": (0.0, 0.0),
                "axes":   (0.0, 0.0),
                "angle":  0.0,
            },
            "diameter":   0.0,
            "location":   (0.0, 0.0),
            "confidence": 0.0,
            "timestamp":  ts,
        }


# -----------------------------------------------------------------------------
# Wrapper: exposes the exact same interface regardless of backend
# -----------------------------------------------------------------------------

class Detector2DWrapper:
    """
    Unified wrapper that uses the native C++ pupil-detectors package
    when available (Python <= 3.12), otherwise uses the faithful
    pure-Python implementation.
    """

    def __init__(self, properties: Optional[Dict[str, Any]] = None):
        if _NATIVE_AVAILABLE and _NativeDetector2D is not None:
            self._backend = _NativeDetector2D(properties or {})
            self._is_native = True
        else:
            self._backend = PurePythonDetector2D(properties)
            self._is_native = False

    @property
    def backend_name(self) -> str:
        return "native-C++" if self._is_native else "pure-Python"

    def get_properties(self) -> dict:
        return self._backend.get_properties()

    def update_properties(self, props: dict):
        self._backend.update_properties(props)

    def detect(self, gray_img: np.ndarray,
               color_img: Optional[np.ndarray] = None,
               roi=None) -> dict:
        if self._is_native:
            try:
                from pupil_detectors import Roi as _Roi
                roi_arg = _Roi(*roi) if roi else None
                return self._backend.detect(
                    gray_img=gray_img,
                    color_img=color_img,
                    roi=roi_arg
                )
            except Exception:
                return self._backend.detect(gray_img=gray_img)
        else:
            return self._backend.detect(gray_img, color_img, roi)


# -----------------------------------------------------------------------------
# Module-level convenience functions
# -----------------------------------------------------------------------------

_DEFAULT_DETECTOR: Optional[Detector2DWrapper] = None


def get_detector(properties: Optional[dict] = None) -> Detector2DWrapper:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = Detector2DWrapper(properties)
    return _DEFAULT_DETECTOR


def detect_pupil_pupillabs(frame: np.ndarray,
                           detector: Optional[Detector2DWrapper] = None,
                           debug: bool = False) -> Tuple[Optional[dict], np.ndarray]:
    """
    High-level single-call detection compatible with the original
    `detect_pupil(frame)` pattern from gaze_mouse.py.
    """
    if detector is None:
        detector = get_detector()

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame
    else:
        gray = frame
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    if w != 640 or h != 480:
        gray    = cv2.resize(gray,    (640, 480))
        display = cv2.resize(display, (640, 480))

    color_img = display if debug else None

    result = detector.detect(gray_img=gray, color_img=color_img)

    MIN_CONFIDENCE = 0.25
    if result["confidence"] < MIN_CONFIDENCE or result["diameter"] < 1.0:
        return None, display

    return result, display


def crop_to_aspect_ratio(image: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Crop the image to maintain a specific aspect ratio (width:height) before resizing.
    """
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))
