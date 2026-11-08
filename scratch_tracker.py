"""
scratch/test_ultimate_tracker.py
Test the combined PuRe + PupilLabs Ramanujan Stabilized SOTA Engine.
"""

import math
import time
import cv2
import numpy as np

def _ellipse_circumference(ma: float, mb: float) -> float:
    a = max(ma, 1e-6) / 2.0
    b = max(mb, 1e-6) / 2.0
    h = ((a - b) / (a + b)) ** 2
    return float(np.pi * (a + b) * (1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h))))

def _ellipse_geometric_distance(ellipse: tuple, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.empty(0, dtype=np.float64)
    (cx, cy), (ma, mb), angle_deg = ellipse
    ma = max(float(ma), 1e-3)
    mb = max(float(mb), 1e-3)
    a, b = ma / 2.0, mb / 2.0
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    x = pts[:, 0].astype(np.float64) - cx
    y = pts[:, 1].astype(np.float64) - cy
    x_rot =  x * cos_a + y * sin_a
    y_rot = -x * sin_a + y * cos_a
    r_pt = np.hypot(x_rot, y_rot)
    r_pt = np.maximum(r_pt, 1e-6)
    theta = np.arctan2(y_rot, x_rot)
    r_ell = (a * b) / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    return np.abs(r_pt - r_ell)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.06, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(rate, 1e-4)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float = None) -> float:
        if t is None:
            t = time.time()
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        rate = 1.0 / max(t - self.t_prev, 1e-5)
        dx = (x - self.x_prev) * rate
        edx = self.dx_prev + self._alpha(rate, self.d_cutoff) * (dx - self.dx_prev)
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        x_filtered = self.x_prev + self._alpha(rate, cutoff) * (x - self.x_prev)
        self.x_prev = x_filtered
        self.t_prev = t
        return x_filtered

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class SOTAStabilizedTracker:
    """
    SOTA Stabilized Pupil Tracker:
    Combines PuRe's curvature discrimination and radial contrast validation
    with Pupil Labs' Ramanujan perimeter edge-support tracking and One-Euro filtering.
    """
    def __init__(self):
        self.filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.08)
        self.filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.08)
        self.prior_ellipse = None
        self.use_prior = False
        self.smooth_axes = None
        self.smooth_angle = 0.0

    def detect(self, frame_640x480: np.ndarray):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        now_t = time.time()
        
        # 1. Determine ROI
        if self.use_prior and self.prior_ellipse is not None:
            (pcx, pcy), (pma, pmb), pang = self.prior_ellipse
            half = max(75, int(max(pma, pmb) * 1.3))
            rx0 = max(0, int(pcx - half))
            ry0 = max(0, int(pcy - half))
            rx1 = min(w, int(pcx + half))
            ry1 = min(h, int(pcy + half))
        else:
            # Full coarse search
            small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            small_b = cv2.GaussianBlur(small, (7, 7), 0)
            _, _, min_loc, _ = cv2.minMaxLoc(small_b)
            cx, cy = min_loc[0] * 4, min_loc[1] * 4
            rx0 = max(0, cx - 100)
            ry0 = max(0, cy - 80)
            rx1 = min(w, cx + 100)
            ry1 = min(h, cy + 80)

        roi = gray[ry0:ry1, rx0:rx1]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 25 or roi_w < 25:
            self.use_prior = False
            dt = (time.perf_counter() - t0) * 1000.0
            return None, dt, 0.0, "SEARCHING"

        # 2. Darkest pocket thresholding + Glint removal
        min_v, max_v, min_l, _ = cv2.minMaxLoc(roi)
        # Pupil threshold: dark well
        pupil_thresh = min(min_v + 35, int((min_v + np.mean(roi)) * 0.55))
        bin_pupil = cv2.inRange(roi, 0, pupil_thresh)
        bin_dilated = cv2.dilate(bin_pupil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
        
        # Glint mask
        if max_v > 210:
            glint_mask = cv2.inRange(roi, 0, max(190, int(max_v - 15)))
        else:
            glint_mask = np.full_like(roi, 255)
            
        # Morphological opening to disconnect eyelashes
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        blurred = cv2.GaussianBlur(opened, (5, 5), 1.2)
        
        # Contrast-adaptive Canny
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        hi_th = max(25, int(np.percentile(mag, 70)))
        lo_th = max(10, int(hi_th * 0.40))
        edges = cv2.Canny(blurred, lo_th, hi_th, L2gradient=True)
        
        # Keep edges only in pupil zone & glint free
        pupil_edges = cv2.min(cv2.min(edges, bin_dilated), glint_mask)
        bin_edge = cv2.Canny(bin_dilated, 50, 150)
        combined_edges = cv2.bitwise_or(pupil_edges, bin_edge)
        
        nz = cv2.findNonZero(combined_edges)
        if nz is None or len(nz) < 10:
            self.use_prior = False
            dt = (time.perf_counter() - t0) * 1000.0
            return None, dt, 0.0, "SEARCHING"

        raw_edges = nz.reshape(-1, 2)
        
        # --- FAST STRONG-PRIOR RE-FIT (< 1ms) ---
        if self.use_prior and self.prior_ellipse is not None:
            prior_local = ((pcx - rx0, pcy - ry0), (pma, pmb), pang)
            dists = _ellipse_geometric_distance(prior_local, raw_edges)
            narrow_mask = dists <= 3.0
            circ = _ellipse_circumference(pma, pmb)
            n_support = float(narrow_mask.sum())
            
            if circ > 0 and n_support >= 10:
                coverage = min(1.0, n_support / circ)
                if coverage >= 0.38:
                    support_pts = raw_edges[narrow_mask]
                    try:
                        refit = cv2.fitEllipse(support_pts.astype(np.int32).reshape(-1, 1, 2))
                        (rcx, rcy), (rma, rmb), rang = refit
                        diam = max(rma, rmb)
                        aspect = diam / max(1e-3, min(rma, rmb))
                        if 15 <= diam <= 220 and aspect <= 2.2:
                            full_cx = rcx + rx0
                            full_cy = rcy + ry0
                            
                            # Filter center with One-Euro adaptive filter
                            sx = self.filter_x.filter(full_cx, now_t)
                            sy = self.filter_y.filter(full_cy, now_t)
                            
                            # Smooth axes & angle (keeps ellipse visually fixed and stable)
                            smooth_ma = pma * 0.82 + rma * 0.18
                            smooth_mb = pmb * 0.82 + rmb * 0.18
                            smooth_ang = pang * 0.82 + rang * 0.18
                            
                            self.prior_ellipse = ((sx, sy), (smooth_ma, smooth_mb), smooth_ang)
                            self.use_prior = True
                            dt = (time.perf_counter() - t0) * 1000.0
                            return self.prior_ellipse, dt, float(coverage), "PuReST_LOCKED"
                    except cv2.error:
                        pass

        # --- FULL CONTOUR & RAMANUJAN SCORING (FALLBACK) ---
        self.use_prior = False
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        best_ell = None
        best_score = 0.30
        
        for c in contours:
            if len(c) >= 5:
                # Curvature check: reject straight lines (eyelashes/eyelids)
                rect = cv2.minAreaRect(c)
                max_d = max(rect[1])
                min_d = min(rect[1])
                if max_d > 0 and (min_d / max_d) < 0.16:
                    continue  # straight line -> reject!
                    
                try:
                    c_cv = c.astype(np.int32).reshape((-1, 1, 2))
                    ell = cv2.fitEllipse(c_cv)
                    (ecx, ecy), (ema, emb), eang = ell
                    diam = max(ema, emb)
                    aspect = diam / max(1e-3, min(ema, emb))
                    if not (15 <= diam <= 220 and aspect <= 2.2):
                        continue
                        
                    # Center must be dark
                    ix, iy = int(round(ecx)), int(round(ecy))
                    if not (0 <= ix < roi_w and 0 <= iy < roi_h):
                        continue
                    if int(roi[iy, ix]) > pupil_thresh + 15:
                        continue
                        
                    # Ramanujan support score
                    dists = _ellipse_geometric_distance(ell, raw_edges)
                    narrow_n = float((dists <= 2.8).sum())
                    circ = _ellipse_circumference(ema, emb)
                    if circ > 0:
                        circ_cov = min(1.0, narrow_n / circ)
                        cnt_cov = min(1.0, narrow_n / float(len(c)))
                        score = circ_cov * (cnt_cov ** 1.5)
                        if score > best_score:
                            best_score = score
                            best_ell = ell
                except cv2.error:
                    continue

        if best_ell is not None:
            (ecx, ecy), (ema, emb), eang = best_ell
            full_cx = ecx + rx0
            full_cy = ecy + ry0
            
            sx = self.filter_x.filter(full_cx, now_t)
            sy = self.filter_y.filter(full_cy, now_t)
            
            if self.smooth_axes is not None:
                smooth_ma = self.smooth_axes[0] * 0.70 + ema * 0.30
                smooth_mb = self.smooth_axes[1] * 0.70 + emb * 0.30
                smooth_ang = self.smooth_angle * 0.70 + eang * 0.30
            else:
                smooth_ma, smooth_mb, smooth_ang = ema, emb, eang
                
            self.smooth_axes = (smooth_ma, smooth_mb)
            self.smooth_angle = smooth_ang
            self.prior_ellipse = ((sx, sy), (smooth_ma, smooth_mb), smooth_ang)
            self.use_prior = True
            dt = (time.perf_counter() - t0) * 1000.0
            return self.prior_ellipse, dt, float(best_score), "PuReST_ACQUIRED"

        dt = (time.perf_counter() - t0) * 1000.0
        return None, dt, 0.0, "SEARCHING"

print("SOTAStabilizedTracker defined.")
