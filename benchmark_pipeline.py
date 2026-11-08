#!/usr/bin/env python3
"""
benchmark_pipeline.py
=====================
Core Data Engine and Pipeline Coordinator for the 4-Algorithm Benchmark Suite.

Manages:
  1. FourWayPipeline: Runs Orlosky, Pupil Labs, Hybrid Fusion, and PuReST on the
     caged camera frame (with Eyeball Boundary Circle).
  2. PolynomialGazeCalibrator: Independent bivariate degree-2 polynomial screen mapping
     for each algorithm.
  3. BenchmarkSession: Records all telemetry across alignment, calibration, static testing,
     and dynamic pursuit testing.
"""

import os
import sys
import time
import math
import json
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

# Ensure local directory is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# -----------------------------------------------------------------------------
# Detectors Import
# -----------------------------------------------------------------------------
from OrloskyPupilDetector import (
    crop_to_aspect_ratio,
    get_darkest_area,
    apply_binary_threshold,
    mask_outside_square,
    process_frames,
)
from PupilLabsDetector import Detector2DWrapper, detect_pupil_pupillabs
from AdvancedPupilDetectors import HybridFusionDetector, PuReSTPupilDetector


# -----------------------------------------------------------------------------
# Algorithm Names & Colors
# -----------------------------------------------------------------------------
ALG_ORLOSKY = "Orlosky"
ALG_PL      = "Pupil Labs"
ALG_HYBRID  = "Hybrid Fusion"
ALG_PUREST  = "PuReST"

ALGORITHMS = [ALG_ORLOSKY, ALG_PL, ALG_HYBRID, ALG_PUREST]

# Visual color mapping (BGR convention for OpenCV)
COLOR_ORLOSKY = (30, 190, 255)   # Amber / Gold
COLOR_PL      = (120, 240, 50)   # Vibrant Emerald / Neon Green
COLOR_HYBRID  = (255, 220, 0)    # Cyan / Sky Blue
COLOR_PUREST  = (210, 80, 240)   # Magenta / Electric Pink

ALG_COLORS = {
    ALG_ORLOSKY: COLOR_ORLOSKY,
    ALG_PL:      COLOR_PL,
    ALG_HYBRID:  COLOR_HYBRID,
    ALG_PUREST:  COLOR_PUREST,
}

# Hex colors for Matplotlib & Markdown reports
ALG_HEX = {
    ALG_ORLOSKY: "#FFBE1E",
    ALG_PL:      "#32F078",
    ALG_HYBRID:  "#00DCFF",
    ALG_PUREST:  "#F050D2",
}


# -----------------------------------------------------------------------------
# 4-Way Pipeline Runner
# -----------------------------------------------------------------------------
class FourWayPipeline:
    def __init__(self):
        self.pl_detector = Detector2DWrapper()
        self.hybrid_detector = HybridFusionDetector()
        self.purest_tracker = PuReSTPupilDetector()

    def run_orlosky(self, frame_640x480: np.ndarray) -> Tuple[Optional[Tuple], float]:
        t0 = time.perf_counter()
        darkest = get_darkest_area(frame_640x480)
        if darkest is None:
            return None, (time.perf_counter() - t0) * 1000.0

        gray = cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2GRAY)
        dpv = gray[darkest[1], darkest[0]]

        t_strict  = mask_outside_square(apply_binary_threshold(gray, dpv,  5), darkest, 250)
        t_medium  = mask_outside_square(apply_binary_threshold(gray, dpv, 15), darkest, 250)
        t_relaxed = mask_outside_square(apply_binary_threshold(gray, dpv, 25), darkest, 250)

        result = process_frames(t_strict, t_medium, t_relaxed, frame_640x480, gray, darkest, False, False)
        dt = (time.perf_counter() - t0) * 1000.0

        if result is None or result == ((0, 0), (0, 0), 0):
            return None, dt
        return result, dt

    def run_pupil_labs(self, frame_640x480: np.ndarray) -> Tuple[Optional[Tuple], float, float]:
        t0 = time.perf_counter()
        result, _ = detect_pupil_pupillabs(frame_640x480, detector=self.pl_detector, debug=False)
        dt = (time.perf_counter() - t0) * 1000.0

        if result is None:
            return None, dt, 0.0

        ell = result["ellipse"]
        rotated_rect = (
            (float(ell["center"][0]), float(ell["center"][1])),
            (float(ell["axes"][0]),   float(ell["axes"][1])),
            float(ell["angle"]),
        )
        conf = float(result.get("confidence", 1.0))
        return rotated_rect, dt, conf

    def run_hybrid(self, frame_640x480: np.ndarray) -> Tuple[Optional[Tuple], float, float, str]:
        return self.hybrid_detector.detect(frame_640x480)

    def run_purest(self, frame_640x480: np.ndarray) -> Tuple[Optional[Tuple], float, float, str]:
        return self.purest_tracker.detect(frame_640x480)

    def process_all(self, caged_frame: np.ndarray, circle_cx: int, circle_cy: int, circle_r: int) -> Dict[str, Dict[str, Any]]:
        """
        Runs all 4 algorithms concurrently on the identical caged frame and validates against the circle boundary.
        """
        r_or, lat_or = self.run_orlosky(caged_frame)
        r_pl, lat_pl, conf_pl = self.run_pupil_labs(caged_frame)
        r_hy, lat_hy, conf_hy, src_hy = self.run_hybrid(caged_frame)
        r_sk, lat_sk, conf_sk, src_sk = self.run_purest(caged_frame)

        # Boundary circle containment check
        def inside_circle(ell):
            if not ell:
                return False
            px, py = ell[0][0], ell[0][1]
            return math.hypot(px - circle_cx, py - circle_cy) <= circle_r

        if not inside_circle(r_or):
            r_or = None
        if not inside_circle(r_pl):
            r_pl = None
        if not inside_circle(r_hy):
            r_hy = None
        if not inside_circle(r_sk):
            r_sk = None

        return {
            ALG_ORLOSKY: {
                "detected": r_or is not None,
                "ellipse": r_or,
                "center": (float(r_or[0][0]), float(r_or[0][1])) if r_or else None,
                "latency_ms": lat_or,
                "confidence": 1.0 if r_or else 0.0,
                "engine": "Multi-Threshold",
            },
            ALG_PL: {
                "detected": r_pl is not None,
                "ellipse": r_pl,
                "center": (float(r_pl[0][0]), float(r_pl[0][1])) if r_pl else None,
                "latency_ms": lat_pl,
                "confidence": conf_pl if r_pl else 0.0,
                "engine": "Canny + Ramanujan",
            },
            ALG_HYBRID: {
                "detected": r_hy is not None,
                "ellipse": r_hy,
                "center": (float(r_hy[0][0]), float(r_hy[0][1])) if r_hy else None,
                "latency_ms": lat_hy,
                "confidence": conf_hy if r_hy else 0.0,
                "engine": src_hy,
            },
            ALG_PUREST: {
                "detected": r_sk is not None,
                "ellipse": r_sk,
                "center": (float(r_sk[0][0]), float(r_sk[0][1])) if r_sk else None,
                "latency_ms": lat_sk,
                "confidence": conf_sk if r_sk else 0.0,
                "engine": src_sk,
            },
        }


# -----------------------------------------------------------------------------
# 1-Euro Adaptive Low-Pass Filter (Casiez, Roussel, & Vogel, CHI 2012)
# -----------------------------------------------------------------------------
class OneEuroFilter:
    """
    1-Euro adaptive low-pass filter for real-time biological gaze signals.
    Dynamically modulates cutoff frequency between fixation (zero jitter) and saccades (zero lag).
    """
    def __init__(self, t0: float, x0: float, dx0: float = 0.0, min_cutoff: float = 0.85, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        edx = self.dx_prev + a_d * (dx - self.dx_prev)
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(cutoff, dt)
        filtered_x = self.x_prev + a * (x - self.x_prev)
        self.x_prev = filtered_x
        return filtered_x

    def reset(self, t0: float, x0: float):
        self.t_prev = float(t0)
        self.x_prev = float(x0)
        self.dx_prev = 0.0


class PointOneEuroFilter:
    """Applies independent 1-Euro adaptive filtering to 2D (x, y) gaze coordinates."""
    def __init__(self, min_cutoff: float = 0.85, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fx: Optional[OneEuroFilter] = None
        self.fy: Optional[OneEuroFilter] = None

    def filter(self, x: float, y: float, t: float) -> Tuple[float, float]:
        if self.fx is None or self.fy is None:
            self.fx = OneEuroFilter(t, x, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            self.fy = OneEuroFilter(t, y, min_cutoff=self.min_cutoff, beta=self.beta, d_cutoff=self.d_cutoff)
            return x, y

        if t - self.fx.t_prev > 0.35:
            self.fx.reset(t, x)
            self.fy.reset(t, y)
            return x, y

        return self.fx.filter(x, t), self.fy.filter(y, t)

    def reset(self):
        self.fx = None
        self.fy = None


# -----------------------------------------------------------------------------
# Bivariate Degree-2 Anisotropic Polynomial Calibrator
# -----------------------------------------------------------------------------
class PolynomialGazeCalibrator:
    """
    Fits polynomial mapping pupil (px, py) -> screen (sx, sy).
    Supports two calibration paradigms:
      1. MODE_SOTA (Default): 7-Term Anisotropic basis [1, u, v, uv, u^2, v^2, (uv)^2] with Ridge regularization.
      2. MODE_CLASSIC: 6-Term Bivariate Degree-2 polynomial [1, px, py, px*py, px^2, py^2] with unweighted OLS.
    """
    MODE_SOTA = "SOTA"
    MODE_CLASSIC = "CLASSIC"

    def __init__(self, alg_name: str, screen_w: int, screen_h: int, mode: str = "SOTA"):
        self.alg_name = alg_name
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.mode = mode.upper()
        self.src_pts = []  # list of (px, py)
        self.dst_pts = []  # list of (sx, sy)
        self.cx = None
        self.cy = None
        self.is_calibrated = False
        self.train_rms_px = 0.0

    def _feat(self, px: float, py: float) -> np.ndarray:
        if self.mode == self.MODE_CLASSIC:
            return np.array([1.0, px, py, px * py, px * px, py * py], dtype=np.float64)
        else:
            u = px / 100.0
            v = py / 100.0
            return np.array([1.0, u, v, u * v, u * u, v * v, (u * v) ** 2], dtype=np.float64)

    def add_calibration_sample(self, pupil_x: float, pupil_y: float, screen_x: float, screen_y: float):
        self.src_pts.append([float(pupil_x), float(pupil_y)])
        self.dst_pts.append([float(screen_x), float(screen_y)])

    def fit(self) -> bool:
        min_pts = 6 if self.mode == self.MODE_CLASSIC else 7
        if len(self.src_pts) < min_pts:
            self.is_calibrated = False
            return False

        src = np.array(self.src_pts, dtype=np.float64)
        dst = np.array(self.dst_pts, dtype=np.float64)
        X = np.array([self._feat(p[0], p[1]) for p in src])

        try:
            if self.mode == self.MODE_CLASSIC:
                self.cx, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
                self.cy, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)
            else:
                XT_X = X.T @ X + 1e-4 * np.eye(X.shape[1])
                self.cx = np.linalg.solve(XT_X, X.T @ dst[:, 0])
                self.cy = np.linalg.solve(XT_X, X.T @ dst[:, 1])

            pred_x = X @ self.cx
            pred_y = X @ self.cy
            err_sq = (pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2
            self.train_rms_px = float(np.sqrt(np.mean(err_sq)))
            self.is_calibrated = True
            return True
        except Exception as e:
            print(f"[Calibrator:{self.alg_name}] Fit error: {e}")
            self.is_calibrated = False
            return False

    def predict(self, pupil_x: float, pupil_y: float) -> Optional[Tuple[float, float]]:
        if not self.is_calibrated or self.cx is None or self.cy is None:
            return None
        feat = self._feat(pupil_x, pupil_y)
        sx = float(np.dot(feat, self.cx))
        sy = float(np.dot(feat, self.cy))
        # Keep predictions within reasonable screen bounds +/- 10%
        margin = 150
        sx = max(-margin, min(self.screen_w + margin, sx))
        sy = max(-margin, min(self.screen_h + margin, sy))
        return sx, sy


# -----------------------------------------------------------------------------
# Outlier Filter (Tukey IQR)
# -----------------------------------------------------------------------------
def filter_tukey_iqr(pts: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Rejects blink and saccadic outliers using Tukey Interquartile Range."""
    if len(pts) < 5:
        return pts
    q25 = np.percentile(pts, 25, axis=0)
    q75 = np.percentile(pts, 75, axis=0)
    iqr = q75 - q25
    lower = q25 - k * iqr
    upper = q75 + k * iqr
    mask = (pts[:, 0] >= lower[0]) & (pts[:, 0] <= upper[0]) & \
           (pts[:, 1] >= lower[1]) & (pts[:, 1] <= upper[1])
    filtered = pts[mask]
    return filtered if len(filtered) >= 4 else pts


# -----------------------------------------------------------------------------
# Benchmark Session Data Storage
# -----------------------------------------------------------------------------
class BenchmarkSession:
    """Holds all empirical data gathered throughout the 4 benchmark phases."""
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.date_str = time.strftime("%B %d, %Y - %H:%M:%S")

        # Eyeball circle parameters
        self.circle_x = 320
        self.circle_y = 240
        self.circle_r = 110

        # Phase 1: Calibration Point Data
        # Format: {alg: {'train_rms': float, 'points_captured': int}}
        self.calibration_summary = {}

        # Phase 2: Static Target Fixation Test Data
        # List of records: {
        #   'target_idx': int,
        #   'target_xy': (tx, ty),
        #   'predictions': {alg: {'gaze_xy': (gx, gy), 'error_px': float, 'error_deg': float, 'latency_ms': float, 'jitter_rms_px': float, 'samples': int}}
        # }
        self.static_test_records: List[Dict[str, Any]] = []

        # Phase 3: Dynamic Smooth Pursuit Tracking Data
        # List of records per frame: {
        #   't': float,
        #   'target_xy': (tx, ty),
        #   'predictions': {alg: {'gaze_xy': (gx, gy), 'tracked': bool, 'latency_ms': float, 'error_px': float}}
        # }
        self.dynamic_frames: List[Dict[str, Any]] = []

        # Computed Benchmark Summary
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.rankings: List[Tuple[str, float]] = []

    def save_json(self, filepath: str):
        payload = {
            "timestamp": self.timestamp,
            "date": self.date_str,
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "circle_config": {
                "cx": self.circle_x,
                "cy": self.circle_y,
                "radius": self.circle_r,
            },
            "calibration_summary": self.calibration_summary,
            "static_test_records": self.static_test_records,
            "dynamic_frames_count": len(self.dynamic_frames),
            "metrics": self.metrics,
            "rankings": self.rankings,
        }
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[BenchmarkSession] Saved raw telemetry to: {filepath}")
