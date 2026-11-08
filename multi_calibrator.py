#!/usr/bin/env python3
"""
multi_calibrator.py
===================
100% Faithful Multi-Algorithm Replication of PrecisionCalibrator.

Extends the medical-grade Tobii / LC Technologies calibration pipeline
(from Eye Tracking Module/calibration_engine.py) to calibrate all 4 algorithms
simultaneously on the exact same visual stimuli and identical video frames:

Phases (Identical to 1st Calibration):
  1. WARMUP      (4.0 s)  — Lissajous moving dot with glowing aura & trail.
  2. APPROACHING (2.0 s)  — Target dot at grid position; outer ring shrinks.
  3. COLLECTING  (1.5 s)  — Cyan countdown arc sweeps 0 -> 360° CW.
                           Simultaneously records pupil coordinates for all 4 algorithms.
                           Tukey IQR outlier rejection applied per algorithm.
  4. FLASHING    (0.55 s) — Quality feedback flash ring (green = good, amber = retry).
  5. Repeat 2-4 for all 9 grid points.
  6. TRACKING    (8 paths!) — Continuous smooth pursuit across 8 paths:
                           Top L->R, Top R->L, Mid L->R, Mid R->L, Bot L->R, Bot R->L, Diag 1, Diag 2.
                           Collects hundreds of continuous (px, py) -> (tx, ty) tracking pairs.
  7. FIT POLYNOMIALS     — Solves independent least-squares Bivariate Degree-2 Polynomials
                           for all 4 algorithms using all static + dynamic tracking points.
  8. VALIDATING  (4 dots) — Validation crosshairs at unseen coordinates.
                           Measures real validation accuracy for all 4 models.
  9. DONE        (2.5 s)  — Displays validation summary screen before advancing to Level 2.
"""

import math
import time
import json
import threading
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from benchmark_pipeline import ALGORITHMS


class MultiAlgorithmPrecisionCalibrator:
    # ── 9-point grid with 8 % margin (100% identical to 1st calibration) ─────
    GRID = [
        (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
        (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
        (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
    ]

    # ── Smooth Pursuit Tracking paths (8 paths! Identical to 1st calibration) ──
    TRACKING_PATHS = [
        ((0.10, 0.10), (0.90, 0.10)),  # Top L->R
        ((0.90, 0.10), (0.10, 0.10)),  # Top R->L
        ((0.10, 0.50), (0.90, 0.50)),  # Mid L->R
        ((0.90, 0.50), (0.10, 0.50)),  # Mid R->L
        ((0.10, 0.90), (0.90, 0.90)),  # Bot L->R
        ((0.90, 0.90), (0.10, 0.90)),  # Bot R->L
        ((0.10, 0.10), (0.90, 0.90)),  # Diag 1
        ((0.90, 0.10), (0.10, 0.90)),  # Diag 2
    ]

    # ── Validation Grid at positions not seen during calibration ─────────────
    VAL_GRID = [
        (0.25, 0.25), (0.75, 0.25),
        (0.25, 0.75), (0.75, 0.75),
    ]

    # ── Phase durations (100% identical to 1st calibration) ──────────────────
    WARMUP_SECS         = 4.0   # Lissajous warm-up
    APPROACH_SECS       = 2.0   # ring shrinks; eye moves to target
    COLLECT_SECS        = 1.5   # cyan arc sweeps 360°; main sample collection
    FLASH_SECS          = 0.55  # quality feedback flash
    TRACK_APPROACH_SECS = 1.5   # eye moves to the start of the path
    TRACKING_SECS       = 3.5   # duration for each smooth pursuit path
    VAL_APPROACH_S      = 1.2   # approach to each validation dot
    VAL_COLLECT_S       = 1.8   # collect at each validation dot

    # ── Quality thresholds ───────────────────────────────────────────────────
    MIN_SAMPLES    = 12
    QUALITY_THRESH = 0.28
    MAX_RETRIES    = 1

    # ── State names ──────────────────────────────────────────────────────────
    S_IDLE     = 'IDLE'
    S_WARMUP   = 'WARMUP'
    S_APPROACH = 'APPROACHING'
    S_COLLECT  = 'COLLECTING'
    S_FLASH    = 'FLASHING'
    S_TRACKING = 'TRACKING'
    S_VAL      = 'VALIDATING'
    S_DONE     = 'DONE'

    MODE_SOTA = 'SOTA'
    MODE_CLASSIC = 'CLASSIC'

    def __init__(self, screen_w: int, screen_h: int, mode: str = 'SOTA'):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.mode = mode.upper()
        self.grid_px = [(int(fx * screen_w), int(fy * screen_h)) for fx, fy in self.GRID]
        self.val_px  = [(int(fx * screen_w), int(fy * screen_h)) for fx, fy in self.VAL_GRID]

        self._lock = threading.Lock()
        self._state = self.S_IDLE
        self._pub: dict = {}

        # Per-algorithm polynomial coefficients (cx, cy)
        self.cx: Dict[str, Optional[np.ndarray]] = {alg: None for alg in ALGORITHMS}
        self.cy: Dict[str, Optional[np.ndarray]] = {alg: None for alg in ALGORITHMS}

        # Data collection buffers per algorithm
        self._samples: Dict[str, List[Tuple[float, float]]] = {alg: [] for alg in ALGORITHMS}
        self._src_pts: Dict[str, List[List[float]]] = {alg: [] for alg in ALGORITHMS}
        self._dst_pts: Dict[str, List[List[float]]] = {alg: [] for alg in ALGORITHMS}
        self._weights: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
        self._quality: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
        self._val_errors: Dict[str, List[float]] = {alg: [] for alg in ALGORITHMS}
        self._val_samples: Dict[str, List[Tuple[float, float]]] = {alg: [] for alg in ALGORITHMS}

        self._reset_internals()

    def start(self, t: Optional[float] = None):
        """Begin the full 100% calibration sequence."""
        self._reset_internals()
        for alg in ALGORITHMS:
            self.cx[alg] = None
            self.cy[alg] = None
        self._transition(self.S_WARMUP, t)
        print("[MultiCalibrator] 100% Precision Calibration Started across ALL 4 algorithms!")

    def reset(self):
        """Return to IDLE and clear data."""
        self._reset_internals()
        for alg in ALGORITHMS:
            self.cx[alg] = None
            self.cy[alg] = None
        self._state = self.S_IDLE
        with self._lock:
            self._pub = {}

    def _reset_internals(self):
        self._phase_t0 = 0.0
        self._pt_idx = 0
        self._retries = 0
        self._track_idx = 0
        self._track_sub = 'APPROACH'
        self._track_target = (0, 0)
        self._val_idx = 0
        self._val_sub = 'APPROACH'
        self._flash_type = 'good'
        self._after_flash = 'next'
        self._flash_target = (0, 0)
        self._last_qual = 1.0

        for alg in ALGORITHMS:
            self._samples[alg] = []
            self._src_pts[alg] = []
            self._dst_pts[alg] = []
            self._weights[alg] = []
            self._quality[alg] = []
            self._val_errors[alg] = []
            self._val_samples[alg] = []

    def _transition(self, new_state: str, t: float = None):
        self._state = new_state
        self._phase_t0 = t if t is not None else time.time()

    # -------------------------------------------------------------------------
    # Frame Tick (Driven every camera frame with all 4 algorithm detections)
    # -------------------------------------------------------------------------
    def tick(self, detections: Dict[str, Dict[str, Any]], t: float) -> dict:
        """
        Drives calibration state machine across all 4 algorithms concurrently.
        """
        # Extract centers
        pupil_coords = {alg: detections[alg].get("center") for alg in ALGORITHMS}

        s = self._state
        if   s == self.S_WARMUP:   self._tick_warmup(t)
        elif s == self.S_APPROACH: self._tick_approach(pupil_coords, t)
        elif s == self.S_COLLECT:  self._tick_collect(pupil_coords, t)
        elif s == self.S_FLASH:    self._tick_flash(t)
        elif s == self.S_TRACKING: self._tick_tracking(pupil_coords, t)
        elif s == self.S_VAL:      self._tick_val(pupil_coords, t)

        self._publish(t)
        return self.get_state()

    def _tick_warmup(self, t: float):
        if t - self._phase_t0 >= self.WARMUP_SECS:
            for alg in ALGORITHMS:
                self._samples[alg] = []
            self._transition(self.S_APPROACH, t)
            print(f"[MultiCalibrator] Warm-up complete -> Point 1 / {len(self.grid_px)}")

    def _tick_approach(self, pupil_coords: Dict[str, Optional[Tuple[float, float]]], t: float):
        # Settling time for the user's eye
        if t - self._phase_t0 >= self.APPROACH_SECS:
            for alg in ALGORITHMS:
                self._samples[alg] = []
            self._transition(self.S_COLLECT, t)

    def _tick_collect(self, pupil_coords: Dict[str, Optional[Tuple[float, float]]], t: float):
        # Accumulate clean fixation samples for all 4 algorithms
        for alg in ALGORITHMS:
            c = pupil_coords.get(alg)
            if c is not None:
                if self.mode == self.MODE_SOTA and len(self._samples[alg]) > 0:
                    prev = self._samples[alg][-1]
                    step_v = math.hypot(c[0] - prev[0], c[1] - prev[1])
                    if step_v > 8.0:
                        continue
                self._samples[alg].append(tuple(c))

        if t - self._phase_t0 >= self.COLLECT_SECS:
            self._evaluate_point(t)

    def _evaluate_point(self, t: float):
        """Evaluates samples for all 4 algorithms using Tukey IQR outlier rejection."""
        self._flash_target = self.grid_px[self._pt_idx]

        # Use available samples as reference for pacing
        ref_samples = (self._samples.get("PuReST", []) or 
                       self._samples.get("Hybrid Fusion", []) or 
                       self._samples.get("Pupil Labs", []) or 
                       self._samples.get("Orlosky", []) or [])
        quality = self._compute_quality(ref_samples)
        self._last_qual = quality

        retry = (quality < self.QUALITY_THRESH) and (self._retries < self.MAX_RETRIES)

        if retry:
            self._retries += 1
            for alg in ALGORITHMS:
                self._samples[alg] = []
            self._flash_type = 'warn'
            self._after_flash = 'retry'
            print(f"[MultiCalibrator] Point {self._pt_idx + 1}: quality {quality:.2f} -- retrying.")
        else:
            # Accept point for all 4 algorithms
            for alg in ALGORITHMS:
                s_list = self._samples[alg]
                if len(s_list) >= 4:
                    pts = np.array(s_list, dtype=np.float64)
                    clean = self._tukey(pts)
                    if len(clean) >= 3:
                        avg = np.median(clean, axis=0)
                        self._src_pts[alg].append(avg.tolist())
                        self._dst_pts[alg].append(list(self.grid_px[self._pt_idx]))
                        static_weight = 35.0 if self.mode == self.MODE_SOTA else 1.0
                        self._weights[alg].append(static_weight)
                        alg_q = self._compute_quality(s_list)
                        self._quality[alg].append(round(alg_q, 3))
                self._samples[alg] = []

            self._retries = 0
            self._pt_idx += 1
            self._flash_type = 'good' if quality >= self.QUALITY_THRESH else 'warn'

            if self._pt_idx >= len(self.grid_px):
                self._after_flash = 'tracking'
                print(f"[MultiCalibrator] All {len(self.grid_px)} static points captured -> moving to 8 Dynamic Pursuit Paths!")
            else:
                self._after_flash = 'next'
                remaining = len(self.grid_px) - self._pt_idx
                print(f"[MultiCalibrator] Point {self._pt_idx} accepted (qual {quality:.2f}) -- {remaining} remaining.")

        self._transition(self.S_FLASH, t)

    def _tick_flash(self, t: float):
        if t - self._phase_t0 < self.FLASH_SECS:
            return
        af = self._after_flash
        if af == 'retry':
            self._transition(self.S_APPROACH, t)
        elif af == 'tracking':
            self._track_idx = 0
            self._track_sub = 'APPROACH'
            self._track_target = (0, 0)
            self._transition(self.S_TRACKING, t)
            print("[MultiCalibrator] Starting Dynamic Pursuit Tracking pass (8 continuous paths).")
        else:
            self._transition(self.S_APPROACH, t)

    def _tick_tracking(self, pupil_coords: Dict[str, Optional[Tuple[float, float]]], t: float):
        elapsed = t - self._phase_t0
        start, end = self.TRACKING_PATHS[self._track_idx]

        if self._track_sub == 'APPROACH':
            tx_px = int(start[0] * self.screen_w)
            ty_px = int(start[1] * self.screen_h)
            self._track_target = (tx_px, ty_px)

            if elapsed >= self.TRACK_APPROACH_SECS:
                self._track_sub = 'COLLECT'
                self._phase_t0 = t

        elif self._track_sub == 'COLLECT':
            frac = min(1.0, elapsed / self.TRACKING_SECS)
            tx = start[0] + (end[0] - start[0]) * frac
            ty = start[1] + (end[1] - start[1]) * frac
            tx_px = int(tx * self.screen_w)
            ty_px = int(ty * self.screen_h)
            self._track_target = (tx_px, ty_px)

            if self.mode == self.MODE_SOTA:
                # Human smooth pursuit physiological lag compensation (tau = 120 ms)
                vx_px = (end[0] - start[0]) * self.screen_w / self.TRACKING_SECS
                vy_px = (end[1] - start[1]) * self.screen_h / self.TRACKING_SECS
                tau = 0.12  # 120 ms oculomotor tracking delay
                comp_tx = tx_px - vx_px * tau
                comp_ty = ty_px - vy_px * tau
            else:
                comp_tx = tx_px
                comp_ty = ty_px

            # Collect continuous tracking points for all 4 algorithms
            for alg in ALGORITHMS:
                c = pupil_coords.get(alg)
                if c is not None:
                    self._src_pts[alg].append(list(c))
                    self._dst_pts[alg].append([comp_tx, comp_ty])
                    self._weights[alg].append(1.0)

            if elapsed >= self.TRACKING_SECS:
                self._track_idx += 1
                self._track_sub = 'APPROACH'
                self._phase_t0 = t

                if self._track_idx >= len(self.TRACKING_PATHS):
                    # Fit polynomials for all 4 algorithms
                    self._fit_all_polynomials()
                    self._val_idx = 0
                    for alg in ALGORITHMS:
                        self._val_samples[alg] = []
                    self._val_sub = 'APPROACH'
                    self._transition(self.S_VAL, t)
                    print("[MultiCalibrator] Dynamic tracking complete -> Starting 4-Point Validation pass.")

    def _tick_val(self, pupil_coords: Dict[str, Optional[Tuple[float, float]]], t: float):
        elapsed = t - self._phase_t0
        if self._val_sub == 'APPROACH':
            if elapsed >= self.VAL_APPROACH_S:
                for alg in ALGORITHMS:
                    self._val_samples[alg] = []
                self._val_sub = 'COLLECT'
                self._phase_t0 = t
        elif self._val_sub == 'COLLECT':
            for alg in ALGORITHMS:
                c = pupil_coords.get(alg)
                if c is not None:
                    self._val_samples[alg].append(tuple(c))

            if elapsed >= self.VAL_COLLECT_S:
                self._record_val_error()
                self._val_idx += 1
                for alg in ALGORITHMS:
                    self._val_samples[alg] = []
                self._val_sub = 'APPROACH'
                self._phase_t0 = t

                if self._val_idx >= len(self.val_px):
                    self._transition(self.S_DONE, t)
                    print("\n" + "="*60)
                    print("100% CALIBRATION COMPLETE -- VALIDATION RESULTS ACROSS ALL 4 ALGORITHMS:")
                    for alg in ALGORITHMS:
                        errs = self._val_errors[alg]
                        mean_e = sum(errs) / len(errs) if errs else float('nan')
                        print(f"  {alg:15s} -> Mean Val Error: {mean_e:.1f} px ({len(self._src_pts[alg])} total training points)")
                    print("="*60 + "\n")

    def _record_val_error(self):
        tx, ty = self.val_px[self._val_idx]
        for alg in ALGORITHMS:
            s_list = self._val_samples[alg]
            if len(s_list) >= 4 and self.cx[alg] is not None:
                pts = np.array(s_list, dtype=np.float64)
                clean = self._tukey(pts)
                avg = np.median(clean, axis=0)
                mapped = self.map(alg, avg[0], avg[1])
                if mapped:
                    err = math.hypot(mapped[0] - tx, mapped[1] - ty)
                    self._val_errors[alg].append(round(err, 1))

    # ── Mathematics & Polynomial Fitting ──────────────────────────────────────

    def _feat(self, px: float, py: float) -> np.ndarray:
        """
        Feature vector generation based on active calibration mode:
          - SOTA: 7-term anisotropic basis [1, u, v, uv, u^2, v^2, (uv)^2] (scaled by 100)
          - CLASSIC: 6-term bivariate degree-2 polynomial [1, px, py, px*py, px^2, py^2]
        """
        if self.mode == self.MODE_CLASSIC:
            return np.array([1.0, px, py, px * py, px * px, py * py], dtype=np.float64)
        else:
            u = px / 100.0
            v = py / 100.0
            return np.array([1.0, u, v, u * v, u * u, v * v, (u * v) ** 2], dtype=np.float64)

    def _fit_all_polynomials(self):
        """Fits independent polynomials for all 4 algorithms using SOTA (WLS + Ridge) or CLASSIC (OLS)."""
        min_pts = 6 if self.mode == self.MODE_CLASSIC else 7
        print(f"[MultiCalibrator] Fitting polynomials using mode: {self.mode}...")
        for alg in ALGORITHMS:
            src = np.array(self._src_pts[alg], dtype=np.float64)
            dst = np.array(self._dst_pts[alg], dtype=np.float64)
            weights = np.array(self._weights.get(alg, []), dtype=np.float64)
            if len(src) < min_pts:
                print(f"[MultiCalibrator] Too few points ({len(src)}) for {alg}")
                continue
            if len(weights) != len(src):
                weights = np.ones(len(src), dtype=np.float64)

            X = np.array([self._feat(p[0], p[1]) for p in src])

            if self.mode == self.MODE_CLASSIC:
                cx, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
                cy, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)
            else:
                # Weighted Least Squares with Ridge Regularization (lambda = 1e-4)
                W = np.diag(weights)
                XT_W = X.T @ W
                XT_W_X = XT_W @ X + 1e-4 * np.eye(X.shape[1])
                try:
                    cx = np.linalg.solve(XT_W_X, XT_W @ dst[:, 0])
                    cy = np.linalg.solve(XT_W_X, XT_W @ dst[:, 1])
                except Exception:
                    cx, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
                    cy, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)

            self.cx[alg] = cx
            self.cy[alg] = cy

            pred_x = X @ cx
            pred_y = X @ cy
            rms_x = float(np.sqrt(np.mean((pred_x - dst[:, 0]) ** 2)))
            rms_y = float(np.sqrt(np.mean((pred_y - dst[:, 1]) ** 2)))
            print(f"  {alg:15s} -> Mode {self.mode} fit with {len(src)} points | Train RMS: X={rms_x:.1f}px, Y={rms_y:.1f}px")

    def map(self, alg: str, px: float, py: float) -> Optional[Tuple[int, int]]:
        """Maps pupil (px, py) -> screen (sx, sy) using algorithm's calibrated polynomial."""
        cx = self.cx.get(alg)
        cy = self.cy.get(alg)
        if cx is None or cy is None:
            return None
        f = self._feat(px, py)
        sx = float(np.dot(f, cx))
        sy = float(np.dot(f, cy))
        return int(round(sx)), int(round(sy))

    def _compute_quality(self, samples: list) -> float:
        if len(samples) < self.MIN_SAMPLES:
            return 0.0
        arr = np.array(samples, dtype=np.float64)
        med = np.median(arr, axis=0)
        dists = np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1])
        std = float(np.std(dists))
        raw = math.exp(-std / 3.0)
        return float(max(0.0, min(1.0, raw)))

    @staticmethod
    def _tukey(pts: np.ndarray) -> np.ndarray:
        if len(pts) < 5:
            return pts
        q1 = np.percentile(pts, 25, axis=0)
        q3 = np.percentile(pts, 75, axis=0)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        res = pts[mask]
        return res if len(res) >= 4 else pts

    # ── State publishing for calibration_overlay.py ───────────────────────────

    def _publish(self, t: float):
        elapsed = t - self._phase_t0
        state = self._state

        # Animated target
        if state == self.S_WARMUP:
            frac = min(1.0, elapsed / self.WARMUP_SECS)
            angle = frac * 4.0 * math.pi
            wx = 0.5 + 0.38 * math.sin(angle)
            wy = 0.5 + 0.22 * math.sin(2 * angle + math.pi / 2)
            target = (int(wx * self.screen_w), int(wy * self.screen_h))
        elif state in (self.S_APPROACH, self.S_COLLECT):
            target = self.grid_px[min(self._pt_idx, len(self.grid_px) - 1)]
        elif state == self.S_FLASH:
            target = self._flash_target
        elif state == self.S_TRACKING:
            target = self._track_target
        elif state == self.S_VAL:
            vi = min(self._val_idx, len(self.val_px) - 1)
            target = self.val_px[vi]
        else:
            target = (self.screen_w // 2, self.screen_h // 2)

        # Animation fractions
        if state == self.S_APPROACH:
            approach_frac = min(1.0, elapsed / self.APPROACH_SECS)
        elif state == self.S_WARMUP:
            approach_frac = min(1.0, elapsed / self.WARMUP_SECS)
        else:
            approach_frac = 1.0

        if state == self.S_COLLECT:
            collect_frac = min(1.0, elapsed / self.COLLECT_SECS)
        elif state == self.S_VAL and self._val_sub == 'COLLECT':
            collect_frac = min(1.0, elapsed / self.VAL_COLLECT_S)
        else:
            collect_frac = 0.0

        flash_frac = min(1.0, elapsed / self.FLASH_SECS) if state == self.S_FLASH else 0.0
        track_app_frac = min(1.0, elapsed / self.TRACK_APPROACH_SECS) if (state == self.S_TRACKING and self._track_sub == 'APPROACH') else 1.0

        # Representative quality and validation errors for overlay
        ref_qual = (self._quality.get("PuReST") or 
                    self._quality.get("Hybrid Fusion") or 
                    self._quality.get("Pupil Labs") or 
                    self._quality.get("Orlosky") or [])
        ref_val_errs = (self._val_errors.get("PuReST") or 
                        self._val_errors.get("Hybrid Fusion") or 
                        self._val_errors.get("Pupil Labs") or 
                        self._val_errors.get("Orlosky") or [])

        with self._lock:
            self._pub = {
                'state': state,
                'target': target,
                'pt_idx': self._pt_idx,
                'total_pts': len(self.grid_px),
                'track_idx': self._track_idx,
                'track_sub': self._track_sub,
                'track_total': len(self.TRACKING_PATHS),
                'track_approach_frac': track_app_frac,
                'approach_frac': approach_frac,
                'collect_frac': collect_frac,
                'flash_frac': flash_frac,
                'flash_type': self._flash_type,
                'quality': list(ref_qual),
                'last_quality': self._last_qual,
                'val_idx': self._val_idx,
                'val_total': len(self.val_px),
                'val_errors': list(ref_val_errs),
                'val_sub': self._val_sub,
                'done': state == self.S_DONE,
                'calibrating': state not in (self.S_IDLE, self.S_DONE),
            }

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._pub)

    @property
    def done(self) -> bool:
        return self._state == self.S_DONE
