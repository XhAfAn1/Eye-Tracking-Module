#!/usr/bin/env python3
"""
calibration_engine.py  —  PrecisionCalibrator for EyeTracker Module
====================================================================
Fully automatic, zero-keypress 9-point gaze calibration engine.

Pipeline (identical to Tobii / GazeRecorder architecture adapted for
a close-up pupil-only camera with no IR hardware):

  1. WARMUP      (4 s)  — Lissajous moving dot; user just watches.
                          Gets the eye muscles warmed up and gives the
                          algorithm a preview of the full pupil range.

  2. APPROACHING (2 s)  — Static dot appears at grid position.
                          Large outer ring shrinks toward dot centre.
                          Eye has time to move to the target.
                          Samples discarded (settling period).

  3. COLLECTING  (1.5 s)— Cyan countdown arc sweeps 0→360°.
                          Pupil samples recorded every frame.
                          At end: Tukey IQR outlier rejection applied.

  4. FLASHING    (0.55 s)— Quality feedback ring (green = good,
                           amber = accepted despite low quality).
                           Then advance to next point OR retry once.

  5. Repeat 2–4 for all 9 grid points.

  6. Bivariate degree-2 polynomial fit (pure numpy, ~6 µs):
       screen_x = a0 + a1·px + a2·py + a3·px·py + a4·px² + a5·py²
       screen_y = b0 + b1·px + b2·py + b3·px·py + b4·px² + b5·py²
     This is the same model used by LC Technologies Eyegaze (medical
     eye tracker used for ALS/plegia patients).

  7. VALIDATING  (4 dots × 3 s each) — accuracy measurement at
     positions not seen during calibration.

  8. DONE — polynomial map is active. Call map(px, py) each frame.

Usage:
    cal = PrecisionCalibrator(screen_w, screen_h)
    cal.start()                         # begin sequence
    ...
    state = cal.tick(pupil_xy, t)       # call every frame
    ...
    if cal.done:
        sx, sy = cal.map(px, py)        # polynomial mapping

Persistence:
    cal.save('gaze_calibration.json')
    cal.load('gaze_calibration.json')   # returns True on success
"""

import math
import time
import json
import threading
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
class PrecisionCalibrator:
    """Automatic precision gaze calibrator.  No keypresses after start()."""

    # ── 9-point grid with 8 % margin ─────────────────────────────────────────
    GRID = [
        (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
        (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
        (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
    ]
    # Smooth Pursuit Tracking paths (Start, End)
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
    # Validation at positions not in the calibration grid
    VAL_GRID = [
        (0.25, 0.25), (0.75, 0.25),
        (0.25, 0.75), (0.75, 0.75),
    ]

    # ── Phase durations (seconds) ─────────────────────────────────────────────
    WARMUP_SECS    = 4.0
    APPROACH_SECS  = 2.0   # ring shrinks; eye moves to target
    COLLECT_SECS   = 1.5   # arc sweeps; main sample collection
    FLASH_SECS     = 0.55  # quality feedback flash
    TRACK_APPROACH_SECS = 1.5  # eye moves to the start of the path
    TRACKING_SECS  = 3.5   # duration for each smooth pursuit path
    VAL_APPROACH_S = 1.2   # approach to each validation dot
    VAL_COLLECT_S  = 1.8   # collect at each validation dot

    # ── Quality thresholds ───────────────────────────────────────────────────
    MIN_SAMPLES    = 15    # discard point if fewer raw samples than this
    QUALITY_THRESH = 0.28  # variance quality [0..1]; below → auto-retry once
    MAX_RETRIES    = 1     # max retries per calibration point

    # ── State names ──────────────────────────────────────────────────────────
    S_IDLE     = 'IDLE'
    S_WARMUP   = 'WARMUP'
    S_APPROACH = 'APPROACHING'
    S_COLLECT  = 'COLLECTING'
    S_FLASH    = 'FLASHING'
    S_TRACKING = 'TRACKING'
    S_VAL      = 'VALIDATING'
    S_DONE     = 'DONE'

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w  = screen_w
        self.screen_h  = screen_h
        self.grid_px   = [(int(fx * screen_w), int(fy * screen_h))
                          for fx, fy in self.GRID]
        self.val_px    = [(int(fx * screen_w), int(fy * screen_h))
                          for fx, fy in self.VAL_GRID]
        self._lock     = threading.Lock()
        self._cx       = None   # polynomial x-axis coefficients (6,)
        self._cy       = None   # polynomial y-axis coefficients (6,)
        self._state    = self.S_IDLE
        self._pub: dict = {}
        self._reset_internals()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Begin the full calibration sequence."""
        self._reset_internals()
        self._cx = None
        self._cy = None
        self._transition(self.S_WARMUP)
        print("[Calibrator] Sequence started -- follow the dot, no keypresses needed.")

    def reset(self):
        """Return to IDLE and clear all data."""
        self._reset_internals()
        self._cx    = None
        self._cy    = None
        self._state = self.S_IDLE
        with self._lock:
            self._pub = {}
        print("[Calibrator] Reset.")

    def tick(self, pupil_xy, t: float) -> dict:
        """
        Drive the calibration state machine.  Call every camera frame.

        Parameters
        ----------
        pupil_xy : tuple(float, float) or None
            Pupil centre in image coordinates from OrloskyPupilDetector.
        t : float
            Current time (time.time()).

        Returns
        -------
        dict
            Public state snapshot for the overlay renderer.
        """
        s = self._state
        if   s == self.S_WARMUP:   self._tick_warmup(t)
        elif s == self.S_APPROACH: self._tick_approach(pupil_xy, t)
        elif s == self.S_COLLECT:  self._tick_collect(pupil_xy, t)
        elif s == self.S_FLASH:    self._tick_flash(t)
        elif s == self.S_TRACKING: self._tick_tracking(pupil_xy, t)
        elif s == self.S_VAL:      self._tick_val(pupil_xy, t)
        self._publish(t)
        return self.get_state()

    def get_state(self) -> dict:
        """Thread-safe copy of the current public state dict."""
        with self._lock:
            return dict(self._pub)

    @property
    def done(self) -> bool:
        return self._state == self.S_DONE

    @property
    def calibrating(self) -> bool:
        return self._state not in (self.S_IDLE, self.S_DONE)

    def map(self, px: float, py: float):
        """
        Map pupil image coords → screen pixel coords via polynomial.

        Returns (sx, sy) or None if not yet calibrated.
        """
        if self._cx is None:
            return None
        f  = self._feat(px, py)
        sx = float(np.dot(f, self._cx))
        sy = float(np.dot(f, self._cy))
        return int(sx), int(sy)

    def save(self, path: str, fine_tune_dict: dict = None):
        """Persist polynomial coefficients and optional fine-tune matrix to JSON."""
        if self._cx is None:
            return
        try:
            data = {
                'cx':      self._cx.tolist(),
                'cy':      self._cy.tolist(),
                'quality': self._quality,
                'val_err': self._val_errors,
                'screen':  [self.screen_w, self.screen_h],
                'model':   'bivariate_poly_deg2',
            }
            if fine_tune_dict:
                data['fine_tune'] = fine_tune_dict
            with open(path, 'w') as fh:
                json.dump(data, fh, indent=2)
            print(f"[Calibrator] Saved → {path}")
        except Exception as e:
            print(f"[Calibrator] Save failed: {e}")

    def load(self, path: str):
        """Load persisted calibration. Returns (success_bool, fine_tune_dict)."""
        try:
            with open(path) as fh:
                d = json.load(fh)
            self._cx         = np.array(d['cx'],      dtype=np.float64)
            self._cy         = np.array(d['cy'],      dtype=np.float64)
            self._quality    = d.get('quality',  [])
            self._val_errors = d.get('val_err',  [])
            self._state      = self.S_DONE
            self._publish(time.time())
            return True, d.get('fine_tune', None)
        except Exception:
            return False, None

    # ── Internal state management ─────────────────────────────────────────────

    def _reset_internals(self):
        self._phase_t0     = 0.0
        self._pt_idx       = 0       # current calibration point index
        self._retries      = 0       # retries on current point
        self._samples: list = []     # raw pupil samples in current phase
        self._src_pts: list = []     # accepted pupil medians
        self._dst_pts: list = []     # corresponding screen points
        self._quality: list = []     # per-point quality score [0..1]
        self._last_qual    = 1.0
        self._flash_type   = 'good'  # 'good' | 'warn'
        self._after_flash  = 'next'  # 'next' | 'retry' | 'tracking'
        self._flash_target = (0, 0)
        self._track_idx    = 0
        self._track_sub    = 'APPROACH'
        self._track_target = (0, 0)
        self._val_idx      = 0
        self._val_samples: list = []
        self._val_errors: list  = []
        self._val_sub      = 'APPROACH'  # 'APPROACH' or 'COLLECT'

    def _transition(self, new_state: str, t: float = None):
        self._state    = new_state
        self._phase_t0 = t if t is not None else time.time()

    # ── Per-state ticks ───────────────────────────────────────────────────────

    def _tick_warmup(self, t: float):
        if t - self._phase_t0 >= self.WARMUP_SECS:
            self._samples = []
            self._transition(self.S_APPROACH, t)
            print(f"[Calibrator] Warm-up done -> Point 1 / {len(self.grid_px)}")

    def _tick_approach(self, pupil_xy, t: float):
        # DO NOT collect samples here — settling time for the eye.
        if t - self._phase_t0 >= self.APPROACH_SECS:
            self._samples = []                    # discard any pre-samples
            self._transition(self.S_COLLECT, t)

    def _tick_collect(self, pupil_xy, t: float):
        if pupil_xy is not None:
            self._samples.append(tuple(pupil_xy))
        if t - self._phase_t0 >= self.COLLECT_SECS:
            self._evaluate_point(t)

    def _evaluate_point(self, t: float):
        """Score collected samples; accept or schedule one retry."""
        samples = self._samples
        quality = self._compute_quality(samples)
        self._last_qual    = quality
        self._flash_target = self.grid_px[self._pt_idx]   # before possible increment

        retry = (quality < self.QUALITY_THRESH) and (self._retries < self.MAX_RETRIES)

        if retry:
            self._retries    += 1
            self._samples     = []
            self._flash_type  = 'warn'
            self._after_flash = 'retry'
            print(f"[Calibrator] Point {self._pt_idx + 1}: quality {quality:.2f} — retrying.")
        else:
            # Accept — compute clean median and record pair
            if len(samples) >= 4:
                pts   = np.array(samples, dtype=np.float64)
                clean = self._tukey(pts)
                if len(clean) >= 4:
                    avg = np.median(clean, axis=0)
                    self._src_pts.append(avg.tolist())
                    self._dst_pts.append(list(self.grid_px[self._pt_idx]))
                    self._quality.append(round(quality, 3))
            self._retries = 0
            self._samples = []
            self._pt_idx += 1
            self._flash_type  = 'good' if quality >= self.QUALITY_THRESH else 'warn'
            if self._pt_idx >= len(self.grid_px):
                self._after_flash = 'tracking'
                print(f"[Calibrator] All {len(self.grid_px)} static points captured -> moving to dynamic tracking.")
            else:
                self._after_flash = 'next'
                remaining = len(self.grid_px) - self._pt_idx
                print(f"[Calibrator] Point accepted (quality {quality:.2f}) — "
                      f"{remaining} remaining.")

        self._transition(self.S_FLASH, t)

    def _tick_flash(self, t: float):
        if t - self._phase_t0 < self.FLASH_SECS:
            return
        af = self._after_flash
        if af == 'retry':
            self._transition(self.S_APPROACH, t)
        elif af == 'tracking':
            self._track_idx    = 0
            self._track_sub    = 'APPROACH'
            self._track_target = (0, 0)
            self._transition(self.S_TRACKING, t)
            print("[Calibrator] Starting dynamic tracking pass.")
        else:   # 'next'
            self._transition(self.S_APPROACH, t)

    def _tick_tracking(self, pupil_xy, t: float):
        elapsed = t - self._phase_t0
        start, end = self.TRACKING_PATHS[self._track_idx]

        if self._track_sub == 'APPROACH':
            tx, ty = start
            tx_px = int(tx * self.screen_w)
            ty_px = int(ty * self.screen_h)
            self._track_target = (tx_px, ty_px)

            if elapsed >= self.TRACK_APPROACH_SECS:
                self._track_sub = 'COLLECT'
                self._phase_t0  = t

        elif self._track_sub == 'COLLECT':
            frac = min(1.0, elapsed / self.TRACKING_SECS)
            tx = start[0] + (end[0] - start[0]) * frac
            ty = start[1] + (end[1] - start[1]) * frac
            tx_px = int(tx * self.screen_w)
            ty_px = int(ty * self.screen_h)
            self._track_target = (tx_px, ty_px)

            # Collect continuous tracking points
            if pupil_xy is not None:
                self._src_pts.append(list(pupil_xy))
                self._dst_pts.append([tx_px, ty_px])

            if elapsed >= self.TRACKING_SECS:
                self._track_idx += 1
                self._track_sub = 'APPROACH'
                self._phase_t0  = t
                if self._track_idx >= len(self.TRACKING_PATHS):
                    self._fit_polynomial()
                    self._val_idx      = 0
                    self._val_samples  = []
                    self._val_sub      = 'APPROACH'
                    self._transition(self.S_VAL, t)
                    print("[Calibrator] Starting validation pass.")

    def _tick_val(self, pupil_xy, t: float):
        elapsed = t - self._phase_t0
        if self._val_sub == 'APPROACH':
            if elapsed >= self.VAL_APPROACH_S:
                self._val_samples = []
                self._val_sub     = 'COLLECT'
                self._phase_t0    = t
        elif self._val_sub == 'COLLECT':
            if pupil_xy is not None:
                self._val_samples.append(tuple(pupil_xy))
            if elapsed >= self.VAL_COLLECT_S:
                self._record_val_error()
                self._val_idx  += 1
                self._val_samples = []
                self._val_sub   = 'APPROACH'
                self._phase_t0  = t
                if self._val_idx >= len(self.val_px):
                    self._transition(self.S_DONE, t)
                    errs = self._val_errors
                    mean_e = sum(errs) / len(errs) if errs else float('nan')
                    print(f"[Calibrator] DONE -- mean validation error: {mean_e:.0f} px")

    def _record_val_error(self):
        """Compute and store the mapping error at the current validation point."""
        if len(self._val_samples) >= 4 and self._cx is not None:
            pts    = np.array(self._val_samples, dtype=np.float64)
            clean  = self._tukey(pts)
            avg    = np.median(clean, axis=0)
            mapped = self.map(avg[0], avg[1])
            if mapped:
                tx, ty = self.val_px[self._val_idx]
                err    = math.hypot(mapped[0] - tx, mapped[1] - ty)
                self._val_errors.append(round(err, 1))

    # ── Mathematics ───────────────────────────────────────────────────────────

    @staticmethod
    def _feat(px: float, py: float) -> np.ndarray:
        """
        6-term bivariate degree-2 polynomial feature vector.
        [1, px, py, px·py, px², py²]
        """
        return np.array([1.0, px, py, px * py, px * px, py * py],
                        dtype=np.float64)

    def _fit_polynomial(self):
        """
        Least-squares fit of bivariate degree-2 polynomial.
        Solves separately for x and y screen coordinates.
        Requires ≥ 6 point pairs (we have 9).
        """
        n = len(self._src_pts)
        if n < 4:
            print(f"[Calibrator] Too few accepted points ({n}) — skipping fit.")
            return
        src = np.array(self._src_pts, dtype=np.float64)   # shape (n, 2)
        dst = np.array(self._dst_pts, dtype=np.float64)   # shape (n, 2)
        X   = np.array([self._feat(p[0], p[1]) for p in src])  # shape (n, 6)
        # Over-determined system: X @ coeffs ≈ dst_axis  (least-squares)
        self._cx, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
        self._cy, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)
        # Compute training residuals for diagnostics
        pred_x = X @ self._cx
        pred_y = X @ self._cy
        rms_x  = float(np.sqrt(np.mean((pred_x - dst[:, 0]) ** 2)))
        rms_y  = float(np.sqrt(np.mean((pred_y - dst[:, 1]) ** 2)))
        print(f"[Calibrator] Polynomial fit with {n} pts | "
              f"train RMS x={rms_x:.1f} px  y={rms_y:.1f} px")

    def _compute_quality(self, samples: list) -> float:
        """
        Variance-based quality score in [0, 1].
        1.0 = perfectly stable pupil; 0.0 = too few / too noisy samples.
        """
        if len(samples) < self.MIN_SAMPLES:
            return 0.0
        pts   = np.array(samples, dtype=np.float64)
        clean = self._tukey(pts)
        if len(clean) < 4:
            return 0.0
        var = float(np.var(clean[:, 0]) + np.var(clean[:, 1]))
        # Normalise: var ≈ 0 → quality 1.0; var ≈ 120 → quality 0.0
        return max(0.0, min(1.0, 1.0 - var / 120.0))

    @staticmethod
    def _tukey(pts: np.ndarray) -> np.ndarray:
        """
        Tukey IQR fence outlier rejection.
        Removes points outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] on either axis.
        Falls back to original array if too many points would be removed.
        """
        if len(pts) < 4:
            return pts
        q1  = np.percentile(pts, 25, axis=0)
        q3  = np.percentile(pts, 75, axis=0)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        mask   = np.all((pts >= lo) & (pts <= hi), axis=1)
        result = pts[mask]
        return result if len(result) >= 4 else pts

    # ── State publishing ──────────────────────────────────────────────────────

    def _publish(self, t: float):
        """Compute and atomically write the public state snapshot."""
        elapsed = t - self._phase_t0
        state   = self._state

        # ── Animated target position ────────────────────────────────────────
        if state == self.S_WARMUP:
            frac   = min(1.0, elapsed / self.WARMUP_SECS)
            angle  = frac * 4.0 * math.pi           # 2 full Lissajous cycles
            wx     = 0.5 + 0.38 * math.sin(angle)
            wy     = 0.5 + 0.22 * math.sin(2 * angle + math.pi / 2)
            target = (int(wx * self.screen_w), int(wy * self.screen_h))
        elif state in (self.S_APPROACH, self.S_COLLECT):
            target = self.grid_px[min(self._pt_idx, len(self.grid_px) - 1)]
        elif state == self.S_FLASH:
            target = self._flash_target
        elif state == self.S_TRACKING:
            target = self._track_target
        elif state == self.S_VAL:
            vi     = min(self._val_idx, len(self.val_px) - 1)
            target = self.val_px[vi]
        else:
            target = (self.screen_w // 2, self.screen_h // 2)

        # ── Phase fractions for animation ───────────────────────────────────
        if   state == self.S_APPROACH:
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

        with self._lock:
            self._pub = {
                'state':         state,
                'target':        target,
                'pt_idx':        self._pt_idx,
                'total_pts':     len(self.grid_px),
                'track_idx':     self._track_idx,
                'track_sub':     self._track_sub,
                'track_total':   len(self.TRACKING_PATHS),
                'track_approach_frac': track_app_frac,
                'approach_frac': approach_frac,
                'collect_frac':  collect_frac,
                'flash_frac':    flash_frac,
                'flash_type':    self._flash_type,
                'quality':       list(self._quality),
                'last_quality':  self._last_qual,
                'val_idx':       self._val_idx,
                'val_total':     len(self.val_px),
                'val_errors':    list(self._val_errors),
                'val_sub':       self._val_sub,
                'done':          state == self.S_DONE,
                'calibrating':   state not in (self.S_IDLE, self.S_DONE),
            }


# ─────────────────────────────────────────────────────────────────────────────
class FineTuneCalibrator:
    """Manual Level 2 Calibration (Affine Fine-Tuning)."""
    
    S_IDLE     = 'IDLE'
    S_WAITING  = 'WAITING'   # Waiting for Spacebar
    S_ACCEPTED = 'ACCEPTED'  # Flashes green
    S_DONE     = 'DONE'
    
    def __init__(self, screen_w: int, screen_h: int, num_points: int = 10):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.num_points = num_points
        self._lock = threading.Lock()
        
        self._ax = None  # Affine X coefficients [1, x, y]
        self._ay = None  # Affine Y coefficients [1, x, y]
        self.reset()
        
    def reset(self):
        self._state = self.S_IDLE
        self._pts_collected = 0
        self._src_pts = [] # (sx, sy) from Level 1
        self._dst_pts = [] # (tx, ty) random targets
        self._target = (0, 0)
        self._pub = {}
        self._phase_t0 = 0
        self._space_t0 = 0.0
        
    def start(self):
        self.reset()
        self._next_target()
        self._state = self.S_WAITING
        self._phase_t0 = time.time()
        print(f"[FineTune] Started Level 2 Calibration ({self.num_points} points).")
        
    def _next_target(self):
        import random
        margin = 0.15
        fx = random.uniform(margin, 1.0 - margin)
        fy = random.uniform(margin, 1.0 - margin)
        self._target = (int(fx * self.screen_w), int(fy * self.screen_h))
            
    def tick(self, sx, sy, space_down: bool, t: float) -> dict:
        """Called every frame to update state and live cursor."""
        if self._state == self.S_WAITING:
            if space_down and sx is not None and sy is not None:
                if self._space_t0 == 0.0:
                    self._space_t0 = t
                elif t - self._space_t0 >= 0.8:  # 0.8 second hold
                    self._src_pts.append([sx, sy])
                    self._dst_pts.append(list(self._target))
                    self._pts_collected += 1
                    self._state = self.S_ACCEPTED
                    self._phase_t0 = t
                    self._space_t0 = 0.0
                    print(f"[FineTune] Accepted point {self._pts_collected}/{self.num_points}")
            else:
                self._space_t0 = 0.0

        elif self._state == self.S_ACCEPTED:
            if t - self._phase_t0 >= 0.4:
                if self._pts_collected >= self.num_points:
                    self._fit_affine()
                    self._state = self.S_DONE
                    self._phase_t0 = t
                else:
                    self._next_target()
                    self._state = self.S_WAITING
                    self._phase_t0 = t
                    
        with self._lock:
            self._pub = {
                'state': self._state,
                'target': self._target,
                'pts_collected': self._pts_collected,
                'total_pts': self.num_points,
                'hold_progress': max(0.0, min(1.0, (t - self._space_t0) / 0.8)) if self._space_t0 > 0 else 0.0,
                'live_cursor': (int(sx), int(sy)) if sx is not None else None,
                'calibrating': self._state not in (self.S_IDLE, self.S_DONE) or (self._state == self.S_DONE and t - self._phase_t0 < 2.0),
                'done': self._state == self.S_DONE
            }
        return self.get_state()
        
    def get_state(self):
        with self._lock:
            return dict(self._pub)
            
    @property
    def calibrating(self) -> bool:
        return self._pub.get('calibrating', False)

    @property
    def done(self) -> bool:
        return self._state == self.S_DONE

    def map(self, sx: float, sy: float):
        """Map Level 1 screen coords to Level 2 corrected coords."""
        if self._ax is None:
            return int(sx), int(sy)
        X = np.array([1.0, sx, sy], dtype=np.float64)
        tx = float(np.dot(X, self._ax))
        ty = float(np.dot(X, self._ay))
        return int(tx), int(ty)
        
    def _fit_affine(self):
        if len(self._src_pts) < 3:
            return
        src = np.array(self._src_pts, dtype=np.float64)
        dst = np.array(self._dst_pts, dtype=np.float64)
        X = np.column_stack((np.ones(len(src)), src[:, 0], src[:, 1]))
        self._ax, *_ = np.linalg.lstsq(X, dst[:, 0], rcond=None)
        self._ay, *_ = np.linalg.lstsq(X, dst[:, 1], rcond=None)
        
        pred_x = X @ self._ax
        pred_y = X @ self._ay
        rms_x  = float(np.sqrt(np.mean((pred_x - dst[:, 0]) ** 2)))
        rms_y  = float(np.sqrt(np.mean((pred_y - dst[:, 1]) ** 2)))
        print(f"[FineTune] Affine fit with {len(src)} pts | RMS x={rms_x:.1f} y={rms_y:.1f}")

    def save_to_dict(self):
        if self._ax is None: return None
        return {'ax': self._ax.tolist(), 'ay': self._ay.tolist()}
        
    def load_from_dict(self, d):
        if d and 'ax' in d and 'ay' in d:
            self._ax = np.array(d['ax'], dtype=np.float64)
            self._ay = np.array(d['ay'], dtype=np.float64)
            self._state = self.S_DONE
            return True
        return False

