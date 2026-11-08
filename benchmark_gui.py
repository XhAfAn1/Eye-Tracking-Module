#!/usr/bin/env python3
"""
benchmark_gui.py
================
Full-screen Interactive Benchmark Suite GUI Renderer.

Manages the 5-State Benchmark Machine:
  1. STATE_ALIGNMENT: Live camera preview with interactive Eyeball Boundary Circle
     positioning (W/A/S/D, +/-, C) and 4-algorithm readiness check.
  2. STATE_CALIBRATION: 9-point simultaneous calibration for all 4 algorithms with
     bivariate degree-2 polynomial fitting.
  3. STATE_STATIC_TEST: Level 2 testing targets with 4 live colored pointers
     (Gold, Green, Cyan, Magenta) and [SPACE] confirmation.
  4. STATE_DYNAMIC_TEST: Smooth pursuit tracking tests with continuous trajectory
     and lag logging.
  5. STATE_RESULTS: In-app podium, summary metrics, and report export notifications.
"""

import os
import sys
import time
import math
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import cv2
import numpy as np

# Ensure local directory is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from benchmark_pipeline import (
    ALGORITHMS,
    ALG_ORLOSKY,
    ALG_PL,
    ALG_HYBRID,
    ALG_PUREST,
    ALG_COLORS,
    COLOR_ORLOSKY,
    COLOR_PL,
    COLOR_HYBRID,
    COLOR_PUREST,
    FourWayPipeline,
    PolynomialGazeCalibrator,
    PointOneEuroFilter,
    BenchmarkSession,
    filter_tukey_iqr,
)
from multi_calibrator import MultiAlgorithmPrecisionCalibrator
from calibration_overlay import draw_frame
from benchmark_reporter import (
    compute_benchmark_metrics,
    generate_benchmark_visualizations,
    generate_markdown_report,
)

# Optional audio feedback
try:
    import winsound
    def play_beep(freq=1200, dur=60):
        winsound.Beep(freq, dur)
except Exception:
    def play_beep(freq=1200, dur=60):
        pass


# -----------------------------------------------------------------------------
# Color Palette
# -----------------------------------------------------------------------------
COLOR_BG        = (18, 14, 12)       # Cinematic deep dark blue-black
COLOR_PANEL_BG  = (30, 24, 20)       # Dark panel card background
COLOR_BORDER    = (60, 50, 45)       # Divider line
COLOR_TEXT_MAIN = (245, 245, 248)    # Crisp white
COLOR_TEXT_MUTED= (160, 160, 175)    # Slate gray
COLOR_TARGET    = (255, 255, 255)    # Target white
COLOR_CYAN_GLOW = (255, 230, 0)      # High-visibility cyan
COLOR_GREEN     = (80, 255, 100)     # Neon Green
COLOR_AMBER     = (0, 165, 255)      # Warm Amber
COLOR_RED       = (60, 60, 255)      # Warning red


# -----------------------------------------------------------------------------
# Pointer Trail Tracker
# -----------------------------------------------------------------------------
class PointerTrail:
    def __init__(self, maxlen=10):
        self.trail = deque(maxlen=maxlen)

    def add(self, pt: Optional[Tuple[float, float]]):
        if pt is not None:
            self.trail.append((float(pt[0]), float(pt[1])))

    def draw(self, img: np.ndarray, color: Tuple[int, int, int]):
        if len(self.trail) < 2:
            return
        pts = list(self.trail)
        for i in range(1, len(pts)):
            alpha = i / float(len(pts))
            thickness = max(1, int(round(alpha * 2.5)))
            pt1 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
            pt2 = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Benchmark Suite GUI Orchestrator
# -----------------------------------------------------------------------------
class BenchmarkGUI:
    # States
    STATE_ALIGNMENT    = 0
    STATE_CALIBRATION  = 1
    STATE_STATIC_TEST  = 2
    STATE_DYNAMIC_TEST = 3
    STATE_RESULTS      = 4

    # 9-Point Calibration Grid (Normalized)
    CALIB_GRID = [
        (0.12, 0.12), (0.50, 0.12), (0.88, 0.12),
        (0.12, 0.50), (0.50, 0.50), (0.88, 0.50),
        (0.12, 0.88), (0.50, 0.88), (0.88, 0.88),
    ]

    # Level 2 Static Testing Targets (Diverse validation positions safely inside screen)
    TEST_TARGETS = [
        (0.25, 0.25), (0.75, 0.25),
        (0.50, 0.50),
        (0.20, 0.75), (0.80, 0.75),
        (0.50, 0.22), (0.20, 0.50),
        (0.80, 0.50), (0.50, 0.78),
    ]

    # Smooth Pursuit Dynamic Paths: (Start, End, Label)
    PURSUIT_PATHS = [
        ((0.15, 0.22), (0.85, 0.22), "Top Horizontal (L -> R)"),
        ((0.85, 0.50), (0.15, 0.50), "Mid Horizontal (R -> L)"),
        ((0.15, 0.78), (0.85, 0.78), "Bottom Horizontal (L -> R)"),
        ((0.50, 0.20), (0.50, 0.80), "Vertical (Top -> Bottom)"),
        ((0.18, 0.22), (0.82, 0.78), "Diagonal (Top-L -> Bot-R)"),
        ((0.82, 0.22), (0.18, 0.78), "Diagonal (Top-R -> Bot-L)"),
    ]

    def __init__(self, cap: cv2.VideoCapture, cam_id: int, available_cams: List[int],
                 screen_w: int, screen_h: int, fullscreen: bool = True, calib_mode: str = "SOTA"):
        self.cap = cap
        self.cam_id = cam_id
        self.available_cams = available_cams
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.fullscreen = fullscreen
        self.calib_mode = calib_mode.upper()

        self.pipeline = FourWayPipeline()
        self.session = BenchmarkSession(screen_w, screen_h)
        self.session.calib_mode = self.calib_mode

        # Eyeball boundary circle
        self.circle_x = 320
        self.circle_y = 240
        self.circle_r = 110

        self.session.circle_x = self.circle_x
        self.session.circle_y = self.circle_y
        self.session.circle_r = self.circle_r

        # Multi-algorithm precision calibrator (Supports SOTA and CLASSIC)
        self.multi_calibrator = MultiAlgorithmPrecisionCalibrator(screen_w, screen_h, mode=self.calib_mode)

        # Calibrators for each algorithm
        self.calibrators: Dict[str, PolynomialGazeCalibrator] = {
            alg: PolynomialGazeCalibrator(alg, screen_w, screen_h, mode=self.calib_mode) for alg in ALGORITHMS
        }

        # 1-Euro Adaptive Gaze Filters (Casiez et al., CHI 2012)
        self.one_euro_filters: Dict[str, PointOneEuroFilter] = {
            alg: PointOneEuroFilter(min_cutoff=0.85, beta=0.007) for alg in ALGORITHMS
        }

        # Pointer trails
        self.pointer_trails = {alg: PointerTrail(maxlen=12) for alg in ALGORITHMS}

        # Current State
        self.state = self.STATE_ALIGNMENT
        self.paused = False
        self.frozen_frame = None

        # State Timers & Steps
        self.state_t0 = time.time()
        self.flash_msg = "ALIGNMENT: Adjust circle with [W/A/S/D / Arrows] & [+/-] | Press [SPACE] to Start"
        self.flash_t = time.time()

        # Calibration state variables
        self.calib_idx = 0
        self.calib_phase = "APPROACH"  # APPROACH, COLLECT, FLASH
        self.calib_phase_t0 = time.time()
        self.calib_samples: Dict[str, List[Tuple[float, float]]] = {alg: [] for alg in ALGORITHMS}

        # Static testing state variables
        self.test_idx = 0
        self.test_fixation_buffer: Dict[str, List[Tuple[float, float]]] = {alg: [] for alg in ALGORITHMS}
        self.test_confirm_flash_t0 = 0.0

        # Dynamic pursuit state variables
        self.pursuit_idx = 0
        self.pursuit_phase = "APPROACH"  # APPROACH, TRACK
        self.pursuit_phase_t0 = time.time()

        # Output paths
        self.output_dir = _THIS_DIR
        self.charts_path = os.path.join(self.output_dir, "benchmark_results_charts.png")
        self.report_path = os.path.join(self.output_dir, "BENCHMARK_REPORT.md")
        self.json_path = os.path.join(self.output_dir, f"benchmark_data_{self.session.timestamp}.json")

    # -------------------------------------------------------------------------
    # Frame Pre-processing & Caging
    # -------------------------------------------------------------------------
    def get_caged_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.paused or self.frozen_frame is None:
            ret, raw_frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                raw_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
            frame = cv2.resize(raw_frame, (640, 480))
            self.frozen_frame = frame.copy()
        else:
            frame = self.frozen_frame.copy()

        # Build circular eyeball mask
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (self.circle_x, self.circle_y), self.circle_r, 255, -1)

        caged = frame.copy()
        caged[mask == 0] = (240, 240, 240)
        return frame, caged

    # -------------------------------------------------------------------------
    # Render State 0: ALIGNMENT & CIRCLE ADJUSTMENT
    # -------------------------------------------------------------------------
    def render_alignment(self, canvas: np.ndarray, frame: np.ndarray, detections: Dict[str, Dict[str, Any]]):
        h, w = canvas.shape[:2]

        # Top Header Banner
        cv2.rectangle(canvas, (0, 0), (w, 75), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, 75), (w, 75), COLOR_BORDER, 1)
        cv2.putText(canvas, "PHASE 1: PRE-BENCHMARK EYE ALIGNMENT & BOUNDARY CAGING",
                    (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Adjust circle to fit your eyeball strictly below your eyebrow before starting",
                    (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Center camera preview (640x480)
        cam_x = 40
        cam_y = 100
        cam_view = frame.copy()

        # Draw circle on camera preview
        cv2.circle(cam_view, (self.circle_x, self.circle_y), self.circle_r, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(cam_view, (self.circle_x, self.circle_y), 3, (0, 230, 255), -1, cv2.LINE_AA)
        
        # Draw detections on camera preview
        for alg in ALGORITHMS:
            d = detections[alg]
            ell = d.get("ellipse")
            if ell:
                cv2.ellipse(cam_view, ell, ALG_COLORS[alg], 2, cv2.LINE_AA)
                cx, cy = int(round(ell[0][0])), int(round(ell[0][1]))
                cv2.circle(cam_view, (cx, cy), 3, ALG_COLORS[alg], -1, cv2.LINE_AA)

        # Cage sizing guidance badge
        if self.circle_r > 140:
            cv2.rectangle(cam_view, (10, 440), (630, 470), (0, 40, 180), -1)
            cv2.putText(cam_view, f"WARNING: Radius {self.circle_r}px is large! Press [-] to shrink to ~100-120px to exclude eyebrows",
                        (15, 461), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(cam_view, (10, 440), (630, 470), (30, 30, 35), -1)
            cv2.putText(cam_view, f"Optimal Eye Cage: Center ({self.circle_x}, {self.circle_y}) | Radius {self.circle_r}px (Eyebrows excluded)",
                        (15, 461), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 255, 120), 1, cv2.LINE_AA)

        canvas[cam_y:cam_y + 480, cam_x:cam_x + 640] = cam_view
        cv2.rectangle(canvas, (cam_x, cam_y), (cam_x + 640, cam_y + 480), COLOR_BORDER, 2)

        # Right Telemetry & Controls Cards
        card_x = cam_x + 640 + 35
        card_w = w - card_x - 40
        card_y = cam_y

        # Card 1: 4 Algorithms Status
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 245), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 245), COLOR_BORDER, 1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 4), (0, 230, 255), -1)
        cv2.putText(canvas, "LIVE ALGORITHM READINESS CHECK", (card_x + 15, card_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

        row_y = card_y + 60
        all_ready = True
        for alg in ALGORITHMS:
            d = detections[alg]
            detected = d["detected"]
            if not detected:
                all_ready = False
            stat_str = "LOCKED" if detected else "SEARCHING"
            stat_col = (80, 255, 100) if detected else (60, 60, 255)

            cv2.circle(canvas, (card_x + 20, row_y - 4), 6, ALG_COLORS[alg], -1, cv2.LINE_AA)
            cv2.putText(canvas, alg, (card_x + 35, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
            cv2.putText(canvas, stat_str, (card_x + 190, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, stat_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{d['latency_ms']:.1f}ms", (card_x + 290, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            row_y += 42

        # Card 2: Interactive Controls
        ctrl_y = card_y + 265
        cv2.rectangle(canvas, (card_x, ctrl_y), (card_x + card_w, ctrl_y + 215), COLOR_PANEL_BG, -1)
        cv2.rectangle(canvas, (card_x, ctrl_y), (card_x + card_w, ctrl_y + 215), COLOR_BORDER, 1)
        cv2.rectangle(canvas, (card_x, ctrl_y), (card_x + card_w, ctrl_y + 4), (80, 255, 120), -1)
        cv2.putText(canvas, "KEYBOARD ADJUSTMENTS", (card_x + 15, ctrl_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 120), 1, cv2.LINE_AA)

        controls_list = [
            ("[W / A / S / D] or [Arrows]", "Move Circle (Up / Left / Down / Right)"),
            ("[+] / [-] or [ = / - ]",     "Resize Eyeball Circle Radius"),
            ("[M]",                         f"Calib Mode: {self.calib_mode} (Toggle SOTA / Classic)"),
            ("[C]",                         f"Switch Active Camera (Cam: {self.cam_id})"),
            ("[Space] or [Enter]",          "CONFIRM & START BENCHMARK"),
            ("[Q] / ESC",                   "Quit Benchmark Suite"),
        ]
        crow_y = ctrl_y + 60
        for key_s, desc in controls_list:
            cv2.putText(canvas, key_s, (card_x + 15, crow_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, desc, (card_x + 235, crow_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            crow_y += 30

        # Bottom Action Bar
        cv2.rectangle(canvas, (0, h - 65), (w, h), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, h - 65), (w, h - 65), COLOR_BORDER, 1)

        btn_w = 480
        btn_h = 44
        bx1 = w // 2 - btn_w // 2
        by1 = h - 55
        btn_col = (80, 255, 100) if all_ready else (0, 180, 255)
        cv2.rectangle(canvas, (bx1, by1), (bx1 + btn_w, by1 + btn_h), btn_col, 2, cv2.LINE_AA)
        msg = "PRESS [SPACE] TO START BENCHMARK" if all_ready else "ADJUST CIRCLE ON EYE -- PRESS [SPACE] TO START"
        cv2.putText(canvas, msg, (bx1 + 22, by1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, btn_col, 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Render State 1: 100% FAITHFUL PRECISION CALIBRATION (Multi-Algorithm)
    # -------------------------------------------------------------------------
    def process_calibration(self, canvas: np.ndarray, detections: Dict[str, Dict[str, Any]]):
        """
        Drives the 100% replicated PrecisionCalibrator across all 4 algorithms:
        Lissajous warmup -> 9 static points (Approach/Collect/Flash) ->
        8 smooth pursuit tracking paths -> degree-2 polynomial fitting ->
        4 validation points -> summary screen.
        """
        now = time.time()
        calib_state = self.multi_calibrator.tick(detections, now)
        overlay_frame = draw_frame(self.screen_w, self.screen_h, calib_state)
        canvas[:] = overlay_frame

        if self.multi_calibrator.done:
            # When complete, display the summary screen for 2.5s before advancing to Level 2
            if now - self.multi_calibrator._phase_t0 >= 2.5:
                self.finish_calibration()

    def finish_calibration(self):
        """Advances from Calibration 1 to Level 2 Testing with synchronized polynomials."""
        print("\n" + "="*70)
        print("100% CALIBRATION COMPLETE -- SYNCHRONIZING ALL 4 ALGORITHMS TO LEVEL 2")
        print("="*70)
        for alg in ALGORITHMS:
            cx = self.multi_calibrator.cx.get(alg)
            cy = self.multi_calibrator.cy.get(alg)
            if cx is not None and cy is not None:
                self.calibrators[alg].cx = cx.copy()
                self.calibrators[alg].cy = cy.copy()
                self.calibrators[alg].is_calibrated = True
                self.calibrators[alg].src_pts = list(self.multi_calibrator._src_pts[alg])
                self.calibrators[alg].dst_pts = list(self.multi_calibrator._dst_pts[alg])

                # Compute train RMS
                src = np.array(self.multi_calibrator._src_pts[alg], dtype=np.float64)
                dst = np.array(self.multi_calibrator._dst_pts[alg], dtype=np.float64)
                X = np.array([self.calibrators[alg]._feat(p[0], p[1]) for p in src])
                pred_x = X @ cx
                pred_y = X @ cy
                rms = float(np.sqrt(np.mean((pred_x - dst[:, 0])**2 + (pred_y - dst[:, 1])**2)))
                self.calibrators[alg].train_rms_px = rms

                val_errs = self.multi_calibrator._val_errors.get(alg, [])
                val_mean = float(np.mean(val_errs)) if val_errs else 0.0

                self.session.calibration_summary[alg] = {
                    "fit_success": True,
                    "train_rms": round(rms, 2),
                    "points": len(src),
                    "val_errors": list(val_errs),
                    "val_mean_error": round(val_mean, 2),
                }
                print(f"  {alg:15s} -> Fit RMS: {rms:.1f}px | Val Error: {val_mean:.1f}px | Points: {len(src)}")
            else:
                self.calibrators[alg].is_calibrated = False
                self.session.calibration_summary[alg] = {
                    "fit_success": False,
                    "train_rms": 999.0,
                    "points": len(self.multi_calibrator._src_pts.get(alg, [])),
                    "val_errors": [],
                    "val_mean_error": 999.0,
                }
                print(f"  {alg:15s} -> FIT FAILED (Insufficient points)")
        print("="*70 + "\n")

        play_beep(1800, 150)
        self.state = self.STATE_STATIC_TEST
        self.state_t0 = time.time()
        self.test_idx = 0
        self.test_fixation_buffer = {alg: [] for alg in ALGORITHMS}
        self.test_confirm_flash_t0 = 0.0

    # -------------------------------------------------------------------------
    # Render State 2: LEVEL 2 TESTING (Static Targets + 4 Colored Pointers + Space Confirm)
    # -------------------------------------------------------------------------
    def process_static_testing(self, canvas: np.ndarray, detections: Dict[str, Dict[str, Any]]):
        now = time.time()
        w = self.screen_w
        h = self.screen_h

        norm_x, norm_y = self.TEST_TARGETS[self.test_idx]
        tx = int(norm_x * w)
        ty = int(norm_y * h)

        # Top Instruction Banner
        cv2.rectangle(canvas, (0, 0), (w, 80), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, 80), (w, 80), COLOR_BORDER, 1)

        cv2.putText(canvas, f"LEVEL 2 BENCHMARK TESTING: Target {self.test_idx + 1} of {len(self.TEST_TARGETS)}",
                    (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Look steadily at the Target Dot, inspect the 4 live algorithm pointers, and press [SPACE] to Confirm",
                    (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Pulsing Target Bullseye
        pulse = 0.5 + 0.5 * math.sin(now * 5.0)
        t_radius = int(14 + 4 * pulse)
        cv2.circle(canvas, (tx, ty), t_radius + 12, (100, 100, 120), 1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), t_radius, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(canvas, (tx - 24, ty), (tx + 24, ty), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (tx, ty - 24), (tx, ty + 24), (255, 255, 255), 1, cv2.LINE_AA)

        # Compute & Render 4 Live Colored Pointers
        live_preds: Dict[str, Optional[Tuple[float, float]]] = {}
        hud_items = []

        for alg in ALGORITHMS:
            c = detections[alg]["center"]
            pred = None
            if c is not None:
                raw_pred = self.calibrators[alg].predict(c[0], c[1])
                if raw_pred is not None:
                    if not (math.isnan(raw_pred[0]) or math.isnan(raw_pred[1]) or math.isinf(raw_pred[0]) or math.isinf(raw_pred[1])):
                        # Apply 1-Euro adaptive low-pass filter (sub-pixel fixation smoothing with zero saccadic delay)
                        filtered_xy = self.one_euro_filters[alg].filter(raw_pred[0], raw_pred[1], now)
                        pred = filtered_xy

            live_preds[alg] = pred
            self.pointer_trails[alg].add(pred)
            self.pointer_trails[alg].draw(canvas, ALG_COLORS[alg])

            if pred is not None:
                col = ALG_COLORS[alg]
                dist_px = math.hypot(pred[0] - tx, pred[1] - ty)

                # Clamp drawn indicator coordinates inside canvas boundary
                px = int(round(max(20, min(w - 20, pred[0]))))
                py = int(round(max(95, min(h - 65, pred[1]))))

                # Draw pointer crosshair & dot
                cv2.circle(canvas, (px, py), 4, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 10, col, 2, cv2.LINE_AA)
                cv2.line(canvas, (px - 14, py), (px + 14, py), col, 1, cv2.LINE_AA)
                cv2.line(canvas, (px, py - 14), (px, py + 14), col, 1, cv2.LINE_AA)

                # Distance line connecting target to pointer
                cv2.line(canvas, (tx, ty), (px, py), col, 1, cv2.LINE_AA)

                # Off-screen indicator tag if prediction lies beyond screen edge
                is_off = (pred[0] < 0 or pred[0] > w or pred[1] < 0 or pred[1] > h)
                lbl_txt = f"{alg}: {dist_px:.0f}px [EDGE]" if is_off else f"{alg}: {dist_px:.0f}px"
                cv2.putText(canvas, lbl_txt, (px + 14, py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

                # Saccadic filter (I-VT): discard high-velocity transition frames when moving between targets
                if len(self.test_fixation_buffer[alg]) > 0:
                    last_pt = self.test_fixation_buffer[alg][-1]
                    step_v = math.hypot(pred[0] - last_pt[0], pred[1] - last_pt[1])
                    if step_v > 45.0:  # Eye is in rapid saccadic motion; flush previous target residual
                        self.test_fixation_buffer[alg].clear()

                # Append to fixation buffer (last 35 settled fixation samples)
                self.test_fixation_buffer[alg].append((float(pred[0]), float(pred[1])))
                if len(self.test_fixation_buffer[alg]) > 35:
                    self.test_fixation_buffer[alg].pop(0)

                hud_items.append((alg, f"{dist_px:.1f} px", col))
            else:
                hud_items.append((alg, "DROPOUT", (60, 60, 255)))

        # Bottom HUD bar showing live errors
        cv2.rectangle(canvas, (0, h - 50), (w, h), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, h - 50), (w, h - 50), COLOR_BORDER, 1)

        hud_x = 30
        for alg, val_s, col in hud_items:
            cv2.circle(canvas, (hud_x, h - 25), 6, col, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{alg}: {val_s}", (hud_x + 14, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
            hud_x += 280

        # Confirmation flash ripple
        if now - self.test_confirm_flash_t0 < 0.35:
            cv2.circle(canvas, (tx, ty), 45, (80, 255, 100), 3, cv2.LINE_AA)
            cv2.putText(canvas, "CONFIRMED!", (tx - 50, ty - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 100), 2, cv2.LINE_AA)

    def confirm_static_test_target(self, detections: Dict[str, Dict[str, Any]]):
        """Called when user presses [SPACE] during static testing."""
        norm_x, norm_y = self.TEST_TARGETS[self.test_idx]
        tx = int(norm_x * self.screen_w)
        ty = int(norm_y * self.screen_h)

        record_preds = {}
        for alg in ALGORITHMS:
            buf = self.test_fixation_buffer[alg]
            if len(buf) >= 3:
                pts = np.array(buf, dtype=np.float64)
                # Compute fixation dispersion (jitter)
                med_xy = np.median(pts, axis=0)
                dispersion = float(np.mean(np.hypot(pts[:, 0] - med_xy[0], pts[:, 1] - med_xy[1])))
                err = float(math.hypot(med_xy[0] - tx, med_xy[1] - ty))
                record_preds[alg] = {
                    "gaze_xy": (float(med_xy[0]), float(med_xy[1])),
                    "error_px": round(err, 1),
                    "jitter_rms_px": round(dispersion, 2),
                    "latency_ms": round(detections[alg]["latency_ms"], 2),
                    "samples": len(buf),
                }
            else:
                record_preds[alg] = {
                    "gaze_xy": None,
                    "error_px": 999.0,
                    "jitter_rms_px": 99.0,
                    "latency_ms": round(detections[alg]["latency_ms"], 2),
                    "samples": 0,
                }

        self.session.static_test_records.append({
            "target_idx": self.test_idx,
            "target_xy": (tx, ty),
            "predictions": record_preds,
        })

        play_beep(1500, 60)
        self.test_confirm_flash_t0 = time.time()
        self.test_idx += 1
        self.test_fixation_buffer = {alg: [] for alg in ALGORITHMS}
        for alg in ALGORITHMS:
            self.one_euro_filters[alg].reset()

        if self.test_idx >= len(self.TEST_TARGETS):
            # Advance to dynamic tracking test
            self.state = self.STATE_DYNAMIC_TEST
            self.pursuit_idx = 0
            self.pursuit_phase = "APPROACH"
            self.pursuit_phase_t0 = time.time()
            play_beep(1800, 100)

    # -------------------------------------------------------------------------
    # Render State 3: LEVEL 3 DYNAMIC PURSUIT TRACKING TEST
    # -------------------------------------------------------------------------
    def process_dynamic_testing(self, canvas: np.ndarray, detections: Dict[str, Dict[str, Any]]):
        now = time.time()
        elapsed = now - self.pursuit_phase_t0
        w = self.screen_w
        h = self.screen_h

        path_start, path_end, path_label = self.PURSUIT_PATHS[self.pursuit_idx]
        sx = int(path_start[0] * w)
        sy = int(path_start[1] * h)
        ex = int(path_end[0] * w)
        ey = int(path_end[1] * h)

        # Header Info Banner
        cv2.rectangle(canvas, (0, 0), (w, 80), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, 80), (w, 80), COLOR_BORDER, 1)

        cv2.putText(canvas, f"DYNAMIC PURSUIT TRACKING: Path {self.pursuit_idx + 1} of {len(self.PURSUIT_PATHS)} ({path_label})",
                    (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Smoothly follow the moving dot with your eyes across the screen",
                    (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Draw guideline path
        cv2.line(canvas, (sx, sy), (ex, ey), (40, 45, 55), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # Approach to path start (1.2 s)
        # ---------------------------------------------------------------------
        if self.pursuit_phase == "APPROACH":
            duration = 1.2
            frac = min(1.0, elapsed / duration)
            r = int(45 - 32 * frac)
            cv2.circle(canvas, (sx, sy), r, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (sx, sy), 7, (0, 230, 255), -1, cv2.LINE_AA)

            if elapsed >= duration:
                self.pursuit_phase = "TRACK"
                self.pursuit_phase_t0 = now

        # ---------------------------------------------------------------------
        # Track along path (3.5 s)
        # ---------------------------------------------------------------------
        elif self.pursuit_phase == "TRACK":
            duration = 3.5
            frac = min(1.0, elapsed / duration)
            cur_tx = int(sx + (ex - sx) * frac)
            cur_ty = int(sy + (ey - sy) * frac)

            # Draw moving target dot with glowing halo
            cv2.circle(canvas, (cur_tx, cur_ty), 18, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (cur_tx, cur_ty), 8, (255, 255, 255), -1, cv2.LINE_AA)

            # Record frame data for each algorithm
            frame_preds = {}
            for alg in ALGORITHMS:
                c = detections[alg]["center"]
                pred = None
                tracked = False
                err = 999.0
                if c is not None:
                    raw_pred = self.calibrators[alg].predict(c[0], c[1])
                    if raw_pred is not None:
                        if not (math.isnan(raw_pred[0]) or math.isnan(raw_pred[1]) or math.isinf(raw_pred[0]) or math.isinf(raw_pred[1])):
                            pred = self.one_euro_filters[alg].filter(raw_pred[0], raw_pred[1], now)
                            tracked = True
                            err = math.hypot(pred[0] - cur_tx, pred[1] - cur_ty)
                            self.pointer_trails[alg].add(pred)
                            self.pointer_trails[alg].draw(canvas, ALG_COLORS[alg])
                            
                            # Clamp drawn indicator dot
                            px = int(round(max(15, min(w - 15, pred[0]))))
                            py = int(round(max(90, min(h - 65, pred[1]))))
                            cv2.circle(canvas, (px, py), 5, ALG_COLORS[alg], -1, cv2.LINE_AA)

                frame_preds[alg] = {
                    "gaze_xy": pred,
                    "tracked": tracked,
                    "error_px": round(err, 1) if tracked else 999.0,
                    "latency_ms": round(detections[alg]["latency_ms"], 2),
                }

            self.session.dynamic_frames.append({
                "t": round(now, 3),
                "target_xy": (cur_tx, cur_ty),
                "predictions": frame_preds,
            })

            # Bottom HUD bar
            cv2.rectangle(canvas, (0, h - 50), (w, h), COLOR_PANEL_BG, -1)
            cv2.line(canvas, (0, h - 50), (w, h - 50), COLOR_BORDER, 1)

            hud_x = 30
            for alg in ALGORITHMS:
                d = frame_preds.get(alg, {})
                col = ALG_COLORS[alg]
                err_str = f"{d.get('error_px', 999):.0f} px" if d.get('tracked') else "DROPOUT"
                cv2.circle(canvas, (hud_x, h - 25), 6, col, -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{alg}: {err_str}", (hud_x + 14, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
                hud_x += 280

            cv2.putText(canvas, "Press [SPACE] to Skip Path  |  [Q] Exit", (w - 330, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

            if elapsed >= duration:
                self.pursuit_idx += 1
                if self.pursuit_idx >= len(self.PURSUIT_PATHS):
                    # Benchmark complete! Advance to Results
                    self.finalize_benchmark()
                else:
                    self.pursuit_phase = "APPROACH"
                    self.pursuit_phase_t0 = now

    # -------------------------------------------------------------------------
    # Finalize Benchmark & Generate Reports
    # -------------------------------------------------------------------------
    def finalize_benchmark(self):
        self.state = self.STATE_RESULTS
        self.state_t0 = time.time()
        play_beep(2000, 250)

        print("\n" + "="*70)
        print("BENCHMARK COMPLETED! COMPUTING METRICS AND GENERATING REPORTS...")
        print("="*70)

        # 1. Compute statistical metrics
        compute_benchmark_metrics(self.session)

        # 2. Generate 6-Panel Visualization Charts
        generate_benchmark_visualizations(self.session, self.charts_path)

        # 3. Generate Detailed Markdown Report
        generate_markdown_report(self.session, self.report_path)

        # 4. Save Raw JSON Telemetry
        self.session.save_json(self.json_path)

        print(f"-> Charts saved to: {self.charts_path}")
        print(f"-> Markdown report saved to: {self.report_path}")
        print(f"-> Raw data JSON saved to: {self.json_path}")
        print("="*70 + "\n")

    # -------------------------------------------------------------------------
    # Render State 4: RESULTS DASHBOARD
    # -------------------------------------------------------------------------
    def render_results(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        metrics = self.session.metrics
        rankings = self.session.rankings

        # Top Header Banner
        cv2.rectangle(canvas, (0, 0), (w, 80), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, 80), (w, 80), COLOR_BORDER, 1)
        cv2.putText(canvas, "4-ALGORITHM EMPIRICAL BENCHMARK -- FINAL RESULTS & RANKINGS",
                    (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Detailed report saved to BENCHMARK_REPORT.md  |  Multi-panel charts saved to benchmark_results_charts.png",
                    (25, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Podium Cards for all 4 algorithms
        card_w = (w - 100) // 4
        card_h = h - 220
        y_top = 110

        medals = ["1st PLACE (WINNER)", "2nd PLACE", "3rd PLACE", "4th PLACE"]
        medal_colors = [(0, 215, 255), (192, 192, 192), (60, 120, 205), (100, 100, 110)]

        for idx, (alg, score) in enumerate(rankings):
            m = metrics[alg]
            cx1 = 40 + idx * (card_w + 20)
            cx2 = cx1 + card_w
            cy1 = y_top
            cy2 = cy1 + card_h

            accent = ALG_COLORS[alg]

            # Card background
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_PANEL_BG, -1)
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_BORDER, 1)

            # Top accent bar
            cv2.rectangle(canvas, (cx1, cy1), (cx2, cy1 + 6), accent, -1)

            # Medal Badge
            cv2.putText(canvas, medals[idx], (cx1 + 15, cy1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, medal_colors[idx], 1, cv2.LINE_AA)
            cv2.putText(canvas, alg, (cx1 + 15, cy1 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, accent, 2, cv2.LINE_AA)

            # Score badge
            cv2.rectangle(canvas, (cx1 + 15, cy1 + 80), (cx2 - 15, cy1 + 125), (40, 35, 30), -1)
            cv2.putText(canvas, f"SCORE: {score:.1f} / 100", (cx1 + 25, cy1 + 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 120), 2, cv2.LINE_AA)

            # Detailed metrics
            stat_rows = [
                ("Mean Gaze Error:", f"{m['mean_error_px']:.1f} px", (255, 255, 255)),
                ("Visual Angle Error:", f"{m['mean_error_deg']:.2f}°", (0, 230, 255)),
                ("95% Worst Error:", f"{m['p95_error_px']:.1f} px", (200, 200, 210)),
                ("Fixation Jitter (RMS):", f"{m['fixation_jitter_rms_px']:.2f} px", (80, 255, 120) if m['fixation_jitter_rms_px'] < 3.0 else (0, 165, 255)),
                ("Dynamic Jitter:", f"{m['dynamic_jitter_rms_px']:.2f} px", COLOR_TEXT_MUTED),
                ("Tracking Uptime:", f"{m['uptime_pct']:.1f}%", (80, 255, 120) if m['uptime_pct'] > 95 else (60, 60, 255)),
                ("60px Button Hit:", f"{m['button_hit_rate_pct']:.1f}%", (0, 230, 255)),
                ("Mean Latency:", f"{m['mean_latency_ms']:.1f} ms", (255, 255, 255)),
                ("Effective Speed:", f"{m['effective_fps']:.0f} FPS", (80, 255, 120)),
            ]

            sy = cy1 + 155
            for lbl, val_s, col in stat_rows:
                cv2.putText(canvas, lbl, (cx1 + 15, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(canvas, val_s, (cx2 - 110, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
                sy += 24

        # Footer Action Bar
        cv2.rectangle(canvas, (0, h - 60), (w, h), COLOR_PANEL_BG, -1)
        cv2.line(canvas, (0, h - 60), (w, h - 60), COLOR_BORDER, 1)

        cv2.putText(canvas, "Press [R] to Run Benchmark Again  |  Press [Q] or [ESC] to Exit",
                    (w // 2 - 250, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # Main Application Loop
    # -------------------------------------------------------------------------
    def run(self):
        # Set High-DPI awareness on Windows so fullscreen spans true native pixels
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

        win_name = "Eye Tracking Module — 4-Algorithm Benchmark Suite"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(win_name, self.screen_w, self.screen_h)

        while True:
            # 1. Grab camera frame and apply eyeball boundary circle cage
            raw_frame, caged_frame = self.get_caged_frame()

            # 2. Run all 4 algorithms concurrently
            detections = self.pipeline.process_all(caged_frame, self.circle_x, self.circle_y, self.circle_r)

            # 3. Create fresh canvas
            canvas = np.full((self.screen_h, self.screen_w, 3), COLOR_BG, dtype=np.uint8)

            # 4. Render active state
            if self.state == self.STATE_ALIGNMENT:
                self.render_alignment(canvas, raw_frame, detections)
            elif self.state == self.STATE_CALIBRATION:
                self.process_calibration(canvas, detections)
            elif self.state == self.STATE_STATIC_TEST:
                self.process_static_testing(canvas, detections)
            elif self.state == self.STATE_DYNAMIC_TEST:
                self.process_dynamic_testing(canvas, detections)
            elif self.state == self.STATE_RESULTS:
                self.render_results(canvas)

            cv2.imshow(win_name, canvas)

            # Handle User Input
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

            # State-specific Key Handlers
            if self.state == self.STATE_ALIGNMENT:
                if key in (ord(' '), 13):  # Space or Enter
                    play_beep(1400, 80)
                    self.state = self.STATE_CALIBRATION
                    self.state_t0 = time.time()
                    self.multi_calibrator.start(t=self.state_t0)
                # Move circle UP
                elif key in (ord('w'), ord('W'), 82, 2490368):
                    self.circle_y = max(20, self.circle_y - 5)
                    self.session.circle_y = self.circle_y
                # Move circle DOWN
                elif key in (ord('s'), ord('S'), 84, 2621440):
                    self.circle_y = min(460, self.circle_y + 5)
                    self.session.circle_y = self.circle_y
                # Move circle LEFT
                elif key in (ord('a'), ord('A'), 81, 2424832):
                    self.circle_x = max(20, self.circle_x - 5)
                    self.session.circle_x = self.circle_x
                # Move circle RIGHT
                elif key in (ord('d'), ord('D'), 83, 2555904):
                    self.circle_x = min(620, self.circle_x + 5)
                    self.session.circle_x = self.circle_x
                # Increase circle radius
                elif key in (ord('+'), ord('='), ord(']')):
                    self.circle_r = min(220, self.circle_r + 5)
                    self.session.circle_r = self.circle_r
                # Decrease circle radius
                elif key in (ord('-'), ord('_'), ord('[')):
                    self.circle_r = max(25, self.circle_r - 5)
                    self.session.circle_r = self.circle_r
                # Toggle calibration mode (SOTA vs CLASSIC)
                elif key in (ord('m'), ord('M')):
                    self.calib_mode = "CLASSIC" if self.calib_mode == "SOTA" else "SOTA"
                    self.session.calib_mode = self.calib_mode
                    self.multi_calibrator.mode = self.calib_mode
                    for alg in ALGORITHMS:
                        self.calibrators[alg].mode = self.calib_mode
                    play_beep(1600, 70)
                    print(f"[BenchmarkGUI] Calibration mode toggled to: {self.calib_mode}")
                # Switch camera
                elif key in (ord('c'), ord('C')):
                    if len(self.available_cams) > 1:
                        ptr = (self.available_cams.index(self.cam_id) + 1) % len(self.available_cams)
                        self.cam_id = self.available_cams[ptr]
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
                        if not self.cap.isOpened():
                            self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_MSMF)

            elif self.state == self.STATE_CALIBRATION:
                if key in (ord(' '), 13) and self.multi_calibrator.done:
                    self.finish_calibration()

            elif self.state == self.STATE_STATIC_TEST:
                if key in (ord(' '), 13):  # User pressed SPACE to confirm static target
                    self.confirm_static_test_target(detections)

            elif self.state == self.STATE_DYNAMIC_TEST:
                if key in (ord(' '), 13):  # User pressed SPACE to skip pursuit path
                    self.pursuit_idx += 1
                    if self.pursuit_idx >= len(self.PURSUIT_PATHS):
                        self.finalize_benchmark()
                    else:
                        self.pursuit_phase = "APPROACH"
                        self.pursuit_phase_t0 = time.time()

            elif self.state == self.STATE_RESULTS:
                if key in (ord('r'), ord('R')):
                    # Reset and run again
                    self.state = self.STATE_ALIGNMENT
                    self.session = BenchmarkSession(self.screen_w, self.screen_h)
                    self.calibrators = {alg: PolynomialGazeCalibrator(alg, self.screen_w, self.screen_h) for alg in ALGORITHMS}
                    self.one_euro_filters = {alg: PointOneEuroFilter(min_cutoff=0.85, beta=0.007) for alg in ALGORITHMS}
                    self.multi_calibrator.reset()

        self.cap.release()
        cv2.destroyAllWindows()

