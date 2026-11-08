#!/usr/bin/env python3
"""
compare_live.py
===============
4-Way Real-Time Algorithm Comparison with Eyeball Boundary Circle Caging:
  1. Top-Left:     Orlosky Pupil Detector (Baseline 1: High uptime, high jitter)
  2. Top-Right:    Pupil Labs Detector2D  (Baseline 2: Ultra-fast, low jitter, occasional dropouts)
  3. Bottom-Left:  Hybrid Fusion Detector (PupilLabs + Constrained Adaptive Fallback)
  4. Bottom-Right: PuReST SOTA Tracker   (World Champion on LPW Benchmark: Santini et al.)

Feeds frames from a single camera to all 4 algorithms simultaneously,
constrained within an adjustable Eyeball Boundary Circle that eliminates
eyebrow and forehead false locks.

Controls:
  [W] / [Up]    → Move circle UP
  [S] / [Down]  → Move circle DOWN
  [A] / [Left]  → Move circle LEFT
  [D] / [Right] → Move circle RIGHT
  [+] / [=]     → Increase circle radius (larger eyeball area)
  [-] / [_]     → Decrease circle radius (smaller eyeball area)
  [Space]       → Pause / unpause live stream
  [M]           → Cycle view mode (4-Way 2x2 Grid / Unified 4-Way Overlay / PupilLabs vs PuReST)
  [P]           → Print current circle coordinates to console
  [C]           → Switch camera (cycles through available camera IDs)
  [Q] / ESC     → Quit
"""

import sys
import os
import time
import math
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

# Silence noisy internal OpenCV C++ driver probe warnings on Windows
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

import numpy as np

# Ensure this folder is at top of path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# =============================================================================
# EYEBALL BOUNDARY CIRCLE CONFIGURATION (Lines 55 - 57)
# >>> MODIFY THESE 3 VALUES TO FIT THE CIRCLE EXACTLY TO YOUR EYE <<<
# =============================================================================
EYEBALL_CIRCLE_X      = 320   # Horizontal center (0 = left edge, 640 = right edge)
EYEBALL_CIRCLE_Y      = 240   # Vertical center   (0 = top edge,  480 = bottom edge)
EYEBALL_CIRCLE_RADIUS = 110   # Circle radius in pixels (increase/decrease to fit eye)
# =============================================================================


# -----------------------------------------------------------------------------
# Import All 4 Detectors
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
# Color Palette
# -----------------------------------------------------------------------------
COLOR_BG        = (24, 24, 30)      # Catppuccin Mantle dark background
COLOR_PANEL_BG  = (32, 32, 40)      # Card / Banner background
COLOR_BORDER    = (60, 60, 75)      # Divider line
COLOR_TEXT_MAIN = (240, 240, 245)   # Crisp white
COLOR_TEXT_MUTED= (160, 160, 175)   # Slate gray
COLOR_ORLOSKY   = (255, 190, 30)    # Amber / Gold (Alg 1)
COLOR_PL        = (50, 240, 120)    # Vibrant Emerald / Neon Green (Alg 2)
COLOR_HYBRID    = (0, 220, 255)     # Cyan / Sky Blue (Alg 3)
COLOR_PUREST    = (240, 80, 210)    # Magenta / Electric Pink (Alg 4: PuReST)
COLOR_CIRCLE    = (0, 230, 255)     # Glowing Cyan (Eyeball boundary ring)
COLOR_RED       = (70, 70, 255)     # Searching / Warning Red


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
        darkest_point = get_darkest_area(frame_640x480)
        if darkest_point is None:
            dt = (time.perf_counter() - t0) * 1000.0
            return None, dt

        gray = cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2GRAY)
        dpv = gray[darkest_point[1], darkest_point[0]]

        t_strict  = mask_outside_square(apply_binary_threshold(gray, dpv,  5), darkest_point, 250)
        t_medium  = mask_outside_square(apply_binary_threshold(gray, dpv, 15), darkest_point, 250)
        t_relaxed = mask_outside_square(apply_binary_threshold(gray, dpv, 25), darkest_point, 250)

        result = process_frames(t_strict, t_medium, t_relaxed, frame_640x480, gray, darkest_point, False, False)
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

    run_starkalman = run_purest


# -----------------------------------------------------------------------------
# Rendering Helpers
# -----------------------------------------------------------------------------
def draw_hud_card(img: np.ndarray, x: int, y: int, w: int, h: int, title: str,
                  items: list, accent_color: Tuple[int, int, int]):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_BORDER, 1)

    # Accent top-bar
    cv2.rectangle(img, (x, y), (x + w, y + 4), accent_color, -1)
    # Title
    cv2.putText(img, title, (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent_color, 1, cv2.LINE_AA)

    # Items
    curr_y = y + 38
    for label, val, val_col in items:
        cv2.putText(img, label, (x + 8, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(img, str(val), (x + 85, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, val_col, 1, cv2.LINE_AA)
        curr_y += 16


def draw_crosshair(img: np.ndarray, cx: float, cy: float, color: Tuple[int, int, int], size: int = 10):
    icx, icy = int(round(cx)), int(round(cy))
    cv2.circle(img, (icx, icy), 3, color, -1, cv2.LINE_AA)
    cv2.circle(img, (icx, icy), 8, color, 1, cv2.LINE_AA)
    cv2.line(img, (icx - size, icy), (icx - 3, icy), color, 1, cv2.LINE_AA)
    cv2.line(img, (icx + 3, icy), (icx + size, icy), color, 1, cv2.LINE_AA)
    cv2.line(img, (icx, icy - size), (icx, icy - 3), color, 1, cv2.LINE_AA)
    cv2.line(img, (icx, icy + 3), (icx, icy + size), color, 1, cv2.LINE_AA)


def draw_eyeball_boundary_circle(img: np.ndarray, cx: int, cy: int, r: int,
                                 accent_color: Tuple[int, int, int] = COLOR_CIRCLE):
    """
    Renders the eyeball boundary circle with center reticle and subtle dark vignette outside.
    """
    # Outer circle ring
    cv2.circle(img, (cx, cy), r, accent_color, 2, cv2.LINE_AA)
    
    # 4 cardinal alignment ticks
    tick_len = 8
    cv2.line(img, (cx - r - tick_len, cy), (cx - r + tick_len, cy), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + r - tick_len, cy), (cx + r + tick_len, cy), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - r - tick_len), (cx, cy - r + tick_len), accent_color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + r - tick_len), (cx, cy + r + tick_len), accent_color, 1, cv2.LINE_AA)

    # Center target dot
    cv2.circle(img, (cx, cy), 2, accent_color, -1, cv2.LINE_AA)


def detect_cameras(max_cams: int = 4) -> List[int]:
    cams = []
    for i in range(max_cams):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    cams.append(i)
                    break
    return cams or [1]


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    print("""
+-------------------------------------------------------------------------------+
|         EYE TRACKING MODULE -- 4-ALGORITHM LIVE COMPARISON & EYEBALL CAGE     |
|    1. Orlosky  |  2. Pupil Labs  |  3. Hybrid Fusion  |  4. PuReST (SOTA)     |
|                                                                               |
|   EYEBALL CIRCLE CONTROLS:                                                    |
|     [W] / [A] / [S] / [D] or [Arrows] : Move Circle (Up / Left / Down / Right)|
|     [+] / [-] or [ = ] / [ - ]        : Resize Circle Radius                  |
|     [P]                               : Print Current Circle Coordinates      |
|     [Space]                           : Pause/Resume Live Stream              |
|     [M]                               : Toggle Layout (2x2 / Overlay / H2H)   |
|     [C]                               : Switch Camera                         |
|     [Q] / ESC                         : Quit Application                      |
+-------------------------------------------------------------------------------+
    """)

    available_cams = detect_cameras()
    
    # Check CLI argument for camera
    requested_cam = None
    if len(sys.argv) > 1:
        for idx, arg in enumerate(sys.argv[1:], 1):
            if arg in ('--camera', '-c') and idx < len(sys.argv) - 1:
                try:
                    requested_cam = int(sys.argv[idx + 1])
                except ValueError:
                    pass
            elif arg.isdigit():
                requested_cam = int(arg)

    if requested_cam is not None:
        cam_id = requested_cam
        cam_idx_ptr = available_cams.index(cam_id) if cam_id in available_cams else 0
    else:
        cam_idx_ptr = 0
        cam_id = available_cams[cam_idx_ptr]

    print(f"Available cameras: {available_cams}  --> Active: Camera {cam_id}")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_id}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pipeline = FourWayPipeline()

    win_title = 'Eye Tracking Module — 4-Way Comparison with Eyeball Boundary Circle'
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 800)

    paused = False
    frozen_frame = None
    view_mode = 0  # 0: 4-Way Grid, 1: Unified Multi-Overlay, 2: PupilLabs vs PuReST Head-to-Head
    mode_names = ["4-WAY 2x2 GRID", "UNIFIED 4-COLOR OVERLAY", "PUPIL LABS vs PuReST (SOTA)"]

    # Initialize circle from config lines 55 - 57
    circle_x = EYEBALL_CIRCLE_X
    circle_y = EYEBALL_CIRCLE_Y
    circle_r = EYEBALL_CIRCLE_RADIUS

    flash_message = f"EYEBALL CIRCLE CAGE: Center=({circle_x}, {circle_y}), Radius={circle_r}px"
    flash_time = time.time()

    # Rolling stats
    hist_lat_or = deque(maxlen=45)
    hist_lat_pl = deque(maxlen=45)
    hist_lat_hy = deque(maxlen=45)
    hist_lat_sk = deque(maxlen=45)
    fps_hist    = deque(maxlen=30)
    last_frame_t = time.time()

    snapshots_dir = os.path.join(_THIS_DIR, 'snapshots')
    os.makedirs(snapshots_dir, exist_ok=True)

    while True:
        now = time.time()

        if not paused or frozen_frame is None:
            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            frame = crop_to_aspect_ratio(raw_frame, 640, 480)
            frozen_frame = frame.copy()
        else:
            frame = frozen_frame.copy()

        dt_frame = max(now - last_frame_t, 1e-5)
        fps_hist.append(1.0 / dt_frame)
        last_frame_t = now
        avg_fps = np.mean(fps_hist)

        # ---------------------------------------------------------------------
        # Build Circular Eyeball Mask & Caged Frame
        # ---------------------------------------------------------------------
        h, w = frame.shape[:2]
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, (circle_x, circle_y), circle_r, 255, -1)

        # Pixels outside the circle are neutralized to bright 240
        # This guarantees eyebrows, eyelids, and hair outside the circle CANNOT be detected!
        caged_frame = frame.copy()
        caged_frame[circle_mask == 0] = (240, 240, 240)

        # ---------------------------------------------------------------------
        # Execute All 4 Pipelines on the CAGED Frame
        # ---------------------------------------------------------------------
        r_or, lat_or = pipeline.run_orlosky(caged_frame)
        r_pl, lat_pl, conf_pl = pipeline.run_pupil_labs(caged_frame)
        r_hy, lat_hy, conf_hy, src_hy = pipeline.run_hybrid(caged_frame)
        r_sk, lat_sk, conf_sk, src_sk = pipeline.run_starkalman(caged_frame)

        # Enforce boundary circle constraint: reject any center outside the circle
        if r_or and math.hypot(r_or[0][0] - circle_x, r_or[0][1] - circle_y) > circle_r:
            r_or = None
        if r_pl and math.hypot(r_pl[0][0] - circle_x, r_pl[0][1] - circle_y) > circle_r:
            r_pl = None
        if r_hy and math.hypot(r_hy[0][0] - circle_x, r_hy[0][1] - circle_y) > circle_r:
            r_hy = None
        if r_sk and math.hypot(r_sk[0][0] - circle_x, r_sk[0][1] - circle_y) > circle_r:
            r_sk = None

        hist_lat_or.append(lat_or)
        hist_lat_pl.append(lat_pl)
        hist_lat_hy.append(lat_hy)
        hist_lat_sk.append(lat_sk)

        circle_tag = f"Circle R={circle_r}"

        # ---------------------------------------------------------------------
        # Render Panel 1: Orlosky (Top-Left)
        # ---------------------------------------------------------------------
        p1 = frame.copy()
        draw_eyeball_boundary_circle(p1, circle_x, circle_y, circle_r, COLOR_CIRCLE)

        if r_or:
            cv2.ellipse(p1, r_or, COLOR_ORLOSKY, 2, cv2.LINE_AA)
            draw_crosshair(p1, r_or[0][0], r_or[0][1], COLOR_ORLOSKY)
            p1_stat, p1_col = "LOCKED", (0, 255, 80)
            p1_pos = f"({r_or[0][0]:.1f}, {r_or[0][1]:.1f})"
        else:
            p1_stat, p1_col = "SEARCHING", COLOR_RED
            p1_pos = "None"

        draw_hud_card(p1, 10, 10, 225, 125, "1. ORLOSKY DETECTOR", [
            ("Status:", p1_stat, p1_col),
            ("Latency:", f"{lat_or:.1f} ms", COLOR_TEXT_MAIN),
            ("Eyeball Cage:", circle_tag, (0, 230, 255)),
            ("Center:", p1_pos, COLOR_TEXT_MAIN),
        ], COLOR_ORLOSKY)

        # ---------------------------------------------------------------------
        # Render Panel 2: Pupil Labs Detector2D (Top-Right)
        # ---------------------------------------------------------------------
        p2 = frame.copy()
        draw_eyeball_boundary_circle(p2, circle_x, circle_y, circle_r, COLOR_CIRCLE)

        if r_pl:
            cv2.ellipse(p2, r_pl, COLOR_PL, 2, cv2.LINE_AA)
            draw_crosshair(p2, r_pl[0][0], r_pl[0][1], COLOR_PL)
            p2_stat, p2_col = "LOCKED", (0, 255, 80)
            p2_pos = f"({r_pl[0][0]:.1f}, {r_pl[0][1]:.1f})"
            p2_conf = f"{conf_pl * 100:.0f}%"
        else:
            p2_stat, p2_col = "DROPOUT", COLOR_RED
            p2_pos = "None"
            p2_conf = "0%"

        draw_hud_card(p2, 10, 10, 225, 125, "2. PUPIL LABS DETECTOR2D", [
            ("Status:", p2_stat, p2_col),
            ("Latency:", f"{lat_pl:.1f} ms", (0, 255, 120)),
            ("Confidence:", p2_conf, (0, 255, 80) if conf_pl > 0.5 else (0, 165, 255)),
            ("Eyeball Cage:", circle_tag, (0, 230, 255)),
        ], COLOR_PL)

        # ---------------------------------------------------------------------
        # Render Panel 3: Hybrid Fusion (Bottom-Left)
        # ---------------------------------------------------------------------
        p3 = frame.copy()
        draw_eyeball_boundary_circle(p3, circle_x, circle_y, circle_r, COLOR_CIRCLE)

        if r_hy:
            cv2.ellipse(p3, r_hy, COLOR_HYBRID, 2, cv2.LINE_AA)
            draw_crosshair(p3, r_hy[0][0], r_hy[0][1], COLOR_HYBRID)
            p3_stat, p3_col = "LOCKED", (0, 255, 80)
            p3_pos = f"({r_hy[0][0]:.1f}, {r_hy[0][1]:.1f})"
        else:
            p3_stat, p3_col = "SEARCHING", COLOR_RED
            p3_pos = "None"

        draw_hud_card(p3, 10, 10, 225, 125, "3. HYBRID FUSION (PROPOSED)", [
            ("Status:", p3_stat, p3_col),
            ("Latency:", f"{lat_hy:.1f} ms", (0, 255, 120)),
            ("Engine:", src_hy, (0, 230, 255)),
            ("Eyeball Cage:", circle_tag, (0, 230, 255)),
        ], COLOR_HYBRID)

        # ---------------------------------------------------------------------
        # Render Panel 4: PuReST SOTA World Champion (Bottom-Right)
        # ---------------------------------------------------------------------
        p4 = frame.copy()
        draw_eyeball_boundary_circle(p4, circle_x, circle_y, circle_r, COLOR_CIRCLE)

        if r_sk:
            cv2.ellipse(p4, r_sk, COLOR_PUREST, 2, cv2.LINE_AA)
            draw_crosshair(p4, r_sk[0][0], r_sk[0][1], COLOR_PUREST)
            p4_stat, p4_col = "SOTA LOCK", (0, 255, 80)
            p4_pos = f"({r_sk[0][0]:.1f}, {r_sk[0][1]:.1f})"
        else:
            p4_stat, p4_col = "SEARCHING", COLOR_RED
            p4_pos = "None"

        draw_hud_card(p4, 10, 10, 225, 125, "4. PuReST (SOTA CHAMPION)", [
            ("Status:", p4_stat, p4_col),
            ("Latency:", f"{lat_sk:.1f} ms", (0, 255, 120)),
            ("Engine:", f"{src_sk} ({conf_sk*100:.0f}%)", (240, 80, 210)),
            ("Eyeball Cage:", circle_tag, (0, 230, 255)),
        ], COLOR_PUREST)

        # ---------------------------------------------------------------------
        # View Modes
        # ---------------------------------------------------------------------
        if view_mode == 0:
            # Mode 0: 4-Way 2x2 Grid (1280x960)
            row_top = np.hstack([p1, p2])
            cv2.line(row_top, (640, 0), (640, 480), COLOR_BORDER, 2)

            row_bot = np.hstack([p3, p4])
            cv2.line(row_bot, (640, 0), (640, 480), COLOR_BORDER, 2)

            content = np.vstack([row_top, row_bot])
            cv2.line(content, (0, 480), (1280, 480), COLOR_BORDER, 2)

        elif view_mode == 1:
            # Mode 1: Unified 4-Color Overlay
            unified = frame.copy()
            draw_eyeball_boundary_circle(unified, circle_x, circle_y, circle_r, COLOR_CIRCLE)

            if r_or:
                cv2.ellipse(unified, r_or, COLOR_ORLOSKY, 2, cv2.LINE_AA)
                draw_crosshair(unified, r_or[0][0], r_or[0][1], COLOR_ORLOSKY)
            if r_pl:
                cv2.ellipse(unified, r_pl, COLOR_PL, 2, cv2.LINE_AA)
                draw_crosshair(unified, r_pl[0][0], r_pl[0][1], COLOR_PL)
            if r_hy:
                cv2.ellipse(unified, r_hy, COLOR_HYBRID, 2, cv2.LINE_AA)
                draw_crosshair(unified, r_hy[0][0], r_hy[0][1], COLOR_HYBRID)
            if r_sk:
                cv2.ellipse(unified, r_sk, COLOR_PUREST, 2, cv2.LINE_AA)
                draw_crosshair(unified, r_sk[0][0], r_sk[0][1], COLOR_PUREST)

            draw_hud_card(unified, 15, 15, 300, 130, "UNIFIED 4-COLOR OVERLAY", [
                ("1. Orlosky (Gold):",     "Original Multi-Threshold", COLOR_ORLOSKY),
                ("2. Pupil Labs (Green):", "Canny + Ramanujan", COLOR_PL),
                ("3. Hybrid (Cyan):",      "PL + Adaptive Fallback", COLOR_HYBRID),
                ("4. PuReST (Pink):",      "LPW SOTA Winner (Santini)", COLOR_PUREST),
                ("Eyeball Circle Cage:",   circle_tag, (0, 230, 255)),
            ], (0, 255, 255))
            content = cv2.resize(unified, (1280, 720))

        else:
            # Mode 2: Head-to-head (Pupil Labs vs PuReST)
            content = np.hstack([p2, p4])
            cv2.line(content, (640, 0), (640, 480), COLOR_BORDER, 2)

        # ---------------------------------------------------------------------
        # Header Banner
        # ---------------------------------------------------------------------
        hdr_w = content.shape[1]
        header = np.full((64, hdr_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
        cv2.line(header, (0, 63), (hdr_w, 63), COLOR_BORDER, 1)

        # Row 1: Title & Telemetry
        cv2.putText(header, "EYE TRACKING MODULE -- 4-ALGORITHM COMPARISON & EYEBALL CAGE", (18, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 200), 2, cv2.LINE_AA)

        avg_sk = np.mean(hist_lat_sk) if hist_lat_sk else 0.0
        cv2.putText(header, f"FPS: {avg_fps:.0f}  |  PuReST: {avg_sk:.1f}ms  |  Cam: {cam_id}",
                    (hdr_w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # Row 2: Eyeball Boundary Circle Coordinates
        badge_text = f"EYEBALL CIRCLE CAGE: Center=({circle_x}, {circle_y}) | Radius={circle_r}px  |  Adjust: [W/A/S/D / Arrows] Move, [+/-] Resize, [P] Print Config"
        cv2.putText(header, badge_text, (18, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 230, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # Footer Banner
        # ---------------------------------------------------------------------
        footer = np.full((40, hdr_w, 3), COLOR_PANEL_BG, dtype=np.uint8)
        cv2.line(footer, (0, 0), (hdr_w, 0), COLOR_BORDER, 1)

        controls = "[W/A/S/D / Arrows] Move Cage | [+/-] Resize Radius | [Space] Pause | [M] View | [P] Print | [Q] Quit"
        cv2.putText(footer, controls, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(footer, f"View: {mode_names[view_mode]}", (hdr_w - 240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 255, 200), 1, cv2.LINE_AA)

        dashboard = np.vstack([header, content, footer])

        # Centered Flash Notification
        if flash_message and (now - flash_time < 2.0):
            box_w = 460
            box_h = 44
            x1 = hdr_w // 2 - box_w // 2
            y1 = 80
            x2 = x1 + box_w
            y2 = y1 + box_h
            overlay = dashboard.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
            cv2.addWeighted(overlay, 0.90, dashboard, 0.10, 0, dashboard)
            cv2.rectangle(dashboard, (x1, y1), (x2, y2), (0, 230, 255), 1)
            cv2.putText(dashboard, flash_message, (x1 + 18, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 220), 1, cv2.LINE_AA)

        cv2.imshow(win_title, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            flash_message = "LIVE STREAM PAUSED" if paused else "LIVE STREAM RESUMED"
            flash_time = now

        # Move circle UP
        elif key in (ord('w'), ord('W'), 82, 2490368):  # 'w' or Up Arrow
            circle_y = max(20, circle_y - 5)
            flash_message = f"CAGE: X={circle_x}, Y={circle_y}, R={circle_r}"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Move circle DOWN
        elif key in (ord('s'), ord('S'), 84, 2621440):  # 's' or Down Arrow
            circle_y = min(460, circle_y + 5)
            flash_message = f"CAGE: X={circle_x}, Y={circle_y}, R={circle_r}"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Move circle LEFT
        elif key in (ord('a'), ord('A'), 81, 2424832):  # 'a' or Left Arrow
            circle_x = max(20, circle_x - 5)
            flash_message = f"CAGE: X={circle_x}, Y={circle_y}, R={circle_r}"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Move circle RIGHT
        elif key in (ord('d'), ord('D'), 83, 2555904):  # 'd' or Right Arrow
            circle_x = min(620, circle_x + 5)
            flash_message = f"CAGE: X={circle_x}, Y={circle_y}, R={circle_r}"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Increase circle radius
        elif key in (ord('+'), ord('='), ord(']')):
            circle_r = min(220, circle_r + 5)
            flash_message = f"CAGE: Radius={circle_r}px"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Decrease circle radius
        elif key in (ord('-'), ord('_'), ord('[')):
            circle_r = max(25, circle_r - 5)
            flash_message = f"CAGE: Radius={circle_r}px"
            flash_time = now
            print(f"[CAGE] X={circle_x}, Y={circle_y}, R={circle_r}")

        # Print current coordinates to console for easy copy-paste
        elif key in (ord('p'), ord('P')):
            print("\n" + "="*60)
            print("CURRENT EYEBALL CIRCLE CONFIGURATION (Lines 55 - 57):")
            print(f"EYEBALL_CIRCLE_X      = {circle_x}")
            print(f"EYEBALL_CIRCLE_Y      = {circle_y}")
            print(f"EYEBALL_CIRCLE_RADIUS = {circle_r}")
            print("="*60 + "\n")
            flash_message = f"PRINTED: X={circle_x}, Y={circle_y}, R={circle_r}"
            flash_time = now

        elif key in (ord('m'), ord('M')):
            view_mode = (view_mode + 1) % 3
            flash_message = f"VIEW MODE: {mode_names[view_mode]}"
            flash_time = now

        elif key in (ord('c'), ord('C')):
            if len(available_cams) > 1:
                cam_idx_ptr = (cam_idx_ptr + 1) % len(available_cams)
                new_cam = available_cams[cam_idx_ptr]
                cap.release()
                cap = cv2.VideoCapture(new_cam, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(new_cam, cv2.CAP_MSMF)
                if cap.isOpened():
                    cam_id = new_cam
                    flash_message = f"Switched to Camera {cam_id}"
                    flash_time = now
            else:
                flash_message = "Only 1 camera found"
                flash_time = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
