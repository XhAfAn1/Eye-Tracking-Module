#!/usr/bin/env python3
"""
benchmark.py
============
4-Algorithm Comprehensive Benchmark Suite for Eye Tracking in Plegia.

Pipeline:
  Phase 0: Interactive Alignment (Adjust Eyeball Boundary Circle on Eye before start)
  Phase 1: Multi-Algorithm Calibration (Simultaneous 9-Point Bivariate Polynomial Fitting)
  Phase 2: Level 2 Testing (Static Targets with 4 Live Colored Pointers + [SPACE] Confirmation)
  Phase 3: Dynamic Tracking Test (Smooth Pursuit Trajectory Tracking & Lag Measurement)
  Phase 4: Analytics Engine (Accuracy, Fixation Jitter, Uptime, Latency, and AAC Usability Score)
  Phase 5: Automated Reporting (Multi-Panel Matplotlib Infographic + Markdown Clinical Report + JSON)

Controls:
  [W/A/S/D] or [Arrows] : Move Eyeball Circle during Phase 0
  [+] / [-]             : Resize Eyeball Circle Radius
  [Space] or [Enter]    : Start Benchmark (Phase 0) / Confirm Fixation Target (Phase 2)
  [C]                   : Switch Camera
  [R]                   : Re-run Benchmark after completion
  [Q] or [ESC]          : Quit Benchmark Suite
"""

import os
import sys
import argparse
from typing import List
import cv2

# Silence OpenCV internal noise
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

# Ensure local directory is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from benchmark_gui import BenchmarkGUI


def detect_cameras(max_cams: int = 4) -> List[int]:
    """Scans for active video capture devices and validates frame reading."""
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


def get_screen_resolution() -> (int, int):
    """Detects primary monitor native resolution with High-DPI awareness."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        if w > 640 and h > 480:
            return w, h
    except Exception:
        pass
    return 1280, 800


def main():
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="4-Algorithm Comprehensive Eye Tracking Benchmark Suite")
    parser.add_argument("--camera", "-c", type=int, default=None, help="Camera device index")
    parser.add_argument("--windowed", "-w", action="store_true", help="Run in windowed mode (1280x800) instead of full screen")
    parser.add_argument("--width", type=int, default=None, help="Custom canvas width")
    parser.add_argument("--height", type=int, default=None, help="Custom canvas height")
    parser.add_argument("--calib", "-m", choices=["sota", "classic"], default="sota", help="Calibration mode: sota (WLS 7-term) or classic (OLS 6-term)")
    args = parser.parse_args()

    print("""
+-------------------------------------------------------------------------------+
|         EYE TRACKING MODULE -- 4-ALGORITHM COMPREHENSIVE BENCHMARK SUITE      |
|    1. Orlosky  |  2. Pupil Labs  |  3. Hybrid Fusion  |  4. PuReST (SOTA)     |
+-------------------------------------------------------------------------------+
    """)

    available_cams = detect_cameras()
    if args.camera is not None:
        cam_id = args.camera
    else:
        cam_id = available_cams[0]

    print(f"Available cameras: {available_cams}  --> Active: Camera {cam_id}")

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_id}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Determine canvas resolution
    if args.windowed:
        screen_w, screen_h = 1280, 800
    elif args.width and args.height:
        screen_w, screen_h = args.width, args.height
    else:
        screen_w, screen_h = get_screen_resolution()

    print(f"Benchmark Display Canvas: {screen_w} x {screen_h} (Fullscreen: {not args.windowed}) | Mode: {args.calib.upper()}")

    gui = BenchmarkGUI(cap, cam_id, available_cams, screen_w, screen_h, fullscreen=not args.windowed, calib_mode=args.calib)
    gui.run()


if __name__ == '__main__':
    main()
