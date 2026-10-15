#!/usr/bin/env python3
"""
EyeTracker Module — Eye-Gaze Mouse Controller  (Precision Edition)
===================================================================
Imports the EXACT, UNCHANGED OrloskyPupilDetector.py from EyeTracker/
so the pupil detection is bit-for-bit identical to the original.

Adds on top:
  - Fully automatic 9-point precision calibration (no keypresses)
      · 4-second Lissajous warm-up animation
      · Auto-dwell: 2 s approach ring + 1.5 s countdown arc
      · Tukey IQR outlier rejection per point
      · Bivariate degree-2 polynomial regression (vs old affine)
      · Quality flash (green / amber) after each point
      · 4-point validation pass with pixel-error reporting
      · Calibration persisted to gaze_calibration.json
  - Exponential smoothing + blink grace period
  - Windows cursor control via ctypes (no extra library)

Controls:
  c      → Start precision calibration  (fully automatic — just follow the dot)
  r      → Reset calibration
  d      → Toggle debug overlay
  s      → Open settings
  -/+    → Cursor gain down / up
  [/]    → More / less smoothing
  q      → Quit
"""

import sys
import os
import cv2
import numpy as np
import ctypes
import math
import time
from collections import deque

# Settings window (Tkinter — stdlib, no extra install)
try:
    from settings_window import load_settings, open_settings, save_settings
    _SETTINGS_AVAILABLE = True
except ImportError:
    _SETTINGS_AVAILABLE = False
    def load_settings():
        return {"dpi": 96.0, "gain": 1.0, "alpha": 0.15, "camera_id": 0, "grace_ms": 250}
    def open_settings(**kw):
        return None

# Precision calibration engine + overlay renderer
from calibration_engine  import PrecisionCalibrator
from calibration_overlay import draw_frame as _draw_cal_frame

# ─────────────────────────────────────────────────────────────────────────────
# ①  IMPORT THE EXACT ORIGINAL ORLOSKY DETECTOR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Import only the functions we need — everything else stays untouched
    from OrloskyPupilDetector import (
        crop_to_aspect_ratio,
        get_darkest_area,
        apply_binary_threshold,
        mask_outside_square,
        process_frames,
    )
    print("OK Loaded OrloskyPupilDetector locally.")
except ImportError as e:
    print(f"ERROR: Could not import OrloskyPupilDetector.py")
    print(f"       {e}")
    sys.exit(1)


def detect_pupil(frame):
    """
    Run the full Orlosky pipeline on a single frame.
    Returns (rotated_rect, cropped_frame).
    rotated_rect is ((cx,cy),(major,minor),angle) or ((0,0),(0,0),0) if not found.
    """
    frame = crop_to_aspect_ratio(frame)
    darkest_point = get_darkest_area(frame)
    if darkest_point is None:
        return None, frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dpv = gray[darkest_point[1], darkest_point[0]]

    t_strict  = mask_outside_square(apply_binary_threshold(gray, dpv,  5), darkest_point, 250)
    t_medium  = mask_outside_square(apply_binary_threshold(gray, dpv, 15), darkest_point, 250)
    t_relaxed = mask_outside_square(apply_binary_threshold(gray, dpv, 25), darkest_point, 250)

    # process_frames with debug=False, render=False
    result = process_frames(t_strict, t_medium, t_relaxed, frame, gray, darkest_point,
                            False, False)

    # The original returns ((0,0),(0,0),0) when no ellipse found
    if result is None or result == ((0, 0), (0, 0), 0):
        return None, frame
    return result, frame


# ─────────────────────────────────────────────────────────────────────────────
# ②  CURSOR CONTROL  (Windows — no extra library needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_screen_size():
    u32 = ctypes.windll.user32
    return u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)


def move_cursor(x, y):
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


# ─────────────────────────────────────────────────────────────────────────────
# ③  EXPONENTIAL SMOOTHING  +  blink grace period
# ─────────────────────────────────────────────────────────────────────────────

class GazeSmoother:
    def __init__(self, alpha=0.15):
        self.alpha = alpha          # 0 = max smooth / 1 = no smooth
        self.sx = None
        self.sy = None
        self.last_good_time = 0.0
        self.GRACE = 0.25           # seconds to hold last position during blinks

    def update(self, x, y):
        if x is None or y is None:
            if time.time() - self.last_good_time < self.GRACE:
                return self.sx, self.sy
            return None, None
        if self.sx is None:
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy
        self.last_good_time = time.time()
        return self.sx, self.sy


# ─────────────────────────────────────────────────────────────────────────────
# ④  CAMERA DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_cameras(max_cams=6):
    found = []
    for i in range(max_cams):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
                cap.release()
                break
    return found


# ─────────────────────────────────────────────────────────────────────────────
# ⑤  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

_DATA_DIR   = os.path.join(_THIS_DIR, 'data')
os.makedirs(_DATA_DIR, exist_ok=True)
_CALIB_FILE = os.path.join(_DATA_DIR, 'default.json')


def main(camera_id=None):
    screen_w, screen_h = get_screen_size()
    print(f"Screen: {screen_w}×{screen_h}")

    # ── Load persisted settings ───────────────────────────────────────────────
    cfg = load_settings()
    if camera_id is None:
        camera_id = cfg.get("camera_id", 0)

    if not isinstance(camera_id, int):
        cams = detect_cameras()
        if not cams:
            print("ERROR: No camera found.")
            sys.exit(1)
        camera_id = cams[0]
        if len(cams) > 1:
            print(f"Cameras found: {cams}  →  using {camera_id}. "
                  f"Pass --camera N to choose another.")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── PrecisionCalibrator replaces old GazeCalibrator ──────────────────────
    calibrator  = PrecisionCalibrator(screen_w, screen_h)
    smoother    = GazeSmoother(alpha=float(cfg.get("alpha", 0.15)))
    gain        = float(cfg.get("gain", 1.0))
    dpi_scale   = float(cfg.get("dpi", 96.0)) / 96.0
    smoother.GRACE = float(cfg.get("grace_ms", 250)) / 1000.0
    fps_hist    = deque(maxlen=30)
    last_time   = time.time()
    debug_mode  = False
    _tk_root    = None
    cal_state   = {}             # last state dict from calibrator
    cal_win_open= False          # is the calibration OpenCV window open?
    done_show_t = 0.0            # time when DONE results screen was first shown

    # ── Try loading saved calibration ────────────────────────────────────────
    if calibrator.load(_CALIB_FILE):
        print(f"✓ Loaded saved calibration from {_CALIB_FILE}")

    cv2.namedWindow('EyeTracker Module \u2014 Eye View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EyeTracker Module \u2014 Eye View', 640, 480)

    print("""
╔══════════════════════════════════════════════════════╗
║    EyeTracker Module — Precision Calibration Edition ║
╠══════════════════════════════════════════════════════╣
║  c      → Start precision calibration (auto-dwell)   ║
║  r      → Reset calibration                          ║
║  d      → Toggle debug view                          ║
║  s      → Settings                                   ║
║  -/+    → Cursor gain down / up                      ║
║  [/]    → More / less smoothing                      ║
║  q      → Quit                                       ║
╚══════════════════════════════════════════════════════╝

TIP: Press 'c' to start. Just follow the dot — no keypresses needed!
     Calibration auto-saves to data/default.json.
""")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        now = time.time()

        # ── Detect pupil (original Orlosky algorithm — unchanged) ─────────
        pupil_rect, frame = detect_pupil(raw_frame)
        pupil_center = None
        if pupil_rect is not None:
            cx, cy = pupil_rect[0]
            if cx > 0 or cy > 0:
                pupil_center = (cx, cy)

        # ── Drive calibration state machine ──────────────────────────────
        if calibrator.calibrating:
            cal_state = calibrator.tick(pupil_center, now)
        else:
            cal_state = calibrator.get_state()

        # ── Smooth ───────────────────────────────────────────────────────
        rx = pupil_center[0] if pupil_center else None
        ry = pupil_center[1] if pupil_center else None
        sx, sy = smoother.update(rx, ry)

        # ── Move cursor (polynomial mapping) ─────────────────────────────
        if calibrator.done and sx is not None:
            mapped = calibrator.map(sx, sy)
            if mapped:
                mx, my = mapped
                effective_gain = gain * dpi_scale
                mx = int(screen_w / 2 + (mx - screen_w / 2) * effective_gain)
                my = int(screen_h / 2 + (my - screen_h / 2) * effective_gain)
                mx = max(0, min(screen_w - 1, mx))
                my = max(0, min(screen_h - 1, my))
                move_cursor(mx, my)

        # ── Draw eye view ─────────────────────────────────────────────────
        display = frame.copy()
        if pupil_rect is not None and pupil_center is not None:
            cv2.ellipse(display, pupil_rect, (55, 255, 0), 2)
            cxi, cyi = int(pupil_center[0]), int(pupil_center[1])
            cv2.circle(display, (cxi, cyi), 4, (255, 255, 0), -1)

        fps_hist.append(1.0 / max(now - last_time, 1e-6))
        last_time = now
        avg_fps   = np.mean(fps_hist)

        if calibrator.calibrating:
            cs   = cal_state.get('state', '')
            pt   = cal_state.get('pt_idx', 0)
            tot  = cal_state.get('total_pts', 9)
            status = f"{cs}  {pt}/{tot}"
            col    = (0, 165, 255)
        elif calibrator.done:
            status, col = 'ACTIVE (poly-deg2)', (0, 255, 80)
        else:
            status, col = "UNCALIBRATED — press 'c'", (0, 60, 255)

        cv2.putText(display, f"FPS: {avg_fps:.0f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(display, status,
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        cv2.putText(display, f"gain:{gain:.1f}  smooth:{smoother.alpha:.2f}  dpi:{cfg.get('dpi',96):.0f}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(display, "c=cal  r=reset  d=dbg  s=settings  -/+=gain  [/]=smooth  q=quit",
                    (10, 476), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1)

        if pupil_center is None:
            cv2.putText(display, "NO PUPIL", (220, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('EyeTracker Module \u2014 Eye View', cv2.flip(display, 1))

        # ── Calibration overlay (fullscreen) ─────────────────────────────
        show_done   = (cal_state.get('state') == 'DONE' and done_show_t > 0.0)
        show_overlay= calibrator.calibrating or show_done

        if show_overlay:
            overlay = _draw_cal_frame(screen_w, screen_h, cal_state)
            if not cal_win_open:
                cv2.namedWindow('EyeTracker Module \u2014 Calibration', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('EyeTracker Module \u2014 Calibration',
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cal_win_open = True
            cv2.imshow('EyeTracker Module \u2014 Calibration', overlay)

            # Handle DONE: save + auto-close after 3 s
            if cal_state.get('state') == 'DONE':
                if done_show_t == 0.0:
                    done_show_t = now
                    calibrator.save(_CALIB_FILE)
                    errs = cal_state.get('val_errors', [])
                    if errs:
                        print(f"✓ Validation mean error: "
                              f"{sum(errs)/len(errs):.0f} px  "
                              f"(individual: {errs})")
                elif now - done_show_t >= 3.0:
                    done_show_t   = 0.0
                    cal_win_open  = False
                    try:
                        cv2.destroyWindow('EyeTracker Module \u2014 Calibration')
                    except Exception:
                        pass
        elif cal_win_open:
            cal_win_open = False
            try:
                cv2.destroyWindow('EyeTracker Module \u2014 Calibration')
            except Exception:
                pass

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrator.start()
            done_show_t = 0.0
            print("[EyeTracker Module] Precision calibration started — just follow the dot!")
        elif key == ord('r'):
            calibrator.reset()
            smoother.sx = smoother.sy = None
            done_show_t = 0.0
            try:
                cv2.destroyWindow('EyeTracker Module \u2014 Calibration')
            except Exception:
                pass
            cal_win_open = False
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('s'):
            if _SETTINGS_AVAILABLE:
                import tkinter as _tk
                if _tk_root is None:
                    _tk_root = _tk.Tk()
                    _tk_root.withdraw()
                new_cfg = open_settings(master=_tk_root, current_cfg=cfg)
                if new_cfg:
                    cfg.update(new_cfg)
                    gain           = float(cfg["gain"])
                    dpi_scale      = float(cfg["dpi"]) / 96.0
                    smoother.alpha = float(cfg["alpha"])
                    smoother.GRACE = float(cfg["grace_ms"]) / 1000.0
                    print(f"Settings applied: DPI={cfg['dpi']:.0f}  gain={gain:.1f}  "
                          f"alpha={smoother.alpha:.2f}  grace={cfg['grace_ms']}ms")
            else:
                print("Settings window not available (settings_window.py missing).")
        elif key in (ord('+'), ord('=')):
            gain = min(3.0, gain + 0.1)
            print(f"Cursor gain: {gain:.1f}")
        elif key == ord('-'):
            gain = max(0.2, gain - 0.1)
            print(f"Cursor gain: {gain:.1f}")
        elif key == ord(']'):
            smoother.alpha = min(0.9, smoother.alpha + 0.05)
            print(f"Smoothing alpha: {smoother.alpha:.2f}")
        elif key == ord('['):
            smoother.alpha = max(0.02, smoother.alpha - 0.05)
            print(f"Smoothing alpha: {smoother.alpha:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    try:
        if _tk_root is not None:
            _tk_root.destroy()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ⑧  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cam = None
    if "--camera" in sys.argv:
        try:
            idx = sys.argv.index("--camera")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
                cam = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            pass
    main(camera_id=cam)

