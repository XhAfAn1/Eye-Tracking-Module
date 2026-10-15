#!/usr/bin/env python3
"""
gaze_controller.py  --  Thread-based Gaze Controller for EyeTracker Module
===========================================================================
Wraps the camera capture + pupil detection + cursor-control loop
into a background thread so that a Tkinter launcher window can run
on the main thread without blocking.

Usage (from launcher.py):
    ctrl = GazeController(screen_w, screen_h, camera_id=0, cfg={},
                          data_dir=DATA_DIR)
    ctrl.load_profile('default')      # load saved calibration (optional)
    ctrl.start()                      # open camera, begin loop
    ctrl.trigger_calibration('home')  # start calibration -> save to 'home'
    status = ctrl.get_status()        # poll UI-friendly status dict
    ctrl.stop()                       # close camera, end loop

The controller thread owns all OpenCV windows (eye view + calibration
overlay).  The Tkinter thread only reads get_status().
"""

import os
import sys
import time
import threading
import math
import numpy as np
import cv2
from collections import deque
import win32api
import win32con

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

# -- Re-use functions and classes already defined in gaze_mouse.py ------------
from gaze_mouse import detect_pupil, move_cursor, get_screen_size, GazeSmoother
from calibration_engine import PrecisionCalibrator, FineTuneCalibrator
from calibration_overlay import draw_frame as _draw_cal_frame, draw_ft_frame as _draw_ft_frame


# =============================================================================
class GazeController:
    """
    Background-thread gaze controller.
    Thread-safe: all public methods may be called from any thread.
    """

    def __init__(self,
                 screen_w: int,
                 screen_h: int,
                 camera_id: int = 0,
                 cfg: dict = None,
                 data_dir: str = None):
        self.screen_w  = screen_w
        self.screen_h  = screen_h
        self.camera_id = camera_id
        self.cfg       = cfg or {}
        self.data_dir  = data_dir or os.path.join(_THIS_DIR, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        self._stop_evt      = threading.Event()
        self._cal_trigger   = threading.Event()
        self._ft_trigger    = threading.Event()
        self._cal_profile   = 'default'   # profile to save calibration into
        self._thread        = None
        self._cursor_enabled = False      # must be explicitly set via set_cursor_enabled()

        self._status_lock   = threading.Lock()
        self._status        = {
            'fps':            0.0,
            'pupil':          None,       # (px, py) raw image coords, or None
            'cursor_active':  False,
            'cursor_enabled': False,      # whether cursor control is turned on
            'cal_state':      'IDLE',     # mirrors PrecisionCalibrator state
            'cal_pt_idx':     0,
            'cal_total':      9,
            'cal_accuracy':   None,       # mean validation error (px), or None
            'calibrated':     False,
            'camera_ok':      False,
            'profile':        'default',
        }

        # Pre-loaded calibration (set by load_profile before start())
        self._preloaded_cx  = None
        self._preloaded_cy  = None
        self._preloaded_acc = None
        self._preloaded_ft  = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_calibrated(self) -> bool:
        with self._status_lock:
            return self._status['calibrated']

    def start(self):
        """Start the camera loop in a daemon thread."""
        if self.is_running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='GazeControllerThread')
        self._thread.start()

    def stop(self):
        """Signal the loop to stop (closes camera + all CV2 windows)."""
        self._stop_evt.set()

    def set_cursor_enabled(self, enabled: bool):
        """Enable or disable cursor movement without touching the camera thread."""
        self._cursor_enabled = enabled
        self._update(cursor_enabled=enabled)

    def trigger_calibration(self, profile_name: str = 'default'):
        """
        Start a new calibration sequence.  Result is saved to
        data/<profile_name>.json when the sequence completes.
        """
        self._cal_profile = profile_name
        self._cal_trigger.set()

    def trigger_fine_tuning(self, profile_name: str = 'default'):
        """Start a Level 2 fine-tuning sequence (manual 10-point affine)."""
        self._cal_profile = profile_name
        self._ft_trigger.set()

    def load_profile(self, profile_name: str) -> bool:
        """
        Load a saved calibration profile so that cursor control starts
        immediately even before a new calibration is done.
        Returns True if the profile was found and loaded.
        """
        path = self._profile_path(profile_name)
        if not os.path.exists(path):
            return False
        try:
            import json
            with open(path) as f:
                d = json.load(f)
            self._preloaded_cx  = np.array(d['cx'])
            self._preloaded_cy  = np.array(d['cy'])
            self._preloaded_ft  = d.get('fine_tune', None)
            errs                = d.get('val_err', [])
            self._preloaded_acc = sum(errs) / len(errs) if errs else None
            with self._status_lock:
                self._status['calibrated']   = True
                self._status['cal_accuracy'] = self._preloaded_acc
                self._status['profile']      = profile_name
            return True
        except Exception as e:
            print(f"[Controller] load_profile({profile_name}) failed: {e}")
            return False

    def get_status(self) -> dict:
        """Thread-safe snapshot of the current status."""
        with self._status_lock:
            return dict(self._status)

    def profile_path(self, name: str) -> str:
        return self._profile_path(name)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _profile_path(self, name: str) -> str:
        return os.path.join(self.data_dir, f'{name}.json')

    def _update(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)

    def _run(self):
        """Main loop -- runs entirely in the controller thread."""
        cfg       = self.cfg
        screen_w  = self.screen_w
        screen_h  = self.screen_h
        spacebar_click_enabled = cfg.get('spacebar_click', True)

        # -- Open camera -------------------------------------------------------
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)
        cam_ok = cap.isOpened()
        self._update(camera_ok=cam_ok)

        if not cam_ok:
            print(f"[Controller] Could not open camera {self.camera_id}.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # -- Calibrator --------------------------------------------------------
        calibrator = PrecisionCalibrator(screen_w, screen_h)
        fine_tuner = FineTuneCalibrator(screen_w, screen_h, num_points=10)

        # Apply pre-loaded calibration if available
        if self._preloaded_cx is not None:
            calibrator._cx = self._preloaded_cx
            calibrator._cy = self._preloaded_cy
            calibrator._state = calibrator.S_DONE
            calibrator._val_errors = []
            if self._preloaded_ft:
                fine_tuner.load_from_dict(self._preloaded_ft)
        else:
            # Try loading the 'default' profile as fallback
            default_path = self._profile_path('default')
            ok, ft_dict = calibrator.load(default_path)
            if ok:
                if ft_dict:
                    fine_tuner.load_from_dict(ft_dict)
                errs = calibrator._val_errors
                acc  = sum(errs) / len(errs) if errs else None
                self._update(calibrated=True, cal_accuracy=acc)

        # -- Smoother & config -------------------------------------------------
        smoother    = GazeSmoother(alpha=float(cfg.get('alpha', 0.15)))
        gain        = float(cfg.get('gain', 1.0))
        dpi_scale   = float(cfg.get('dpi', 96.0)) / 96.0
        smoother.GRACE = float(cfg.get('grace_ms', 250)) / 1000.0

        fps_hist     = deque(maxlen=30)
        last_t       = time.time()
        cal_state    = {}
        ft_state     = {}
        cal_win_open = False
        done_show_t  = 0.0
        _cal_saved   = calibrator.done   # one-shot flag: did we already save this run?
        prev_space_down = False          # for edge detection (press, not hold)
        prev_f5_down    = False
        prev_f6_down    = False
        prev_esc_down   = False

        # -- Eye view window ---------------------------------------------------
        cv2.namedWindow('EyeTracker Module -- Eye View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('EyeTracker Module -- Eye View', 640, 480)

        # ======================================================================
        while not self._stop_evt.is_set():
            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()

            key = cv2.waitKey(1) & 0xFF

            # -- Handle calibration trigger ------------------------------------
            if self._cal_trigger.is_set():
                self._cal_trigger.clear()
                calibrator.start()
                fine_tuner.reset()  # Reset fine tuning if full recalibration starts
                done_show_t = 0.0
                _cal_saved  = False   # allow saving for the new run

            if self._ft_trigger.is_set():
                self._ft_trigger.clear()
                if calibrator.done:
                    fine_tuner.start()
                    done_show_t = 0.0
                    _cal_saved  = False

            # -- Detect pupil --------------------------------------------------
            pupil_rect, frame = detect_pupil(raw_frame)
            pupil_center = None
            if pupil_rect is not None:
                cx, cy = pupil_rect[0]
                if cx > 0 or cy > 0:
                    pupil_center = (cx, cy)

            # -- Drive calibrator state machine --------------------------------
            if calibrator.calibrating:
                cal_state = calibrator.tick(pupil_center, now)
            else:
                cal_state = calibrator.get_state()
            cs = cal_state.get('state', 'IDLE')
                
            raw_mapped = None
            if calibrator.done and pupil_center:
                raw_mapped = calibrator.map(pupil_center[0], pupil_center[1])
                
            sx = raw_mapped[0] if raw_mapped else None
            sy = raw_mapped[1] if raw_mapped else None
            
            # -- Global Hotkeys ------------------------------------------------
            space_down = (win32api.GetAsyncKeyState(win32con.VK_SPACE) < 0)
            space_pressed = space_down and not prev_space_down  # fresh press edge
            prev_space_down = space_down

            f5_down = (win32api.GetAsyncKeyState(win32con.VK_F5) < 0)
            if f5_down and not prev_f5_down:
                self.set_cursor_enabled(True)
            prev_f5_down = f5_down

            f6_down = (win32api.GetAsyncKeyState(win32con.VK_F6) < 0)
            if f6_down and not prev_f6_down:
                self.set_cursor_enabled(False)
            prev_f6_down = f6_down

            esc_down = (win32api.GetAsyncKeyState(win32con.VK_ESCAPE) < 0)
            if esc_down and not prev_esc_down:
                if calibrator.calibrating:
                    calibrator.reset()
                if fine_tuner.calibrating:
                    fine_tuner.reset()
            prev_esc_down = esc_down
            # ------------------------------------------------------------------
            ft_state = fine_tuner.tick(sx, sy, space_down, now)

            # -- Spacebar-to-click: instantly click whatever is under the cursor --
            # Only fires on a fresh tap (not during fine-tuning hold)
            if (spacebar_click_enabled
                    and space_pressed
                    and not calibrator.calibrating
                    and not fine_tuner.calibrating
                    and self._cursor_enabled
                    and cursor_active):
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.03)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            # -- Cursor control ------------------------------------------------
            screen_pos = None
            if calibrator.done and raw_mapped:
                if fine_tuner.done:
                    screen_pos = fine_tuner.map(raw_mapped[0], raw_mapped[1])
                else:
                    screen_pos = raw_mapped

            cursor_active = False
            if screen_pos:
                sx, sy = smoother.update(screen_pos[0], screen_pos[1])
                eff_gain = gain * dpi_scale
                mx = int(screen_w / 2 + (sx - screen_w / 2) * eff_gain)
                my = int(screen_h / 2 + (sy - screen_h / 2) * eff_gain)
                mx = max(0, min(screen_w - 1, mx))
                my = max(0, min(screen_h - 1, my))
                if self._cursor_enabled and not (calibrator.calibrating or fine_tuner.calibrating):
                    move_cursor(mx, my)
                    cursor_active = True

            # -- Auto-save after calibration DONE (one-shot per calibration run)
            if calibrator.done and not calibrator.calibrating and not fine_tuner.calibrating and not _cal_saved:
                _cal_saved  = True        # ← prevents re-triggering after done_show_t resets
                done_show_t = now
                prof = self._cal_profile
                save_path = self._profile_path(prof)
                calibrator.save(save_path, fine_tuner.save_to_dict())
                errs = cal_state.get('val_errors', [])
                acc  = sum(errs) / len(errs) if errs else None
                self._update(calibrated=True, cal_accuracy=acc, profile=prof)
                print(f"[Controller] Calibration saved -> data/{prof}.json"
                      + (f"  ({acc:.0f} px mean error)" if acc else ""))

            # -- FPS -----------------------------------------------------------
            fps_hist.append(1.0 / max(now - last_t, 1e-6))
            last_t = now
            avg_fps = float(np.mean(fps_hist))

            # -- Update public status ------------------------------------------
            errs = cal_state.get('val_errors', [])
            acc  = (sum(errs) / len(errs)) if errs else self._status.get('cal_accuracy')
            
            if fine_tuner.calibrating:
                active_cs = ft_state.get('state', 'IDLE')
            else:
                active_cs = cs
                
            self._update(
                fps            = avg_fps,
                pupil          = pupil_center,
                cursor_active  = cursor_active,
                cursor_enabled = self._cursor_enabled,
                cal_state      = active_cs,
                cal_pt_idx     = cal_state.get('pt_idx', 0),
                cal_total      = cal_state.get('total_pts', 9),
                ft_pt_idx      = ft_state.get('pts_collected', 0),
                ft_total       = ft_state.get('total_pts', 10),
                cal_accuracy   = acc,
                calibrated     = calibrator.done or self._status.get('calibrated', False),
            )

            # -- Draw eye view -------------------------------------------------
            display = frame.copy()
            if pupil_rect is not None and pupil_center is not None:
                cv2.ellipse(display, pupil_rect, (55, 255, 0), 2)
                cxi, cyi = int(pupil_center[0]), int(pupil_center[1])
                cv2.circle(display, (cxi, cyi), 4, (255, 255, 0), -1)

            if cs in ('WARMUP', 'APPROACHING', 'COLLECTING', 'FLASHING', 'VALIDATING'):
                pt  = cal_state.get('pt_idx', 0)
                tot = cal_state.get('total_pts', 9)
                lbl = f"CAL {cs}  {pt}/{tot}"
                col = (0, 165, 255)
            elif calibrator.done:
                lbl = 'ACTIVE (poly-deg2)'
                col = (0, 255, 80)
            else:
                lbl = "UNCALIBRATED"
                col = (0, 60, 255)

            cv2.putText(display, f"FPS:{avg_fps:.0f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            cv2.putText(display, lbl, (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if pupil_center is None:
                cv2.putText(display, "NO PUPIL", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow('EyeTracker Module -- Eye View', cv2.flip(display, 1))

            # -- Calibration overlay -------------------------------------------
            show_done    = ((calibrator.done and not fine_tuner.calibrating) and done_show_t > 0.0)
            show_overlay = calibrator.calibrating or fine_tuner.calibrating or show_done

            if show_overlay:
                if calibrator.calibrating or (show_done and not fine_tuner.done and not fine_tuner.calibrating):
                    overlay = _draw_cal_frame(screen_w, screen_h, cal_state)
                else:
                    overlay = _draw_ft_frame(screen_w, screen_h, ft_state)
                    
                if not cal_win_open:
                    cv2.namedWindow('EyeTracker Module -- Calibration', cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty('EyeTracker Module -- Calibration',
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                    cal_win_open = True
                cv2.imshow('EyeTracker Module -- Calibration', overlay)

                if show_done and now - done_show_t >= 3.0:
                    done_show_t   = 0.0
                    cal_win_open  = False
                    try:
                        cv2.destroyWindow('EyeTracker Module -- Calibration')
                    except Exception:
                        pass
            elif cal_win_open:
                cal_win_open = False
                try:
                    cv2.destroyWindow('EyeTracker Module -- Calibration')
                except Exception:
                    pass

            cv2.waitKey(1)
        # ======================================================================

        cap.release()
        cv2.destroyAllWindows()
        self._update(camera_ok=False, cursor_active=False,
                     cal_state='IDLE', fps=0.0, pupil=None)
        print("[Controller] Thread stopped.")

