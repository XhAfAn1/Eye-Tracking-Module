#!/usr/bin/env python3
"""
launcher.py  --  Eye Tracking Module Control Panel
================================================
Premium dark-theme Tkinter launcher with a full menu bar.

Features:
  * File    | Show Eye View   / Exit
  * Calibration | Start Calibration / New Profile / Rename / Delete /
                  View Data
  * Control | Start / Stop Cursor Control / Settings
  * Help    | About

Status panel shows camera, calibration, and cursor-control health with
colour-coded indicators (green / amber / red / dim).

Calibration profiles are stored in  data/<name>.json
Run:
    python launcher.py
"""

import os
import sys
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# ── Resolve paths ─────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
DATA_DIR  = os.path.join(_THIS_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = '#0d1117'
SURFACE = '#161b22'
SURF2   = '#21262d'
BORDER  = '#30363d'
CYAN    = '#00e5ff'
GREEN   = '#3fb950'
AMBER   = '#e3b341'
RED     = '#f85149'
TEXT    = '#f0f6fc'
SUBTEXT = '#8b949e'
DIM     = '#484f58'
BLACK   = '#000000'

FONT    = ('Segoe UI', 10)
FONT_B  = ('Segoe UI', 10, 'bold')
FONT_LG = ('Segoe UI', 13, 'bold')
FONT_SM = ('Segoe UI', 9)
FONT_XS = ('Segoe UI', 8)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_profiles() -> list:
    """Return sorted list of profile names (filenames without .json)."""
    try:
        files = [f[:-5] for f in os.listdir(DATA_DIR)
                 if f.endswith('.json')]
        return sorted(files) or ['default']
    except Exception:
        return ['default']


def _profile_meta(name: str) -> str:
    """One-line description of a profile for display."""
    path = os.path.join(DATA_DIR, f'{name}.json')
    if not os.path.exists(path):
        return 'No calibration data yet'
    mtime = os.path.getmtime(path)
    stamp = time.strftime('%d %b %Y  %H:%M', time.localtime(mtime))
    try:
        with open(path) as f:
            d = json.load(f)
        errs = d.get('val_err', [])
        acc  = f'{sum(errs)/len(errs):.0f} px accuracy' if errs else 'accuracy unknown'
        return f'Last calibrated: {stamp}  \u00b7  {acc}'
    except Exception:
        return f'Last modified: {stamp}'


def _btn(parent, text, cmd, bg=SURF2, fg=TEXT, font=FONT, pad=(14, 9),
         hover_bg=BORDER, hover_fg=TEXT, **kw):
    """Flat button with hover effect."""
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, font=font,
                  relief='flat', bd=0, cursor='hand2',
                  activebackground=hover_bg, activeforeground=hover_fg,
                  padx=pad[0], pady=pad[1], **kw)
    b.bind('<Enter>', lambda e: b.config(bg=hover_bg, fg=hover_fg))
    b.bind('<Leave>', lambda e: b.config(bg=bg, fg=fg))
    return b


def _label(parent, text='', fg=TEXT, font=FONT, bg=None, **kw):
    bg = bg or parent.cget('bg')
    return tk.Label(parent, text=text, fg=fg, font=font, bg=bg, **kw)


def _sep(parent, color=BORDER):
    tk.Frame(parent, bg=color, height=1).pack(fill='x', pady=6)


# =============================================================================
class EyeTrackerModuleApp:
    """Main application window."""

    POLL_MS = 250   # status polling interval

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('Eye Tracking Module  --  Control Panel')
        root.geometry('510x460')
        root.resizable(False, False)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self._on_close)

        # ── State ─────────────────────────────────────────────────────────────
        self._controller   = None
        self._profile_var  = tk.StringVar(value='default')
        self._profiles     = []
        self._running      = False

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_menu()
        self._build_header()
        self._build_status_card()
        self._build_profile_card()
        self._build_actions()
        self._build_statusbar()

        # ── Keyboard shortcuts ────────────────────────────────────────────────
        root.bind('<F5>', lambda e: self._start_control())
        root.bind('<F6>', lambda e: self._stop_control())

        # ── Initial refresh ────────────────────────────────────────────────────
        self._refresh_profiles()
        self._poll()

    # =========================================================================
    # Menu
    # =========================================================================

    def _build_menu(self):
        mbar = tk.Menu(self.root,
                       bg=SURF2, fg=TEXT, relief='flat', bd=0,
                       activebackground=BORDER, activeforeground=CYAN)
        self.root.config(menu=mbar)

        def _menu(label, items):
            m = tk.Menu(mbar, tearoff=0, bg=SURF2, fg=TEXT, relief='flat',
                        activebackground=BORDER, activeforeground=TEXT,
                        font=FONT_SM)
            for item in items:
                if item is None:
                    m.add_separator()
                else:
                    txt, cmd, *rest = item
                    acc = rest[0] if rest else ''
                    m.add_command(label=txt, command=cmd, accelerator=acc)
            mbar.add_cascade(label=label, menu=m)
            return m

        _menu('File', [
            ('Show Eye View',         self._show_eye_view),
            ('Close Eye View',        self._close_eye_view),
            None,
            ('Exit',                  self._on_close,       'Alt+F4'),
        ])

        _menu('Calibration', [
            ('Start Calibration',    self._do_calibrate,   'Ctrl+G'),
            ('Start Fine-Tuning (L2)', self._do_fine_tune, 'Ctrl+T'),
            None,
            ('New Profile...',       self._new_profile),
            ('Rename Profile...',    self._rename_profile),
            ('Delete Profile',       self._delete_profile),
            None,
            ('View Calibration Data',self._view_cal_data),
        ])

        _menu('Control', [
            ('Start Cursor Control',  self._start_control,  'F5'),
            ('Stop Cursor Control',   self._stop_control,   'F6'),
            None,
            ('Settings...',           self._open_settings),
        ])

        _menu('Help', [
            ('About Eye Tracking Module', self._about),
        ])

        self.root.bind('<Control-g>', lambda e: self._do_calibrate())
        self.root.bind('<Control-t>', lambda e: self._do_fine_tune())

    # =========================================================================
    # UI sections
    # =========================================================================

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=SURFACE, pady=0)
        hdr.pack(fill='x')

        canvas = tk.Canvas(hdr, height=58, bg=SURFACE, highlightthickness=0)
        canvas.pack(fill='x')

        # Cyan accent circle
        canvas.create_oval(16, 14, 42, 40, fill=CYAN, outline='')
        canvas.create_text(28, 27, text='ET', font=('Segoe UI', 10, 'bold'),
                           fill=BLACK)

        canvas.create_text(54, 19, text='Eye Tracking Module', anchor='w',
                           fill=TEXT, font=FONT_B)
        canvas.create_text(54, 38, text='Prototype',
                           fill=SUBTEXT, font=FONT_SM, anchor='nw')

        # Thin border line below header
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x')

    def _build_status_card(self):
        wrap = tk.Frame(self.root, bg=BG, padx=16, pady=10)
        wrap.pack(fill='x')

        _label(wrap, 'STATUS', fg=SUBTEXT, font=FONT_XS).pack(anchor='w', pady=(0, 4))

        card = tk.Frame(wrap, bg=SURFACE,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill='x')

        inner = tk.Frame(card, bg=SURFACE, padx=14, pady=10)
        inner.pack(fill='x')

        self._dots = {}
        self._dot_labels = {}

        rows = [
            ('camera',  'Camera'),
            ('cal',     'Calibration'),
            ('control', 'Cursor Control'),
        ]
        for key, display in rows:
            row = tk.Frame(inner, bg=SURFACE)
            row.pack(fill='x', pady=3)

            cv = tk.Canvas(row, width=12, height=12,
                           bg=SURFACE, highlightthickness=0)
            cv.pack(side='left', padx=(0, 10))
            dot = cv.create_oval(1, 1, 11, 11, fill=DIM, outline='')
            self._dots[key] = (cv, dot)

            _label(row, display, fg=SUBTEXT, font=FONT_SM, bg=SURFACE,
                   width=15, anchor='w').pack(side='left')

            lbl = _label(row, '\u2014', fg=TEXT, font=FONT_SM, bg=SURFACE, anchor='w')
            lbl.pack(side='left', fill='x')
            self._dot_labels[key] = lbl

    def _build_profile_card(self):
        wrap = tk.Frame(self.root, bg=BG, padx=16)
        wrap.pack(fill='x')

        _label(wrap, 'CALIBRATION PROFILE', fg=SUBTEXT,
               font=FONT_XS).pack(anchor='w', pady=(0, 4))

        card = tk.Frame(wrap, bg=SURFACE,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill='x')

        inner = tk.Frame(card, bg=SURFACE, padx=14, pady=10)
        inner.pack(fill='x')

        row1 = tk.Frame(inner, bg=SURFACE)
        row1.pack(fill='x')

        _label(row1, 'Active profile:', fg=SUBTEXT, font=FONT_SM,
               bg=SURFACE, width=15, anchor='w').pack(side='left')

        self._combo = ttk.Combobox(
            row1, textvariable=self._profile_var,
            state='readonly', width=17, font=FONT_SM)
        self._combo.pack(side='left', padx=(0, 8))
        self._combo.bind('<<ComboboxSelected>>', self._on_profile_change)

        _btn(row1, 'New', self._new_profile,
             bg=SURF2, font=FONT_SM, pad=(6, 3)).pack(side='left', padx=2)
        _btn(row1, 'Rename', self._rename_profile,
             bg=SURF2, font=FONT_SM, pad=(6, 3)).pack(side='left', padx=2)

        self._prof_meta = _label(inner, '', fg=SUBTEXT, font=FONT_SM, bg=SURFACE)
        self._prof_meta.pack(anchor='w', pady=(6, 0))

    def _build_actions(self):
        wrap = tk.Frame(self.root, bg=BG, padx=16, pady=12)
        wrap.pack(fill='x')

        # Primary action: Start / Stop
        self._start_btn = _btn(
            wrap,
            text='\u25b6   Start Cursor Control',
            cmd=self._toggle_control,
            bg=CYAN, fg=BLACK,
            font=('Segoe UI', 11, 'bold'),
            hover_bg='#00b8d4', hover_fg=BLACK,
            pad=(20, 11))
        self._start_btn.pack(side='left', expand=True, fill='x', padx=(0, 10))

        # Secondary: Calibrate
        self._cal_btn = _btn(
            wrap,
            text='\u2699  Calibrate',
            cmd=self._do_calibrate,
            bg=SURF2, fg=TEXT,
            font=FONT,
            hover_bg=BORDER, hover_fg=TEXT,
            pad=(14, 11))
        self._cal_btn.pack(side='right')

    def _build_statusbar(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x')
        bar = tk.Frame(self.root, bg=SURFACE)
        bar.pack(fill='x', side='bottom')
        self._sbar_var = tk.StringVar(value='Ready  \u00b7  Camera not started')
        tk.Label(bar, textvariable=self._sbar_var,
                 bg=SURFACE, fg=SUBTEXT, font=FONT_SM,
                 anchor='w', padx=10, pady=4).pack(fill='x')

    # =========================================================================
    # Polling
    # =========================================================================

    def _poll(self):
        """Update UI from controller status every POLL_MS ms."""
        try:
            if self._controller and self._controller.is_running:
                self._apply_status(self._controller.get_status())
        except Exception:
            pass
        self.root.after(self.POLL_MS, self._poll)

    def _apply_status(self, s: dict):
        # Camera dot
        if s['camera_ok']:
            self._set_dot('camera', GREEN)
            self._dot_labels['camera'].config(
                text=f"Connected  \u00b7  Camera {self._controller.camera_id}")
        else:
            self._set_dot('camera', RED)
            self._dot_labels['camera'].config(text='Not found')

        # Calibration dot
        cs = s['cal_state']
        cal = s['calibrated']
        acc = s['cal_accuracy']
        if cs in ('WARMUP', 'APPROACHING', 'COLLECTING', 'FLASHING', 'VALIDATING', 'TRACKING', 'WAITING', 'ACCEPTED'):
            self._set_dot('cal', AMBER)
            if cs in ('WAITING', 'ACCEPTED'):
                pt  = s.get('ft_pt_idx', 0)
                tot = s.get('ft_total', 10)
                self._dot_labels['cal'].config(text=f"Fine-Tuning  \u00b7  {pt}/{tot} accepted")
            else:
                pt  = s.get('cal_pt_idx', 0)
                tot = s.get('cal_total', 9)
                self._dot_labels['cal'].config(text=f"Calibrating  \u00b7  {cs}  {pt}/{tot}")
        elif cal:
            self._set_dot('cal', GREEN)
            acc_txt = f'{acc:.0f} px accuracy' if acc else 'Active'
            self._dot_labels['cal'].config(text=f"Active  \u00b7  {acc_txt}")
        else:
            self._set_dot('cal', DIM)
            self._dot_labels['cal'].config(text="Not calibrated  \u00b7  press Calibrate")

        # Cursor dot
        cur_enabled = s.get('cursor_enabled', False)
        cur_active  = s.get('cursor_active',  False)
        if cur_active:
            self._set_dot('control', GREEN)
            self._dot_labels['control'].config(text='Active  \u00b7  Cursor moving')
        elif cur_enabled:
            self._set_dot('control', AMBER)
            self._dot_labels['control'].config(text='Enabled  \u00b7  No pupil / not calibrated')
        elif self._controller and self._controller.is_running:
            self._set_dot('control', DIM)
            self._dot_labels['control'].config(text='Eye View only  \u00b7  cursor not active')
        else:
            self._set_dot('control', DIM)
            self._dot_labels['control'].config(text='Stopped')

        # Start button label
        if cur_enabled != self._running:
            self._running = cur_enabled
            if self._running:
                self._start_btn.config(
                    text='\u25a0   Stop Cursor Control',
                    bg=SURF2, fg=RED,
                    activebackground=BORDER, activeforeground=RED)
                self._start_btn.bind('<Enter>', lambda e: self._start_btn.config(bg=BORDER))
                self._start_btn.bind('<Leave>', lambda e: self._start_btn.config(bg=SURF2))
            else:
                self._start_btn.config(
                    text='\u25b6   Start Cursor Control',
                    bg=CYAN, fg=BLACK,
                    activebackground='#00b8d4', activeforeground=BLACK)
                self._start_btn.bind('<Enter>',
                    lambda e: self._start_btn.config(bg='#00b8d4'))
                self._start_btn.bind('<Leave>',
                    lambda e: self._start_btn.config(bg=CYAN))

        # Status bar
        fps    = s['fps']
        pupil  = s['pupil']
        p_txt  = (f"Pupil: ({pupil[0]:.0f}, {pupil[1]:.0f})"
                  if pupil else "Pupil: not detected")
        a_txt  = f"Accuracy: {acc:.0f} px" if acc else ""
        self._sbar_var.set(
            f"FPS: {fps:.0f}  \u00b7  {p_txt}"
            + (f"  \u00b7  {a_txt}" if a_txt else ""))

    def _set_dot(self, key: str, color: str):
        canvas, dot = self._dots[key]
        canvas.itemconfig(dot, fill=color)

    # =========================================================================
    # Actions
    # =========================================================================

    def _ensure_controller(self) -> bool:
        """Create and start the controller camera thread if not already running.
        Does NOT enable cursor movement -- call set_cursor_enabled(True) separately.
        Returns True if controller is running."""
        if self._controller and self._controller.is_running:
            return True
        from gaze_mouse import get_screen_size
        try:
            from settings_window import load_settings
            cfg = load_settings()
        except Exception:
            cfg = {}
        sw, sh = get_screen_size()
        cam    = int(cfg.get('camera_id', 0))
        from gaze_controller import GazeController
        self._controller = GazeController(sw, sh, cam, cfg, DATA_DIR)
        prof = self._profile_var.get()
        self._controller.load_profile(prof)
        self._controller.start()
        # Note: _running stays False here -- cursor NOT enabled yet
        return self._controller.is_running

    def _toggle_control(self):
        """Toggle cursor control on/off (camera stays open either way)."""
        if self._running:
            self._stop_control()
        else:
            self._start_control()

    def _start_control(self):
        """Start camera (if needed) and enable cursor movement."""
        if not self._ensure_controller():
            messagebox.showerror('Camera Error',
                                 'Could not open camera.\n'
                                 'Check camera connection and try again.')
            return
        self._controller.set_cursor_enabled(True)
        self._running = True
        self._start_btn.config(text='\u25a0   Stop Cursor Control',
                               bg=SURF2, fg=RED,
                               activebackground=BORDER, activeforeground=RED)

    def _stop_control(self):
        """Disable cursor movement -- camera + eye view stay open."""
        if self._controller:
            self._controller.set_cursor_enabled(False)
        self._running = False
        self._start_btn.config(text='\u25b6   Start Cursor Control',
                               bg=CYAN, fg=BLACK,
                               activebackground='#00b8d4', activeforeground=BLACK)

    def _close_eye_view(self):
        """Fully stop the controller thread and close all CV2 windows."""
        self._running = False
        if self._controller:
            self._controller.stop()
        self._controller = None
        self._start_btn.config(text='\u25b6   Start Cursor Control',
                               bg=CYAN, fg=BLACK,
                               activebackground='#00b8d4', activeforeground=BLACK)
        for k in ('camera', 'cal', 'control'):
            self._set_dot(k, DIM)
        self._dot_labels['camera'].config(text='\u2014')
        self._dot_labels['cal'].config(text='\u2014')
        self._dot_labels['control'].config(text='Stopped')
        self._sbar_var.set('Eye View closed  \u00b7  Camera stopped')

    def _do_calibrate(self):
        if not self._ensure_controller():
            return
        self._controller.trigger_calibration(self._profile_var.get())

    def _do_fine_tune(self):
        if not self._ensure_controller():
            return
        if not self._controller.get_status().get('calibrated'):
            tk.messagebox.showwarning("Not Calibrated", "Please perform a standard Level 1 calibration before fine-tuning.")
            return
        self._controller.trigger_fine_tuning(self._profile_var.get())
        self._sbar_var.set(f"Fine-tuning started  \u00b7  profile: {self._profile_var.get()}")

    # =========================================================================
    # Profile management
    # =========================================================================

    def _refresh_profiles(self):
        self._profiles = _list_profiles()
        self._combo['values'] = self._profiles
        cur = self._profile_var.get()
        if cur not in self._profiles:
            self._profile_var.set(self._profiles[0] if self._profiles else 'default')
        self._update_prof_meta()

    def _update_prof_meta(self):
        name = self._profile_var.get()
        self._prof_meta.config(text=_profile_meta(name))

    def _on_profile_change(self, _event=None):
        self._update_prof_meta()
        # Load the newly selected profile into the running controller
        if self._controller and self._controller.is_running:
            self._controller.load_profile(self._profile_var.get())

    def _new_profile(self):
        name = simpledialog.askstring(
            'New Profile', 'Enter a name for the new profile:',
            parent=self.root)
        if not name:
            return
        name = name.strip().replace(' ', '_').lower()
        if not name:
            return
        # Just select it (file created when calibration is saved)
        self._profile_var.set(name)
        self._profiles = _list_profiles()
        if name not in self._profiles:
            self._profiles.append(name)
            self._profiles.sort()
        self._combo['values'] = self._profiles
        self._update_prof_meta()
        messagebox.showinfo('New Profile',
                            f'Profile "{name}" created.\n'
                            'Press Calibrate to capture data for it.',
                            parent=self.root)

    def _rename_profile(self):
        old = self._profile_var.get()
        new = simpledialog.askstring(
            'Rename Profile', f'New name for "{old}":',
            parent=self.root, initialvalue=old)
        if not new or new == old:
            return
        new = new.strip().replace(' ', '_').lower()
        old_path = os.path.join(DATA_DIR, f'{old}.json')
        new_path = os.path.join(DATA_DIR, f'{new}.json')
        if os.path.exists(old_path):
            try:
                os.rename(old_path, new_path)
            except Exception as e:
                messagebox.showerror('Error', str(e), parent=self.root)
                return
        self._profile_var.set(new)
        self._refresh_profiles()

    def _delete_profile(self):
        name = self._profile_var.get()
        if not messagebox.askyesno(
                'Delete Profile',
                f'Delete profile "{name}"?\nThis cannot be undone.',
                parent=self.root):
            return
        path = os.path.join(DATA_DIR, f'{name}.json')
        if os.path.exists(path):
            os.remove(path)
        self._refresh_profiles()
        if self._profiles:
            self._profile_var.set(self._profiles[0])
        self._update_prof_meta()

    def _view_cal_data(self):
        """Show the raw JSON of the current profile in a read-only dialog."""
        name = self._profile_var.get()
        path = os.path.join(DATA_DIR, f'{name}.json')
        if not os.path.exists(path):
            messagebox.showinfo('No Data',
                                f'Profile "{name}" has not been calibrated yet.',
                                parent=self.root)
            return
        try:
            with open(path) as f:
                raw = f.read()
            d = json.loads(raw)
        except Exception as e:
            messagebox.showerror('Error', str(e), parent=self.root)
            return

        # Parse for human-readable summary
        errs = d.get('val_err', [])
        acc  = f'{sum(errs)/len(errs):.1f} px' if errs else 'N/A'
        qual = d.get('quality', [])
        mean_q = f'{sum(qual)/len(qual)*100:.0f}%' if qual else 'N/A'
        screen = d.get('screen', ['?', '?'])

        win = tk.Toplevel(self.root)
        win.title(f'Calibration Data  --  {name}')
        win.geometry('500x400')
        win.configure(bg=BG)
        win.resizable(False, False)

        tk.Label(win, text=f'Profile: {name}', font=FONT_LG,
                 bg=BG, fg=TEXT).pack(anchor='w', padx=16, pady=(14, 2))
        tk.Label(win, text=_profile_meta(name), font=FONT_SM,
                 bg=BG, fg=SUBTEXT).pack(anchor='w', padx=16)

        tk.Frame(win, bg=BORDER, height=1).pack(fill='x', pady=10)

        rows_info = [
            ('Model',         d.get('model', 'bivariate_poly_deg2')),
            ('Screen',        f"{screen[0]} x {screen[1]}"),
            ('Val. accuracy', acc),
            ('Avg quality',   mean_q),
            ('Points',        str(len(qual))),
        ]
        for lbl, val in rows_info:
            row = tk.Frame(win, bg=BG)
            row.pack(fill='x', padx=16, pady=2)
            tk.Label(row, text=lbl + ':', font=FONT_SM, fg=SUBTEXT,
                     bg=BG, width=18, anchor='w').pack(side='left')
            tk.Label(row, text=val, font=FONT_SM, fg=TEXT,
                     bg=BG, anchor='w').pack(side='left')

        tk.Frame(win, bg=BORDER, height=1).pack(fill='x', pady=10)

        tk.Label(win, text='Polynomial coefficients (cx  |  cy):',
                 font=FONT_SM, fg=SUBTEXT, bg=BG).pack(anchor='w', padx=16)
        txt = tk.Text(win, bg=SURFACE, fg=SUBTEXT, font=('Consolas', 8),
                      relief='flat', bd=0, padx=10, pady=8)
        txt.pack(fill='both', expand=True, padx=16, pady=(4, 16))
        cx = d.get('cx', [])
        cy = d.get('cy', [])
        terms = ['1', 'px', 'py', 'px*py', 'px^2', 'py^2']
        txt.insert('end', 'screen_x  =\n')
        for t, c in zip(terms, cx):
            txt.insert('end', f"   {c:+.6f}  *  {t}\n")
        txt.insert('end', '\nscreen_y  =\n')
        for t, c in zip(terms, cy):
            txt.insert('end', f"   {c:+.6f}  *  {t}\n")
        txt.config(state='disabled')

        _btn(win, 'Close', win.destroy,
             bg=SURF2, pad=(14, 7)).pack(pady=(0, 12))

    # =========================================================================
    # Misc menu actions
    # =========================================================================

    def _show_eye_view(self):
        """Open camera and show eye view WITHOUT enabling cursor movement."""
        if not self._ensure_controller():
            messagebox.showerror('Camera Error',
                                 'Could not open camera.\n'
                                 'Check camera connection and try again.')
            return
        # Explicitly keep cursor disabled -- user must click Start Cursor Control
        self._controller.set_cursor_enabled(False)
        self._sbar_var.set('Eye View open  \u00b7  Press Start Cursor Control to move the cursor')

    def _open_settings(self):
        try:
            from settings_window import open_settings, load_settings
            cfg = load_settings()
            new_cfg = open_settings(master=self.root, current_cfg=cfg)
            if new_cfg and self._controller:
                # Apply non-camera settings on the fly
                self._controller.cfg.update(new_cfg)
        except ImportError:
            messagebox.showinfo('Settings',
                                'settings_window.py not found.\n'
                                'Edit gaze_settings.json manually.',
                                parent=self.root)

    def _about(self):
        win = tk.Toplevel(self.root)
        win.title('About Eye Tracking Module')
        win.geometry('380x280')
        win.configure(bg=BG)
        win.resizable(False, False)

        c = tk.Canvas(win, height=70, bg=BG, highlightthickness=0)
        c.pack(fill='x')
        c.create_oval(155, 14, 195, 54, fill=CYAN, outline='')
        c.create_text(175, 34, text='ET', font=('Segoe UI', 13, 'bold'), fill=BLACK)

        for txt, fnt, col in [
            ('Eye Tracking Module',          FONT_LG, TEXT),
            ('Prototype',  FONT_SM, SUBTEXT),
        ]:
            tk.Label(win, text=txt, font=fnt, bg=BG, fg=col).pack()

        tk.Frame(win, bg=BORDER, height=1).pack(fill='x', pady=10)

        rows = [
            'Pupil tracking by OrloskyPupilDetector',
            'Calibration: bivariate polynomial (degree 2)',
            'Outlier rejection: Tukey IQR fence',
            'No ML required -- pure NumPy mathematics',
            'Designed for Plegia / motor-impaired users',
        ]
        for r in rows:
            tk.Label(win, text=r, font=FONT_SM, bg=BG, fg=SUBTEXT).pack(pady=1)

        _btn(win, 'Close', win.destroy,
             bg=SURF2, pad=(14, 7)).pack(pady=14)

    def _on_close(self):
        if self._controller:
            self._controller.stop()
        self.root.after(300, self.root.destroy)


# =============================================================================
# Entry point
# =============================================================================

def main():
    root = tk.Tk()
    # Attempt to set a dark window title bar on Windows 11
    try:
        root.update()
        from ctypes import windll, c_int, byref, sizeof
        HWND = windll.user32.GetParent(root.winfo_id())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        windll.dwmapi.DwmSetWindowAttribute(
            HWND, DWMWA_USE_IMMERSIVE_DARK_MODE,
            byref(c_int(1)), sizeof(c_int))
    except Exception:
        pass

    app = EyeTrackerModuleApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
