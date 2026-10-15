#!/usr/bin/env python3
"""
EyeTracker Module — Settings Window
===================================
A modern Tkinter-based settings panel that lets you configure:
  • Custom DPI (scales cursor movement to match your monitor's real DPI)
  • Cursor Gain
  • Smoothing Alpha
  • Camera Index

Settings are persisted in 'gaze_settings.json' next to this script.

Usage (standalone):
    python settings_window.py

Usage (from gaze_mouse.py — open on top of the running loop):
    from settings_window import open_settings, load_settings
    cfg = load_settings()
"""

import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import math

# ─────────────────────────────────────────────────────────────────────────────
# Paths & defaults
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_THIS_DIR, "gaze_settings.json")

DEFAULTS = {
    "dpi":                96.0,   # Monitor DPI  (Windows default = 96)
    "gain":               1.0,    # Cursor gain  (0.2 – 3.0)
    "alpha":              0.15,   # Smoothing    (0.02 – 0.90)
    "camera_id":          0,      # Camera index
    "grace_ms":           250,    # Blink grace period in ms
    "spacebar_click":     True,   # Spacebar simulates a left click at cursor
}

COMMON_DPIS = [
    ("96 DPI  — Standard (100 %)",           96),
    ("120 DPI — High DPI (125 %)",          120),
    ("144 DPI — QHD / Retina-like (150 %)", 144),
    ("192 DPI — 4 K (200 %)",               192),
]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    """Return settings dict, falling back to DEFAULTS for missing keys."""
    cfg = dict(DEFAULTS)
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: saved[k] for k in DEFAULTS if k in saved})
        except Exception:
            pass
    return cfg


def save_settings(cfg: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

PAL = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "card":      "#22263a",
    "accent":    "#3b82f6",
    "accent2":   "#06b6d4",
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "text":      "#e2e8f0",
    "subtext":   "#94a3b8",
    "border":    "#2d3748",
    "hover":     "#2563eb",
    "danger":    "#ef4444",
}


# ─────────────────────────────────────────────────────────────────────────────
# Custom widgets
# ─────────────────────────────────────────────────────────────────────────────

class ModernSlider(tk.Frame):
    """A slider with a live value label and optional unit string."""

    def __init__(self, parent, label, from_, to, resolution, initial,
                 unit="", fmt="{:.2f}", command=None, **kwargs):
        super().__init__(parent, bg=PAL["card"], **kwargs)
        self.fmt     = fmt
        self.unit    = unit
        self._cmd    = command
        self.var     = tk.DoubleVar(value=initial)

        # Row 1 — label + live value
        top = tk.Frame(self, bg=PAL["card"])
        top.pack(fill="x", padx=12, pady=(10, 2))

        tk.Label(top, text=label, font=("Segoe UI", 10, "bold"),
                 fg=PAL["text"], bg=PAL["card"]).pack(side="left")

        self.val_lbl = tk.Label(top, text=self._fmt(initial),
                                font=("Segoe UI", 10),
                                fg=PAL["accent"], bg=PAL["card"])
        self.val_lbl.pack(side="right")

        # Row 2 — the slider track
        slider_frame = tk.Frame(self, bg=PAL["card"])
        slider_frame.pack(fill="x", padx=12, pady=(0, 10))

        self.slider = tk.Scale(
            slider_frame,
            variable=self.var,
            from_=from_, to=to,
            resolution=resolution,
            orient="horizontal",
            showvalue=False,
            bg=PAL["card"], fg=PAL["text"],
            troughcolor=PAL["border"],
            activebackground=PAL["hover"],
            highlightthickness=0,
            bd=0,
            command=self._on_change,
        )
        self.slider.pack(fill="x")

        # Min / Max hints
        hints = tk.Frame(self, bg=PAL["card"])
        hints.pack(fill="x", padx=14, pady=(0, 4))
        tk.Label(hints, text=f"{from_}{unit}", font=("Segoe UI", 8),
                 fg=PAL["subtext"], bg=PAL["card"]).pack(side="left")
        tk.Label(hints, text=f"{to}{unit}", font=("Segoe UI", 8),
                 fg=PAL["subtext"], bg=PAL["card"]).pack(side="right")

    def _fmt(self, v):
        return self.fmt.format(v) + self.unit

    def _on_change(self, val):
        self.val_lbl.config(text=self._fmt(float(val)))
        if self._cmd:
            self._cmd(float(val))

    def get(self):
        return self.var.get()

    def set(self, v):
        self.var.set(v)
        self.val_lbl.config(text=self._fmt(float(v)))


class SectionHeader(tk.Frame):
    """Coloured section divider with icon + title."""

    def __init__(self, parent, icon, title, color=None, **kwargs):
        color = color or PAL["accent"]
        super().__init__(parent, bg=PAL["surface"], **kwargs)
        tk.Label(self, text=f"  {icon}  {title}",
                 font=("Segoe UI", 11, "bold"),
                 fg=color, bg=PAL["surface"],
                 anchor="w").pack(fill="x", pady=4)
        tk.Frame(self, height=1, bg=color).pack(fill="x")


# ─────────────────────────────────────────────────────────────────────────────
# Settings Window
# ─────────────────────────────────────────────────────────────────────────────

class SettingsWindow(tk.Toplevel):
    """
    Opens a modern settings panel.

    Parameters
    ----------
    master      : Tk root (or None -> creates its own root)
    cfg         : current settings dict (from load_settings())
    on_apply    : optional callback(new_cfg) called when user clicks Apply
    """

    def __init__(self, master=None, cfg=None, on_apply=None):
        if master is None:
            master = tk.Tk()
            master.withdraw()
            self._owns_root = True
        else:
            self._owns_root = False

        super().__init__(master)
        self.master_ref  = master
        self._on_apply   = on_apply
        self._cfg        = dict(cfg or load_settings())
        self._result     = None

        self._build_ui()
        self._centre_window(540, 760)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.focus_force()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.title("EyeTracker Module — Settings")
        self.resizable(False, False)
        self.configure(bg=PAL["bg"])

        # ── Title bar ─────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PAL["accent"], height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="EyeTracker Module Settings",
                 font=("Segoe UI", 14, "bold"),
                 fg="white", bg=PAL["accent"]).pack(
            side="left", padx=18, pady=12)

        # ── Scrollable body ────────────────────────────────────────
        body = tk.Frame(self, bg=PAL["bg"])
        body.pack(fill="both", expand=True)

        canvas = tk.Canvas(body, bg=PAL["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas, bg=PAL["bg"])

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left",  fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        sf = self._scroll_frame

        # ═══════════════════════════════════════════════════════════
        # 1 — DPI SETTINGS
        # ═══════════════════════════════════════════════════════════
        SectionHeader(sf, "Monitor DPI", "", color=PAL["accent2"]).pack(
            fill="x", padx=16, pady=(16, 4))

        dpi_card = tk.Frame(sf, bg=PAL["card"], relief="flat")
        dpi_card.pack(fill="x", padx=16, pady=4)

        tk.Label(dpi_card,
                 text=("DPI controls how cursor movement scales to your physical screen.\n"
                       "96 DPI = standard (100%), 192 DPI = 4K/HiDPI (200%)."),
                 font=("Segoe UI", 9), fg=PAL["subtext"], bg=PAL["card"],
                 justify="left", wraplength=460).pack(padx=12, pady=(10, 6), anchor="w")

        # Preset radio buttons
        preset_frame = tk.Frame(dpi_card, bg=PAL["card"])
        preset_frame.pack(fill="x", padx=12, pady=4)

        self._dpi_preset_var = tk.StringVar(value="custom")
        current_dpi = self._cfg["dpi"]

        for label, val in COMMON_DPIS:
            rb = tk.Radiobutton(
                preset_frame,
                text=label,
                variable=self._dpi_preset_var,
                value=str(val),
                command=lambda v=val: self._apply_dpi_preset(v),
                font=("Segoe UI", 9),
                fg=PAL["text"], bg=PAL["card"],
                selectcolor=PAL["surface"],
                activebackground=PAL["card"],
                activeforeground=PAL["accent"],
                cursor="hand2",
            )
            rb.pack(anchor="w", pady=1)
            if abs(current_dpi - val) < 0.5:
                self._dpi_preset_var.set(str(val))

        tk.Frame(dpi_card, height=1, bg=PAL["border"]).pack(fill="x", padx=12, pady=6)

        # Custom DPI entry
        custom_row = tk.Frame(dpi_card, bg=PAL["card"])
        custom_row.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(custom_row, text="Custom DPI:",
                 font=("Segoe UI", 10, "bold"),
                 fg=PAL["text"], bg=PAL["card"]).pack(side="left")

        self._dpi_var = tk.StringVar(value=str(int(self._cfg["dpi"])))
        dpi_entry = tk.Entry(
            custom_row,
            textvariable=self._dpi_var,
            width=7,
            font=("Segoe UI Mono", 11),
            bg=PAL["surface"], fg=PAL["accent"],
            insertbackground=PAL["accent"],
            relief="flat",
            bd=4,
        )
        dpi_entry.pack(side="left", padx=10)
        dpi_entry.bind("<FocusIn>",
                       lambda e: self._dpi_preset_var.set("custom"))

        tk.Label(custom_row, text="px/inch",
                 font=("Segoe UI", 9),
                 fg=PAL["subtext"], bg=PAL["card"]).pack(side="left")

        self._dpi_info_lbl = tk.Label(
            dpi_card,
            text=self._dpi_info_text(current_dpi),
            font=("Segoe UI", 9, "italic"),
            fg=PAL["success"], bg=PAL["card"],
        )
        self._dpi_info_lbl.pack(anchor="w", padx=14, pady=(0, 6))
        self._dpi_var.trace_add("write", lambda *_: self._update_dpi_info())

        # Detect DPI button
        detect_btn = tk.Button(
            dpi_card,
            text="  Auto-detect System DPI",
            font=("Segoe UI", 9, "bold"),
            fg=PAL["text"], bg=PAL["surface"],
            activebackground=PAL["hover"],
            activeforeground="white",
            relief="flat", cursor="hand2", bd=0, padx=10, pady=6,
            command=self._detect_system_dpi,
        )
        detect_btn.pack(anchor="w", padx=12, pady=(0, 12))
        detect_btn.bind("<Enter>", lambda e: detect_btn.config(bg=PAL["hover"], fg="white"))
        detect_btn.bind("<Leave>", lambda e: detect_btn.config(bg=PAL["surface"], fg=PAL["text"]))

        # ═══════════════════════════════════════════════════════════
        # 2 — CURSOR SETTINGS
        # ═══════════════════════════════════════════════════════════
        SectionHeader(sf, "Cursor Settings", "", color=PAL["accent"]).pack(
            fill="x", padx=16, pady=(16, 4))

        gain_card = tk.Frame(sf, bg=PAL["card"])
        gain_card.pack(fill="x", padx=16, pady=4)

        tk.Label(gain_card,
                 text="Gain amplifies cursor movement around the screen centre.",
                 font=("Segoe UI", 9), fg=PAL["subtext"], bg=PAL["card"],
                 justify="left").pack(padx=12, pady=(8, 0), anchor="w")

        self._gain_slider = ModernSlider(
            gain_card,
            label="Cursor Gain",
            from_=0.2, to=3.0, resolution=0.1,
            initial=self._cfg["gain"],
            fmt="{:.1f}", unit="x",
        )
        self._gain_slider.pack(fill="x")

        tk.Label(gain_card,
                 text="Smoothing: lower = smoother but slower, higher = faster but jittery.",
                 font=("Segoe UI", 9), fg=PAL["subtext"], bg=PAL["card"],
                 justify="left").pack(padx=12, pady=(4, 0), anchor="w")

        self._alpha_slider = ModernSlider(
            gain_card,
            label="Smoothing Alpha",
            from_=0.02, to=0.90, resolution=0.01,
            initial=self._cfg["alpha"],
            fmt="{:.2f}",
        )
        self._alpha_slider.pack(fill="x")

        tk.Label(gain_card,
                 text="Blink grace: cursor freezes for this many ms during blinks.",
                 font=("Segoe UI", 9), fg=PAL["subtext"], bg=PAL["card"],
                 justify="left").pack(padx=12, pady=(4, 0), anchor="w")

        self._grace_slider = ModernSlider(
            gain_card,
            label="Blink Grace Period",
            from_=50, to=800, resolution=10,
            initial=self._cfg.get("grace_ms", 250),
            fmt="{:.0f}", unit=" ms",
        )
        self._grace_slider.pack(fill="x", pady=(0, 6))

        # ═══════════════════════════════════════════════════════════
        # 3 — CAMERA
        # ═══════════════════════════════════════════════════════════
        SectionHeader(sf, "Camera", "", color=PAL["warning"]).pack(
            fill="x", padx=16, pady=(16, 4))

        cam_card = tk.Frame(sf, bg=PAL["card"])
        cam_card.pack(fill="x", padx=16, pady=4)

        cam_row = tk.Frame(cam_card, bg=PAL["card"])
        cam_row.pack(fill="x", padx=12, pady=10)

        tk.Label(cam_row, text="Camera Index:",
                 font=("Segoe UI", 10, "bold"),
                 fg=PAL["text"], bg=PAL["card"]).pack(side="left")

        self._cam_var = tk.IntVar(value=self._cfg.get("camera_id", 0))
        cam_spin = tk.Spinbox(
            cam_row,
            from_=0, to=9,
            textvariable=self._cam_var,
            width=4,
            font=("Segoe UI Mono", 11),
            bg=PAL["surface"], fg=PAL["warning"],
            insertbackground=PAL["warning"],
            relief="flat", bd=4,
            buttonbackground=PAL["surface"],
        )
        cam_spin.pack(side="left", padx=10)

        tk.Label(cam_row, text="(0 = built-in, 1 = external, ...)",
                 font=("Segoe UI", 9),
                 fg=PAL["subtext"], bg=PAL["card"]).pack(side="left")

        tk.Frame(sf, height=16, bg=PAL["bg"]).pack()

        # ═══════════════════════════════════════════════════════════
        # 4 — INPUT SHORTCUTS
        # ═══════════════════════════════════════════════════════════
        SectionHeader(sf, "Input Shortcuts", "", color=PAL["success"]).pack(
            fill="x", padx=16, pady=(16, 4))

        sc_card = tk.Frame(sf, bg=PAL["card"])
        sc_card.pack(fill="x", padx=16, pady=4)

        self._spacebar_click_var = tk.BooleanVar(
            value=self._cfg.get("spacebar_click", True))

        sc_row = tk.Frame(sc_card, bg=PAL["card"])
        sc_row.pack(fill="x", padx=12, pady=10)

        tk.Checkbutton(
            sc_row,
            text="Spacebar simulates a click at cursor position",
            variable=self._spacebar_click_var,
            font=("Segoe UI", 10),
            fg=PAL["text"], bg=PAL["card"],
            selectcolor=PAL["surface"],
            activebackground=PAL["card"],
            activeforeground=PAL["success"],
            cursor="hand2",
        ).pack(side="left")

        tk.Label(
            sc_card,
            text=("When enabled: pressing Spacebar sends a left-click at whatever\n"
                  "the cursor is pointing at. Disable if the Flutter app handles\n"
                  "Spacebar internally to avoid double-firing."),
            font=("Segoe UI", 9), fg=PAL["subtext"], bg=PAL["card"],
            justify="left", wraplength=460,
        ).pack(padx=12, pady=(0, 10), anchor="w")

        tk.Frame(sf, height=16, bg=PAL["bg"]).pack()

        # ═══════════════════════════════════════════════════════════
        # Bottom button bar
        # ═══════════════════════════════════════════════════════════
        bar = tk.Frame(self, bg=PAL["surface"], height=62)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        def _btn(parent, text, fg, bg, cmd):
            b = tk.Button(parent, text=text,
                          font=("Segoe UI", 10, "bold"),
                          fg=fg, bg=bg,
                          activebackground=PAL["hover"],
                          activeforeground="white",
                          relief="flat", cursor="hand2",
                          padx=18, pady=8, command=cmd)
            b.bind("<Enter>", lambda e: b.config(bg=PAL["hover"], fg="white"))
            b.bind("<Leave>", lambda e: b.config(bg=bg, fg=fg))
            return b

        _btn(bar, "Cancel",  PAL["subtext"], PAL["card"],   self._on_cancel).pack(
            side="right", padx=12, pady=10)
        _btn(bar, "Reset",   PAL["warning"], PAL["card"],   self._on_reset ).pack(
            side="right", padx=4,  pady=10)
        _btn(bar, "Apply",   "white",         PAL["accent"], self._on_apply_click).pack(
            side="right", padx=4,  pady=10)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _centre_window(self, w, h):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x  = (sw - w) // 2
        y  = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _dpi_info_text(self, dpi_val):
        try:
            dpi = float(dpi_val)
            scale = dpi / 96.0
            return f"  -> Scale factor: {scale:.3f}x  ({dpi:.0f} / 96)"
        except (ValueError, ZeroDivisionError):
            return "  -> Invalid DPI value"

    def _update_dpi_info(self):
        try:
            v = float(self._dpi_var.get())
            self._dpi_info_lbl.config(
                text=self._dpi_info_text(v),
                fg=PAL["success"] if 48 <= v <= 600 else PAL["danger"],
            )
        except ValueError:
            self._dpi_info_lbl.config(
                text="  -> Invalid DPI value", fg=PAL["danger"])

    def _apply_dpi_preset(self, val):
        self._dpi_var.set(str(val))

    def _detect_system_dpi(self):
        """Try to read the actual system DPI via ctypes (Windows)."""
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            hdc  = ctypes.windll.user32.GetDC(0)
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
            ctypes.windll.user32.ReleaseDC(0, hdc)
            self._dpi_var.set(str(dpi_x))
            for _, v in COMMON_DPIS:
                if v and abs(dpi_x - v) < 6:
                    self._dpi_preset_var.set(str(v))
                    return
            self._dpi_preset_var.set("custom")
        except Exception as exc:
            messagebox.showerror("DPI Detection Failed",
                                 f"Could not read system DPI:\n{exc}",
                                 parent=self)

    def _collect_cfg(self):
        """Validate inputs and return config dict, or None on error."""
        try:
            dpi = float(self._dpi_var.get())
            if not (48 <= dpi <= 600):
                raise ValueError("DPI must be between 48 and 600.")
        except ValueError as e:
            messagebox.showerror("Invalid DPI",
                                 f"Please enter a valid DPI value.\n\n{e}",
                                 parent=self)
            return None

        return {
            "dpi":            dpi,
            "gain":           round(self._gain_slider.get(), 1),
            "alpha":          round(self._alpha_slider.get(), 3),
            "grace_ms":       int(self._grace_slider.get()),
            "camera_id":      self._cam_var.get(),
            "spacebar_click": self._spacebar_click_var.get(),
        }

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_apply_click(self):
        cfg = self._collect_cfg()
        if cfg is None:
            return
        save_settings(cfg)
        self._result = cfg
        if self._on_apply:
            self._on_apply(cfg)
        messagebox.showinfo("Settings Saved",
                            "Settings saved successfully.\n\n"
                            "Restart EyeTracker Module (or apply live if supported) "
                            "for all changes to take effect.",
                            parent=self)
        self._close()

    def _on_reset(self):
        if messagebox.askyesno("Reset to Defaults",
                               "Reset all settings to factory defaults?",
                               parent=self):
            d = dict(DEFAULTS)
            self._dpi_var.set(str(int(d["dpi"])))
            self._dpi_preset_var.set(str(int(d["dpi"])))
            self._gain_slider.set(d["gain"])
            self._alpha_slider.set(d["alpha"])
            self._grace_slider.set(d["grace_ms"])
            self._cam_var.set(d["camera_id"])
            self._spacebar_click_var.set(d["spacebar_click"])

    def _on_cancel(self):
        self._close()

    def _close(self):
        self.grab_release()
        self.destroy()
        if self._owns_root:
            self.master_ref.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers for use from gaze_mouse.py
# ─────────────────────────────────────────────────────────────────────────────

def open_settings(master=None, current_cfg=None, on_apply=None):
    """
    Open the settings window and block until the user closes it.
    Returns the new config dict if Apply was clicked, else None.
    """
    win = SettingsWindow(master=master, cfg=current_cfg, on_apply=on_apply)
    win.wait_window()
    return win._result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    cfg    = load_settings()
    result = open_settings(master=root, current_cfg=cfg)

    if result:
        print("\nSettings applied:")
        for k, v in result.items():
            print(f"   {k:12s} = {v}")
    else:
        print("Settings cancelled — no changes saved.")

    root.destroy()
