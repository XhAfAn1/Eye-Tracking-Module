#!/usr/bin/env python3
"""
calibration_overlay.py  —  Precision Calibration Visual Renderer
=================================================================
Renders the full-screen calibration animation onto a BGR numpy array
using pure OpenCV drawing primitives (no extra dependencies).

Visual design inspired by GazeRecorder / iPhone Eye Tracking:
  • Dark cinematic background (#080c14)
  • Lissajous moving dot with glow trail (warm-up)
  • Large outer ring that shrinks toward target (approach)
  • Cyan countdown arc sweeping 360° CW (collecting)
  • Quality flash — neon-green (good) / amber (retry) (flashing)
  • Crosshair target with arc (validation)
  • Results summary screen (done)

Public API:
    from calibration_overlay import draw_frame
    frame = draw_frame(screen_w, screen_h, state_dict)
    cv2.imshow('Calibration', frame)
"""

import math
import time
import cv2
import numpy as np

# ── Colour palette (BGR, OpenCV convention) ───────────────────────────────────
BG        = ( 20,  12,   8)   # very dark blue-black
CYAN      = (255, 229,   0)   # electric cyan     RGB(0,229,255)
CYAN_DIM  = (120, 110,   0)   # dim cyan
GREEN     = (100, 255,  80)   # neon green        RGB(80,255,100)
AMBER     = (  0, 165, 255)   # warm amber        RGB(255,165,0)
WHITE     = (255, 255, 255)
DIM       = (130, 130, 130)
DARK_DIM  = ( 45,  35,  25)   # barely-visible grid dots
BLACK     = (  0,   0,   0)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

# ── Geometry ──────────────────────────────────────────────────────────────────
OUTER_R   = 58    # approach ring max radius (px)
INNER_R   = 14    # approach ring min radius (px) — shrinks to here
DOT_R     = 8     # solid inner dot radius (px)
COUNT_R   = 44    # countdown arc radius (px)
VAL_CROSS = 24    # validation crosshair arm length (px)


# ─────────────────────────────────────────────────────────────────────────────
def draw_frame(screen_w: int, screen_h: int, state: dict) -> np.ndarray:
    """
    Render one calibration overlay frame.

    Parameters
    ----------
    screen_w, screen_h : int  Full-screen canvas dimensions.
    state : dict              State dict from PrecisionCalibrator.get_state().

    Returns
    -------
    np.ndarray  BGR image ready for cv2.imshow().
    """
    frame = np.full((screen_h, screen_w, 3), BG, dtype=np.uint8)
    s     = state.get('state', 'IDLE')
    t     = time.time()

    if s == 'IDLE':
        return frame

    # Fixed infrastructure drawn on every active frame
    _draw_grid_guides(frame, screen_w, screen_h)
    _draw_progress(frame, screen_w, state, t)
    _draw_instruction(frame, screen_w, screen_h, state, t)

    # State-specific target animation
    tx, ty = state.get('target', (screen_w // 2, screen_h // 2))
    ctr    = (int(tx), int(ty))

    if   s == 'WARMUP':       _draw_warmup(frame, ctr, state, t)
    elif s == 'APPROACHING':  _draw_approach(frame, ctr, state, t)
    elif s == 'COLLECTING':   _draw_collect(frame, ctr, state, t)
    elif s == 'FLASHING':     _draw_flash(frame, ctr, state, t)
    elif s == 'TRACKING':     _draw_tracking(frame, ctr, state, t)
    elif s == 'VALIDATING':   _draw_validation(frame, ctr, state, t)
    elif s == 'DONE':         _draw_done(frame, screen_w, screen_h, state)

    return frame


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _draw_grid_guides(frame, sw, sh):
    """
    Draw very faint markers at all 9 calibration positions.
    Gives spatial context without distracting from the target dot.
    """
    grid = [
        (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
        (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
        (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
    ]
    for fx, fy in grid:
        cv2.circle(frame, (int(fx * sw), int(fy * sh)), 3, DARK_DIM, -1, cv2.LINE_AA)


def _draw_progress(frame, sw, state, t):
    """
    Progress dots row at top-centre.
    Green = captured (good quality), Amber = captured (low quality),
    Cyan ring = current target, hollow = upcoming.
    """
    total = state.get('total_pts', 9)
    done  = state.get('pt_idx',    0)
    quals = state.get('quality',   [])
    s     = state.get('state', '')

    D   = 13       # dot diameter
    GAP = 7        # gap between dots
    row = total * D + (total - 1) * GAP
    x0  = (sw - row) // 2
    y   = 26

    for i in range(total):
        cx = x0 + i * (D + GAP) + D // 2
        if i < done:
            q   = quals[i] if i < len(quals) else 0.0
            col = GREEN if q >= 0.28 else AMBER
            cv2.circle(frame, (cx, y), D // 2, col, -1, cv2.LINE_AA)
        elif i == done and s in ('APPROACHING', 'COLLECTING', 'FLASHING'):
            pulse = int(2 * abs(math.sin(t * 4)))
            cv2.circle(frame, (cx, y), D // 2 + 1 + pulse, CYAN, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, y), D // 2, (50, 38, 28), 1, cv2.LINE_AA)


def _draw_instruction(frame, sw, sh, state, t):
    """Two-line instruction text near the bottom of the screen."""
    s      = state.get('state', '')
    pt_idx = state.get('pt_idx', 0)
    total  = state.get('total_pts', 9)
    vi     = state.get('val_idx', 0)
    vt     = state.get('val_total', 4)
    ft     = state.get('flash_type', 'good')
    errs   = state.get('val_errors', [])

    text_map = {
        'WARMUP':     ('Follow the dot with your eyes',
                       'Calibration begins automatically...'),
        'APPROACHING':('Look at the dot',
                       f'Point  {pt_idx + 1}  of  {total}'),
        'COLLECTING': ('Hold perfectly still \u2022\u2022\u2022', ''),
        'FLASHING':   (('\u2713  Point captured!'     if ft == 'good'
                        else '\u21ba  Retrying — hold still'),
                       ('Great fixation.' if ft == 'good' else 'Keep eyes on dot')),
        'TRACKING':   (('Look at the dot' if state.get('track_sub') == 'APPROACH' else 'Follow the moving dot smoothly'),
                       f'Path  {state.get("track_idx", 0) + 1}  of  {state.get("track_total", 8)}'),
        'VALIDATING': ('Validation \u2014 look at the cross',
                       f'Dot  {vi + 1}  of  {vt}'),
        'DONE':       ('\u2713  Calibration Complete!',
                       (f'Mean error: {sum(errs)/len(errs):.0f} px'
                        if errs else 'Cursor is now gaze-controlled')),
    }

    lines = text_map.get(s, ('', ''))
    yb    = sh - 65

    for i, txt in enumerate(lines):
        if not txt:
            continue
        scale = 0.75 if i == 0 else 0.52
        col   = WHITE if i == 0 else DIM
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
        x = (sw - tw) // 2
        y = yb + i * 30
        # Drop shadow
        cv2.putText(frame, txt, (x + 1, y + 1), FONT, scale, BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (x,     y    ), FONT, scale, col,   1, cv2.LINE_AA)


def _draw_warmup(frame, ctr, state, t):
    """
    Warm-up dot: Lissajous path, large glowing ring, pulsing centre.
    """
    # Soft glow aura
    _glow(frame, ctr, 45, CYAN, intensity=0.20)

    # Pulsing outer ring
    pulse = int(5 * abs(math.sin(t * 2.2)))
    cv2.circle(frame, ctr, 26 + pulse, CYAN, 2, cv2.LINE_AA)

    # Solid inner dot
    cv2.circle(frame, ctr, DOT_R + 1, WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, ctr, DOT_R - 2, CYAN,  -1, cv2.LINE_AA)

    # Small label above dot
    lbl = 'WARM-UP'
    (lw, _), _ = cv2.getTextSize(lbl, FONT, 0.40, 1)
    lx = ctr[0] - lw // 2
    ly = ctr[1] - 35
    cv2.putText(frame, lbl, (lx + 1, ly + 1), FONT, 0.40, BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, lbl, (lx,     ly    ), FONT, 0.40, CYAN_DIM, 1, cv2.LINE_AA)


def _draw_approach(frame, ctr, state, t):
    """
    Approach: large outer ring shrinks toward the dot (cosine-eased).
    """
    frac = state.get('approach_frac', 0.0)
    # Cosine ease-out: fast start, decelerates at end
    ease    = 0.5 - 0.5 * math.cos(frac * math.pi)
    ring_r  = int(OUTER_R - (OUTER_R - INNER_R) * ease)

    # Glow scaled by ring size
    _glow(frame, ctr, ring_r + 12, CYAN, intensity=0.15)

    # Shrinking outer ring (fades slightly as it shrinks)
    ring_col = _scale(CYAN, 0.6 + 0.4 * (1.0 - frac))
    cv2.circle(frame, ctr, ring_r, ring_col, 2, cv2.LINE_AA)

    # Gentle pulsing inner dot
    pulse = int(2 * abs(math.sin(t * 3.0)))
    cv2.circle(frame, ctr, DOT_R + pulse,     WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, ctr, DOT_R - 2 + pulse, CYAN,  -1, cv2.LINE_AA)


def _draw_collect(frame, ctr, state, t):
    """
    Collecting: cyan countdown arc sweeps 0→360° CW from 12-o'clock.
    """
    frac = state.get('collect_frac', 0.0)

    # Background full-circle guide
    cv2.circle(frame, ctr, COUNT_R, (55, 45, 32), 1, cv2.LINE_AA)

    # Glow (steady)
    _glow(frame, ctr, COUNT_R + 16, CYAN, intensity=0.22)

    # Countdown arc
    if frac > 0.005:
        cv2.ellipse(frame, ctr,
                    (COUNT_R, COUNT_R),
                    0,            # ellipse rotation
                    -90,          # start at 12-o'clock
                    -90 + 360 * frac,   # sweep CW
                    CYAN, 5, cv2.LINE_AA)

    # Solid inner dot
    cv2.circle(frame, ctr, DOT_R,     WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, ctr, DOT_R - 3, CYAN,  -1, cv2.LINE_AA)

    # Subtle "hold" label beneath dot (fades in after first 20 % of arc)
    if frac > 0.20:
        alpha = min(1.0, (frac - 0.20) / 0.20)
        col   = _scale(DIM, alpha * 0.85)
        lbl   = 'hold\u2026'
        (lw, _), _ = cv2.getTextSize(lbl, FONT, 0.42, 1)
        lx = ctr[0] - lw // 2
        ly = ctr[1] + COUNT_R + 22
        cv2.putText(frame, lbl, (lx, ly), FONT, 0.42, col, 1, cv2.LINE_AA)


def _draw_flash(frame, ctr, state, t):
    """
    Quality flash: expanding ring in green (good) or amber (retry).
    """
    frac = state.get('flash_frac', 0.0)
    ft   = state.get('flash_type', 'good')
    col  = GREEN if ft == 'good' else AMBER

    # Expanding ring
    ring_r = int(INNER_R + (OUTER_R - INNER_R) * frac)
    alpha  = max(0.0, 1.0 - frac * 1.2)
    rc     = _scale(col, alpha)

    _glow(frame, ctr, ring_r + 8, col, intensity=0.30 * alpha)
    cv2.circle(frame, ctr, ring_r, rc,  3, cv2.LINE_AA)
    cv2.circle(frame, ctr, DOT_R,  col, -1, cv2.LINE_AA)

    # Check-mark or retry icon
    icon = '\u2713' if ft == 'good' else '\u21ba'
    (iw, ih), _ = cv2.getTextSize(icon, FONTB, 0.55, 2)
    ix = ctr[0] - iw // 2
    iy = ctr[1] + ih // 2
    cv2.putText(frame, icon, (ix + 1, iy + 1), FONTB, 0.55, BLACK, 3, cv2.LINE_AA)
    cv2.putText(frame, icon, (ix,     iy    ), FONTB, 0.55, BLACK, 2, cv2.LINE_AA)


def _draw_tracking(frame, ctr, state, t):
    """
    Smooth pursuit tracking: moving dot with cyan glow.
    """
    sub = state.get('track_sub', 'COLLECT')
    if sub == 'APPROACH':
        frac = state.get('track_approach_frac', 1.0)
        ease    = 0.5 - 0.5 * math.cos(frac * math.pi)
        ring_r  = int(OUTER_R - (OUTER_R - INNER_R) * ease)
        _glow(frame, ctr, ring_r + 12, CYAN, intensity=0.15)
        ring_col = _scale(CYAN, 0.6 + 0.4 * (1.0 - frac))
        cv2.circle(frame, ctr, ring_r, ring_col, 2, cv2.LINE_AA)
        cv2.circle(frame, ctr, DOT_R + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, ctr, DOT_R - 2, CYAN,  -1, cv2.LINE_AA)
    else:
        # Soft glow aura
        _glow(frame, ctr, 35, CYAN, intensity=0.20)

        # Pulsing outer ring
        pulse = int(3 * abs(math.sin(t * 4.0)))
        cv2.circle(frame, ctr, DOT_R * 2 + pulse, CYAN, 1, cv2.LINE_AA)

        # Solid inner dot
        cv2.circle(frame, ctr, DOT_R + 1, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, ctr, DOT_R - 2, CYAN,  -1, cv2.LINE_AA)


def _draw_validation(frame, ctr, state, t):
    """
    Validation: crosshair target + collecting arc.
    """
    sub  = state.get('val_sub', 'APPROACH')
    frac = state.get('collect_frac', 0.0)
    errs = state.get('val_errors', [])

    cx, cy = ctr
    arm    = VAL_CROSS
    col    = WHITE if sub == 'COLLECT' else CYAN

    # Crosshair lines
    cv2.line(frame, (cx - arm, cy), (cx + arm, cy), col, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - arm), (cx, cy + arm), col, 1, cv2.LINE_AA)
    # Centre dot
    cv2.circle(frame, ctr, 4, col, -1, cv2.LINE_AA)

    # Collect arc
    if sub == 'COLLECT' and frac > 0.005:
        cv2.ellipse(frame, ctr, (28, 28), 0,
                    -90, -90 + 360 * frac, CYAN, 3, cv2.LINE_AA)

    # Running error tally (small, beneath crosshair)
    if errs:
        avg_e = sum(errs) / len(errs)
        etxt  = f'avg err so far: {avg_e:.0f} px'
        (ew, _), _ = cv2.getTextSize(etxt, FONT, 0.40, 1)
        ex = cx - ew // 2
        ey = cy + arm + 22
        cv2.putText(frame, etxt, (ex, ey), FONT, 0.40, DIM, 1, cv2.LINE_AA)


def _draw_done(frame, sw, sh, state):
    """Final summary screen shown for ~3 s after calibration completes."""
    errs  = state.get('val_errors', [])
    quals = state.get('quality',    [])
    mean_e = sum(errs)  / len(errs)  if errs  else 0.0
    mean_q = sum(quals) / len(quals) if quals else 0.0

    rows = [
        ('Calibration Complete!',               1.10, GREEN),
        (f'Validation error  :  {mean_e:.0f} px  avg over 4 points',
                                                0.62, WHITE),
        (f'Point quality     :  {mean_q:.2f} / 1.0  (9 points)',
                                                0.62, WHITE),
        ('Cursor control is now active.',        0.55, CYAN),
        ('Press r to recalibrate at any time.',  0.48, DIM),
    ]

    y = sh // 2 - 90
    for txt, scale, col in rows:
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 2)
        x = (sw - tw) // 2
        cv2.putText(frame, txt, (x + 1, y + 1), FONT, scale, BLACK, 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (x,     y    ), FONT, scale, col,   2, cv2.LINE_AA)
        y += int(scale * 58 + 10)

    # Quality dots row
    if quals:
        _draw_quality_summary(frame, sw, sh, quals)


def _draw_quality_summary(frame, sw, sh, quals):
    """Small colour-coded quality row near the bottom of the done screen."""
    total = len(quals)
    D = 15; GAP = 5
    row = total * D + (total - 1) * GAP
    x0  = (sw - row) // 2
    y   = sh - 55
    for i, q in enumerate(quals):
        cx  = x0 + i * (D + GAP) + D // 2
        col = GREEN if q >= 0.28 else AMBER if q > 0 else DARK_DIM
        cv2.circle(frame, (cx, y), D // 2, col, -1, cv2.LINE_AA)
        ptxt = f'{int(q * 100)}%'
        (pw, _), _ = cv2.getTextSize(ptxt, FONT, 0.30, 1)
        cv2.putText(frame, ptxt, (cx - pw // 2, y + D // 2 + 14),
                    FONT, 0.30, DIM, 1, cv2.LINE_AA)


# ── Fine-Tuning Renderer ──────────────────────────────────────────────────────

def draw_ft_frame(screen_w: int, screen_h: int, state: dict) -> np.ndarray:
    """Render the Level 2 Fine-Tuning overlay."""
    frame = np.full((screen_h, screen_w, 3), BG, dtype=np.uint8)
    s = state.get('state', 'IDLE')
    if s == 'IDLE': return frame

    # Draw live cursor
    live = state.get('live_cursor')
    if live:
        cx, cy = live
        cv2.circle(frame, (cx, cy), 15, CYAN, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, CYAN, -1, cv2.LINE_AA)
        cv2.putText(frame, "Predicted Gaze", (cx + 20, cy + 5), FONT, 0.4, CYAN, 1, cv2.LINE_AA)

    if s == 'DONE':
        txt = "Level 2 Fine-Tuning Complete!"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 1.10, 2)
        x = (screen_w - tw) // 2
        y = screen_h // 2
        cv2.putText(frame, txt, (x, y), FONT, 1.10, GREEN, 2, cv2.LINE_AA)
        return frame

    # Draw target
    tx, ty = state.get('target', (screen_w // 2, screen_h // 2))
    ctr = (int(tx), int(ty))
    
    col = GREEN if s == 'ACCEPTED' else WHITE
    arm = 20
    cv2.line(frame, (ctr[0] - arm, ctr[1]), (ctr[0] + arm, ctr[1]), col, 2, cv2.LINE_AA)
    cv2.line(frame, (ctr[0], ctr[1] - arm), (ctr[0], ctr[1] + arm), col, 2, cv2.LINE_AA)
    cv2.circle(frame, ctr, 5, col, -1, cv2.LINE_AA)
    if s == 'ACCEPTED':
        _glow(frame, ctr, 50, GREEN, 0.4)
        
    prog = state.get('hold_progress', 0.0)
    if prog > 0.0 and s != 'ACCEPTED':
        # Draw circular progress bar around target
        radius = arm + 10
        cv2.ellipse(frame, ctr, (radius, radius), -90, 0, prog * 360, CYAN, 3, cv2.LINE_AA)

    # Instructions
    idx = state.get('pts_collected', 0)
    tot = state.get('total_pts', 10)
    
    lines = [
        "Look exactly at the white crosshair.",
        "Assistant: Hold SPACEBAR to confirm.",
        f"Point {idx} of {tot}"
    ]
    yb = screen_h - 90
    for i, txt in enumerate(lines):
        scale = 0.6 if i < 2 else 0.5
        c = WHITE if i == 0 else (GREEN if i == 1 else DIM)
        (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
        x = (screen_w - tw) // 2
        y = yb + i * 30
        cv2.putText(frame, txt, (x + 1, y + 1), FONT, scale, BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, y), FONT, scale, c, 1, cv2.LINE_AA)
        
    return frame

# ── Drawing helpers ───────────────────────────────────────────────────────────

def _glow(frame, center, radius: int, color: tuple, intensity: float = 0.25):
    """
    Additive glow effect: concentric filled circles blended additively.
    Using cv2.add() avoids float alpha blending cost.
    """
    overlay = np.zeros_like(frame)
    layers  = 4
    for i in range(layers, 0, -1):
        r      = max(1, radius * i // layers)
        bright = int(255 * intensity * i / layers)
        c      = tuple(min(255, int(ch * bright // 255)) for ch in color)
        cv2.circle(overlay, center, r, c, -1, cv2.LINE_AA)
    cv2.add(frame, overlay, frame)


def _scale(color: tuple, factor: float) -> tuple:
    """Scale a BGR colour by a [0..1] factor."""
    f = max(0.0, min(1.0, factor))
    return tuple(int(c * f) for c in color)

