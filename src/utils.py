"""
╔══════════════════════════════════════════════════════════════════╗
║               EchoVision  –  src/utils.py                        ║
║  Shared helpers: logging, low-light image enhancement,           ║
║  FPS counter, HUD drawing.                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations  # allows tuple[...] / X | Y on Python 3.9+

import logging
import sys
import time

import cv2
import numpy as np

from . import config


# ─────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────
def setup_logger(name="EchoVision"):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(config.LOG_DIR / "echovision.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_logger()


# ─────────────────────────────────────────────────────────────────
#  LOW-LIGHT ENHANCEMENT
# ─────────────────────────────────────────────────────────────────
def compute_brightness(frame: np.ndarray) -> float:
    """Mean grayscale brightness, 0 (black) – 255 (white)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def enhance_low_light(frame: np.ndarray, force: bool = False) -> tuple[np.ndarray, bool]:
    """
    Automatically brightens frames that are too dark for reliable detection.

    Pipeline:
      1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
         L-channel in LAB colour space — boosts local contrast without
         blowing out highlights.
      2. If the frame is *very* dark, additionally applies gamma
         correction to lift shadow detail.

    NOTE (honesty check): this measurably helps YOLO recover detections in
    dim/indoor/dusk conditions, but it cannot manufacture detail that the
    sensor never captured (e.g. near-total darkness). For genuinely
    low-light-critical use, pair this with a phone's night-mode camera
    pipeline or an IR-assisted sensor — this is a software-only mitigation.

    Returns (possibly-enhanced frame, was_enhanced: bool)
    """
    brightness = compute_brightness(frame)
    if not force and brightness >= config.LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        return frame, False

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                             tileGridSize=config.CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    enhanced = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    if brightness < config.VERY_DARK_THRESHOLD:
        inv_gamma = 1.0 / config.GAMMA_CORRECTION
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                           for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    return enhanced, True


# ─────────────────────────────────────────────────────────────────
#  FPS COUNTER
# ─────────────────────────────────────────────────────────────────
class FPSMeter:
    def __init__(self, smoothing=0.9):
        self._t = time.time()
        self._fps = 0.0
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        dt = max(now - self._t, 1e-6)
        self._t = now
        inst_fps = 1.0 / dt
        self._fps = (self._smoothing * self._fps +
                     (1 - self._smoothing) * inst_fps) if self._fps else inst_fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ─────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
PALETTE = {
    "coco":     (34,  197, 94),    # green   – general daily-life object
    "hazard":   (239, 68,  68),    # red     – navigation hazard (pothole, stairs…)
    "warning":  (251, 146, 60),    # orange  – close-range warning
    "hud_bg":   (0, 0, 0),
}


def draw_box(frame, x1, y1, x2, y2, label, distance, direction, kind="coco"):
    color = PALETTE.get(kind, PALETTE["coco"])
    thickness = 3 if kind == "hazard" else 2
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    dist_str = f" {distance:.1f}m" if distance is not None else ""
    dir_str = f" [{direction}]" if direction else ""
    text = f"{label}{dist_str}{dir_str}"
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 2)
    pad = 4
    cv2.rectangle(frame,
                   (int(x1), int(y1) - th - pad * 2 - baseline),
                   (int(x1) + tw + pad * 2, int(y1)),
                   color, -1)
    cv2.putText(frame, text, (int(x1) + pad, int(y1) - pad - baseline),
                font, scale, (255, 255, 255), 2)


def draw_hud(frame, frame_n, fps, obj_count, tts_in, low_light: bool):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), PALETTE["hud_bg"], -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    ll_tag = "  [LOW-LIGHT BOOST]" if low_light else ""
    cv2.putText(frame,
                f"EchoVision | FPS:{fps:.1f}  Frame:{frame_n}  Objects:{obj_count}  "
                f"TTS in:{tts_in}f{ll_tag}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    legends = [
        ("■ Daily object", PALETTE["coco"]),
        ("■ Navigation hazard", PALETTE["hazard"]),
        ("■ Close-range warning", PALETTE["warning"]),
    ]
    lx = 10
    for txt, col in legends:
        cv2.putText(frame, txt, (lx, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        lx += 210
