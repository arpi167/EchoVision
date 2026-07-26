"""
╔══════════════════════════════════════════════════════════════════╗
║           EchoVision  –  src/direction_detector.py               ║
║  Classifies WHERE in the frame an object is: Left / Center /     ║
║  Right — using bounding-box centre position vs frame thirds.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

from . import config


def get_direction(bbox, frame_width: int) -> str:
    """
    bbox: (x1, y1, x2, y2) in pixel coordinates
    Returns "left" | "center" | "right"
    """
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2.0
    ratio = cx / max(frame_width, 1)

    if ratio < config.LEFT_ZONE_RATIO:
        return "left"
    elif ratio > config.RIGHT_ZONE_RATIO:
        return "right"
    return "center"


def direction_phrase(direction: str) -> str:
    """Natural-language fragment used by sentence_generator.py."""
    return {
        "left":   "on your left",
        "right":  "on your right",
        "center": "ahead",
    }.get(direction, "nearby")
