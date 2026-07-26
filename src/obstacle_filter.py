"""
╔══════════════════════════════════════════════════════════════════╗
║             EchoVision  –  src/obstacle_filter.py                ║
║  Cleans up the raw detections from detector.py:                  ║
║    • removes duplicate/overlapping boxes across the two models   ║
║    • tags each detection as "hazard" (navigation-critical) or    ║
║      "general" (everyday object)                                 ║
║    • caps the number of detections passed downstream             ║
╚══════════════════════════════════════════════════════════════════╝
"""

from . import config


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def filter_detections(detections):
    """
    detections: list[Detection] from detector.py
    Returns a deduplicated, capped list, sorted by confidence, each
    still a Detection but with a `.hazard` attribute (bool) attached.
    """
    # Sort by confidence, greedy-suppress overlapping lower-confidence boxes
    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept = []
    for det in detections:
        is_dup = any(_iou(det.bbox, k.bbox) > config.NMS_IOU_THRES
                     and det.label == k.label for k in kept)
        if is_dup:
            continue
        det.hazard = det.label in config.HAZARD_CLASSES or det.source == "nav"
        kept.append(det)
        if len(kept) >= config.MAX_DETECTIONS_PER_FRAME:
            break

    # Hazards first, then by confidence — most navigation-relevant first
    kept.sort(key=lambda d: (not d.hazard, -d.conf))
    return kept
