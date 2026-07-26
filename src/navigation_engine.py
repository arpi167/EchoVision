"""
╔══════════════════════════════════════════════════════════════════╗
║            EchoVision  –  src/navigation_engine.py               ║
║  Combines detection + distance + direction into a small set of   ║
║  prioritized "navigation events" worth speaking aloud this tick. ║
╚══════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass
class NavEvent:
    label: str
    distance: Optional[float]     # metres, or None if depth unavailable
    direction: str                # "left" | "center" | "right"
    hazard: bool
    urgent: bool                  # within NEAR_DISTANCE_M
    critical: bool                # within CRITICAL_DISTANCE_M


class NavigationEngine:
    def process(self, detections, frame_width, depth_estimator, dmap):
        """
        detections: filtered list[Detection] (with `.hazard` set) from
                    obstacle_filter.py
        Returns list[NavEvent], most important first, capped at
        MAX_EVENTS_PER_ANNOUNCEMENT.
        """
        from .direction_detector import get_direction

        events = []
        for det in detections:
            distance = depth_estimator.box_distance(dmap, det.bbox) if dmap is not None else None
            direction = get_direction(det.bbox, frame_width)
            urgent = distance is not None and distance <= config.NEAR_DISTANCE_M
            critical = distance is not None and distance <= config.CRITICAL_DISTANCE_M

            events.append(NavEvent(
                label=det.label,
                distance=distance,
                direction=direction,
                hazard=det.hazard,
                urgent=urgent,
                critical=critical,
            ))

        # Priority: critical > hazard+urgent > hazard > urgent > everything else,
        # then closer distance first (None distances sort last).
        def sort_key(e: NavEvent):
            return (
                not e.critical,
                not (e.hazard and e.urgent),
                not e.hazard,
                not e.urgent,
                e.distance if e.distance is not None else float("inf"),
            )

        events.sort(key=sort_key)
        return events[:config.MAX_EVENTS_PER_ANNOUNCEMENT]
