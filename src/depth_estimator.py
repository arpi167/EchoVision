"""
╔══════════════════════════════════════════════════════════════════╗
║             EchoVision  –  src/depth_estimator.py                ║
║  Monocular depth via MiDaS-small (torch.hub, intel-isl/MiDaS)    ║
║  → converted to an approximate metric distance per detection.    ║
║                                                                    ║
║  ⚠ HONEST ACCURACY NOTE (please read):                            ║
║  MiDaS predicts *relative* inverse depth, not metric distance.    ║
║  There is no true metric ground-truth from a single RGB camera    ║
║  without either (a) a calibrated reference object of known size   ║
║  in-frame, (b) a second camera (stereo), or (c) a depth sensor    ║
║  (LiDAR/ToF/IR, e.g. an iPhone Pro's LiDAR or an Intel RealSense). ║
║  `DEPTH_SCALE` in config.py is a manual calibration constant you  ║
║  must tune against a real, measured reference distance — see      ║
║  `calibrate_scale()` below. Treat the numbers as "close / medium  ║
║  / far" guidance, not survey-grade measurement. For genuinely      ║
║  accurate distance, pair this software with a phone that exposes  ║
║  LiDAR/ToF (ARKit/ARCore depth API) in the Phase-2 Android app.   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations  # allows `float | None` on Python 3.9+

import cv2
import numpy as np
import torch

from . import config
from .utils import log


class DepthEstimator:
    def __init__(self):
        log.info(f"Loading MiDaS depth model ({config.MIDAS_MODEL_NAME}) …")
        self.model = torch.hub.load("intel-isl/MiDaS", config.MIDAS_MODEL_NAME,
                                     trust_repo=True)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.scale = config.DEPTH_SCALE
        log.info("MiDaS [OK]")

    def depth_map(self, frame) -> np.ndarray:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False,
            ).squeeze()
        dmap = pred.cpu().numpy()
        dmap = dmap / (dmap.max() + 1e-6)   # normalise 0-1 (near=1 for MiDaS)
        return dmap

    def box_distance(self, dmap: np.ndarray, bbox) -> float | None:
        x1, y1, x2, y2 = bbox
        roi = dmap[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        if roi.size == 0:
            return None
        d = float(np.median(roi))
        return round(self.scale / (d + 1e-6), 2)

    def calibrate_scale(self, dmap: np.ndarray, bbox, known_distance_m: float) -> float:
        """
        Call this once with an object of a KNOWN, measured distance
        (e.g. a chair you placed exactly 2.0 m away) to derive a better
        DEPTH_SCALE for config.py:

            new_scale = median_depth_value * known_distance_m
        """
        x1, y1, x2, y2 = bbox
        roi = dmap[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
        d = float(np.median(roi)) if roi.size else 0.0
        new_scale = d * known_distance_m
        log.info(f"Calibration suggestion: set DEPTH_SCALE = {new_scale:.4f} "
                  f"in config.py")
        return new_scale
