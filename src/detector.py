"""
╔══════════════════════════════════════════════════════════════════╗
║               EchoVision  –  src/detector.py                     ║
║  Runs TWO Ultralytics YOLO models on every frame:                ║
║    1. General model  (yolov8n.pt, COCO-pretrained, 80 classes)   ║
║       → every "daily life" object: person, car, bicycle, dog,    ║
║         chair, bottle, laptop, backpack …                        ║
║    2. Navigation-hazard model (navigation_best.pt, custom-        ║
║       trained — see training/) → stairs, potholes, speed          ║
║       breakers, open drains, road barriers, construction          ║
║       blocks, traffic cones, curbs, low hanging obstacles.        ║
║                                                                    ║
║  Both models are 100% open-source / run fully offline — no paid   ║
║  or rate-limited third-party detection API is required or used.   ║
╚══════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

from . import config
from .utils import log, enhance_low_light, compute_brightness


class Detection:
    """Simple structured detection result."""
    __slots__ = ("label", "conf", "bbox", "source", "hazard")

    def __init__(self, label, conf, bbox, source):
        self.label = label
        self.conf = conf
        self.bbox = bbox
        self.source = source
        self.hazard = False       # "coco" | "nav"

    def __repr__(self):
        return f"Detection({self.label}, conf={self.conf:.2f}, src={self.source})"


class ObjectDetector:
    def __init__(self, auto_low_light=True):
        from ultralytics import YOLO

        self.auto_low_light = auto_low_light

        # ── General "daily life objects" model (COCO, 80 classes) ──────
        general_path = config.GENERAL_MODEL_PATH
        if not Path(general_path).exists():
            log.warning(
                f"{general_path} not found locally — Ultralytics will "
                f"auto-download '{config.GENERAL_MODEL_HUB_NAME}'."
            )
            general_path = config.GENERAL_MODEL_HUB_NAME
        log.info(f"Loading general COCO model: {general_path}")
        self.general_model = YOLO(general_path)

        # ── Custom navigation-hazard model (optional) ──────────────────
        self.nav_model = None
        if Path(config.NAV_MODEL_PATH).exists():
            log.info(f"Loading navigation-hazard model: {config.NAV_MODEL_PATH}")
            self.nav_model = YOLO(config.NAV_MODEL_PATH)
        else:
            log.warning(
                "No navigation-hazard model found at "
                f"{config.NAV_MODEL_PATH} — running general-object "
                "detection only. Train one with training/train.py."
            )

    def detect(self, frame):
        """
        Returns (detections: list[Detection], enhanced_frame, was_low_light: bool)
        `enhanced_frame` is what was actually fed to the models — draw
        boxes on this frame (coordinates match it 1:1).
        """
        was_low_light = False
        infer_frame = frame
        if self.auto_low_light:
            infer_frame, was_low_light = enhance_low_light(frame)

        detections = []

        # ── General COCO pass ───────────────────────────────────────────
        res = self.general_model(infer_frame, conf=config.GENERAL_CONF_THRES,
                                  verbose=False)[0]
        if res.boxes is not None and res.boxes.xyxy.numel() > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls  = res.boxes.cls.cpu().numpy()
            names = res.names
            for i in range(len(xyxy)):
                label = names[int(cls[i])]
                if label in config.GENERAL_SKIP_CLASSES:
                    continue
                detections.append(Detection(label, float(conf[i]),
                                             tuple(xyxy[i]), "coco"))

        # ── Navigation-hazard pass ──────────────────────────────────────
        if self.nav_model is not None:
            res_n = self.nav_model(infer_frame, conf=config.NAV_CONF_THRES,
                                    verbose=False)[0]
            if res_n.boxes is not None and res_n.boxes.xyxy.numel() > 0:
                xyxy = res_n.boxes.xyxy.cpu().numpy()
                conf = res_n.boxes.conf.cpu().numpy()
                cls  = res_n.boxes.cls.cpu().numpy()
                names = res_n.names
                for i in range(len(xyxy)):
                    label = names[int(cls[i])]
                    detections.append(Detection(label, float(conf[i]),
                                                 tuple(xyxy[i]), "nav"))

        return detections, infer_frame, was_low_light
