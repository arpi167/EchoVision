"""
╔══════════════════════════════════════════════════════════════════╗
║               EchoVision  –  src/config.py                       ║
║  Single source of truth for paths, models, thresholds.           ║
║  Every other module imports from here — change values once.      ║
╚══════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
MODEL_DIR     = BASE_DIR / "models"
DATASET_DIR   = BASE_DIR / "datasets"
OUTPUT_DIR    = BASE_DIR / "outputs"
LOG_DIR       = OUTPUT_DIR / "logs"
RECORDING_DIR = OUTPUT_DIR / "recordings"

# ─────────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────────
# General "daily-life objects" model — Ultralytics YOLOv8n pretrained on
# COCO (80 classes: person, car, bicycle, dog, cat, chair, bottle, laptop …).
# Auto-downloaded by Ultralytics on first run if not already in models/.
GENERAL_MODEL_PATH = str(MODEL_DIR / "yolov8n.pt")
GENERAL_MODEL_HUB_NAME = "yolov8n.pt"          # fallback: ultralytics auto-download

# Custom navigation-hazard model (stairs, potholes, speed breakers, open
# drains, road barriers, construction blocks, traffic cones, curbs, low
# hanging obstacles). Produced by training/train.py. Optional at runtime —
# detect.py runs fine on general objects only if this file is missing.
NAV_MODEL_PATH = str(MODEL_DIR / "navigation_best.pt")

# MiDaS monocular depth model (torch.hub, intel-isl/MiDaS)
MIDAS_MODEL_NAME = "MiDaS_small"

# TFLite export target (produced by training/export_tflite.py, used by the
# separate Android/Phase-2 project — not required for the desktop app)
TFLITE_MODEL_PATH = str(MODEL_DIR / "best.tflite")

# ─────────────────────────────────────────────────────────────────
#  CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────────
# 9 navigation-specific hazard classes (custom-trained model).
# Index order MUST match training/dataset.yaml.
NAV_CLASSES = [
    "stairs",
    "potholes",
    "open_drains",
    "road_barriers",
    "doors",
]

# COCO classes we don't bother announcing (too small / not navigation-relevant)
GENERAL_SKIP_CLASSES = {
    "cell phone", "remote", "scissors", "toothbrush", "spoon",
    "fork", "knife", "clock", "book", "vase", "tie", "frisbee",
    "sports ball", "kite", "wine glass", "cup", "bowl",
}

# All navigation-hazard classes are always treated as high priority.
HAZARD_CLASSES = set(NAV_CLASSES)

# ─────────────────────────────────────────────────────────────────
#  DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────
GENERAL_CONF_THRES = 0.40   # confidence threshold — general COCO model
NAV_CONF_THRES      = 0.35  # confidence threshold — custom hazard model
NMS_IOU_THRES        = 0.45  # IoU used to de-duplicate overlapping boxes
                              # across the two models in obstacle_filter.py
MAX_DETECTIONS_PER_FRAME = 10

# ─────────────────────────────────────────────────────────────────
#  LOW-LIGHT HANDLING
# ─────────────────────────────────────────────────────────────────
# Mean pixel brightness (0-255) below which utils.enhance_low_light()
# kicks in automatically before running detection.
LOW_LIGHT_BRIGHTNESS_THRESHOLD = 90
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID   = (8, 8)
GAMMA_CORRECTION  = 1.6      # >1 brightens midtones; used only when very dark
VERY_DARK_THRESHOLD = 50     # below this, also apply gamma correction

# ─────────────────────────────────────────────────────────────────
#  DISTANCE ESTIMATION (monocular depth — see depth_estimator.py docstring
#  for an honest explanation of the accuracy limits of this approach)
# ─────────────────────────────────────────────────────────────────
DEPTH_SCALE = 0.4            # calibration constant — tune with a known
                              # reference distance, see depth_estimator.py
NEAR_DISTANCE_M   = 4.0      # objects closer than this are flagged urgent
CRITICAL_DISTANCE_M = 1.5    # objects this close get a "very close" warning

# ─────────────────────────────────────────────────────────────────
#  DIRECTION ZONES (as a fraction of frame width)
# ─────────────────────────────────────────────────────────────────
LEFT_ZONE_RATIO  = 0.33      # bbox center x < 33% of width  → "left"
RIGHT_ZONE_RATIO = 0.66      # bbox center x > 66% of width  → "right"
                              # otherwise                     → "center"

# ─────────────────────────────────────────────────────────────────
#  VOICE / ANNOUNCEMENT
# ─────────────────────────────────────────────────────────────────
TTS_INTERVAL_FRAMES = 60     # announce roughly every N frames (~2s @30fps)
MAX_EVENTS_PER_ANNOUNCEMENT = 3
TTS_RATE   = 150
TTS_VOLUME = 1.0
