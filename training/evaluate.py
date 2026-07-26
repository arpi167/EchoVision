"""
╔══════════════════════════════════════════════════════════════════╗
║             EchoVision  –  training/evaluate.py                  ║
║  Validates the trained navigation-hazard model and prints mAP,   ║
║  precision and recall, overall and per-class.                    ║
║                                                                    ║
║  Usage:                                                            ║
║      python training/evaluate.py                                   ║
║      python training/evaluate.py --model models/navigation_best.pt ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src import config  # noqa: E402

DATASET_CFG = BASE_DIR / "training" / "dataset.yaml"


def evaluate(model_path: str, imgsz: int = 640, device: str = "cpu"):
    from ultralytics import YOLO

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Train first: python training/train.py")
        return

    model = YOLO(model_path)
    metrics = model.val(data=str(DATASET_CFG), imgsz=imgsz, device=device)

    print("\n" + "=" * 55)
    print("  Validation results")
    print("=" * 55)
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall   : {metrics.box.mr:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=config.NAV_MODEL_PATH)
    parser.add_argument("--img",    type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.model, args.img, args.device)
