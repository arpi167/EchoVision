"""
╔══════════════════════════════════════════════════════════════════╗
║           EchoVision  –  training/export_tflite.py               ║
║  Exports the trained navigation-hazard model to TensorFlow Lite,  ║
║  ready to drop into the separate Android Studio (Phase-2) app's   ║
║  app/src/main/assets/ folder.                                      ║
║                                                                    ║
║  Requires the extra 'tensorflow' export dependency — see the      ║
║  commented-out block in requirements.txt.                          ║
║                                                                    ║
║  Usage:                                                            ║
║      python training/export_tflite.py                              ║
║      python training/export_tflite.py --int8      # smaller/faster,║
║                                                     # slight accuracy drop
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src import config  # noqa: E402


def export(model_path: str, imgsz: int = 640, int8: bool = False):
    from ultralytics import YOLO

    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Train first: python training/train.py")
        return

    model = YOLO(model_path)
    print(f"[INFO] Exporting {model_path} → TFLite (imgsz={imgsz}, int8={int8}) …")
    exported_path = model.export(format="tflite", imgsz=imgsz, int8=int8)

    dst = Path(config.TFLITE_MODEL_PATH)
    shutil.copy2(exported_path, dst)
    print(f"[OK] TFLite model saved → {dst}")
    print("  Copy this file into EchoVisionApp/app/src/main/assets/ for the Android build.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.NAV_MODEL_PATH)
    parser.add_argument("--img",   type=int, default=640)
    parser.add_argument("--int8",  action="store_true",
                        help="Quantize to int8 (smaller & faster, small accuracy cost)")
    args = parser.parse_args()
    export(args.model, args.img, args.int8)
