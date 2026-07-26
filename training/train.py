"""
╔══════════════════════════════════════════════════════════════════╗
║               EchoVision  –  training/train.py                   ║
║  Fine-tunes YOLOv8n on the merged navigation-hazard dataset        ║
║  (9 classes — stairs, potholes, speed breakers, open drains,      ║
║  road barriers, construction blocks, traffic cones, curbs, low    ║
║  hanging obstacles). Uses the Ultralytics training API directly   ║
║  — no repo cloning required.                                       ║
║                                                                    ║
║  Prerequisites:                                                    ║
║    1. python training/prepare_dataset.py                           ║
║    2. python training/train.py --epochs 100                        ║
║    3. python training/evaluate.py                                  ║
║    4. python main.py   (auto-loads models/navigation_best.pt)      ║
║                                                                    ║
║  Usage:                                                            ║
║      python training/train.py                                     ║
║      python training/train.py --epochs 150 --batch 8               ║
║      python training/train.py --epochs 150 --device 0   # GPU      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src import config  # noqa: E402

DATASET_CFG = BASE_DIR / "training" / "dataset.yaml"
RUN_DIR     = BASE_DIR / "outputs" / "runs"


def verify_dataset():
    prepared = BASE_DIR / "datasets" / "navigation_dataset" / "prepared"
    for split in ["train", "val"]:
        img_dir = prepared / "images" / split
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
        print(f"  [{split}] images={len(imgs)}")
        if not imgs:
            print(f"\n[ERROR] No images in {img_dir}")
            print("  Run first: python training/prepare_dataset.py")
            sys.exit(1)


def train(epochs=100, batch=16, imgsz=640, device="cpu", workers=2):
    from ultralytics import YOLO

    print("─" * 60)
    print("  Verifying dataset …")
    print("─" * 60)
    verify_dataset()

    base_weights = config.GENERAL_MODEL_PATH
    if not Path(base_weights).exists():
        print(f"[INFO] {base_weights} not found locally — Ultralytics will "
              f"auto-download '{config.GENERAL_MODEL_HUB_NAME}' as the starting point.")
        base_weights = config.GENERAL_MODEL_HUB_NAME

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(base_weights)

    print("─" * 60)
    print(f"  Training  epochs={epochs}  batch={batch}  imgsz={imgsz}  device={device}")
    print("─" * 60)

    results = model.train(
        data=str(DATASET_CFG),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        patience=20,
        project=str(RUN_DIR),
        name="navigation_train",
        exist_ok=True,
    )

    best_pt = RUN_DIR / "navigation_train" / "weights" / "best.pt"
    if best_pt.exists():
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, config.NAV_MODEL_PATH)
        print("─" * 60)
        print(f"  ✓ Custom navigation-hazard model saved → {config.NAV_MODEL_PATH}")
        print("  [NEXT] python training/evaluate.py")
        print("         python main.py")
        print("─" * 60)
    else:
        print(f"[WARNING] Could not find {best_pt} after training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the EchoVision navigation-hazard model")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (100+ recommended for 9-class model)")
    parser.add_argument("--batch",  type=int, default=16, help="Batch size")
    parser.add_argument("--img",    type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="'cpu' or GPU index e.g. '0'")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    args = parser.parse_args()
    train(args.epochs, args.batch, args.img, args.device, args.workers)
