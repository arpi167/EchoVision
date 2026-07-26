"""
╔══════════════════════════════════════════════════════════════════╗
║          EchoVision  –  training/prepare_dataset.py              ║
║                                                                    ║
║  One-command dataset preparation for Roboflow YOLOv11 exports.    ║
║                                                                    ║
║  This script assumes EVERY class dataset under                    ║
║  datasets/navigation_dataset/<class>/ is a standard Roboflow       ║
║  YOLOv8/YOLOv11 export (the two formats use an identical layout   ║
║  and label format — no conversion needed), i.e.:                   ║
║                                                                    ║
║      <class>/train/images/*.jpg   <class>/train/labels/*.txt       ║
║      <class>/valid/images/*.jpg   <class>/valid/labels/*.txt       ║
║      <class>/test/images/*.jpg    <class>/test/labels/*.txt  (opt.)║
║      <class>/data.yaml                                (ignored —   ║
║        each export was labelled independently with its own local   ║
║        single-class index 0; src/config.py::NAV_CLASSES is the     ║
║        single source of truth for the GLOBAL class index instead)  ║
║                                                                    ║
║  What this script does, end to end:                                ║
║    1. Reads config.py::NAV_CLASSES — ONLY these classes are         ║
║       processed. Any other folder under navigation_dataset/         ║
║       (e.g. curbs, traffic_cones, speed_breakers, construction_     ║
║       blocks, low_hanging_obstacle, fan) is ignored unless it's      ║
║       actually listed in NAV_CLASSES.                               ║
║    2. Wipes and recreates datasets/navigation_dataset/prepared/     ║
║       so re-running this script is always safe/idempotent — no      ║
║       stale files from a previous run linger.                       ║
║    3. For each active class, auto-detects which of train/valid/     ║
║       test exist and processes only those that do.                  ║
║    4. PRESERVES Roboflow's own train/valid split (does not          ║
║       reshuffle) — train/ → prepared train, valid/ → prepared val.  ║
║       test/, if present, is optionally merged into val too          ║
║       (--merge-test / --no-merge-test, default: merge).             ║
║    5. Remaps every label's local class index 0 → the correct        ║
║       GLOBAL index (the class's position in NAV_CLASSES).           ║
║    6. Verifies every image has a matching, readable label file,     ║
║       and that the image itself is readable — corrupted or          ║
║       unlabeled samples are skipped with a warning, never crash     ║
║       the run.                                                      ║
║    7. Prefixes every copied filename with the class name to         ║
║       prevent collisions between classes (e.g. two datasets that    ║
║       both happen to have an "IMG_0001.jpg").                       ║
║    8. Prints a detailed per-class + total summary at the end.       ║
║                                                                    ║
║  After this script finishes, training/train.py can be run           ║
║  immediately — no manual file organisation needed.                  ║
║                                                                    ║
║  WHERE TO GET THE IMAGES (open / free — no API key, no cost, but   ║
║  each must be downloaded manually due to per-site licensing):      ║
║    • stairs, open_drain ("manhole"), road_barrier ("barrier"),     ║
║      door → Roboflow Universe, https://universe.roboflow.com       ║
║      (search the term, filter to Object Detection, Export →         ║
║      "YOLOv8" or "YOLOv11" — identical format here)                 ║
║    • pothole → Kaggle "pothole-detection-dataset", or Roboflow      ║
║                                                                    ║
║  IMPORTANT — each extracted folder's name must exactly match the    ║
║  class name in src/config.py::NAV_CLASSES. A "manhole" export       ║
║  goes in a folder named open_drain/, a "barrier" export goes in     ║
║  a folder named road_barrier/ — rename on extraction if needed.     ║
║                                                                    ║
║  Usage:                                                            ║
║      python training/prepare_dataset.py                            ║
║      python training/prepare_dataset.py --no-merge-test            ║
║      python training/prepare_dataset.py --keep-existing            ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src import config  # noqa: E402

NAV_DIR      = BASE_DIR / "datasets" / "navigation_dataset"
PREPARED_DIR = NAV_DIR / "prepared"
IMG_TRAIN    = PREPARED_DIR / "images" / "train"
IMG_VAL      = PREPARED_DIR / "images" / "val"
LBL_TRAIN    = PREPARED_DIR / "labels" / "train"
LBL_VAL      = PREPARED_DIR / "labels" / "val"

IMG_EXTS = (".jpg", ".jpeg", ".png")
VALID_SPLIT_DIR_NAMES = ("valid", "val")   # Roboflow uses "valid"; accept "val" too


# ─────────────────────────────────────────────────────────────────
#  STEP 0 — reset the output directory so every run is idempotent
# ─────────────────────────────────────────────────────────────────
def reset_prepared_dir():
    if PREPARED_DIR.exists():
        shutil.rmtree(PREPARED_DIR)
    for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  STEP 1 — find each class's train/valid/test split directories
# ─────────────────────────────────────────────────────────────────
def find_class_splits(class_dir: Path) -> dict:
    """
    Returns e.g. {"train": (images_dir, labels_dir), "val": (...), "test": (...)}
    for whichever splits actually exist under class_dir. Any split missing
    both an images/ and labels/ subfolder is simply omitted (not an error —
    some exports don't include a test/ split, for example).
    """
    splits = {}

    train_dir = class_dir / "train"
    if (train_dir / "images").is_dir() and (train_dir / "labels").is_dir():
        splits["train"] = (train_dir / "images", train_dir / "labels")

    for name in VALID_SPLIT_DIR_NAMES:
        val_dir = class_dir / name
        if (val_dir / "images").is_dir() and (val_dir / "labels").is_dir():
            splits["val"] = (val_dir / "images", val_dir / "labels")
            break

    test_dir = class_dir / "test"
    if (test_dir / "images").is_dir() and (test_dir / "labels").is_dir():
        splits["test"] = (test_dir / "images", test_dir / "labels")

    return splits


# ─────────────────────────────────────────────────────────────────
#  STEP 2 — remap one label file's class index, validate its content
# ─────────────────────────────────────────────────────────────────
def remap_label_lines(label_path: Path, global_class_idx: int) -> Optional[list]:
    """
    Reads a Roboflow single-class label file (class index always 0) and
    rewrites every line's class to global_class_idx. Returns None (skip
    this sample) if the file is empty, unreadable, or malformed.
    """
    try:
        raw_lines = label_path.read_text().splitlines()
    except Exception:
        return None

    out_lines = []
    for line in raw_lines:
        parts = line.split()
        if not parts:
            continue
        try:
            # sanity-check the numeric fields are actually parseable floats
            _ = [float(p) for p in parts[1:]]
        except ValueError:
            continue   # malformed line — skip just this line, not the file
        parts[0] = str(global_class_idx)
        out_lines.append(" ".join(parts))

    return out_lines if out_lines else None


def _is_readable_image(img_path: Path) -> bool:
    img = cv2.imread(str(img_path))
    return img is not None


# ─────────────────────────────────────────────────────────────────
#  STEP 3 — copy one split (train/val/test) for one class
# ─────────────────────────────────────────────────────────────────
def copy_split(images_dir: Path, labels_dir: Path, class_name: str,
                global_class_idx: int, dest_images_dir: Path,
                dest_labels_dir: Path) -> tuple:
    """Returns (copied_count, skipped_count)."""
    copied = 0
    skipped = 0

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    for img_path in images:
        label_path = labels_dir / (img_path.stem + ".txt")

        if not label_path.exists():
            skipped += 1
            continue

        if not _is_readable_image(img_path):
            skipped += 1
            continue

        remapped = remap_label_lines(label_path, global_class_idx)
        if remapped is None:
            skipped += 1
            continue

        # Prefix with class name to prevent filename collisions across classes
        dst_img = dest_images_dir / f"{class_name}_{img_path.name}"
        dst_lbl = dest_labels_dir / f"{class_name}_{img_path.stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(remapped) + "\n")
        copied += 1

    return copied, skipped


# ─────────────────────────────────────────────────────────────────
#  MAIN — orchestrate all classes, then print a summary
# ─────────────────────────────────────────────────────────────────
def prepare(merge_test: bool = True, keep_existing: bool = False):
    if keep_existing:
        for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
            d.mkdir(parents=True, exist_ok=True)
        print("[INFO] --keep-existing set: not clearing prepared/ before this run.")
    else:
        reset_prepared_dir()
        print("[INFO] Cleared and recreated datasets/navigation_dataset/prepared/")

    print(f"[INFO] Active classes (from src/config.py::NAV_CLASSES): "
          f"{', '.join(config.NAV_CLASSES)}\n")

    summary = []   # (class_name, n_train, n_val, n_skipped)
    total_train = total_val = total_skipped = 0

    for class_idx, class_name in enumerate(config.NAV_CLASSES):
        class_dir = NAV_DIR / class_name

        if not class_dir.exists():
            print(f"  [skip] {class_name:<16} folder not found: {class_dir}")
            summary.append((class_name, 0, 0, 0))
            continue

        splits = find_class_splits(class_dir)
        if not splits:
            print(f"  [skip] {class_name:<16} no train/valid/test split found "
                  f"under {class_dir} (expected Roboflow layout: "
                  f"train/images+labels, valid/images+labels)")
            summary.append((class_name, 0, 0, 0))
            continue

        n_train = n_val = n_skipped = 0

        if "train" in splits:
            imgs_dir, lbls_dir = splits["train"]
            c, s = copy_split(imgs_dir, lbls_dir, class_name, class_idx,
                               IMG_TRAIN, LBL_TRAIN)
            n_train += c
            n_skipped += s

        if "val" in splits:
            imgs_dir, lbls_dir = splits["val"]
            c, s = copy_split(imgs_dir, lbls_dir, class_name, class_idx,
                               IMG_VAL, LBL_VAL)
            n_val += c
            n_skipped += s

        if "test" in splits and merge_test:
            imgs_dir, lbls_dir = splits["test"]
            c, s = copy_split(imgs_dir, lbls_dir, class_name, class_idx,
                               IMG_VAL, LBL_VAL)
            n_val += c
            n_skipped += s

        found_desc = "+".join(splits.keys())
        print(f"  [ok]   {class_name:<16} train={n_train:<5} val={n_val:<5} "
              f"skipped={n_skipped:<4} (splits found: {found_desc})")

        summary.append((class_name, n_train, n_val, n_skipped))
        total_train += n_train
        total_val += n_val
        total_skipped += n_skipped

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  EchoVision dataset preparation — summary")
    print("=" * 64)
    print(f"  {'Class':<18}{'Train':>8}{'Val':>8}{'Skipped':>10}")
    print("  " + "-" * 44)
    for class_name, n_train, n_val, n_skipped in summary:
        print(f"  {class_name:<18}{n_train:>8}{n_val:>8}{n_skipped:>10}")
    print("  " + "-" * 44)
    print(f"  {'TOTAL':<18}{total_train:>8}{total_val:>8}{total_skipped:>10}")
    print("=" * 64)

    if total_train + total_val == 0:
        print("\n[ACTION NEEDED] No usable images were found for any class.")
        print("  Extract each Roboflow export into datasets/navigation_dataset/<class>/")
        print("  so that <class>/train/images, <class>/valid/images (etc.) exist.")
        print("  See the docstring at the top of this file for details + dataset links.")
    else:
        print(f"\n[NEXT] python training/train.py --epochs 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge per-class Roboflow YOLOv8/YOLOv11 exports into one training-ready dataset"
    )
    parser.add_argument(
        "--no-merge-test", dest="merge_test", action="store_false",
        help="Do NOT fold each class's test/ split into the validation set "
             "(default: test IS merged into val)"
    )
    parser.add_argument(
        "--keep-existing", action="store_true",
        help="Do not wipe datasets/navigation_dataset/prepared/ before this run "
             "(default: prepared/ is cleared for a clean, idempotent rebuild)"
    )
    parser.set_defaults(merge_test=True)
    args = parser.parse_args()
    prepare(merge_test=args.merge_test, keep_existing=args.keep_existing)