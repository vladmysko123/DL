#!/usr/bin/env python3
"""
make_yolo_cls_split.py

Create YOLO classification-style train/val folders from an existing dataset.

Assumes you currently have something like:

    DATA_ROOT/
        train/
            class_1/
                img001.jpg
                ...
            class_2/
                ...
            ...

This script will create (or reuse) the following:

    DATA_ROOT/
        train/
            class_1/
            class_2/
            ...
        val/
            class_1/
            class_2/
            ...

and move/copy a percentage of images from train -> val for each class.
"""

import argparse
import random
from pathlib import Path
import shutil


def split_dataset(
    data_root: Path,
    source_subdir: str = "train",
    train_subdir: str = "train",
    val_subdir: str = "val",
    val_ratio: float = 0.1,
    move_files: bool = False,
    seed: int = 42,
):
    random.seed(seed)

    source_dir = data_root / source_subdir
    out_train_dir = data_root / train_subdir
    out_val_dir = data_root / val_subdir

    if not source_dir.is_dir():
        raise SystemExit(f"[ERROR] Source directory does not exist: {source_dir}")

    # If source_subdir == train_subdir, we are splitting in-place:
    # we will keep most images in train/ and just move/copy some into val/.
    in_place = (source_dir.resolve() == out_train_dir.resolve())

    print(f"[INFO] Data root:       {data_root}")
    print(f"[INFO] Source dir:      {source_dir}")
    print(f"[INFO] Train dir out:   {out_train_dir}")
    print(f"[INFO] Val dir out:     {out_val_dir}")
    print(f"[INFO] Val ratio:       {val_ratio}")
    print(f"[INFO] Move files:      {move_files}")
    print(f"[INFO] In-place split:  {in_place}")

    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_val_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over each class folder in the source directory
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"[ERROR] No class subdirectories found in: {source_dir}")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\n[CLASS] {class_name}")

        # Collect all image files
        image_paths = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not image_paths:
            print(f"  [WARN] No images found in {class_dir}, skipping.")
            continue

        random.shuffle(image_paths)
        n_total = len(image_paths)
        n_val = max(1, int(n_total * val_ratio))  # at least 1 image in val if possible
        val_images = set(image_paths[:n_val])

        # Create class subdirs in output train/val
        cls_train_dir = out_train_dir / class_name
        cls_val_dir = out_val_dir / class_name
        cls_train_dir.mkdir(parents=True, exist_ok=True)
        cls_val_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [INFO] Total images: {n_total}, val: {n_val}, train: {n_total - n_val}")

        for img_path in image_paths:
            if img_path in val_images:
                dest = cls_val_dir / img_path.name
            else:
                # If splitting in-place, train images can stay where they are,
                # but we might want to move them under out_train_dir/class_name
                if in_place:
                    # Already in train/class_name, just keep them.
                    # If you want to "normalize" structure you could also copy
                    # them explicitly, but it's not required.
                    dest = img_path  # effectively a no-op
                else:
                    dest = cls_train_dir / img_path.name

            if dest == img_path:
                # No-op (in-place). Just skip.
                continue

            if move_files:
                shutil.move(str(img_path), str(dest))
            else:
                shutil.copy2(str(img_path), str(dest))


def main():
    parser = argparse.ArgumentParser(description="Make YOLO classification train/val split.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="lnu-butterflies",
        help="Root directory of the dataset (default: lnu-butterflies)",
    )
    parser.add_argument(
        "--source_subdir",
        type=str,
        default="train",
        help="Subdirectory under data_root containing class folders (default: train)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of images to put in val for each class (default: 0.1)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )

    args = parser.parse_args()

    split_dataset(
        data_root=Path(args.data_root),
        source_subdir=args.source_subdir,
        train_subdir="train",
        val_subdir="val",
        val_ratio=args.val_ratio,
        move_files=args.move,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()