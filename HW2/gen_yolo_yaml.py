#!/usr/bin/env python3
"""
generate_yolo_yaml.py

Generate a YOLO classification .yaml file that points to your dataset:

lnu-butterflies/
  train/
    class_1/
    class_2/
    ...
  val/
    class_1/
    class_2/
    ...

It scans `train/` to get the list of classes and writes a YAML like:

path: lnu-butterflies
train: train
val: val
names:
  0: class_1
  1: class_2
  ...
"""

import argparse
from pathlib import Path
import yaml  # pip install pyyaml if you don't have it


def generate_yaml(
    data_root: Path,
    yaml_path: Path,
    train_subdir: str = "train",
    val_subdir: str = "val",
):
    train_dir = data_root / train_subdir
    if not train_dir.is_dir():
        raise SystemExit(f"[ERROR] Train directory not found: {train_dir}")

    # Class names = subdirectory names under train/
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"[ERROR] No class subdirectories found in: {train_dir}")

    # Sort for deterministic mapping
    class_names = sorted(d.name for d in class_dirs)

    data = {
        "path": str(data_root),     # root
        "train": train_subdir,      # relative to path
        "val": val_subdir,          # relative to path
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] YAML written to: {yaml_path}")
    print("[INFO] Class mapping:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")


def main():
    parser = argparse.ArgumentParser(description="Generate YOLO classification .yaml config.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="lnu-butterflies",
        help="Root directory of dataset (contains train/ and val/).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="butterflies_cls.yaml",
        help="Output YAML file name (default: butterflies_cls.yaml)",
    )
    parser.add_argument(
        "--train_subdir",
        type=str,
        default="train",
        help="Train subdir relative to data_root (default: train)",
    )
    parser.add_argument(
        "--val_subdir",
        type=str,
        default="val",
        help="Val subdir relative to data_root (default: val)",
    )

    args = parser.parse_args()

    generate_yaml(
        data_root=Path(args.data_root),
        yaml_path=Path(args.out),
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
    )


if __name__ == "__main__":
    main()