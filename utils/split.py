#!/usr/bin/env python
"""
Split the dataset into train and validation sets.

Reads paired images from dataset/input/ and dataset/target/,
shuffles them with a fixed seed, then writes four .txt files
(one path per line) to the splits directory.

Run this ONCE before training:
    python -m utils.split

The paths are controlled by configs/config.yaml - edit it to change:
- dataset_root: where input/ and target/ are located
- splits_dir: where to save the split files
- train_ratio: fraction for training
- seed: random seed for reproducibility

Usage:
    python -m utils.split                           # use config defaults
    python -m utils.split --dataset /path/to/data   # override dataset
    python -m utils.split --ratio 0.9               # change train ratio
    python -m utils.split --seed 123                # change random seed
"""
import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_data_path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def main():
    p = argparse.ArgumentParser(description="Dataset splitter")
    p.add_argument(
        "--dataset",
        default=None,
        help="Dataset root (overrides config: data.dataset_root)"
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Training fraction (overrides config: data.train_ratio)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config: data.seed)"
    )
    args = p.parse_args()

    config = get_config()

    dataset_root = Path(args.dataset) if args.dataset else get_data_path("dataset_root")
    input_dir = dataset_root / config.get("data.input_subdir", "input")
    target_dir = dataset_root / config.get("data.target_subdir", "target")
    out_dir = get_data_path("splits_dir")

    train_ratio = args.ratio if args.ratio is not None else config.get("data.train_ratio", 0.85)
    seed = args.seed if args.seed is not None else config.get("data.seed", 42)

    out_dir.mkdir(parents=True, exist_ok=True)

    inp_names = {f.name for f in input_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    tgt_names = {f.name for f in target_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    common = sorted(inp_names & tgt_names)

    if not common:
        print(f"ERROR: No matching files found.")
        print(f"  input/  : {len(inp_names)} images in {input_dir}")
        print(f"  target/ : {len(tgt_names)} images in {target_dir}")
        print(f"  Both dirs must contain files with identical names.")
        return

    random.seed(seed)
    random.shuffle(common)
    n_train = int(len(common) * train_ratio)
    train = common[:n_train]
    val = common[n_train:]

    def _write(names, sub, filename):
        rel_root = dataset_root.relative_to(PROJECT_ROOT) if dataset_root.is_absolute() else dataset_root
        with open(out_dir / filename, "w") as f:
            for name in names:
                f.write(str((rel_root / sub / name).as_posix()) + "\n")

    _write(train, config.get("data.input_subdir", "input"), "train_input.txt")
    _write(train, config.get("data.target_subdir", "target"), "train_target.txt")
    _write(val, config.get("data.input_subdir", "input"), "val_input.txt")
    _write(val, config.get("data.target_subdir", "target"), "val_target.txt")

    print(f"Dataset root : {dataset_root}")
    rel_root = dataset_root.relative_to(PROJECT_ROOT) if dataset_root.is_absolute() else dataset_root
    print(f"Total        : {len(common)} matched pairs")
    print(f"Train        : {len(train)} ({train_ratio*100:.0f}%)")
    print(f"Val          : {len(val)} ({(1-train_ratio)*100:.0f}%)")
    print(f"Splits saved : {out_dir}/")
    print()
    print("Next: python -m training.train")


if __name__ == "__main__":
    main()