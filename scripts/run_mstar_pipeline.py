"""
MSTAR SAR pipeline: load Phoenix/image chip → preprocess → optional FFT view → viz.

Does not require PyTorch. For training use scripts/train_mstar_cnn.py.

Example:
  python scripts/run_mstar_pipeline.py --chip path/to/file.001 --out mstar_view.png
  python scripts/run_mstar_pipeline.py --summarize --mstar-root archive/MSTAR
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.mstar_phoenix import iter_mstar_dataset, load_mstar_chip
from signal_processing.sar_pipeline import (
    build_two_channel_tensor,
    fft_log_magnitude,
    preprocess_magnitude,
)


def parse_args():
    p = argparse.ArgumentParser(description="MSTAR SAR chip visualization / summary")
    p.add_argument(
        "--mstar-root",
        default=os.environ.get("MSTAR_ROOT", "archive/MSTAR"),
    )
    p.add_argument("--chip", default="", help="Single Phoenix or image chip path")
    p.add_argument("--out", default="mstar_pipeline_view.png")
    p.add_argument("--target-size", type=int, default=88)
    p.add_argument("--fft", action="store_true", help="Include FFT log-magnitude panel")
    p.add_argument(
        "--summarize",
        action="store_true",
        help="Count files per class under train/ and test/",
    )
    return p.parse_args()


def summarize(root: str):
    for split in ("train", "test"):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            print(f"Missing {base}")
            continue
        print(f"=== {split} ===")
        for c in sorted(os.listdir(base)):
            d = os.path.join(base, c)
            if not os.path.isdir(d):
                continue
            n = sum(len(files) for _, _, files in os.walk(d))
            print(f"  {c}: {n} files")


def main():
    args = parse_args()
    if args.summarize:
        summarize(os.path.abspath(args.mstar_root))
        return

    path = args.chip
    if not path:
        # pick one random training chip
        items = list(iter_mstar_dataset(os.path.abspath(args.mstar_root), "train"))
        if not items:
            print("No chips found; set --chip or add data under archive/MSTAR/train/")
            sys.exit(1)
        path = random.choice(items)[0]
        print(f"Using random chip: {path}")

    chip = load_mstar_chip(path)
    th = tw = args.target_size
    spat = preprocess_magnitude(chip.magnitude, target_hw=(th, tw))

    ncols = 2 if args.fft else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    axes[0].imshow(spat, cmap="gray")
    axes[0].set_title("log |A| (normalized)")
    axes[0].axis("off")
    if args.fft:
        spec = fft_log_magnitude(spat, use_gpu=False)
        axes[1].imshow(spec, cmap="viridis")
        axes[1].set_title("log |FFT2| (normalized)")
        axes[1].axis("off")
    plt.suptitle(os.path.basename(path))
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")

    if args.fft:
        _ = build_two_channel_tensor(chip.magnitude, target_hw=(th, tw), use_gpu_fft=False)
        print("Two-channel tensor shape (for CNN):", _.shape)


if __name__ == "__main__":
    main()
