"""
Train the MSTAR SAR target classifier (PyTorch CNN).

Expects:
  <MSTAR_ROOT>/train/<CLASS_NAME>/**/*.{phoenix binary or png}
  <MSTAR_ROOT>/test/<CLASS_NAME>/...

Example:
  python scripts/train_mstar_cnn.py --mstar-root archive/MSTAR --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.mstar_phoenix import collect_classes, iter_mstar_dataset, load_mstar_chip
from models.mstar_cnn import MSTAR_CNN
from signal_processing.sar_pipeline import build_two_channel_tensor, preprocess_magnitude


class MSTARTorchDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        class_to_idx: Dict[str, int],
        target_hw: Tuple[int, int],
        fft_branch: bool,
        gpu_fft: bool,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.target_hw = target_hw
        self.fft_branch = fft_branch
        self.gpu_fft = gpu_fft

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, yname = self.samples[idx]
        chip = load_mstar_chip(path)
        mag = chip.magnitude
        if self.fft_branch:
            t = build_two_channel_tensor(
                mag, target_hw=self.target_hw, use_gpu_fft=self.gpu_fft
            )
        else:
            t = preprocess_magnitude(mag, target_hw=self.target_hw)[np.newaxis, ...]
        x = torch.from_numpy(t.astype(np.float32))
        y = self.class_to_idx[yname]
        return x, y


def gather_split(root: str, split: str) -> List[Tuple[str, str]]:
    return list(iter_mstar_dataset(root, split))


def parse_args():
    p = argparse.ArgumentParser(description="Train MSTAR CNN (SAR ATR)")
    p.add_argument(
        "--mstar-root",
        default=os.environ.get("MSTAR_ROOT", "archive/MSTAR"),
        help="Root with train/ and test/ subfolders",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-size", type=int, default=88)
    p.add_argument("--fft-branch", action="store_true", help="2-ch: spatial + FFT log")
    p.add_argument("--gpu-fft", action="store_true", help="CuPy FFT for FFT branch")
    p.add_argument("--out", default="checkpoints/mstar_cnn.pt")
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.mstar_root)
    train_items = gather_split(root, "train")
    test_items = gather_split(root, "test")
    if not train_items:
        print(f"No training chips under {root}/train. Place MSTAR classes there.")
        sys.exit(1)
    if not test_items:
        print("No test/ split — holding out 15% of train.")
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(train_items))
        n_val = max(1, len(train_items) // 7)
        test_items = [train_items[i] for i in idx[:n_val]]
        train_items = [train_items[i] for i in idx[n_val:]]

    classes = sorted(
        set(collect_classes(root, "train")) | set(collect_classes(root, "test"))
    )
    if not classes:
        classes = sorted({y for _, y in train_items})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    meta = {"classes": classes, "class_to_idx": class_to_idx}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    th = tw = args.target_size
    train_ds = MSTARTorchDataset(
        train_items, class_to_idx, (th, tw), args.fft_branch, args.gpu_fft
    )
    test_ds = MSTARTorchDataset(
        test_items, class_to_idx, (th, tw), args.fft_branch, args.gpu_fft
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_ch = 2 if args.fft_branch else 1
    model = MSTAR_CNN(num_classes=len(classes), in_channels=in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        tr_acc = correct / max(1, total)

        model.eval()
        te_correct, te_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                te_correct += (pred == y).sum().item()
                te_total += y.numel()
        te_acc = te_correct / max(1, te_total)
        print(
            f"epoch {epoch+1:03d}/{args.epochs}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}"
        )

    torch.save(
        {"model": model.state_dict(), "meta": meta, "in_channels": in_ch},
        args.out,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
