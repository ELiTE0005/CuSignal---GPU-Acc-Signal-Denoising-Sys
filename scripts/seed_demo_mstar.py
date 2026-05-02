"""Seed Phoenix chips under archive/MSTAR (visible in Docker as /data/MSTAR)."""
import argparse
import os
import struct

import numpy as np

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "archive", "MSTAR")


def write_phoenix(path: str, mag_u16: np.ndarray) -> None:
    rows, cols = mag_u16.shape
    header = (
        f"PhoenixHeaderVer = 1.0\nNumberOfRows = {rows}\nNumberOfColumns = {cols}\n"
        "EndofPhoenixHeader\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for v in mag_u16.ravel():
            f.write(struct.pack(">H", int(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_DEFAULT, help="MSTAR root (train/ + test/)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    for split, n_per in (("train", 8), ("test", 3)):
        for ci, cls in enumerate(("BMP2", "T72")):
            rng = np.random.default_rng(hash((split, cls)) % (2**32))
            for i in range(n_per):
                mag = (rng.random((32, 32)) * 8000 + 100 * (ci + 1)).astype(np.uint16)
                sub = os.path.join(root, split, cls, f"sn_{i}")
                os.makedirs(sub, exist_ok=True)
                write_phoenix(os.path.join(sub, f"chip_{i:03d}.001"), mag)
    print("Wrote chips under", root)


if __name__ == "__main__":
    main()
