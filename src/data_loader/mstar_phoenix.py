"""
MSTAR public-release chips in Phoenix (ASCII header + binary magnitude/phase).

Typical layout on disk (you organize under archive/MSTAR):

  train/BMP2/sn_9563/*.001
  train/T72/sn_132/...
  test/...

Files may also be converted to PNG/TIF; those load via Pillow when available.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


@dataclass
class MSTARChip:
    """One SAR chip: magnitude (required), phase optional (radians or raw uint)."""

    magnitude: np.ndarray  # float32, shape (H, W), linear amplitude
    phase: Optional[np.ndarray] = None  # float32 or None
    header: Optional[Dict[str, str]] = None
    path: str = ""


def _parse_phoenix_header(f) -> Tuple[Dict[str, str], int]:
    """
    Read lines until EndofPhoenixHeader; return header dict and file offset after header.
    File must be opened 'rb'. Leaves f positioned at start of binary payload.
    """
    header: Dict[str, str] = {}
    while True:
        line = f.readline()
        if not line:
            break
        try:
            s = line.decode("ascii", errors="ignore").strip()
        except Exception:
            continue
        if "PhoenixHeaderVer" in s:
            continue
        if "EndofPhoenixHeader" in s:
            break
        m = re.match(r"^\s*([^=\s]+)\s*[=:]\s*(.+?)\s*$", s)
        if m:
            header[m.group(1).strip()] = m.group(2).strip()
    return header, f.tell()


def _get_hw(header: Dict[str, str]) -> Tuple[int, int]:
    def _int(keys):
        for k in keys:
            if k in header:
                return int(float(header[k]))
        return 0

    rows = _int(("NumberOfRows", "nrows", "NumRows", "rows"))
    cols = _int(("NumberOfColumns", "ncols", "NumCols", "cols"))
    return rows, cols


def _decode_binary(
    blob: bytes, rows: int, cols: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (magnitude float32, phase uint16-as-float or None)."""
    n = rows * cols
    if rows <= 0 or cols <= 0 or len(blob) < 2:
        raise ValueError("Invalid dimensions or empty binary payload")
    L = len(blob)

    if L == n * 2:
        mag = np.frombuffer(blob, dtype=">u2").reshape(rows, cols).astype(np.float32)
        return mag, None

    # Interleaved uint16 magnitude / phase (common MSTAR layout)
    if L == n * 4:
        u = np.frombuffer(blob, dtype=">u2")
        mag = u[0::2].reshape(rows, cols).astype(np.float32)
        ph = u[1::2].reshape(rows, cols).astype(np.float32)
        return mag, ph

    if L == n * 2:
        mag = np.frombuffer(blob, dtype="<u2").reshape(rows, cols).astype(np.float32)
        return mag, None

    raise ValueError(
        f"Cannot map {L} bytes to {rows}x{cols} (try PNG/TIF conversion)."
    )


def read_phoenix_chip(path: str) -> MSTARChip:
    with open(path, "rb") as f:
        header, _ = _parse_phoenix_header(f)
        blob = f.read()
    rows, cols = _get_hw(header)
    if rows == 0 or cols == 0:
        raise ValueError(f"Missing NumberOfRows/Columns in header: {path}")
    mag, ph = _decode_binary(blob, rows, cols)
    return MSTARChip(magnitude=mag, phase=ph, header=header, path=path)


def read_image_chip(path: str) -> MSTARChip:
    if not _HAS_PIL:
        raise ImportError("Install pillow to load PNG/TIF MSTAR chips: pip install pillow")
    img = Image.open(path).convert("L")
    mag = np.asarray(img, dtype=np.float32)
    return MSTARChip(magnitude=mag, phase=None, header=None, path=path)


def load_mstar_chip(path: str) -> MSTARChip:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        return read_image_chip(path)
    return read_phoenix_chip(path)


def iter_mstar_dataset(
    root: str,
    split: str = "train",
) -> Iterator[Tuple[str, str]]:
    """Yield (path, class_name) walking root/split/<CLASS>/..."""
    base = os.path.join(root, split)
    if not os.path.isdir(base):
        return
    skip_ext = {".txt", ".json", ".md", ".csv"}
    for class_name in sorted(os.listdir(base)):
        cdir = os.path.join(base, class_name)
        if not os.path.isdir(cdir):
            continue
        for dirpath, _dirnames, filenames in os.walk(cdir):
            for fn in filenames:
                if fn.startswith("."):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in skip_ext:
                    continue
                yield os.path.join(dirpath, fn), class_name


def collect_classes(root: str, split: str = "train") -> List[str]:
    base = os.path.join(root, split)
    if not os.path.isdir(base):
        return []
    return sorted(
        d
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
    )
