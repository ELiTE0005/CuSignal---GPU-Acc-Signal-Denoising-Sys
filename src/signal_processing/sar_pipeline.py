"""
SAR chip preprocessing aligned with classical MSTAR ATR: log scaling, resize,
optional 2D spectrum branch (FFT) for visualization or multi-channel CNN input.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import cupy as cp

    _HAS_CP = True
except ImportError:
    cp = None
    _HAS_CP = False

try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from scipy.ndimage import zoom


def preprocess_magnitude(
    magnitude: np.ndarray,
    target_hw: Tuple[int, int] = (88, 88),
    eps: float = 1.0,
) -> np.ndarray:
    """
    Log compress + per-chip standardize; returns float32 (H, W).
    """
    x = np.log(np.maximum(magnitude.astype(np.float64), eps))
    x = (x - x.mean()) / (x.std() + 1e-8)
    th, tw = target_hw
    if _HAS_PIL:
        # PIL expects (W,H) for resize
        img = Image.fromarray(x.astype(np.float32))
        img = img.resize((tw, th), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)
    h, w = x.shape
    return zoom(x, (th / h, tw / w), order=1).astype(np.float32)


def fft_log_magnitude(
    magnitude: np.ndarray,
    use_gpu: bool = False,
) -> np.ndarray:
    """
    |FFT2(chip)| in log domain for a second "frequency" view (real-valued FFT).
    """
    if use_gpu and _HAS_CP:
        z = cp.asarray(magnitude, dtype=cp.float32)
        spec = cp.fft.fftshift(cp.fft.fft2(z))
        out = cp.log(cp.abs(spec) + 1e-8)
        out = out - out.mean()
        out = out / (out.std() + 1e-8)
        return cp.asnumpy(out)
    z = np.fft.fftshift(np.fft.fft2(magnitude.astype(np.float64)))
    out = np.log(np.abs(z) + 1e-8)
    out = (out - out.mean()) / (out.std() + 1e-8)
    return out.astype(np.float32)


def build_two_channel_tensor(
    magnitude: np.ndarray,
    target_hw: Tuple[int, int] = (88, 88),
    use_gpu_fft: bool = False,
) -> np.ndarray:
    """Stack [preprocessed spatial, FFT log-mag] as shape (2, H, W)."""
    spatial = preprocess_magnitude(magnitude, target_hw=target_hw)
    fft_ch = fft_log_magnitude(spatial, use_gpu=use_gpu_fft)
    return np.stack([spatial, fft_ch], axis=0)
