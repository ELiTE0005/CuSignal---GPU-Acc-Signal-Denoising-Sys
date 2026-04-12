"""
GPU sonar beamforming on real audio (WAV).

RadarScenes does not include sonar; use a multi-channel or mono WAV as the
sensor input. Mono is split across a synthetic ULA with a phase ramp (same
idea as the former run_sonar_wav.py).

Optional toy simulator (no WAV): scripts/run_sonar_synthetic_demo.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import cupy as cp
import matplotlib.pyplot as plt
from scipy.io import wavfile

from signal_processing.sonar_pipeline import SonarPipeline


def load_mono_normalized(path: str, max_samples: int) -> np.ndarray:
    _rate, data = wavfile.read(path)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak
    n = min(int(max_samples), x.shape[0])
    return x[:n]


def mono_to_array_channels(mono: np.ndarray, num_elements: int) -> cp.ndarray:
    base = cp.asarray(mono, dtype=cp.complex64)
    ph = cp.linspace(0, 0.25 * cp.pi, num_elements, dtype=cp.float32)[:, None]
    w = cp.exp(1j * ph)
    return base[None, :] * w


def parse_args():
    p = argparse.ArgumentParser(description="Sonar spatial FFT beamforming from WAV")
    p.add_argument(
        "--wav",
        required=True,
        help="Path to .wav (mono or multi-channel; multi is averaged to mono)",
    )
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--elements", type=int, default=32)
    p.add_argument("--output", default="sonar_output.png")
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--fc", type=float, default=100e3, help="Carrier Hz (metadata for pipeline)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.wav):
        print(f"File not found: {args.wav}")
        sys.exit(1)

    mono = load_mono_normalized(args.wav, args.max_samples)
    if mono.size < 8:
        print("WAV too short after trim.")
        sys.exit(1)

    print(f"Loaded {mono.size} samples → {args.elements}-channel array (GPU)...")
    array_data = mono_to_array_channels(mono, args.elements)

    pipeline = SonarPipeline(num_elements=args.elements, fc=args.fc)
    power_map = pipeline.process_spatial_fft(array_data)
    det, _ = pipeline.extract_targets(power_map, threshold=args.threshold)
    print(f"Detection cells (median × threshold): {int(cp.sum(det))}")

    db_map = 10 * cp.asnumpy(cp.log10(power_map.T + 1e-12))
    plt.figure(figsize=(8, 6))
    plt.imshow(db_map, aspect="auto", cmap="viridis")
    plt.title("Sonar beamformed map from WAV (dB)")
    plt.xlabel("Spatial frequency (angle bins)")
    plt.ylabel("Time / range bins")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
