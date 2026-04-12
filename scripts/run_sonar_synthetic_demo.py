"""
Offline sonar demo only: simulated targets, no WAV / no real recording.
For real audio use: python scripts/run_sonar_pipeline.py --wav your.wav
"""
import os
import sys

import cupy as cp
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.synthetic_sonar import SyntheticSonarGenerator
from signal_processing.sonar_pipeline import SonarPipeline


def main():
    print("Synthetic sonar demo only (not real recordings).")
    sonar_gen = SyntheticSonarGenerator(num_elements=32, fc=100e3, max_range=150.0)
    pipeline = SonarPipeline(num_elements=32, fc=100e3)

    targets = [
        {"range": 60.0, "angle": 0.2, "snr": 50.0},
        {"range": 110.0, "angle": -0.4, "snr": 30.0},
    ]
    array_data = sonar_gen.generate_array_data(targets)
    print(f"Array data shape: {array_data.shape}")

    power_map = pipeline.process_spatial_fft(array_data)
    det, _ = pipeline.extract_targets(power_map, threshold=10.0)
    print(f"Detection cells: {int(cp.sum(det))}")

    db_map = 10 * cp.asnumpy(cp.log10(power_map.T + 1e-12))
    plt.figure(figsize=(8, 6))
    plt.imshow(db_map, aspect="auto", cmap="viridis")
    plt.title("Sonar range–angle map (dB) [synthetic demo]")
    plt.xlabel("Spatial frequency (angle bins)")
    plt.ylabel("Time (range bins)")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    out = "sonar_synthetic_demo_output.png"
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
