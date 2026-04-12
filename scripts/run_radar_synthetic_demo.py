"""
Offline demo only: two synthetic targets, no RadarScenes.
For real data use: python scripts/run_radar_pipeline.py (RADARSCENES_ROOT set).
"""
import os
import sys

from numba import cuda

cuda.current_context()

import cupy as cp
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from signal_processing.radar_pipeline import RadarPipeline


def main():
    print("Synthetic demo only (not RadarScenes).")
    adc_gen = SyntheticADCGenerator(num_chirps=128, pulse_duration=10e-6, pulse_bw=1e9)
    pipeline = RadarPipeline(num_chirps=128, num_samples=adc_gen.num_adc_samples)

    points = {
        "range": cp.array([50.0, 120.0]),
        "velocity": cp.array([15.0, -10.0]),
        "rcs": cp.array([10.0, 5.0]),
    }
    adc_data = adc_gen.generate_adc_data(points)
    range_doppler = pipeline.process_range_doppler(adc_data)
    detections, power_map, _ = pipeline.ca_cfar_2d(
        range_doppler, train_cells=(8, 4), guard_cells=(4, 2), pfa=1e-5
    )
    df_targets = pipeline.cluster_detections(detections, power_map, eps=5.0, min_samples=1)
    if len(df_targets) > 0:
        print(df_targets.to_string(index=False))
    det_np = cp.asnumpy(detections)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(10 * cp.asnumpy(cp.log10(power_map + 1e-12)), aspect="auto", cmap="jet")
    plt.title("Range-Doppler (dB) [synthetic demo]")
    plt.subplot(1, 2, 2)
    plt.imshow(det_np, aspect="auto", cmap="gray")
    plt.title("CA-CFAR")
    plt.tight_layout()
    out = "radar_synthetic_demo_output.png"
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
