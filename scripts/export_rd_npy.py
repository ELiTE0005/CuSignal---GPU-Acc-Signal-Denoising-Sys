"""
Export Range–Doppler power maps (dB, numpy) from RadarScenes-driven IF for offline
CNN / sklearn training. Does not require labels beyond what you add later.

Example:
  RADARSCENES_ROOT=/data/RadarScenes/data python scripts/export_rd_npy.py \\
    --sequence 1 --frames 20 --out-dir ./rd_exports
"""
import argparse
import json
import os
import sys

from numba import cuda

cuda.current_context()

import cupy as cp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.radarscenes_loader import (
    RadarScenesLoader,
    frame_to_synthetic_points,
    resolve_radar_data_h5,
)
from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from signal_processing.radar_pipeline import RadarPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--radarscenes-root",
        default=os.environ.get("RADARSCENES_ROOT", ""),
    )
    ap.add_argument("--sequence", type=int, default=1)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--sensor", type=int, default=None)
    ap.add_argument("--max-targets", type=int, default=96)
    ap.add_argument("--out-dir", default="rd_exports")
    args = ap.parse_args()

    root = args.radarscenes_root.strip()
    if not root:
        print("Set --radarscenes-root or RADARSCENES_ROOT")
        sys.exit(1)

    resolve_radar_data_h5(root, args.sequence)
    os.makedirs(args.out_dir, exist_ok=True)

    loader = RadarScenesLoader(root)
    adc_gen = SyntheticADCGenerator(num_chirps=128, pulse_duration=10e-6, pulse_bw=1e9)
    pipeline = RadarPipeline(num_chirps=128, num_samples=adc_gen.num_adc_samples)

    meta = []
    for idx, (ts, frame_rows) in enumerate(
        loader.iter_frames(args.sequence, sensor_id=args.sensor, max_frames=args.frames)
    ):
        pts = frame_to_synthetic_points(
            frame_rows, max_targets=args.max_targets, velocity_field="vr_compensated"
        )
        if pts["range"] is None:
            continue
        cp_pts = {
            "range": cp.asarray(pts["range"], dtype=cp.float32),
            "velocity": cp.asarray(pts["velocity"], dtype=cp.float32),
            "rcs": cp.asarray(pts["rcs"], dtype=cp.float32),
        }
        adc = adc_gen.generate_adc_data(cp_pts, max_targets=args.max_targets)
        rd = pipeline.process_range_doppler(adc)
        power = cp.abs(rd) ** 2
        db = 10.0 * np.log10(cp.asnumpy(power) + 1e-12).astype(np.float32)

        fname = os.path.join(args.out_dir, f"seq{args.sequence}_f{idx:04d}_ts{ts:.0f}.npy")
        np.save(fname, db)

        lbls = pts["meta"]["label_id"]
        meta.append(
            {
                "file": os.path.basename(fname),
                "timestamp": ts,
                "n_points": int(lbls.size),
                "unique_labels": [int(x) for x in np.unique(lbls).tolist()],
            }
        )
        print("Wrote", fname)

    with open(os.path.join(args.out_dir, f"meta_seq{args.sequence}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.", len(meta), "frames")


if __name__ == "__main__":
    main()
