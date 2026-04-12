"""
GPU radar pipeline on RadarScenes (Range–Doppler → CA-CFAR → DBSCAN).

RadarScenes ships detector point clouds in radar_data.h5, not raw IF ADC. This
pipeline converts each frame's detections (range, vr, RCS, …) into a synthetic
IF cube for the same FFT/CFAR stack — driven by real sequence data, not toy targets.

Set RADARSCENES_ROOT (e.g. /data/RadarScenes/data in Docker) or pass --radarscenes-root.

Optional synthetic-only demo (no dataset): scripts/run_radar_synthetic_demo.py
"""
import argparse
import os
import sys

from numba import cuda

cuda.current_context()

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader.radarscenes_loader import (
    RadarScenesLoader,
    frame_positions_for_tracking,
    frame_to_synthetic_points,
    resolve_radar_data_h5,
)
from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from signal_processing.ego_motion import adjust_radial_velocities, nearest_odometry_vx
from signal_processing.radar_pipeline import RadarPipeline
from tracking import MultiObjectTracker2D


def parse_args():
    p = argparse.ArgumentParser(
        description="RadarScenes → Range–Doppler → CA-CFAR → DBSCAN (GPU)"
    )
    p.add_argument(
        "--radarscenes-root",
        default=os.environ.get("RADARSCENES_ROOT", ""),
        help="Folder containing sequence_<N>/radar_data.h5 (or env RADARSCENES_ROOT)",
    )
    p.add_argument("--sequence", type=int, default=1)
    p.add_argument("--frames", type=int, default=6, help="Number of radar frames to process")
    p.add_argument("--sensor", type=int, default=None, help="Optional sensor_id filter")
    p.add_argument(
        "--max-targets",
        type=int,
        default=96,
        help="Max detections per frame (strongest RCS) for IF synthesis",
    )
    p.add_argument(
        "--velocity-field",
        choices=("vr_compensated", "vr"),
        default="vr_compensated",
    )
    p.add_argument("--track", action="store_true", help="Kalman + Hungarian on (x_cc,y_cc)")
    p.add_argument("--output", default="radar_output.png", help="Output PNG path")
    p.add_argument(
        "--frame-index",
        type=int,
        default=-1,
        help="Which processed frame to plot (default: last)",
    )
    p.add_argument(
        "--ego-adjust",
        choices=("none", "subtract", "add"),
        default="none",
        help="Odometry vx·cos(az) on vr (use --velocity-field vr to avoid double correction)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = args.radarscenes_root.strip()
    if not root:
        print(
            "Set RADARSCENES_ROOT or --radarscenes-root to your RadarScenes directory "
            "(e.g. .../data with sequence_1/radar_data.h5).\n"
            "Docker default: /data/RadarScenes/data\n"
            "Optional offline demo (no dataset): python scripts/run_radar_synthetic_demo.py"
        )
        sys.exit(1)

    try:
        h5_path = resolve_radar_data_h5(root, args.sequence)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"RadarScenes sequence file: {h5_path}")

    loader = RadarScenesLoader(root)
    odom = loader.load_odometry(args.sequence)
    if odom.size:
        print(f"Odometry samples: {odom.shape[0]}")

    adc_gen = SyntheticADCGenerator(num_chirps=128, pulse_duration=10e-6, pulse_bw=1e9)
    pipeline = RadarPipeline(num_chirps=128, num_samples=adc_gen.num_adc_samples)
    tracker = MultiObjectTracker2D() if args.track else None

    summary = []
    plot_snapshots = []

    for idx, (ts, frame_rows) in enumerate(
        loader.iter_frames(args.sequence, sensor_id=args.sensor, max_frames=args.frames)
    ):
        pts = frame_to_synthetic_points(
            frame_rows,
            max_targets=args.max_targets,
            velocity_field=args.velocity_field,
        )
        if pts["range"] is None:
            print(f"Frame {idx} ts={ts}: empty, skip")
            continue

        if args.ego_adjust != "none" and odom.size:
            vx = nearest_odometry_vx(odom, ts)
            pts["velocity"] = adjust_radial_velocities(
                pts["velocity"],
                pts["meta"]["azimuth_sc"],
                vx,
                args.ego_adjust,
            )

        cp_pts = {
            "range": cp.asarray(pts["range"], dtype=cp.float32),
            "velocity": cp.asarray(pts["velocity"], dtype=cp.float32),
            "rcs": cp.asarray(pts["rcs"], dtype=cp.float32),
        }
        adc = adc_gen.generate_adc_data(cp_pts, max_targets=args.max_targets)
        rd = pipeline.process_range_doppler(adc)
        det, power_map, _ = pipeline.ca_cfar_2d(
            rd, train_cells=(8, 4), guard_cells=(4, 2), pfa=1e-5
        )
        n_det = int(cp.sum(det))
        labels = pts["meta"]["label_id"]
        uniq_lbl = len(np.unique(labels[labels != 11])) if labels.size else 0
        df = pipeline.cluster_detections(det, power_map, eps=5.0, min_samples=1)
        n_clu = df["cluster_id"].nunique() if len(df) else 0

        summary.append((ts, n_det, n_clu, uniq_lbl))
        plot_snapshots.append((rd, det, power_map))

        print(
            f"Frame {idx} ts={ts:.3f} | CFAR cells={n_det} | "
            f"DBSCAN clusters={n_clu} | non-static label groups≈{uniq_lbl}"
        )

        if tracker is not None:
            xy = frame_positions_for_tracking(frame_rows)
            states = tracker.step(ts, xy)
            print(f"  Active tracks: {len(states)}")

    if not plot_snapshots:
        print("No frames processed — check dataset path and --sensor filter.")
        sys.exit(1)

    fi = args.frame_index
    if fi < 0:
        fi = len(plot_snapshots) + fi
    fi = max(0, min(fi, len(plot_snapshots) - 1))
    _, detections, power_map = plot_snapshots[fi]

    print("\n--- Per-frame (timestamp, CFAR_count, clusters, unique_non_static_labels) ---")
    for row in summary:
        print(row)
    print("------------------------------------------------------------------\n")

    print(f"Saving frame {fi + 1}/{len(plot_snapshots)} → {args.output}")
    db_map = 10 * cp.asnumpy(cp.log10(power_map + 1e-12))
    det_np = cp.asnumpy(detections)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(db_map, aspect="auto", cmap="jet")
    plt.title("Range-Doppler (dB) — RadarScenes frame → IF → FFT")
    plt.xlabel("Fast Time (Range Bins)")
    plt.ylabel("Slow Time (Doppler Bins)")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(det_np, aspect="auto", cmap="gray")
    plt.title("CA-CFAR Detections")
    plt.xlabel("Fast Time (Range Bins)")
    plt.ylabel("Slow Time (Doppler Bins)")

    plt.tight_layout()
    plt.savefig(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
