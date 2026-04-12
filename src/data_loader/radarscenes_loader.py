"""
Load RadarScenes HDF5 sequences (https://radar-scenes.com/).

Expected layout (either):
  <root>/sequence_<N>/radar_data.h5
  <root>/data/sequence_<N>/radar_data.h5
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional, Tuple

import h5py
import numpy as np


def resolve_radar_data_h5(base_path: str, sequence_id: int) -> str:
    """Return path to radar_data.h5 for a sequence, or raise FileNotFoundError."""
    candidates = [
        os.path.join(base_path, f"sequence_{sequence_id}", "radar_data.h5"),
        os.path.join(base_path, "data", f"sequence_{sequence_id}", "radar_data.h5"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"radar_data.h5 not found for sequence_{sequence_id}. Tried: {candidates}"
    )


def load_odometry(h5_path: str) -> np.ndarray:
    """Load odometry table: timestamp, x_seq, y_seq, yaw_seq, vx, yaw_rate."""
    with h5py.File(h5_path, "r") as f:
        if "odometry" not in f:
            return np.zeros((0,), dtype=np.float64)
        return np.asarray(f["odometry"][:])


def _radar_structured_to_records(radar: np.ndarray) -> np.ndarray:
    """Ensure structured array of radar rows (RadarScenes compound dataset)."""
    return np.asarray(radar)


def iter_radar_frames(
    h5_path: str,
    sensor_id: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[float, np.ndarray]]:
    """
    Yield (timestamp, frame_rows) for each unique radar timestamp.

    frame_rows is a structured numpy array (subset of radar_data) for that frame.
    """
    with h5py.File(h5_path, "r") as f:
        radar = _radar_structured_to_records(f["radar_data"][:])

    if radar.size == 0:
        return

    ts = np.asarray(radar["timestamp"], dtype=np.float64)
    order = np.argsort(ts)
    radar = radar[order]
    ts = ts[order]

    unique_ts = np.unique(ts)
    if max_frames is not None:
        unique_ts = unique_ts[: max_frames]

    for t in unique_ts:
        mask = ts == t
        rows = radar[mask]
        if sensor_id is not None:
            sid = np.asarray(rows["sensor_id"])
            rows = rows[sid == sensor_id]
        if rows.size == 0:
            continue
        yield float(t), rows


def frame_to_synthetic_points(
    frame_rows: np.ndarray,
    max_targets: int = 128,
    velocity_field: str = "vr_compensated",
) -> Dict[str, Any]:
    """
    Build a dict suitable for SyntheticADCGenerator.generate_adc_data.

    Subsamples to the strongest returns by RCS when there are too many points.
    velocity_field: 'vr_compensated' (default) or 'vr'
    """
    if frame_rows.size == 0:
        return {
            "range": None,
            "velocity": None,
            "rcs": None,
            "meta": {"label_id": np.array([], dtype=np.int32)},
        }

    n = len(frame_rows)
    if n > max_targets:
        rcs_all = np.asarray(frame_rows["rcs"], dtype=np.float64)
        order = np.argsort(-rcs_all)[:max_targets]
        frame_rows = frame_rows[order]

    r = np.asarray(frame_rows["range_sc"], dtype=np.float64)
    rcs = np.asarray(frame_rows["rcs"], dtype=np.float64)
    if velocity_field in frame_rows.dtype.names:
        v = np.asarray(frame_rows[velocity_field], dtype=np.float64)
    else:
        v = np.asarray(frame_rows["vr"], dtype=np.float64)

    labels = np.asarray(frame_rows["label_id"], dtype=np.int32)

    return {
        "range": r,
        "velocity": v,
        "rcs": rcs,
        "meta": {
            "label_id": labels,
            "azimuth_sc": np.asarray(frame_rows["azimuth_sc"], dtype=np.float64),
            "x_cc": np.asarray(frame_rows["x_cc"], dtype=np.float64),
            "y_cc": np.asarray(frame_rows["y_cc"], dtype=np.float64),
        },
    }


def frame_positions_for_tracking(frame_rows: np.ndarray) -> np.ndarray:
    """Nx2 array of (x_cc, y_cc) in meters for association / Kalman tracking."""
    if frame_rows.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    x = np.asarray(frame_rows["x_cc"], dtype=np.float64)
    y = np.asarray(frame_rows["y_cc"], dtype=np.float64)
    return np.column_stack([x, y])


class RadarScenesLoader:
    """Convenience wrapper around path resolution and iteration."""

    def __init__(self, radar_scenes_root: str):
        self.base_path = os.path.abspath(radar_scenes_root)

    def radar_h5(self, sequence_id: int) -> str:
        return resolve_radar_data_h5(self.base_path, sequence_id)

    def iter_frames(
        self,
        sequence_id: int,
        sensor_id: Optional[int] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[Tuple[float, np.ndarray]]:
        yield from iter_radar_frames(self.radar_h5(sequence_id), sensor_id, max_frames)

    def load_odometry(self, sequence_id: int) -> np.ndarray:
        return load_odometry(self.radar_h5(sequence_id))

    def load_sequence_to_gpu(self, sequence_id: int):
        """
        Legacy: load entire sequence into CuPy arrays (may use a lot of memory).

        Prefer iter_frames + per-frame processing for large sequences.
        """
        import cupy as cp

        h5_file = self.radar_h5(sequence_id)
        with h5py.File(h5_file, "r") as f:
            radar_data = f["radar_data"][:]

        ranges = cp.asarray(radar_data["range_sc"])
        azimuths = cp.asarray(radar_data["azimuth_sc"])
        if "vr_compensated" in radar_data.dtype.names:
            velocities = cp.asarray(radar_data["vr_compensated"])
        else:
            velocities = cp.asarray(radar_data["vr"])
        rcs = cp.asarray(radar_data["rcs"])
        labels = cp.asarray(radar_data["label_id"])
        x_cc = cp.asarray(radar_data["x_cc"])
        y_cc = cp.asarray(radar_data["y_cc"])

        return {
            "range": ranges,
            "azimuth": azimuths,
            "velocity": velocities,
            "rcs": rcs,
            "labels": labels,
            "x": x_cc,
            "y": y_cc,
        }
