"""
Approximate ego-motion effect on radial velocity for IF synthesis.

RadarScenes provides ``vr`` (sensor frame) and ``vr_compensated`` (ego-corrected).
Use this module when starting from ``vr`` and applying your own correction from
odometry (longitudinal ``vx`` only — simplified automotive model).
"""
from __future__ import annotations

import numpy as np


def nearest_odometry_vx(odometry: np.ndarray, timestamp: float) -> float:
    """Return ego longitudinal velocity (m/s) at closest odometry timestamp."""
    if odometry is None or odometry.size == 0:
        return 0.0
    if "timestamp" not in odometry.dtype.names or "vx" not in odometry.dtype.names:
        return 0.0
    t = np.asarray(odometry["timestamp"], dtype=np.float64)
    vx = np.asarray(odometry["vx"], dtype=np.float64)
    i = int(np.argmin(np.abs(t - float(timestamp))))
    return float(vx[i])


def ego_radial_velocity_mps(vx_ego: float, azimuth_rad: np.ndarray) -> np.ndarray:
    """
    Radial component of ego velocity along each azimuth in the sensor frame
    (forward ≈ +x, broadside azimuth 0): v_r,ego ≈ vx * cos(azimuth).
    """
    a = np.asarray(azimuth_rad, dtype=np.float64)
    return float(vx_ego) * np.cos(a)


def adjust_radial_velocities(
    vr: np.ndarray,
    azimuth_sc: np.ndarray,
    vx_ego: float,
    mode: str,
) -> np.ndarray:
    """
    mode:
      none — return vr unchanged
      subtract — vr - v_r,ego (use with raw ``vr`` when synthesizing IF)
      add — vr + v_r,ego (rare; for experiments)
    """
    v = np.asarray(vr, dtype=np.float64).copy()
    if mode == "none":
        return v
    ego_r = ego_radial_velocity_mps(vx_ego, azimuth_sc)
    if mode == "subtract":
        return v - ego_r
    if mode == "add":
        return v + ego_r
    return v
