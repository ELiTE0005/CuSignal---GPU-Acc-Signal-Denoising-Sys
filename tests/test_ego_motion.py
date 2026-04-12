import numpy as np

from signal_processing.ego_motion import (
    adjust_radial_velocities,
    ego_radial_velocity_mps,
    nearest_odometry_vx,
)


def test_ego_radial_velocity():
    vx = 10.0
    az = np.array([0.0, np.pi / 2])
    v = ego_radial_velocity_mps(vx, az)
    assert np.isclose(v[0], 10.0)
    assert np.isclose(v[1], 0.0, atol=1e-7)


def test_adjust_subtract():
    vr = np.array([5.0, 5.0])
    az = np.array([0.0, np.pi / 2])
    out = adjust_radial_velocities(vr, az, vx_ego=10.0, mode="subtract")
    assert np.isclose(out[0], -5.0)
    assert np.isclose(out[1], 5.0)


def test_nearest_odometry_vx():
    dt = np.dtype(
        [
            ("timestamp", "f8"),
            ("x_seq", "f8"),
            ("y_seq", "f8"),
            ("yaw_seq", "f8"),
            ("vx", "f8"),
            ("yaw_rate", "f8"),
        ]
    )
    o = np.zeros(3, dtype=dt)
    o["timestamp"] = [0.0, 100.0, 200.0]
    o["vx"] = [1.0, 2.0, 3.0]
    assert nearest_odometry_vx(o, 99.0) == 2.0
    assert nearest_odometry_vx(np.array([]), 0.0) == 0.0
