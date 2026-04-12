"""
Phase-2 style multi-object tracking: constant-velocity Kalman filter in 2D (x, y)
with Hungarian assignment (scipy) between predicted states and measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class KalmanTrack:
    x: np.ndarray  # shape (4,)  [px, py, vx, vy]
    P: np.ndarray  # shape (4, 4)
    age_missed: int = 0


class ConstantVelocityKalman2D:
    """Linear Kalman: state [x, y, vx, vy], observation [x, y]."""

    def __init__(
        self,
        q_pos: float = 0.5,
        q_vel: float = 1.0,
        meas_noise: float = 1.0,
    ):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.meas_noise = meas_noise

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        Q = np.diag(
            [self.q_pos * dt, self.q_pos * dt, self.q_vel * dt, self.q_vel * dt]
        ).astype(np.float64)
        return Q

    def predict(self, track: KalmanTrack, dt: float) -> None:
        F = self._F(dt)
        track.x = F @ track.x
        track.P = F @ track.P @ F.T + self._Q(dt)

    def update(self, track: KalmanTrack, z: np.ndarray) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * self.meas_noise
        y = z - H @ track.x
        S = H @ track.P @ H.T + R
        K = track.P @ H.T @ np.linalg.inv(S)
        track.x = track.x + K @ y
        I = np.eye(4, dtype=np.float64)
        track.P = (I - K @ H) @ track.P


class MultiObjectTracker2D:
    """
    Greedy optimal assignment via Hungarian algorithm on Euclidean distance cost.
    """

    def __init__(
        self,
        max_missed: int = 5,
        gate_m: float = 12.0,
        kalman: Optional[ConstantVelocityKalman2D] = None,
    ):
        self.max_missed = max_missed
        self.gate_m = gate_m
        self.kf = kalman or ConstantVelocityKalman2D()
        self._tracks: Dict[int, KalmanTrack] = {}
        self._next_id = 0
        self._last_t: Optional[float] = None

    @property
    def tracks(self) -> Dict[int, KalmanTrack]:
        return self._tracks

    def step(self, timestamp: float, measurements_xy: np.ndarray) -> Dict[int, np.ndarray]:
        """
        measurements_xy: (N, 2) positions in meters.
        Returns dict track_id -> predicted/updated state vector (4,).
        """
        if self._last_t is None:
            dt = 0.05
        else:
            dt = max(1e-4, float(timestamp - self._last_t))
        self._last_t = float(timestamp)

        z = np.asarray(measurements_xy, dtype=np.float64).reshape(-1, 2)
        for tr in self._tracks.values():
            self.kf.predict(tr, dt)

        track_ids = list(self._tracks.keys())
        n_t, n_m = len(track_ids), z.shape[0]

        if n_t == 0 and n_m == 0:
            return {}

        if n_t == 0:
            for j in range(n_m):
                self._spawn(z[j])
            return {tid: self._tracks[tid].x.copy() for tid in self._tracks}

        if n_m == 0:
            dead = []
            for tid in track_ids:
                self._tracks[tid].age_missed += 1
                if self._tracks[tid].age_missed > self.max_missed:
                    dead.append(tid)
            for tid in dead:
                del self._tracks[tid]
            return {tid: self._tracks[tid].x.copy() for tid in self._tracks}

        pred = np.stack([self._tracks[tid].x[:2] for tid in track_ids], axis=0)
        cost = np.linalg.norm(pred[:, None, :] - z[None, :, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_t = set()
        assigned_m = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.gate_m:
                tid = track_ids[r]
                self.kf.update(self._tracks[tid], z[c])
                self._tracks[tid].age_missed = 0
                assigned_t.add(tid)
                assigned_m.add(c)

        for j in range(n_m):
            if j not in assigned_m:
                self._spawn(z[j])

        for tid in track_ids:
            if tid not in assigned_t:
                self._tracks[tid].age_missed += 1

        dead = [tid for tid, tr in self._tracks.items() if tr.age_missed > self.max_missed]
        for tid in dead:
            del self._tracks[tid]

        return {tid: self._tracks[tid].x.copy() for tid in self._tracks}

    def _spawn(self, z: np.ndarray) -> None:
        tid = self._next_id
        self._next_id += 1
        x0 = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        P0 = np.eye(4, dtype=np.float64) * 10.0
        self._tracks[tid] = KalmanTrack(x=x0, P=P0, age_missed=0)
