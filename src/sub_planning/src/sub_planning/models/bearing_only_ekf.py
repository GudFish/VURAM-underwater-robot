#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


class BearingOnlyCVEKF2D:
    """Bearing-only EKF for a 2D constant-velocity target."""

    def __init__(self, process_noise_diag, bearing_noise_var, min_range=0.25):
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float)
        self.Q_base = np.array(process_noise_diag, dtype=float).reshape(4)
        self.R = float(bearing_noise_var)
        self.min_range = float(min_range)
        self.initialized = False

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def set_state(self, state, covariance):
        self.x = np.array(state, dtype=float).reshape(4)
        self.P = np.array(covariance, dtype=float).reshape(4, 4)
        self.P = 0.5 * (self.P + self.P.T)
        self.initialized = True

    def predict(self, dt):
        if not self.initialized:
            return self.x.copy(), self.P.copy()

        dt = float(max(dt, 1e-3))
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        qx, qy, qvx, qvy = self.Q_base.tolist()
        q = np.array(
            [
                [qx * dt ** 3 / 3.0, 0.0, qx * dt ** 2 / 2.0, 0.0],
                [0.0, qy * dt ** 3 / 3.0, 0.0, qy * dt ** 2 / 2.0],
                [qx * dt ** 2 / 2.0, 0.0, qvx * dt, 0.0],
                [0.0, qy * dt ** 2 / 2.0, 0.0, qvy * dt],
            ],
            dtype=float,
        )

        self.x = f.dot(self.x)
        self.P = f.dot(self.P).dot(f.T) + q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy()

    def _linearize_bearing(self, observer_xy, observer_yaw):
        ox, oy = float(observer_xy[0]), float(observer_xy[1])
        dx = self.x[0] - ox
        dy = self.x[1] - oy
        r2 = max(dx * dx + dy * dy, self.min_range * self.min_range)
        pred = self.wrap_pi(math.atan2(dy, dx) - float(observer_yaw))
        h = np.array([-dy / r2, dx / r2, 0.0, 0.0], dtype=float)
        return pred, h

    def update(self, observer_xy, observer_yaw, measured_bearing):
        x, p, innovation = self.update_multi([observer_xy], [observer_yaw], [measured_bearing])
        innovation_scalar = 0.0 if innovation.size == 0 else float(innovation[0])
        return x, p, innovation_scalar

    def update_multi(self, observers_xy, observer_yaws, measured_bearings, bearing_noise_vars=None):
        if not self.initialized:
            return self.x.copy(), self.P.copy(), np.zeros(0, dtype=float)

        num = min(len(observers_xy), len(observer_yaws), len(measured_bearings))
        if num <= 0:
            return self.x.copy(), self.P.copy(), np.zeros(0, dtype=float)

        h_rows = []
        innovations = []
        for idx in range(num):
            pred, h = self._linearize_bearing(observers_xy[idx], observer_yaws[idx])
            h_rows.append(h)
            innovations.append(self.wrap_pi(float(measured_bearings[idx]) - pred))

        h_mat = np.vstack(h_rows)
        innovation = np.array(innovations, dtype=float)
        if bearing_noise_vars is None:
            r_mat = np.eye(num, dtype=float) * self.R
        else:
            diag = np.array(bearing_noise_vars, dtype=float).reshape(num)
            r_mat = np.diag(diag)

        s = h_mat.dot(self.P).dot(h_mat.T) + r_mat
        k = self.P.dot(h_mat.T).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(innovation)
        i_kh = np.eye(4) - k.dot(h_mat)
        self.P = i_kh.dot(self.P).dot(i_kh.T) + k.dot(r_mat).dot(k.T)
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy(), innovation

    def measurement_stats(self, observer_xy, observer_yaw, measured_bearing):
        pred, h = self._linearize_bearing(observer_xy, observer_yaw)
        innovation = self.wrap_pi(float(measured_bearing) - pred)
        h_col = h.reshape(1, 4)
        s = float(h_col.dot(self.P).dot(h_col.T) + self.R)
        nis = innovation * innovation / max(s, 1e-9)
        return innovation, s, nis
