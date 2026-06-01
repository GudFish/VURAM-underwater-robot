#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


class RangeOnlyCVEKF2D:
    """Range-only EKF for a 2D constant-velocity target."""

    def __init__(self, process_noise_diag, range_noise_var, min_range=1e-3):
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float)
        self.Q_base = np.array(process_noise_diag, dtype=float).reshape(4)
        self.R = float(range_noise_var)
        self.min_range = float(min_range)
        self.initialized = False

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

    def update(self, observer_xy, measured_range):
        if not self.initialized:
            return self.x.copy(), self.P.copy(), 0.0

        ox, oy = float(observer_xy[0]), float(observer_xy[1])
        dx = self.x[0] - ox
        dy = self.x[1] - oy
        pred_range = max(math.hypot(dx, dy), self.min_range)

        h = np.array([[dx / pred_range, dy / pred_range, 0.0, 0.0]], dtype=float)
        innovation = float(measured_range) - pred_range
        s = float(h.dot(self.P).dot(h.T) + self.R)
        k = self.P.dot(h.T) / max(s, 1e-9)

        self.x = self.x + (k[:, 0] * innovation)
        i_kh = np.eye(4) - k.dot(h)
        self.P = i_kh.dot(self.P).dot(i_kh.T) + k.dot(np.array([[self.R]])).dot(k.T)
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy(), innovation

    def measurement_stats(self, observer_xy, measured_range):
        ox, oy = float(observer_xy[0]), float(observer_xy[1])
        dx = self.x[0] - ox
        dy = self.x[1] - oy
        pred_range = max(math.hypot(dx, dy), self.min_range)
        h = np.array([[dx / pred_range, dy / pred_range, 0.0, 0.0]], dtype=float)
        innovation = float(measured_range) - pred_range
        s = float(h.dot(self.P).dot(h.T) + self.R)
        nis = innovation * innovation / max(s, 1e-9)
        return innovation, s, nis
