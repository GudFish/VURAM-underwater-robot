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

    def _linearize_range_bearing(self, observer_xy, observer_yaw):
        ox, oy = float(observer_xy[0]), float(observer_xy[1])
        dx = self.x[0] - ox
        dy = self.x[1] - oy
        r2 = max(dx * dx + dy * dy, self.min_range * self.min_range)
        r = math.sqrt(r2)
        range_pred = r
        bearing_pred = self.wrap_pi(math.atan2(dy, dx) - float(observer_yaw))
        h_range = np.array([dx / r, dy / r, 0.0, 0.0], dtype=float)
        h_bearing = np.array([-dy / r2, dx / r2, 0.0, 0.0], dtype=float)
        return np.array([range_pred, bearing_pred], dtype=float), np.vstack((h_range, h_bearing))

    def _apply_update(self, h_mat, innovation, r_mat):
        s = h_mat.dot(self.P).dot(h_mat.T) + r_mat
        k = self.P.dot(h_mat.T).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(innovation)
        i_kh = np.eye(4) - k.dot(h_mat)
        self.P = i_kh.dot(self.P).dot(i_kh.T) + k.dot(r_mat).dot(k.T)
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy(), s

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

        self._apply_update(h_mat, innovation, r_mat)
        return self.x.copy(), self.P.copy(), innovation

    def update_range_bearing(self, observer_xy, observer_yaw, measured_range, measured_bearing, range_noise_var, bearing_noise_var):
        if not self.initialized:
            return self.x.copy(), self.P.copy(), np.zeros(2, dtype=float)

        pred, h_mat = self._linearize_range_bearing(observer_xy, observer_yaw)
        innovation = np.array(
            [
                float(measured_range) - pred[0],
                self.wrap_pi(float(measured_bearing) - pred[1]),
            ],
            dtype=float,
        )
        r_mat = np.diag([float(range_noise_var), float(bearing_noise_var)])
        self._apply_update(h_mat, innovation, r_mat)
        return self.x.copy(), self.P.copy(), innovation

    def update_range_bearing_pdaf(
        self,
        observer_xy,
        observer_yaw,
        candidates,
        range_noise_var,
        bearing_noise_var,
        detection_probability=0.85,
        missed_weight=0.20,
        gate_chi2=9.21,
        min_beta_sum=0.05,
        range_weight_min=0.35,
        range_weight_power=1.0,
    ):
        if not self.initialized:
            return self.x.copy(), self.P.copy(), {
                'updated': False,
                'num_candidates': len(candidates),
                'num_gated': 0,
                'beta0': 1.0,
                'best_beta': 0.0,
                'weighted_range': 0.0,
                'weighted_bearing': 0.0,
                'best_nis': 0.0,
                'innovation': np.zeros(2, dtype=float),
            }

        pred, h_mat = self._linearize_range_bearing(observer_xy, observer_yaw)
        r_mat = np.diag([float(range_noise_var), float(bearing_noise_var)])
        s = h_mat.dot(self.P).dot(h_mat.T) + r_mat
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)

        det_s = max(float(np.linalg.det(s)), 1.0e-12)
        norm = math.sqrt(max(((2.0 * math.pi) ** 2) * det_s, 1.0e-12))
        gated = []
        for candidate in candidates:
            innovation = np.array(
                [
                    float(candidate['range']) - pred[0],
                    self.wrap_pi(float(candidate['bearing']) - pred[1]),
                ],
                dtype=float,
            )
            nis = float(innovation.T.dot(s_inv).dot(innovation))
            if not math.isfinite(nis) or nis > float(gate_chi2):
                continue

            score = max(0.0, min(1.0, float(candidate.get('score', 1.0))))
            measured_range = max(float(candidate['range']), self.min_range)
            range_weight = max(float(range_weight_min), min(1.0, pred[0] / measured_range)) ** max(float(range_weight_power), 0.0)
            likelihood = math.exp(-0.5 * nis) / norm
            weight = max(0.0, float(detection_probability)) * max(score, 1.0e-3) * range_weight * likelihood
            gated.append(
                {
                    'candidate': candidate,
                    'innovation': innovation,
                    'nis': nis,
                    'weight': weight,
                    'range_weight': range_weight,
                }
            )

        weight_sum = sum(item['weight'] for item in gated)
        denominator = max(float(missed_weight), 0.0) + weight_sum
        if not gated or denominator <= 1.0e-12:
            return self.x.copy(), self.P.copy(), {
                'updated': False,
                'num_candidates': len(candidates),
                'num_gated': len(gated),
                'beta0': 1.0,
                'best_beta': 0.0,
                'weighted_range': float(pred[0]),
                'weighted_bearing': float(pred[1]),
                'best_nis': 0.0,
                'innovation': np.zeros(2, dtype=float),
            }

        beta0 = max(float(missed_weight), 0.0) / denominator
        betas = np.array([item['weight'] / denominator for item in gated], dtype=float)
        beta_sum = float(np.sum(betas))
        if beta_sum < max(float(min_beta_sum), 0.0):
            return self.x.copy(), self.P.copy(), {
                'updated': False,
                'num_candidates': len(candidates),
                'num_gated': len(gated),
                'beta0': float(beta0),
                'best_beta': float(np.max(betas)),
                'weighted_range': float(pred[0]),
                'weighted_bearing': float(pred[1]),
                'best_nis': float(min(item['nis'] for item in gated)),
                'innovation': np.zeros(2, dtype=float),
            }

        innovations = np.vstack([item['innovation'] for item in gated])
        innovation_bar = betas.dot(innovations)
        k_gain = self.P.dot(h_mat.T).dot(s_inv)
        state_pred = self.x.copy()
        p_pred = self.P.copy()

        self.x = state_pred + k_gain.dot(innovation_bar)

        p_corrected = p_pred - k_gain.dot(s).dot(k_gain.T)
        spread = np.zeros((2, 2), dtype=float)
        for beta, innovation in zip(betas, innovations):
            delta = innovation - innovation_bar
            spread += beta * np.outer(delta, delta)
        spread += beta0 * np.outer(innovation_bar, innovation_bar)
        self.P = beta0 * p_pred + (1.0 - beta0) * p_corrected + k_gain.dot(spread).dot(k_gain.T)
        self.P = 0.5 * (self.P + self.P.T)

        weighted_range = pred[0] + innovation_bar[0]
        weighted_bearing = self.wrap_pi(pred[1] + innovation_bar[1])
        best_idx = int(np.argmax(betas))
        return self.x.copy(), self.P.copy(), {
            'updated': True,
            'num_candidates': len(candidates),
            'num_gated': len(gated),
            'beta0': float(beta0),
            'best_beta': float(betas[best_idx]),
            'weighted_range': float(weighted_range),
            'weighted_bearing': float(weighted_bearing),
            'best_nis': float(gated[best_idx]['nis']),
            'innovation': innovation_bar,
            'betas': betas,
        }

    def measurement_stats(self, observer_xy, observer_yaw, measured_bearing):
        pred, h = self._linearize_bearing(observer_xy, observer_yaw)
        innovation = self.wrap_pi(float(measured_bearing) - pred)
        h_col = h.reshape(1, 4)
        s = float(h_col.dot(self.P).dot(h_col.T) + self.R)
        nis = innovation * innovation / max(s, 1e-9)
        return innovation, s, nis

    def range_bearing_stats(self, observer_xy, observer_yaw, measured_range, measured_bearing, range_noise_var, bearing_noise_var):
        pred, h_mat = self._linearize_range_bearing(observer_xy, observer_yaw)
        innovation = np.array(
            [
                float(measured_range) - pred[0],
                self.wrap_pi(float(measured_bearing) - pred[1]),
            ],
            dtype=float,
        )
        r_mat = np.diag([float(range_noise_var), float(bearing_noise_var)])
        s = h_mat.dot(self.P).dot(h_mat.T) + r_mat
        try:
            nis = float(innovation.T.dot(np.linalg.inv(s)).dot(innovation))
        except np.linalg.LinAlgError:
            nis = float('inf')
        return innovation, s, nis
