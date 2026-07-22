#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：定义二维匀速目标的 EKF 模型，支持纯方位更新、距离-方位联合更新
      以及 PDAF（概率数据关联）多候选融合，供估计器节点复用。
输入：通过函数参数或上层调用提供动力学、噪声和状态等模型输入。
输出：返回模型计算结果，供规划器、估计器或演示脚本直接调用。
"""
import math

import numpy as np


class TargetTrackingEKF2D:
    """
    功能：封装二维匀速目标的 EKF 状态、预测与量测更新逻辑。
          支持三种更新模式：
            - update():              纯方位量测更新
            - update_range_bearing(): 距离-方位联合量测更新
            - update_range_bearing_pdaf(): PDAF 多候选融合更新
    输入：构造阶段接收过程噪声对角阵、方位噪声方差和最小距离阈值。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """

    def __init__(self, process_noise_diag, bearing_noise_var, min_range=0.25):
        """
        功能：初始化当前对象，设置过程噪声、量测噪声和最小距离参数。
        输入：`process_noise_diag` 过程噪声对角元素、`bearing_noise_var` 方位量测噪声方差、
              `min_range` 避免除零的最小距离。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float)
        self.Q_base = np.array(process_noise_diag, dtype=float).reshape(4)
        self.R = float(bearing_noise_var)
        self.min_range = float(min_range)
        self.initialized = False

    @staticmethod
    def wrap_pi(angle):
        """
        功能：将输入角度归一化到 [-pi, pi] 区间。
        输入：`angle` 为该函数执行所需的输入参数。
        输出：返回归一化后的弧度角。
        """
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def set_state(self, state, covariance):
        """
        功能：直接设置滤波器状态向量及协方差矩阵。
        输入：`state`、`covariance` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        self.x = np.array(state, dtype=float).reshape(4)
        self.P = np.array(covariance, dtype=float).reshape(4, 4)
        self.P = 0.5 * (self.P + self.P.T)
        self.initialized = True

    def predict(self, dt):
        """
        功能：执行一步 CV 模型状态预测。
        输入：`dt` 为时间步长。
        输出：返回预测后的状态与协方差。
        """
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
        """
        功能：线性化纯方位量测模型，返回预测量测与雅可比矩阵。
        输入：`observer_xy` 观测者位置、`observer_yaw` 观测者偏航角。
        输出：返回 (预测量测, 1x4 雅可比行向量)。
        """
        ox, oy = float(observer_xy[0]), float(observer_xy[1])
        dx = self.x[0] - ox
        dy = self.x[1] - oy
        r2 = max(dx * dx + dy * dy, self.min_range * self.min_range)
        pred = self.wrap_pi(math.atan2(dy, dx) - float(observer_yaw))
        h = np.array([-dy / r2, dx / r2, 0.0, 0.0], dtype=float)
        return pred, h

    def _linearize_range_bearing(self, observer_xy, observer_yaw):
        """
        功能：线性化距离加方位的联合量测模型。
        输入：`observer_xy` 观测者位置、`observer_yaw` 观测者偏航角。
        输出：返回 (2 维预测量测, 2x4 雅可比矩阵)。
        """
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
        """
        功能：根据线性化结果执行一次标准卡尔曼更新。
        输入：`h_mat` 雅可比矩阵、`innovation` 新息向量、`r_mat` 量测噪声协方差。
        输出：返回更新后的状态、协方差和新息协方差 S。
        """
        s = h_mat.dot(self.P).dot(h_mat.T) + r_mat
        k = self.P.dot(h_mat.T).dot(np.linalg.inv(s))

        self.x = self.x + k.dot(innovation)
        i_kh = np.eye(4) - k.dot(h_mat)
        self.P = i_kh.dot(self.P).dot(i_kh.T) + k.dot(r_mat).dot(k.T)
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy(), self.P.copy(), s

    def update(self, observer_xy, observer_yaw, measured_bearing):
        """
        功能：使用单次纯方位量测更新滤波器状态。
        输入：`observer_xy`、`observer_yaw`、`measured_bearing` 为该函数执行所需的输入参数。
        输出：返回更新后的状态、协方差和标量新息。
        """
        x, p, innovation = self.update_multi([observer_xy], [observer_yaw], [measured_bearing])
        innovation_scalar = 0.0 if innovation.size == 0 else float(innovation[0])
        return x, p, innovation_scalar

    def update_multi(self, observers_xy, observer_yaws, measured_bearings, bearing_noise_vars=None):
        """
        功能：使用多条方位观测批量更新滤波器状态。
        输入：`observers_xy`、`observer_yaws`、`measured_bearings`、`bearing_noise_vars` 为该函数执行所需的输入参数。
        输出：返回更新后的状态、协方差和新息向量。
        """
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
        """
        功能：使用距离和方位联合量测更新滤波器状态。
        输入：`observer_xy`、`observer_yaw`、`measured_range`、`measured_bearing`、
              `range_noise_var`、`bearing_noise_var` 为该函数执行所需的输入参数。
        输出：返回更新后的状态、协方差和二维新息向量。
        """
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
        """
        功能：使用 PDAF 方式融合多个距离-方位候选量测。
              包含 χ² 门控、似然加权关联概率、漏检权重和协方差扩散项修正。
        输入：`observer_xy`、`observer_yaw`、`candidates`、`range_noise_var`、
              `bearing_noise_var`、PDAF 关联参数。
        输出：返回更新后的状态、协方差及详细 PDAF 调试信息字典。
        """
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
        """
        功能：计算当前方位量测对应的新息统计量（NIS）。
        输入：`observer_xy`、`observer_yaw`、`measured_bearing` 为该函数执行所需的输入参数。
        输出：返回 (新息, 新息协方差, NIS) 三元组。
        """
        pred, h = self._linearize_bearing(observer_xy, observer_yaw)
        innovation = self.wrap_pi(float(measured_bearing) - pred)
        h_col = h.reshape(1, 4)
        s = float(h_col.dot(self.P).dot(h_col.T) + self.R)
        nis = innovation * innovation / max(s, 1e-9)
        return innovation, s, nis

    def range_bearing_stats(self, observer_xy, observer_yaw, measured_range, measured_bearing, range_noise_var, bearing_noise_var):
        """
        功能：计算距离与方位联合量测的统计指标。
        输入：`observer_xy`、`observer_yaw`、`measured_range`、`measured_bearing`、
              `range_noise_var`、`bearing_noise_var` 为该函数执行所需的输入参数。
        输出：返回 (二维新息, 2x2 新息协方差, NIS) 三元组。
        """
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
