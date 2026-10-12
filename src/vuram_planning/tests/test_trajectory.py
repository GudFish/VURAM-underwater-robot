"""trajectory.py 单元测试：端点边界、终点保持、钳位与 w_z 公式。"""
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / 'scripts'
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from planning.trajectory import make_trajectory  # noqa: E402


class TestGoToGoal(unittest.TestCase):

    def test_endpoint_boundary(self):
        traj = make_trajectory([0.0, 0.0, -2.0], [0.2, 0.0, 0.0],
                               [5.0, 0.0, -2.0], start_yaw=0.0, v_des=0.3)
        pos0, rpy0, vel0, _ = traj.eval(0.0)
        self.assertTrue(np.allclose(pos0, [0.0, 0.0, -2.0]))
        self.assertTrue(np.allclose(vel0, [0.2, 0.0, 0.0]))
        posT, _, velT, wT = traj.eval(traj.T)
        self.assertTrue(np.allclose(posT, [5.0, 0.0, -2.0]))
        self.assertTrue(np.allclose(velT, 0.0))
        self.assertEqual(wT, 0.0)

    def test_hold_past_end(self):
        """终点保持：T、T+1、T+30 求值逐分量完全相等。"""
        traj = make_trajectory([0.0, 0.0, -2.0], [0.0, 0.0, 0.0],
                               [5.0, 0.0, -2.0], start_yaw=0.0, v_des=0.3)
        refs = [traj.eval(traj.T + dt) for dt in (0.0, 1.0, 30.0)]
        for pos, rpy, vel, w_z in refs:
            self.assertTrue(np.allclose(pos, [5.0, 0.0, -2.0]))
            self.assertTrue(np.allclose(vel, 0.0))
            self.assertEqual(w_z, 0.0)
            self.assertAlmostEqual(rpy[2], 0.0)  # final_yaw = atan2(0,5)
        for a, b in zip(refs, refs[1:]):
            for x, y in zip(a, b):
                self.assertTrue(np.allclose(np.asarray(x), np.asarray(y)))

    def test_hold_trajectory(self):
        """目标极近 -> is_hold，t=0 起即保持。"""
        traj = make_trajectory([0.0, 0.0, -2.0], [0.3, 0.0, 0.0],
                               [0.01, 0.0, -2.0], start_yaw=0.5, v_des=0.3)
        self.assertTrue(traj.is_hold)
        pos, rpy, vel, w_z = traj.eval(0.0)
        self.assertTrue(np.allclose(pos, [0.01, 0.0, -2.0]))
        self.assertTrue(np.allclose(vel, 0.0))
        self.assertAlmostEqual(rpy[2], 0.5)  # 保持起点航向

    def test_duration_clamp(self):
        # 远目标：T 钳位到 t_max
        traj = make_trajectory([0.0, 0.0, -2.0], [0.0, 0.0, 0.0],
                               [100.0, 0.0, -2.0], start_yaw=0.0,
                               v_des=0.3, t_max=10.0)
        self.assertAlmostEqual(traj.T, 10.0)
        # 近目标（超过 hold_dist）：T 钳位到 t_min
        traj = make_trajectory([0.0, 0.0, -2.0], [0.0, 0.0, 0.0],
                               [0.1, 0.0, -2.0], start_yaw=0.0,
                               v_des=0.3, t_min=0.5)
        self.assertAlmostEqual(traj.T, 0.5)

    def test_backward_v0_clamp(self):
        """目标在身后：反向初速幅值被强钳位（防 S 形后摆）。"""
        traj = make_trajectory([0.0, 0.0, -2.0], [-1.0, 0.0, 0.0],
                               [5.0, 0.0, -2.0], start_yaw=0.0, v_des=0.3)
        v0_rest = 1.5 * 5.0 / traj.T
        self.assertAlmostEqual(np.linalg.norm(traj.start_vel),
                               0.25 * v0_rest, places=6)

    def test_forward_v0_clamp(self):
        traj = make_trajectory([0.0, 0.0, -2.0], [3.0, 0.0, 0.0],
                               [5.0, 0.0, -2.0], start_yaw=0.0, v_des=0.3)
        v0_rest = 1.5 * 5.0 / traj.T
        self.assertAlmostEqual(np.linalg.norm(traj.start_vel),
                               1.5 * v0_rest, places=6)

    def test_wz_formula(self):
        """曲线轨迹中段 w_z = (vx·ay - vy·ax)/|v|²。"""
        traj = make_trajectory([0.0, 0.0, -2.0], [0.0, 0.5, 0.0],
                               [3.0, 3.0, -2.0], start_yaw=0.0, v_des=0.3)
        t = 0.5 * traj.T
        pos, rpy, vel, w_z = traj.eval(t)
        # 数值差分验证加速度与 w_z 一致性
        dt = 1e-4
        _, _, vel_p, _ = traj.eval(t + dt)
        _, _, vel_m, _ = traj.eval(t - dt)
        acc = (vel_p - vel_m) / (2 * dt)
        v2 = vel[0] ** 2 + vel[1] ** 2
        w_expected = (vel[0] * acc[1] - vel[1] * acc[0]) / max(v2, 1e-9)
        self.assertAlmostEqual(w_z, w_expected, places=5)
        # yaw 与速度航向一致
        self.assertAlmostEqual(rpy[2], np.arctan2(vel[1], vel[0]), places=6)


if __name__ == '__main__':
    unittest.main()
