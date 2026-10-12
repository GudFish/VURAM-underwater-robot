"""点到点平滑轨迹（三次 Hermite）+ 终点保持语义（纯 numpy，可单测）。

时间轴由期望速度近似确定：总时长 T = 距离 / v_des，钳位到 [t_min, t_max]
（太近的目标走慢些，太远的目标走快些）。起点边界速度取机器人当前速度
（方向保持、幅值钳位），终点边界速度为零，位置/速度/加速度连续；到达
终点后 eval 恒返回常量（目标位姿 + 零速度 + 终点航向），实现无限期保持。
深度 z 由调用方保持为规划时刻的当前深度（目标 z 忽略）。

Hermite 数学与 vuram_control/scripts/vuram_policy/reference_generator.py
的 _ref_waypoints/_hermite 一致（此处复制而非跨包 import，避免依赖私有
函数）。eval 的返回契约与 ReferenceGenerator.get_ref 一致：
(pos[3], rpy[3], vel[3], w_z)。
"""
import numpy as np


def _heading_from_vel(v, w_z, yaw_prev=None):
    """Yaw from velocity direction; falls back to the previous yaw at ~0 speed."""
    speed = float(np.hypot(v[0], v[1]))
    if speed > 1e-3:
        yaw = np.arctan2(v[1], v[0])
    else:
        yaw = yaw_prev if yaw_prev is not None else 0.0
    return yaw, float(w_z)


def _hermite(p0, p1, v0, v1, T, tau):
    """Cubic Hermite segment at normalized time tau in [0,1]:
    returns (pos, vel, acc)."""
    tau = float(np.clip(tau, 0.0, 1.0))
    h00 = 2 * tau ** 3 - 3 * tau ** 2 + 1
    h10 = tau ** 3 - 2 * tau ** 2 + tau
    h01 = -2 * tau ** 3 + 3 * tau ** 2
    h11 = tau ** 3 - tau ** 2
    pos = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    dh00 = 6 * tau ** 2 - 6 * tau
    dh10 = 3 * tau ** 2 - 4 * tau + 1
    dh01 = -6 * tau ** 2 + 6 * tau
    dh11 = 3 * tau ** 2 - 2 * tau
    vel = (dh00 * p0 + dh10 * T * v0 + dh01 * p1 + dh11 * T * v1) / T
    ddh00 = 12 * tau - 6
    ddh10 = 6 * tau - 4
    ddh01 = -12 * tau + 6
    ddh11 = 6 * tau - 2
    acc = (ddh00 * p0 + ddh10 * T * v0 + ddh01 * p1 + ddh11 * T * v1) / T ** 2
    return pos, vel, acc


def yaw_to_quat(yaw):
    """yaw -> xyzw 四元数（roll/pitch 恒为 0）。"""
    return [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]


class GoToGoalTraj(object):
    """单段三次 Hermite 轨迹：p0=start_pos, p1=goal_pos, v0=start_vel, v1=0。"""

    def __init__(self, start_pos, start_vel, goal_pos, T, start_yaw,
                 final_yaw, is_hold=False):
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.start_vel = np.asarray(start_vel, dtype=np.float64)
        self.goal_pos = np.asarray(goal_pos, dtype=np.float64)
        self.T = float(T)
        self.start_yaw = float(start_yaw)
        self.final_yaw = float(final_yaw)
        self.is_hold = bool(is_hold)

    def eval(self, t):
        """t 秒处的参考状态 -> (pos[3], rpy[3], vel[3], w_z)。

        终点保持语义：t >= T 或 is_hold 时恒返回常量
        (goal_pos, [0,0,final_yaw], 零速度, 0.0) —— 轨迹函数自身的终端
        分支，调用方持续求值即可得到逐帧相同的参考。
        """
        if self.is_hold or t >= self.T:
            return (self.goal_pos.copy(),
                    np.array([0.0, 0.0, self.final_yaw]),
                    np.zeros(3), 0.0)
        tau = float(np.clip(t / max(self.T, 1e-9), 0.0, 1.0))
        pos, vel, acc = _hermite(self.start_pos, self.goal_pos,
                                 self.start_vel, np.zeros(3), self.T, tau)
        v2 = vel[0] ** 2 + vel[1] ** 2
        w_z = (vel[0] * acc[1] - vel[1] * acc[0]) / max(v2, 1e-9)
        yaw, w_z = _heading_from_vel(vel, w_z, self.start_yaw)
        return pos, np.array([0.0, 0.0, yaw]), vel, w_z

    def sample_positions(self, dt=0.5):
        """可视化用：沿线采样位置（含终点，is_hold 时仅终点）。"""
        if self.is_hold:
            return [self.goal_pos.copy()]
        ts = np.arange(0.0, self.T + 1e-9, dt)
        pts = [self.eval(t)[0] for t in ts]
        pts.append(self.goal_pos.copy())
        return pts


def make_trajectory(start_pos, start_vel, goal_pos, start_yaw, v_des=0.3,
                    t_min=0.5, t_max=180.0, hold_dist=0.05):
    """从当前位置/速度规划到目标点的平滑轨迹。

    Args:
        start_pos: (3,) 当前位置（世界系）
        start_vel: (3,) 当前速度（世界系，作为起点边界条件）
        goal_pos:  (3,) 目标位置（z 由调用方置为当前深度）
        start_yaw: 当前航向（低速时姿态回退值）
        v_des:     期望速度 m/s（时间轴 = 距离/v_des；经时长钳位后实际
                   平均速度可能偏离）
        t_min/t_max: 总时长钳位 [s]
        hold_dist: 目标距离小于该值视为到位 -> 纯保持轨迹
    """
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    goal_pos = np.asarray(goal_pos, dtype=np.float64)
    d_vec = goal_pos - start_pos
    d = float(np.linalg.norm(d_vec))

    if d < hold_dist:
        return GoToGoalTraj(start_pos, np.zeros(3), goal_pos, 0.0,
                            start_yaw, start_yaw, is_hold=True)

    T = float(np.clip(d / max(v_des, 1e-6), t_min, t_max))
    d_hat = d_vec / d

    # v0 钳位：方向保持，幅值限制为"静止-静止 Hermite 峰值速度
    # (1.5·d/T)"的比例——同向最多 1.5 倍、反向最多 0.25 倍，防止
    # "目标在身后"（反向速度大）时 S 形后摆过远。
    v0_rest = 1.5 * d / T
    v0_norm = float(np.linalg.norm(start_vel))
    if v0_norm > 1e-9:
        v0_proj = float(np.dot(start_vel, d_hat))
        scale = 1.5 if v0_proj >= 0.0 else 0.25
        v0 = start_vel * (min(v0_norm, scale * v0_rest) / v0_norm)
    else:
        v0 = np.zeros(3)

    final_yaw = float(np.arctan2(d_vec[1], d_vec[0]))
    return GoToGoalTraj(start_pos, v0, goal_pos, T, start_yaw, final_yaw)
