#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：结合目标置信度、声学观测与边界约束，执行 belief-aware MPC 跟踪控制。
输入：通过 ROS 参数以及目标、里程计、边界和感知话题接收规划输入。
输出：发布目标、速度、偏航、障碍物或规划结果，并维护优化器内部状态。
"""
import sys
from pathlib import Path
import math
import casadi as ca
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.boundary_constraints import load_boundary_halfspaces


def wrap_pi(angle):
    """
    功能：将输入角度归一化到 [-pi, pi] 区间。
    输入：`angle` 为该函数执行所需的输入参数。
    输出：返回归一化后的弧度角。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class BeliefAwareMPCTracker:
    """
    功能：结合目标 belief、声呐候选与边界约束执行鲁棒 MPC 跟踪。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('mpc_belief_target_tracker', anonymous=True)

        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.scene_candidate_topic = rospy.get_param('~scene_candidate_topic', '/forward_sonar/scene_candidates')
        self.vel_topic = rospy.get_param('~vel_topic', '/target_velocity_xy')
        self.yaw_topic = rospy.get_param('~yaw_topic', '/target_yaw')

        self.control_rate = float(rospy.get_param('~control_rate', 5.0))
        self.dt = float(rospy.get_param('~mpc_dt', 0.20))
        self.N = int(rospy.get_param('~horizon_steps', 12))

        self.desired_standoff = float(rospy.get_param('~desired_standoff', 3.0))
        self.exploration_standoff = float(rospy.get_param('~exploration_standoff', 4.5))
        self.goal_tolerance = float(rospy.get_param('~goal_tolerance', 0.8))

        self.u_max = float(rospy.get_param('~u_max', 0.8))
        self.v_max = float(rospy.get_param('~v_max', 0.6))
        self.r_cmd_max = float(rospy.get_param('~r_max', 0.6))
        self.du_step_max = float(rospy.get_param('~du_step_max', 0.25))
        self.dv_step_max = float(rospy.get_param('~dv_step_max', 0.20))
        self.dyaw_step_max = float(rospy.get_param('~dyaw_step_max', 0.30))

        self.psi_kp = float(rospy.get_param('~psi_kp', 1.6))
        self.k_tau_u = float(rospy.get_param('~k_tau_u', 120.0))
        self.k_tau_v = float(rospy.get_param('~k_tau_v', 120.0))
        self.k_tau_r = float(rospy.get_param('~k_tau_r', 80.0))
        self.tau_u_max = float(rospy.get_param('~tau_u_max', 200.0))
        self.tau_v_max = float(rospy.get_param('~tau_v_max', 200.0))
        self.tau_r_max = float(rospy.get_param('~tau_r_max', 120.0))

        self.w_track = float(rospy.get_param('~w_track', 2.0))
        self.w_terminal_track = float(rospy.get_param('~w_terminal_track', 10.0))
        self.w_yaw = float(rospy.get_param('~w_yaw', 0.8))
        self.w_ctrl = float(rospy.get_param('~w_ctrl', 0.2))
        self.w_sway = float(rospy.get_param('~w_sway', 0.3))
        self.w_du = float(rospy.get_param('~w_du', 0.6))
        self.w_dyaw = float(rospy.get_param('~w_dyaw', 0.4))
        self.w_cte = float(rospy.get_param('~w_cte', 2.5))
        self.w_sigma = float(rospy.get_param('~w_sigma', 1.2))
        self.w_bearing_trace = float(rospy.get_param('~w_bearing_trace', self.w_sigma))
        self.w_bearing_logdet = float(rospy.get_param('~w_bearing_logdet', 0.0))
        self.bearing_noise_std = float(rospy.get_param('~bearing_noise_std', 0.08))
        self.bearing_process_pos = float(rospy.get_param('~bearing_process_pos', 0.05))
        self.bearing_process_vel = float(rospy.get_param('~bearing_process_vel', 0.10))
        self.min_bearing_range = float(rospy.get_param('~min_bearing_range', 0.5))
        self.cov_floor = float(rospy.get_param('~cov_floor', 1e-6))
        self.sonar_fov_deg = float(rospy.get_param('~sonar_fov_deg', 90.0))
        self.sonar_min_range = float(rospy.get_param('~sonar_min_range', 0.8))
        self.sonar_max_range = float(rospy.get_param('~sonar_max_range', 35.0))
        self.w_sonar_fov = float(rospy.get_param('~w_sonar_fov', 8.0))
        self.w_sonar_range = float(rospy.get_param('~w_sonar_range', 2.0))
        self.w_sonar_center = float(rospy.get_param('~w_sonar_center', 1.5))
        self.w_sonar_front = float(rospy.get_param('~w_sonar_front', 4.0))
        self.sonar_soft_eps = float(rospy.get_param('~sonar_soft_eps', 1e-6))
        self.boundary_param = rospy.get_param('~boundary_param', '/boundary_loader/enu_points')
        self.boundary_margin = float(rospy.get_param('~boundary_margin', 1.0))
        self.boundary_wait_timeout = float(rospy.get_param('~boundary_wait_timeout', 5.0))
        self.boundary_required = bool(rospy.get_param('~boundary_required', False))

        self.unc_pos_gain = float(rospy.get_param('~unc_pos_gain', 1.0))
        self.unc_vel_gain = float(rospy.get_param('~unc_vel_gain', 0.4))
        self.unc_low = float(rospy.get_param('~unc_low', 1.0))
        self.unc_high = float(rospy.get_param('~unc_high', 8.0))
        self.sigma_min = float(rospy.get_param('~sigma_min', 0.1))
        self.search_enable = bool(rospy.get_param('~search_enable', True))
        self.search_forward_speed = float(rospy.get_param('~search_forward_speed', 0.15))
        self.search_yaw_rate = float(rospy.get_param('~search_yaw_rate', 0.25))
        self.bearing_search_enable = bool(rospy.get_param('~bearing_search_enable', True))
        self.bearing_search_timeout = float(rospy.get_param('~bearing_search_timeout', 1.0))
        self.bearing_search_forward_min_scale = float(rospy.get_param('~bearing_search_forward_min_scale', 0.25))
        self.bearing_search_lateral_gain = float(rospy.get_param('~bearing_search_lateral_gain', 0.35))
        self.active_scene_enable = bool(rospy.get_param('~active_scene_enable', True))
        self.active_scene_timeout = float(rospy.get_param('~active_scene_timeout', 0.8))
        self.active_scene_max_points = int(rospy.get_param('~active_scene_max_points', 8))
        self.active_scene_min_score = float(rospy.get_param('~active_scene_min_score', 0.08))
        self.active_scene_sep_sigma_deg = float(rospy.get_param('~active_scene_sep_sigma_deg', 6.0))
        self.w_active_scene_separation = float(rospy.get_param('~w_active_scene_separation', 2.0))

        dyn = rospy.get_param('~dynamics', {})
        self.m11 = float(dyn.get('m11', 40.0))
        self.m22 = float(dyn.get('m22', 55.0))
        self.m23 = float(dyn.get('m23', 0.0))
        self.m33 = float(dyn.get('m33', 12.0))
        self.d11 = float(dyn.get('d11', 20.0))
        self.d22 = float(dyn.get('d22', 28.0))
        self.d23 = float(dyn.get('d23', 0.0))
        self.d32 = float(dyn.get('d32', 0.0))
        self.d33 = float(dyn.get('d33', 8.0))
        self.n11 = float(dyn.get('n11', 18.0))
        self.n22 = float(dyn.get('n22', 24.0))
        self.n23 = float(dyn.get('n23', 0.0))
        self.n32 = float(dyn.get('n32', 0.0))
        self.n33 = float(dyn.get('n33', 6.0))

        self.have_odom = False
        self.have_belief = False
        self.latest_bearing = None
        self.latest_bearing_stamp = None
        self.scene_candidates_world = []
        self.scene_candidates_stamp = None

        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_cov_xx = 0.0
        self.target_cov_xy = 0.0
        self.target_cov_yy = 0.0
        self.target_cov_vxvx = 0.0
        self.target_cov_vxvy = 0.0
        self.target_cov_vyvy = 0.0

        self.last_cmd_u = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_yaw = 0.0
        self.U_init = [0.0] * self.N
        self.V_init = [0.0] * self.N
        self.YAW_init = [0.0] * self.N
        self.last_lam_x = None
        self.last_lam_g = None
        self.boundary_halfspaces = []
        self.boundary_ready = False
        self.boundary_wait_start = rospy.Time.now()
        self.solver = None
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self._build_nlp()

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.belief_sub = rospy.Subscriber(self.belief_topic, Odometry, self.belief_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.scene_candidate_sub = rospy.Subscriber(
            self.scene_candidate_topic,
            Float32MultiArray,
            self.scene_candidates_cb,
            queue_size=2,
        )
        self.vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=10)
        self.yaw_pub = rospy.Publisher(self.yaw_topic, Float32, queue_size=10)

        rospy.loginfo('[BeliefAwareMPC] started. belief topic: %s', self.belief_topic)

    @staticmethod
    def _clip(val, vmin, vmax):
        """
        功能：对数值进行上下限裁剪，避免控制量或中间变量超界。
        输入：`val`、`vmin`、`vmax` 为该函数执行所需的输入参数。
        输出：返回裁剪后的数值。
        """
        return max(vmin, min(vmax, val))

    def belief_scalar_uncertainty(self):
        """
        功能：从 belief 消息中提取可用于控制的标量不确定度。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        pos_unc = max(0.0, self.target_cov_xx + self.target_cov_yy)
        vel_unc = max(0.0, self.target_cov_vxvx + self.target_cov_vyvy)
        return self.unc_pos_gain * pos_unc + self.unc_vel_gain * vel_unc

    def _refresh_boundary_constraints(self):
        """
        功能：刷新边界半空间约束缓存。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        if self.boundary_ready:
            return True

        elapsed = (rospy.Time.now() - self.boundary_wait_start).to_sec()
        wait_timeout = self.boundary_wait_timeout if self.boundary_required else 0.0
        if self.boundary_required and elapsed < wait_timeout and not rospy.has_param(self.boundary_param):
            rospy.logwarn_throttle(
                2.0,
                '[BeliefAwareMPC] waiting for boundary param %s before enabling MPC constraints...',
                self.boundary_param,
            )
            return False

        if not rospy.has_param(self.boundary_param):
            if self.boundary_required:
                rospy.logwarn_throttle(
                    2.0,
                    '[BeliefAwareMPC] boundary param %s still unavailable; MPC remains paused.',
                    self.boundary_param,
                )
                return False
            return True

        halfspaces = load_boundary_halfspaces(
            self.boundary_param,
            wait_timeout=0.0,
            required=False,
            margin=self.boundary_margin,
        )
        if self.boundary_required and not halfspaces:
            rospy.logwarn_throttle(
                2.0,
                '[BeliefAwareMPC] boundary param exists but valid constraints are not ready yet.',
            )
            return False
        if halfspaces != self.boundary_halfspaces:
            self.boundary_halfspaces = halfspaces
            self._build_nlp()
            self.last_lam_x = None
            self.last_lam_g = None
        self.boundary_ready = True
        return True

    def _build_nlp(self):
        """
        功能：构建 MPC 或优化器所需的非线性规划模型。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        u_cmd = ca.SX.sym('U', self.N)
        v_cmd = ca.SX.sym('V', self.N)
        yaw_cmd = ca.SX.sym('Y', self.N)
        w = ca.vertcat(u_cmd, v_cmd, yaw_cmd)

        scene_slots = max(self.active_scene_max_points, 0)
        p = ca.SX.sym('P', 20 + 3 * scene_slots)
        rx = p[0]
        ry = p[1]
        rpsi = p[2]
        ru = p[3]
        rv = p[4]
        rr = p[5]
        tx = p[6]
        ty = p[7]
        tvx = p[8]
        tvy = p[9]
        sigma = p[10]
        prev_u = p[11]
        prev_v = p[12]
        prev_yaw = p[13]
        pxx = p[14]
        pxy = p[15]
        pyy = p[16]
        pvxvx = p[17]
        pvyvy = p[18]
        pvxvy = p[19]
        scene_base = 20

        m = ca.DM([[self.m11, 0.0, 0.0], [0.0, self.m22, self.m23], [0.0, self.m23, self.m33]])
        m_inv = ca.inv(m)

        j = 0
        g = []
        lbg = []
        ubg = []
        sonar_half_fov = max(math.radians(self.sonar_fov_deg) * 0.5, 1e-3)
        sonar_max_range = max(self.sonar_max_range, self.sonar_min_range + 1e-3)
        sonar_range_span = max(sonar_max_range - self.sonar_min_range, 1e-3)
        active_sep_sigma = max(math.radians(self.active_scene_sep_sigma_deg), 1e-3)

        def softplus_hinge(expr):
            return 0.5 * (expr + ca.sqrt(expr * expr + self.sonar_soft_eps))

        for _k in range(self.N):
            alpha = ca.fmin(1.0, ca.fmax(0.0, (sigma - self.unc_low) / max(self.unc_high - self.unc_low, 1e-6)))
            desired_standoff = (1.0 - alpha) * self.desired_standoff + alpha * self.exploration_standoff

            e_psi = ca.atan2(ca.sin(yaw_cmd[_k] - rpsi), ca.cos(yaw_cmd[_k] - rpsi))
            r_cmd = self.psi_kp * e_psi
            r_cmd = ca.fmin(self.r_cmd_max, ca.fmax(-self.r_cmd_max, r_cmd))

            tau_u = self.k_tau_u * (u_cmd[_k] - ru)
            tau_v = self.k_tau_v * (v_cmd[_k] - rv)
            tau_r = self.k_tau_r * (r_cmd - rr)
            tau_u = ca.fmin(self.tau_u_max, ca.fmax(-self.tau_u_max, tau_u))
            tau_v = ca.fmin(self.tau_v_max, ca.fmax(-self.tau_v_max, tau_v))
            tau_r = ca.fmin(self.tau_r_max, ca.fmax(-self.tau_r_max, tau_r))

            c_rb = ca.SX.zeros(3, 3)
            c_rb[0, 2] = -self.m22 * rv - self.m23 * rr
            c_rb[1, 2] = self.m11 * ru
            c_rb[2, 0] = self.m22 * rv + self.m23 * rr
            c_rb[2, 1] = -self.m11 * ru

            d11 = self.d11 + self.n11 * ca.fabs(ru)
            d22 = self.d22 + self.n22 * ca.fabs(rv)
            d33 = self.d33 + self.n33 * ca.fabs(rr)
            d23 = self.d23 + self.n23 * ca.fabs(rr)
            d32 = self.d32 + self.n32 * ca.fabs(rv)
            d_rb = ca.SX.zeros(3, 3)
            d_rb[0, 0] = d11
            d_rb[1, 1] = d22
            d_rb[1, 2] = d23
            d_rb[2, 1] = d32
            d_rb[2, 2] = d33

            nu = ca.vertcat(ru, rv, rr)
            tau = ca.vertcat(tau_u, tau_v, tau_r)
            nu_dot = ca.mtimes(m_inv, (tau - ca.mtimes(c_rb + d_rb, nu)))
            ru = ru + nu_dot[0] * self.dt
            rv = rv + nu_dot[1] * self.dt
            rr = rr + nu_dot[2] * self.dt

            x_dot = ca.cos(rpsi) * ru - ca.sin(rpsi) * rv
            y_dot = ca.sin(rpsi) * ru + ca.cos(rpsi) * rv
            rx = rx + x_dot * self.dt
            ry = ry + y_dot * self.dt
            rpsi = ca.atan2(ca.sin(rpsi + rr * self.dt), ca.cos(rpsi + rr * self.dt))

            tx = tx + tvx * self.dt
            ty = ty + tvy * self.dt

            dx = tx - rx
            dy = ty - ry
            dist = ca.sqrt(dx * dx + dy * dy + 1e-9)
            yaw_ref = ca.atan2(dy, dx)
            yaw_err = ca.atan2(ca.sin(yaw_ref - rpsi), ca.cos(yaw_ref - rpsi))
            range_err = dist - desired_standoff
            cte = -ca.sin(rpsi) * dx + ca.cos(rpsi) * dy

            x_body = ca.cos(rpsi) * dx + ca.sin(rpsi) * dy
            y_body = -ca.sin(rpsi) * dx + ca.cos(rpsi) * dy
            sonar_bearing = ca.atan2(y_body, x_body)
            abs_sonar_bearing = ca.sqrt(sonar_bearing * sonar_bearing + self.sonar_soft_eps)
            fov_violation = softplus_hinge(abs_sonar_bearing - sonar_half_fov)
            too_close = softplus_hinge(self.sonar_min_range - dist)
            too_far = softplus_hinge(dist - sonar_max_range)
            behind = softplus_hinge(-x_body)

            du = u_cmd[_k] - prev_u
            dv = v_cmd[_k] - prev_v
            dyaw = ca.atan2(ca.sin(yaw_cmd[_k] - prev_yaw), ca.cos(yaw_cmd[_k] - prev_yaw))

            pxx_pred = pxx + pvxvx * self.dt * self.dt + self.bearing_process_pos * self.dt
            pxy_pred = pxy + pvxvy * self.dt * self.dt
            pyy_pred = pyy + pvyvy * self.dt * self.dt + self.bearing_process_pos * self.dt
            pvxvx = pvxvx + self.bearing_process_vel * self.dt
            pvyvy = pvyvy + self.bearing_process_vel * self.dt

            r2_bearing = ca.fmax(dx * dx + dy * dy, self.min_bearing_range * self.min_bearing_range)
            hx = -dy / r2_bearing
            hy = dx / r2_bearing
            r_bearing = self.bearing_noise_std * self.bearing_noise_std
            s_bearing = hx * (pxx_pred * hx + pxy_pred * hy) + hy * (pxy_pred * hx + pyy_pred * hy) + r_bearing
            kx = (pxx_pred * hx + pxy_pred * hy) / s_bearing
            ky = (pxy_pred * hx + pyy_pred * hy) / s_bearing
            pxx = ca.fmax(self.cov_floor, pxx_pred - kx * kx * s_bearing)
            pxy = pxy_pred - kx * ky * s_bearing
            pyy = ca.fmax(self.cov_floor, pyy_pred - ky * ky * s_bearing)
            det_p = ca.fmax(self.cov_floor, pxx * pyy - pxy * pxy)
            sigma = ca.fmax(self.sigma_min, pxx + pyy + self.unc_vel_gain * (pvxvx + pvyvy))

            track_scale = 0.25 + 0.75 * (1.0 - alpha)
            cte_scale = 0.2 + 0.8 * (1.0 - alpha)
            active_scene_overlap = 0.0
            if self.active_scene_enable and scene_slots >= 2:
                scene_bearings = []
                scene_weights = []
                for idx in range(scene_slots):
                    base = scene_base + 3 * idx
                    sx = p[base]
                    sy = p[base + 1]
                    sw = p[base + 2]
                    sdx = sx - rx
                    sdy = sy - ry
                    bearing_i = ca.atan2(sdy, sdx) - rpsi
                    bearing_i = ca.atan2(ca.sin(bearing_i), ca.cos(bearing_i))
                    scene_bearings.append(bearing_i)
                    scene_weights.append(sw)
                for i in range(scene_slots):
                    for jj in range(i + 1, scene_slots):
                        dtheta = ca.atan2(
                            ca.sin(scene_bearings[i] - scene_bearings[jj]),
                            ca.cos(scene_bearings[i] - scene_bearings[jj]),
                        )
                        active_scene_overlap = active_scene_overlap + scene_weights[i] * scene_weights[jj] * ca.exp(
                            -0.5 * (dtheta / active_sep_sigma) * (dtheta / active_sep_sigma)
                        )

            j = j + self.w_track * track_scale * (range_err * range_err)
            j = j + self.w_yaw * yaw_err * yaw_err
            j = j + self.w_ctrl * (u_cmd[_k] * u_cmd[_k] + v_cmd[_k] * v_cmd[_k])
            j = j + self.w_sway * (rv * rv)
            j = j + self.w_du * (du * du + dv * dv)
            j = j + self.w_dyaw * (dyaw * dyaw)
            j = j + self.w_cte * cte_scale * (cte * cte)
            j = j + self.w_bearing_trace * (pxx + pyy)
            j = j + self.w_bearing_logdet * ca.log(det_p)
            j = j + self.w_sonar_center * (sonar_bearing / sonar_half_fov) * (sonar_bearing / sonar_half_fov)
            j = j + self.w_sonar_fov * (fov_violation / sonar_half_fov) * (fov_violation / sonar_half_fov)
            j = j + self.w_sonar_range * (
                (too_close / sonar_range_span) * (too_close / sonar_range_span)
                + (too_far / sonar_range_span) * (too_far / sonar_range_span)
            )
            j = j + self.w_sonar_front * (behind / max(self.sonar_min_range, 1e-3)) * (behind / max(self.sonar_min_range, 1e-3))
            j = j + alpha * self.w_active_scene_separation * active_scene_overlap

            prev_u = u_cmd[_k]
            prev_v = v_cmd[_k]
            prev_yaw = yaw_cmd[_k]

            g.append(du)
            lbg.append(-self.du_step_max)
            ubg.append(self.du_step_max)
            g.append(dv)
            lbg.append(-self.dv_step_max)
            ubg.append(self.dv_step_max)
            g.append(dyaw)
            lbg.append(-self.dyaw_step_max)
            ubg.append(self.dyaw_step_max)
            for nx, ny, c_boundary in self.boundary_halfspaces:
                g.append(nx * rx + ny * ry)
                lbg.append(c_boundary)
                ubg.append(ca.inf)

        term_dx = tx - rx
        term_dy = ty - ry
        term_dist = ca.sqrt(term_dx * term_dx + term_dy * term_dy + 1e-9)
        term_alpha = ca.fmin(1.0, ca.fmax(0.0, (sigma - self.unc_low) / max(self.unc_high - self.unc_low, 1e-6)))
        term_desired = (1.0 - term_alpha) * self.desired_standoff + term_alpha * self.exploration_standoff
        term_range_err = term_dist - term_desired
        term_det = ca.fmax(self.cov_floor, pxx * pyy - pxy * pxy)
        j = j + self.w_terminal_track * (term_range_err * term_range_err)
        j = j + self.w_bearing_trace * (pxx + pyy)
        j = j + self.w_bearing_logdet * ca.log(term_det)
        term_x_body = ca.cos(rpsi) * term_dx + ca.sin(rpsi) * term_dy
        term_y_body = -ca.sin(rpsi) * term_dx + ca.cos(rpsi) * term_dy
        term_bearing = ca.atan2(term_y_body, term_x_body)
        term_abs_bearing = ca.sqrt(term_bearing * term_bearing + self.sonar_soft_eps)
        term_fov_violation = softplus_hinge(term_abs_bearing - sonar_half_fov)
        term_too_close = softplus_hinge(self.sonar_min_range - term_dist)
        term_too_far = softplus_hinge(term_dist - sonar_max_range)
        term_behind = softplus_hinge(-term_x_body)
        j = j + self.w_terminal_track * 0.5 * (
            self.w_sonar_center * (term_bearing / sonar_half_fov) * (term_bearing / sonar_half_fov)
            + self.w_sonar_fov * (term_fov_violation / sonar_half_fov) * (term_fov_violation / sonar_half_fov)
            + self.w_sonar_range * (
                (term_too_close / sonar_range_span) * (term_too_close / sonar_range_span)
                + (term_too_far / sonar_range_span) * (term_too_far / sonar_range_span)
            )
            + self.w_sonar_front * (term_behind / max(self.sonar_min_range, 1e-3)) * (term_behind / max(self.sonar_min_range, 1e-3))
        )
        for nx, ny, c_boundary in self.boundary_halfspaces:
            g.append(nx * rx + ny * ry)
            lbg.append(c_boundary)
            ubg.append(ca.inf)

        nlp = {'x': w, 'f': j, 'g': ca.vertcat(*g), 'p': p}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': int(rospy.get_param('~ipopt_max_iter', 80)),
            'ipopt.tol': float(rospy.get_param('~ipopt_tol', 1e-3)),
            'ipopt.acceptable_tol': float(rospy.get_param('~ipopt_acceptable_tol', 5e-3)),
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }
        self.solver = ca.nlpsol('belief_target_solver', 'ipopt', nlp, opts)

        self.lbg = lbg
        self.ubg = ubg
        self.lbx = ([-self.u_max] * self.N) + ([-self.v_max] * self.N) + ([-math.pi] * self.N)
        self.ubx = ([self.u_max] * self.N) + ([self.v_max] * self.N) + ([math.pi] * self.N)

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.psi = math.atan2(siny_cosp, cosy_cosp)
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        c = math.cos(self.psi)
        s = math.sin(self.psi)
        self.u = c * vx_w + s * vy_w
        self.v = -s * vx_w + c * vy_w
        self.r = msg.twist.twist.angular.z
        self.have_odom = True

    def belief_cb(self, msg):
        """
        功能：处理目标 belief 消息并更新内部估计状态。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.target_x = msg.pose.pose.position.x
        self.target_y = msg.pose.pose.position.y
        self.target_vx = msg.twist.twist.linear.x
        self.target_vy = msg.twist.twist.linear.y
        self.target_cov_xx = msg.pose.covariance[0]
        self.target_cov_xy = msg.pose.covariance[1]
        self.target_cov_yy = msg.pose.covariance[7]
        self.target_cov_vxvx = msg.twist.covariance[0]
        self.target_cov_vxvy = msg.twist.covariance[1]
        self.target_cov_vyvy = msg.twist.covariance[7]
        self.have_belief = True

    def bearing_cb(self, msg):
        """
        功能：处理方位消息并更新当前方位缓存。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.latest_bearing = wrap_pi(float(msg.data))
        self.latest_bearing_stamp = rospy.Time.now()

    def fresh_bearing(self):
        """
        功能：判断声学方位观测是否仍然有效。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回布尔值或状态判定结果。
        """
        if self.latest_bearing is None or self.latest_bearing_stamp is None:
            return None
        age = (rospy.Time.now() - self.latest_bearing_stamp).to_sec()
        if age > max(self.bearing_search_timeout, 0.0):
            return None
        return self.latest_bearing

    def scene_candidates_cb(self, msg):
        """
        功能：处理场景候选量测消息并更新候选缓存。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if not self.have_odom or len(msg.data) < 1:
            return

        count = max(0, int(round(float(msg.data[0]))))
        stride = 8
        candidates = []
        for idx in range(count):
            base = 1 + idx * stride
            if base + stride > len(msg.data):
                break

            measured_range = max(0.0, float(msg.data[base]))
            bearing = wrap_pi(float(msg.data[base + 1]))
            score = max(0.0, min(1.0, float(msg.data[base + 2])))
            uncertainty = max(0.0, min(1.0, float(msg.data[base + 3])))
            if score < self.active_scene_min_score:
                continue

            world_bearing = wrap_pi(self.psi + bearing)
            wx = self.x + measured_range * math.cos(world_bearing)
            wy = self.y + measured_range * math.sin(world_bearing)
            weight = score * (1.0 - uncertainty) / math.sqrt(max(measured_range, 1.0))
            candidates.append(
                {
                    'x': wx,
                    'y': wy,
                    'weight': max(weight, 0.0),
                    'range': measured_range,
                    'score': score,
                    'uncertainty': uncertainty,
                }
            )

        candidates.sort(key=lambda item: item['weight'], reverse=True)
        self.scene_candidates_world = candidates[:max(self.active_scene_max_points, 0)]
        self.scene_candidates_stamp = rospy.Time.now()

    def active_scene_is_fresh(self):
        """
        功能：判断当前场景候选缓存是否仍然新鲜。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回布尔值或状态判定结果。
        """
        if not self.active_scene_enable or self.scene_candidates_stamp is None:
            return False
        age = (rospy.Time.now() - self.scene_candidates_stamp).to_sec()
        return age <= max(self.active_scene_timeout, 0.0)

    def active_scene_parameters(self):
        """
        功能：提取当前激活场景候选的控制参数。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        params = []
        if self.active_scene_is_fresh():
            total_weight = sum(item['weight'] for item in self.scene_candidates_world)
            norm = max(total_weight, 1.0e-6)
            selected = self.scene_candidates_world[:max(self.active_scene_max_points, 0)]
        else:
            norm = 1.0
            selected = []

        for idx in range(max(self.active_scene_max_points, 0)):
            if idx < len(selected):
                item = selected[idx]
                params.extend([item['x'], item['y'], self._clip(item['weight'] / norm, 0.0, 1.0)])
            else:
                params.extend([0.0, 0.0, 0.0])
        return params

    def _shift_warm_start(self):
        """
        功能：平移 warm start 序列以加速下一次优化求解。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        if self.N <= 1:
            return
        self.U_init = self.U_init[1:] + [self.U_init[-1]]
        self.V_init = self.V_init[1:] + [self.V_init[-1]]
        self.YAW_init = self.YAW_init[1:] + [self.YAW_init[-1]]

    def solve_mpc(self):
        """
        功能：根据当前状态求解一次 MPC 控制问题。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回当前求解得到的控制结果或求解状态。
        """
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        sigma0 = self.belief_scalar_uncertainty()

        if dist < self.goal_tolerance and sigma0 < self.unc_low:
            yaw_hold = self.last_cmd_yaw
            self.U_init = [0.0] * self.N
            self.V_init = [0.0] * self.N
            self.YAW_init = [yaw_hold] * self.N
            return 0.0, 0.0, yaw_hold, True, sigma0

        yaw_ref = math.atan2(dy, dx)
        ex_b = math.cos(self.psi) * dx + math.sin(self.psi) * dy
        ey_b = -math.sin(self.psi) * dx + math.cos(self.psi) * dy
        u_ref = self._clip(0.5 * ex_b, -self.u_max, self.u_max)
        v_ref = self._clip(0.2 * ey_b, -self.v_max, self.v_max)

        self._shift_warm_start()
        for k in range(self.N):
            self.U_init[k] = self._clip(0.7 * self.U_init[k] + 0.3 * u_ref, -self.u_max, self.u_max)
            self.V_init[k] = self._clip(0.7 * self.V_init[k] + 0.3 * v_ref, -self.v_max, self.v_max)
            self.YAW_init[k] = wrap_pi(0.7 * self.YAW_init[k] + 0.3 * yaw_ref)

        x0 = self.U_init + self.V_init + self.YAW_init
        p = [
            self.x,
            self.y,
            self.psi,
            self.u,
            self.v,
            self.r,
            self.target_x,
            self.target_y,
            self.target_vx,
            self.target_vy,
            sigma0,
            self.last_cmd_u,
            self.last_cmd_v,
            self.last_cmd_yaw,
            max(self.cov_floor, self.target_cov_xx),
            self.target_cov_xy,
            max(self.cov_floor, self.target_cov_yy),
            max(self.cov_floor, self.target_cov_vxvx),
            max(self.cov_floor, self.target_cov_vyvy),
            self.target_cov_vxvy,
        ] + self.active_scene_parameters()

        args = {'x0': x0, 'lbx': self.lbx, 'ubx': self.ubx, 'lbg': self.lbg, 'ubg': self.ubg, 'p': p}
        if self.last_lam_x is not None and self.last_lam_g is not None:
            args['lam_x0'] = self.last_lam_x
            args['lam_g0'] = self.last_lam_g

        sol = self.solver(**args)
        w_opt = sol['x'].full().flatten().tolist()
        self.last_lam_x = sol['lam_x']
        self.last_lam_g = sol['lam_g']

        self.U_init = w_opt[0:self.N]
        self.V_init = w_opt[self.N:2 * self.N]
        self.YAW_init = [wrap_pi(a) for a in w_opt[2 * self.N:3 * self.N]]
        return self.U_init[0], self.V_init[0], self.YAW_init[0], False, sigma0

    def publish_cmd(self, vx_body, vy_body, yaw):
        """
        功能：发布求解得到的速度与偏航控制命令。
        输入：`vx_body`、`vy_body`、`yaw` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        vel_msg = Twist()
        vel_msg.linear.x = vx_body
        vel_msg.linear.y = vy_body
        vel_msg.linear.z = 0.0
        yaw_msg = Float32()
        yaw_msg.data = yaw
        self.vel_pub.publish(vel_msg)
        self.yaw_pub.publish(yaw_msg)

    def publish_search_cmd(self):
        """
        功能：发布搜索模式下的控制命令。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        yaw_cmd = wrap_pi(self.last_cmd_yaw + self.search_yaw_rate / max(self.control_rate, 1e-3))
        vx_cmd = self._clip(self.search_forward_speed, -self.u_max, self.u_max)
        self.last_cmd_u = vx_cmd
        self.last_cmd_v = 0.0
        self.last_cmd_yaw = yaw_cmd
        self.publish_cmd(vx_cmd, 0.0, yaw_cmd)

    def publish_bearing_search_cmd(self, bearing):
        """
        功能：发布结合方位信息的搜索控制命令。
        输入：`bearing` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        world_bearing = wrap_pi(self.psi + bearing)
        forward_scale = max(
            self.bearing_search_forward_min_scale,
            max(0.0, math.cos(bearing)),
        )
        vx_cmd = self._clip(self.search_forward_speed * forward_scale, -self.u_max, self.u_max)
        vy_cmd = self._clip(
            self.bearing_search_lateral_gain * self.search_forward_speed * math.sin(bearing),
            -self.v_max,
            self.v_max,
        )

        self.last_cmd_u = vx_cmd
        self.last_cmd_v = vy_cmd
        self.last_cmd_yaw = world_bearing
        self.publish_cmd(vx_cmd, vy_cmd, world_bearing)

    def run(self):
        """
        功能：执行 `run` 对应的核心处理逻辑。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if not self.have_odom:
                rospy.logwarn_throttle(2.0, '[BeliefAwareMPC] waiting for odom...')
                rate.sleep()
                continue
            if not self.have_belief:
                rospy.logwarn_throttle(2.0, '[BeliefAwareMPC] waiting for target belief; running search behavior...')
                if self.search_enable:
                    bearing = self.fresh_bearing() if self.bearing_search_enable else None
                    if bearing is None:
                        self.publish_search_cmd()
                    else:
                        self.publish_bearing_search_cmd(bearing)
                        rospy.loginfo_throttle(
                            1.0,
                            '[BeliefAwareMPC] bearing-guided search: bearing=%.2f rad yaw=%.2f'
                            % (bearing, self.last_cmd_yaw),
                        )
                rate.sleep()
                continue
            if not self._refresh_boundary_constraints():
                self.publish_cmd(0.0, 0.0, self.last_cmd_yaw)
                rate.sleep()
                continue

            try:
                vx_cmd, vy_cmd, yaw_cmd, reached, sigma0 = self.solve_mpc()
            except Exception as exc:
                rospy.logwarn_throttle(1.0, '[BeliefAwareMPC] solver failed: %s' % str(exc))
                vx_cmd, vy_cmd, yaw_cmd, reached, sigma0 = 0.0, 0.0, self.last_cmd_yaw, False, self.belief_scalar_uncertainty()

            self.last_cmd_u = vx_cmd
            self.last_cmd_v = vy_cmd
            self.last_cmd_yaw = yaw_cmd
            self.publish_cmd(vx_cmd, vy_cmd, yaw_cmd)

            dist = math.hypot(self.target_x - self.x, self.target_y - self.y)
            rospy.loginfo_throttle(
                1.0,
                '[BeliefAwareMPC] belief=(%.2f, %.2f), dist=%.2f, sigma=%.2f, cmd=(%.2f, %.2f), yaw=%.2f'
                % (self.target_x, self.target_y, dist, sigma0, vx_cmd, vy_cmd, yaw_cmd),
            )
            if reached:
                rospy.loginfo_throttle(2.0, '[BeliefAwareMPC] low-uncertainty standoff reached.')
            rate.sleep()


if __name__ == '__main__':
    try:
        BeliefAwareMPCTracker().run()
    except rospy.ROSInterruptException:
        pass
