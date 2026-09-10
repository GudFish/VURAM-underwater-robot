#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：面向动态目标执行 CE-NMPC 跟踪控制，在跟踪丢失时触发脚本化恢复行为。
      与 proposed 方法共享 PDAF-EKF 估计器，但使用确定性等价 NMPC（无信息代价），
      并以被动、脚本化的恢复策略代替主动信息采集（基线方法 B3）。
输入：通过 ROS 参数以及里程计、belief、声学方位和目标话题接收数据。
输出：发布目标速度（/target_velocity_xy）与偏航角（/target_yaw）。
"""
import sys
from pathlib import Path
import math
import time
import casadi as ca
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from utils.boundary_constraints import load_boundary_halfspaces
from utils.sonar_obstacle_avoidance import SonarObstacleAvoidance


def wrap_pi(angle):
    """
    功能：将输入角度归一化到 [-pi, pi] 区间。
    输入：`angle` 为该函数执行所需的输入参数。
    输出：返回归一化后的弧度角。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class RecoveringDynamicTargetMPCTracker:
    """
    功能：CE-NMPC 跟踪 + 脚本化恢复状态机（基线方法 B3）。
          正常模式下执行与 B1 相同的 CE-NMPC 跟踪；检测到跟踪丢失后
          切换到脚本化恢复行为（旋转搜索 / 螺旋搜索 / 方位制导搜索）。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('mpc_dynamic_target_tracker_recovery', anonymous=True)

        # --- 话题接口参数 ---
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.target_topic = rospy.get_param('~target_topic', '/target_estimate/pose')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.vel_topic = rospy.get_param('~vel_topic', '/target_velocity_xy')
        self.yaw_topic = rospy.get_param('~yaw_topic', '/target_yaw')

        # --- MPC 时域参数 ---
        self.control_rate = rospy.get_param('~control_rate', 5.0)
        self.dt = rospy.get_param('~mpc_dt', 0.20)
        self.N = rospy.get_param('~horizon_steps', 12)

        # --- 跟踪参数 ---
        self.desired_standoff = rospy.get_param('~desired_standoff', 3.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.8)
        self.use_goal_yaw = rospy.get_param('~use_goal_yaw', False)
        self.yaw_switch_dist = rospy.get_param('~yaw_switch_dist', 1.5)

        # --- 控制约束 ---
        self.u_max = rospy.get_param('~u_max', 0.8)
        self.v_max = rospy.get_param('~v_max', 0.6)
        self.r_cmd_max = rospy.get_param('~r_max', 0.6)
        self.du_step_max = rospy.get_param('~du_step_max', 0.25)
        self.dv_step_max = rospy.get_param('~dv_step_max', 0.20)
        self.dyaw_step_max = rospy.get_param('~dyaw_step_max', 0.30)

        # --- 虚拟控制映射 ---
        self.psi_kp = rospy.get_param('~psi_kp', 1.6)
        self.k_tau_u = rospy.get_param('~k_tau_u', 120.0)
        self.k_tau_v = rospy.get_param('~k_tau_v', 120.0)
        self.k_tau_r = rospy.get_param('~k_tau_r', 80.0)
        self.tau_u_max = rospy.get_param('~tau_u_max', 200.0)
        self.tau_v_max = rospy.get_param('~tau_v_max', 200.0)
        self.tau_r_max = rospy.get_param('~tau_r_max', 120.0)

        # --- MPC 代价权重 ---
        self.w_track = rospy.get_param('~w_track', 2.0)
        self.w_terminal_track = rospy.get_param('~w_terminal_track', 10.0)
        self.w_yaw = rospy.get_param('~w_yaw', 0.8)
        self.w_ctrl = rospy.get_param('~w_ctrl', 0.2)
        self.w_sway = rospy.get_param('~w_sway', 0.3)
        self.w_du = rospy.get_param('~w_du', 0.6)
        self.w_dyaw = rospy.get_param('~w_dyaw', 0.4)
        self.w_cte = rospy.get_param('~w_cte', 2.5)

        # --- 声呐 FOV 约束 ---
        self.sonar_fov_deg = float(rospy.get_param('~sonar_fov_deg', 130.0))
        self.sonar_min_range = float(rospy.get_param('~sonar_min_range', 0.2))
        self.sonar_max_range = float(rospy.get_param('~sonar_max_range', 30.0))
        self.w_sonar_fov = float(rospy.get_param('~w_sonar_fov', 8.0))
        self.w_sonar_range = float(rospy.get_param('~w_sonar_range', 2.0))
        self.w_sonar_center = float(rospy.get_param('~w_sonar_center', 1.5))
        self.w_sonar_front = float(rospy.get_param('~w_sonar_front', 4.0))
        self.sonar_soft_eps = float(rospy.get_param('~sonar_soft_eps', 1e-6))

        # --- 边界约束 ---
        self.boundary_param = rospy.get_param('~boundary_param', '/boundary_loader/enu_points')
        self.boundary_margin = float(rospy.get_param('~boundary_margin', 1.0))
        self.boundary_wait_timeout = float(rospy.get_param('~boundary_wait_timeout', 8.0))
        self.boundary_required = bool(rospy.get_param('~boundary_required', True))

        # --- 动力学参数 ---
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

        # --- 跟踪丢失检测阈值 ---
        self.cov_trace_threshold = float(rospy.get_param('~cov_trace_threshold', 10.0))
        self.lost_track_missed_updates = int(rospy.get_param('~lost_track_missed_updates', 5))
        self.belief_timeout = float(rospy.get_param('~belief_timeout', 1.0))

        # --- 恢复行为参数 ---
        self.recovery_rotate_yaw_rate = float(rospy.get_param('~recovery_rotate_yaw_rate', 0.30))
        self.recovery_spiral_speed = float(rospy.get_param('~recovery_spiral_speed', 0.20))
        self.recovery_spiral_yaw_rate = float(rospy.get_param('~recovery_spiral_yaw_rate', 0.15))
        self.recovery_spiral_radius_growth = float(rospy.get_param('~recovery_spiral_radius_growth', 0.05))
        self.recovery_bearing_search_speed = float(rospy.get_param('~recovery_bearing_search_speed', 0.20))
        self.recovery_bearing_search_lateral_gain = float(rospy.get_param('~recovery_bearing_search_lateral_gain', 0.35))
        self.recovery_max_duration = float(rospy.get_param('~recovery_max_duration', 25.0))
        self.recovery_mode_switch_time = float(rospy.get_param('~recovery_mode_switch_time', 12.0))
        self.bearing_search_timeout = float(rospy.get_param('~bearing_search_timeout', 1.0))

        # --- 跟踪恢复判定阈值 ---
        self.recovery_cov_trace_threshold = float(rospy.get_param('~recovery_cov_trace_threshold', 2.0))
        self.recovery_min_consecutive_good = int(rospy.get_param('~recovery_min_consecutive_good', 3))

        # ================================================================
        #  运行时状态
        # ================================================================

        # 机器人状态
        self.have_odom = False
        self.have_target = False
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # 目标估计（来自 /target_estimate/pose）
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_yaw = 0.0

        # 目标 belief 协方差（来自 /target_estimate/belief）
        self.target_cov_xx = 0.0
        self.target_cov_xy = 0.0
        self.target_cov_yy = 0.0
        self.last_belief_stamp = None

        # 声学方位
        self.latest_bearing = None
        self.latest_bearing_stamp = None

        # 上一周期指令
        self.last_cmd_u = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_yaw = 0.0

        # MPC warm start 序列
        self.U_init = [0.0] * self.N
        self.V_init = [0.0] * self.N
        self.YAW_init = [0.0] * self.N
        self.last_lam_x = None
        self.last_lam_g = None

        # 边界约束
        self.boundary_halfspaces = []
        self.boundary_ready = False
        self.boundary_wait_start = rospy.Time.now()
        self.avoidance = SonarObstacleAvoidance()

        # NLP 求解器
        self.solver = None
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
        self._build_nlp()

        # --- 恢复状态机 ---
        self.track_lost = False
        self.track_lost_time = None
        self.track_lost_duration = rospy.Duration(0)
        self.recovery_mode = 'rotate'  # 'rotate' | 'spiral' | 'bearing_search'
        self.recovery_start_time = None
        self.recovery_mode_start_time = None
        self.recovery_count = 0
        self.consecutive_missed = 0
        self.consecutive_good = 0
        self.spiral_accumulated_angle = 0.0

        # --- 订阅者 ---
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.target_sub = rospy.Subscriber(self.target_topic, PoseStamped, self.target_cb, queue_size=1)
        self.belief_sub = rospy.Subscriber(self.belief_topic, Odometry, self.belief_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)

        # --- 发布者 ---
        self.vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=10)
        self.yaw_pub = rospy.Publisher(self.yaw_topic, Float32, queue_size=10)
        self.solver_time_pub = rospy.Publisher('/planner/solver_time', Float32MultiArray, queue_size=10)
        self.recovery_status_pub = rospy.Publisher('/planner/recovery_status', Float32MultiArray, queue_size=10)

        rospy.loginfo('[RecoveryMPC] started. belief topic: %s', self.belief_topic)

    # ========================================================================
    #  NLP 构建（与 B1 CE-NMPC 相同）
    # ========================================================================

    @staticmethod
    def _clip(val, vmin, vmax):
        return max(vmin, min(vmax, val))

    def _refresh_boundary_constraints(self):
        """
        功能：刷新边界半空间约束缓存。
        """
        if self.boundary_ready:
            return True

        elapsed = (rospy.Time.now() - self.boundary_wait_start).to_sec()
        wait_timeout = self.boundary_wait_timeout if self.boundary_required else 0.0
        if self.boundary_required and elapsed < wait_timeout and not rospy.has_param(self.boundary_param):
            rospy.logwarn_throttle(
                2.0,
                '[RecoveryMPC] waiting for boundary param %s ...',
                self.boundary_param,
            )
            return False

        if not rospy.has_param(self.boundary_param):
            if self.boundary_required:
                rospy.logwarn_throttle(2.0, '[RecoveryMPC] boundary param %s still unavailable.', self.boundary_param)
                return False
            return True

        halfspaces = load_boundary_halfspaces(
            self.boundary_param,
            wait_timeout=0.0,
            required=False,
            margin=self.boundary_margin,
        )
        if self.boundary_required and not halfspaces:
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
        功能：构建 CE-NMPC 非线性规划模型（与 B1 相同的 Fossen + IPOPT 框架）。
        """
        u_cmd = ca.SX.sym('U', self.N)
        v_cmd = ca.SX.sym('V', self.N)
        yaw_cmd = ca.SX.sym('Y', self.N)
        w = ca.vertcat(u_cmd, v_cmd, yaw_cmd)

        obstacle_slots = max(self.avoidance.max_obstacles, 0)
        p = ca.SX.sym('P', 13 + 4 * obstacle_slots)
        rx = p[0]; ry = p[1]; rpsi = p[2]; ru = p[3]; rv = p[4]; rr = p[5]
        tx = p[6]; ty = p[7]; gyaw = p[8]
        prev_u = p[9]; prev_v = p[10]; prev_yaw = p[11]; use_gyaw = p[12]
        obstacle_base = 13

        m = ca.DM([[self.m11, 0.0, 0.0], [0.0, self.m22, self.m23], [0.0, self.m23, self.m33]])
        m_inv = ca.inv(m)

        j = 0
        g = []; lbg = []; ubg = []
        sonar_half_fov = max(math.radians(self.sonar_fov_deg) * 0.5, 1e-3)
        sonar_max_range = max(self.sonar_max_range, self.sonar_min_range + 1e-3)
        sonar_range_span = max(sonar_max_range - self.sonar_min_range, 1e-3)

        def softplus_hinge(expr):
            return 0.5 * (expr + ca.sqrt(expr * expr + self.sonar_soft_eps))

        for k in range(self.N):
            e_psi = ca.atan2(ca.sin(yaw_cmd[k] - rpsi), ca.cos(yaw_cmd[k] - rpsi))
            r_cmd = self.psi_kp * e_psi
            r_cmd = ca.fmin(self.r_cmd_max, ca.fmax(-self.r_cmd_max, r_cmd))

            tau_u = self.k_tau_u * (u_cmd[k] - ru)
            tau_v = self.k_tau_v * (v_cmd[k] - rv)
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
            d_rb[0, 0] = d11; d_rb[1, 1] = d22; d_rb[1, 2] = d23
            d_rb[2, 1] = d32; d_rb[2, 2] = d33

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

            dx = tx - rx; dy = ty - ry
            dist = ca.sqrt(dx * dx + dy * dy + 1e-9)
            path_yaw = ca.atan2(dy, dx)
            yaw_ref = use_gyaw * ca.if_else(dist < self.yaw_switch_dist, gyaw, path_yaw) \
                      + (1.0 - use_gyaw) * path_yaw
            yaw_err = ca.atan2(ca.sin(yaw_ref - rpsi), ca.cos(yaw_ref - rpsi))

            range_err = dist - self.desired_standoff
            du = u_cmd[k] - prev_u
            dv = v_cmd[k] - prev_v
            dyaw = ca.atan2(ca.sin(yaw_cmd[k] - prev_yaw), ca.cos(yaw_cmd[k] - prev_yaw))
            cte = -ca.sin(rpsi) * dx + ca.cos(rpsi) * dy

            x_body = ca.cos(rpsi) * dx + ca.sin(rpsi) * dy
            y_body = -ca.sin(rpsi) * dx + ca.cos(rpsi) * dy
            sonar_bearing = ca.atan2(y_body, x_body)
            abs_sonar_bearing = ca.sqrt(sonar_bearing * sonar_bearing + self.sonar_soft_eps)
            fov_violation = softplus_hinge(abs_sonar_bearing - sonar_half_fov)
            too_close = softplus_hinge(self.sonar_min_range - dist)
            too_far = softplus_hinge(dist - sonar_max_range)
            behind = softplus_hinge(-x_body)

            j = j + self.w_track * (range_err * range_err)
            j = j + self.w_yaw * yaw_err * yaw_err
            j = j + self.w_ctrl * (u_cmd[k] * u_cmd[k] + v_cmd[k] * v_cmd[k])
            j = j + self.w_sway * (rv * rv)
            j = j + self.w_du * (du * du + dv * dv)
            j = j + self.w_dyaw * (dyaw * dyaw)
            j = j + self.w_cte * (cte * cte)
            j = j + self.w_sonar_center * (sonar_bearing / sonar_half_fov) * (sonar_bearing / sonar_half_fov)
            j = j + self.w_sonar_fov * (fov_violation / sonar_half_fov) * (fov_violation / sonar_half_fov)
            j = j + self.w_sonar_range * (
                (too_close / sonar_range_span) * (too_close / sonar_range_span)
                + (too_far / sonar_range_span) * (too_far / sonar_range_span)
            )
            j = j + self.w_sonar_front * (behind / max(self.sonar_min_range, 1e-3)) * (behind / max(self.sonar_min_range, 1e-3))

            prev_u = u_cmd[k]; prev_v = v_cmd[k]; prev_yaw = yaw_cmd[k]

            g.append(du); lbg.append(-self.du_step_max); ubg.append(self.du_step_max)
            g.append(dv); lbg.append(-self.dv_step_max); ubg.append(self.dv_step_max)
            g.append(dyaw); lbg.append(-self.dyaw_step_max); ubg.append(self.dyaw_step_max)
            for nx, ny, c_boundary in self.boundary_halfspaces:
                g.append(nx * rx + ny * ry)
                lbg.append(c_boundary); ubg.append(ca.inf)
            for idx in range(obstacle_slots):
                base = obstacle_base + 4 * idx
                ox, oy, safe_radius, active = p[base], p[base + 1], p[base + 2], p[base + 3]
                g.append((rx - ox) * (rx - ox) + (ry - oy) * (ry - oy) - active * safe_radius * safe_radius)
                lbg.append(0.0); ubg.append(ca.inf)

        # 终端代价
        term_dx = tx - rx; term_dy = ty - ry
        term_dist = ca.sqrt(term_dx * term_dx + term_dy * term_dy + 1e-9)
        term_range_err = term_dist - self.desired_standoff
        j = j + self.w_terminal_track * (term_range_err * term_range_err)
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
            lbg.append(c_boundary); ubg.append(ca.inf)

        nlp = {'x': w, 'f': j, 'g': ca.vertcat(*g), 'p': p}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': int(rospy.get_param('~ipopt_max_iter', 80)),
            'ipopt.tol': float(rospy.get_param('~ipopt_tol', 1e-3)),
            'ipopt.acceptable_tol': float(rospy.get_param('~ipopt_acceptable_tol', 5e-3)),
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }
        self.solver = ca.nlpsol('recovery_target_solver', 'ipopt', nlp, opts)

        self.lbg = lbg; self.ubg = ubg
        self.lbx = ([-self.u_max] * self.N) + ([-self.v_max] * self.N) + ([-math.pi] * self.N)
        self.ubx = ([self.u_max] * self.N) + ([self.v_max] * self.N) + ([math.pi] * self.N)

    # ========================================================================
    #  回调方法
    # ========================================================================

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.psi = math.atan2(siny_cosp, cosy_cosp)
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        c = math.cos(self.psi); s = math.sin(self.psi)
        self.u = c * vx_w + s * vy_w
        self.v = -s * vx_w + c * vy_w
        self.r = msg.twist.twist.angular.z
        self.have_odom = True

    def target_cb(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        q = msg.pose.orientation
        has_orientation = abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) > 1e-8
        if has_orientation:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.target_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_target = True
        self.consecutive_missed = 0  # 收到目标更新即重置

    def belief_cb(self, msg):
        """提取协方差信息用于跟踪丢失检测。"""
        cov = msg.pose.covariance
        self.target_cov_xx = float(cov[0])
        self.target_cov_xy = float(cov[1])
        self.target_cov_yy = float(cov[7])
        self.last_belief_stamp = rospy.Time.now()

    def bearing_cb(self, msg):
        self.latest_bearing = msg.data
        self.latest_bearing_stamp = rospy.Time.now()

    # ========================================================================
    #  跟踪丢失检测与恢复判定
    # ========================================================================

    def cov_trace(self):
        """返回位置协方差迹 tr(P_xy)。"""
        return max(0.0, self.target_cov_xx + self.target_cov_yy)

    def is_belief_fresh(self):
        if self.last_belief_stamp is None:
            return False
        return (rospy.Time.now() - self.last_belief_stamp).to_sec() < self.belief_timeout

    def is_bearing_fresh(self):
        if self.latest_bearing_stamp is None:
            return False
        return (rospy.Time.now() - self.latest_bearing_stamp).to_sec() < self.bearing_search_timeout

    def detect_track_loss(self):
        """
        功能：检测当前是否处于跟踪丢失状态。
        返回 True 如果任一条件满足:
          - 协方差迹超出阈值
          - 目标估计距离超出声呐量程
          - 连续多周期未收到 belief 更新且 bearing 也已过时
        """
        if not self.have_target:
            return True

        # 条件 1: 协方差增长
        if self.is_belief_fresh() and self.cov_trace() > self.cov_trace_threshold:
            return True

        # 条件 2: 估计距离超出声呐量程
        dist = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist > self.sonar_max_range:
            return True

        # 条件 3: 长时间无 belief 且无 bearing
        if not self.is_belief_fresh() and not self.is_bearing_fresh():
            self.consecutive_missed += 1
            if self.consecutive_missed >= self.lost_track_missed_updates:
                return True
        else:
            self.consecutive_missed = 0

        return False

    def check_track_recovered(self):
        """
        功能：检测是否已从跟踪丢失状态恢复。
        返回 True 如果:
          - 协方差迹低于恢复阈值
          - 目标在声呐量程内
          - 连续多个周期满足以上条件
        """
        if not self.have_target:
            return False

        cov_ok = not self.is_belief_fresh() or self.cov_trace() < self.recovery_cov_trace_threshold
        dist = math.hypot(self.target_x - self.x, self.target_y - self.y)
        range_ok = dist < self.sonar_max_range

        if cov_ok and range_ok:
            self.consecutive_good += 1
        else:
            self.consecutive_good = 0

        return self.consecutive_good >= self.recovery_min_consecutive_good

    def select_recovery_mode(self):
        """
        功能：根据当前传感器可用性选择恢复模式。
        优先: bearing_search > spiral > rotate
        """
        if self.is_bearing_fresh():
            return 'bearing_search'
        # 检查是否在边界附近，若靠近边界则旋转而非螺旋
        dist_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist_to_target < 8.0:
            return 'spiral'
        return 'rotate'

    def reset_recovery_state(self):
        """重置恢复相关状态变量。"""
        self.track_lost = False
        self.track_lost_time = None
        self.recovery_mode = 'rotate'
        self.recovery_start_time = None
        self.recovery_mode_start_time = None
        self.consecutive_good = 0
        self.spiral_accumulated_angle = 0.0

    def _next_recovery_mode(self, current_mode):
        """Return the next available recovery behavior in a cyclic sequence."""
        modes = ['rotate', 'spiral']
        if self.is_bearing_fresh():
            modes.append('bearing_search')
        if current_mode not in modes:
            return modes[0]
        return modes[(modes.index(current_mode) + 1) % len(modes)]

    def publish_recovery_status(self):
        """Publish [active, mode, elapsed_s, recovery_count] for evaluation."""
        mode_codes = {'rotate': 1.0, 'spiral': 2.0, 'bearing_search': 3.0}
        elapsed = 0.0
        if self.track_lost and self.recovery_start_time is not None:
            elapsed = max(0.0, (rospy.Time.now() - self.recovery_start_time).to_sec())
        msg = Float32MultiArray()
        msg.data = [
            1.0 if self.track_lost else 0.0,
            mode_codes.get(self.recovery_mode, 0.0) if self.track_lost else 0.0,
            elapsed,
            float(self.recovery_count),
        ]
        self.recovery_status_pub.publish(msg)

    # ========================================================================
    #  恢复行为执行
    # ========================================================================

    def execute_recovery(self):
        """
        功能：根据当前恢复模式生成控制指令。
        """
        # 定期切换恢复模式以避免卡死
        if self.recovery_mode_start_time is not None:
            mode_elapsed = (rospy.Time.now() - self.recovery_mode_start_time).to_sec()
            if mode_elapsed > self.recovery_mode_switch_time:
                self.recovery_mode = self._next_recovery_mode(self.recovery_mode)
                self.recovery_mode_start_time = rospy.Time.now()
                rospy.loginfo('[RecoveryMPC] switching recovery mode to: %s', self.recovery_mode)

        mode = self.recovery_mode
        if mode == 'rotate':
            self._recovery_rotate()
        elif mode == 'spiral':
            self._recovery_spiral()
        elif mode == 'bearing_search':
            self._recovery_bearing_search()
        else:
            self._recovery_rotate()

    def _recovery_rotate(self):
        """原地旋转搜索：零前进速度，固定偏航角速率。"""
        yaw_cmd = wrap_pi(self.last_cmd_yaw + self.recovery_rotate_yaw_rate / self.control_rate)
        self.publish_cmd(0.0, 0.0, yaw_cmd)
        self.last_cmd_yaw = yaw_cmd

    def _recovery_spiral(self):
        """螺旋搜索：低速前进 + 缓慢偏航，逐步扩大搜索半径。"""
        self.spiral_accumulated_angle += self.recovery_spiral_yaw_rate / self.control_rate
        yaw_cmd = wrap_pi(self.last_cmd_yaw + self.recovery_spiral_yaw_rate / self.control_rate)
        # 前向速度随螺旋半径增长而略微增加
        u_cmd = self.recovery_spiral_speed
        self.publish_cmd(u_cmd, 0.0, yaw_cmd)
        self.last_cmd_yaw = yaw_cmd

    def _recovery_bearing_search(self):
        """方位制导搜索：转向最新声学方位，低速前进。"""
        if self.latest_bearing is None:
            self._recovery_rotate()
            return

        # 世界系下的目标方位 = 机器人航向 + 声学相对方位
        yaw_cmd = wrap_pi(self.psi + self.latest_bearing)
        u_cmd = self.recovery_bearing_search_speed

        # 侧向速度辅助对准
        yaw_err = wrap_pi(yaw_cmd - self.last_cmd_yaw)
        v_cmd = self.recovery_bearing_search_lateral_gain * yaw_err
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))

        self.publish_cmd(u_cmd, v_cmd, yaw_cmd)
        self.last_cmd_yaw = yaw_cmd

    # ========================================================================
    #  MPC 求解
    # ========================================================================

    def _shift_warm_start(self):
        if self.N <= 1:
            return
        self.U_init = self.U_init[1:] + [self.U_init[-1]]
        self.V_init = self.V_init[1:] + [self.V_init[-1]]
        self.YAW_init = self.YAW_init[1:] + [self.YAW_init[-1]]

    def _reset_warm_start(self):
        """恢复后重置 warm start 序列。"""
        yaw_hold = self.last_cmd_yaw
        self.U_init = [0.0] * self.N
        self.V_init = [0.0] * self.N
        self.YAW_init = [yaw_hold] * self.N
        self.last_lam_x = None
        self.last_lam_g = None

    def solve_mpc(self):
        """
        功能：根据当前状态求解一次 CE-NMPC 控制问题。
        """
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tolerance:
            yaw_hold = self.target_yaw if self.use_goal_yaw else self.last_cmd_yaw
            self.U_init = [0.0] * self.N
            self.V_init = [0.0] * self.N
            self.YAW_init = [yaw_hold] * self.N
            return 0.0, 0.0, yaw_hold, True

        if self.use_goal_yaw and dist < self.yaw_switch_dist:
            yaw_ref = self.target_yaw
        else:
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
            self.x, self.y, self.psi, self.u, self.v, self.r,
            self.target_x, self.target_y, self.target_yaw,
            self.last_cmd_u, self.last_cmd_v, self.last_cmd_yaw,
            1.0 if self.use_goal_yaw else 0.0,
        ] + [value for item in self.avoidance.obstacles_world(self.x, self.y, self.psi) for value in item]

        args = {'x0': x0, 'lbx': self.lbx, 'ubx': self.ubx, 'lbg': self.lbg, 'ubg': self.ubg, 'p': p}
        if self.last_lam_x is not None and self.last_lam_g is not None:
            args['lam_x0'] = self.last_lam_x
            args['lam_g0'] = self.last_lam_g

        t_start = time.time()
        sol = self.solver(**args)
        solver_ms = 1000.0 * (time.time() - t_start)
        w_opt = sol['x'].full().flatten().tolist()
        self.last_lam_x = sol['lam_x']
        self.last_lam_g = sol['lam_g']

        # 发布求解耗时
        try:
            stats = self.solver.stats()
            iters = stats.get('iter_count', 0)
            feasible = 1.0 if stats.get('return_status', '') == 'Solve_Succeeded' else 0.0
        except Exception:
            iters = 0
            feasible = 1.0
        tm_msg = Float32MultiArray()
        tm_msg.data = [float(solver_ms), float(iters), feasible]
        self.solver_time_pub.publish(tm_msg)

        self.U_init = w_opt[0:self.N]
        self.V_init = w_opt[self.N:2 * self.N]
        self.YAW_init = [wrap_pi(a) for a in w_opt[2 * self.N:3 * self.N]]
        return self.U_init[0], self.V_init[0], self.YAW_init[0], False

    # ========================================================================
    #  指令发布
    # ========================================================================

    def publish_cmd(self, vx_body, vy_body, yaw):
        vx_body, vy_body, yaw, _ = self.avoidance.filter_command(vx_body, vy_body, yaw, self.u_max, self.v_max)
        vel_msg = Twist()
        vel_msg.linear.x = vx_body
        vel_msg.linear.y = vy_body
        vel_msg.linear.z = 0.0
        yaw_msg = Float32()
        yaw_msg.data = yaw
        self.vel_pub.publish(vel_msg)
        self.yaw_pub.publish(yaw_msg)

    # ========================================================================
    #  主循环（状态机）
    # ========================================================================

    def run(self):
        """
        功能：执行带恢复状态机的 MPC 跟踪主循环。
        正常模式 → CE-NMPC 求解 → 发布指令。
        跟踪丢失 → 脚本化恢复 → 检查恢复条件 → 回到正常模式。
        """
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            # --- 等待必要数据 ---
            if not self.have_odom:
                rospy.logwarn_throttle(2.0, '[RecoveryMPC] waiting for odom...')
                rate.sleep()
                continue

            if not self._refresh_boundary_constraints():
                self.publish_cmd(0.0, 0.0, self.last_cmd_yaw)
                rate.sleep()
                continue

            if not self.have_target and not self.track_lost:
                rospy.logwarn_throttle(2.0, '[RecoveryMPC] waiting for target; running search behavior...')
                if self.is_bearing_fresh():
                    self._recovery_bearing_search()
                else:
                    self._recovery_rotate()
                rate.sleep()
                continue

            # --- 跟踪丢失检测 ---
            if not self.track_lost:
                if self.detect_track_loss():
                    self.track_lost = True
                    self.track_lost_time = rospy.Time.now()
                    self.recovery_start_time = rospy.Time.now()
                    self.recovery_mode = self.select_recovery_mode()
                    self.recovery_mode_start_time = rospy.Time.now()
                    self.recovery_count += 1
                    self._reset_warm_start()
                    rospy.logwarn(
                        '[RecoveryMPC] TRACK LOST! cov_trace=%.2f, mode=%s',
                        self.cov_trace(),
                        self.recovery_mode,
                    )

            # --- 状态机执行 ---
            if self.track_lost:
                # 检查恢复条件
                if self.check_track_recovered():
                    rospy.loginfo(
                        '[RecoveryMPC] Track recovered! cov_trace=%.2f, recovery_duration=%.1fs',
                        self.cov_trace(),
                        (rospy.Time.now() - self.recovery_start_time).to_sec() if self.recovery_start_time else 0.0,
                    )
                    self.reset_recovery_state()
                    # 执行一次正常 MPC（fall through）
                else:
                    # 仍在恢复中：检查是否超过最大恢复时间
                    if self.recovery_start_time is not None:
                        elapsed = (rospy.Time.now() - self.recovery_start_time).to_sec()
                        if elapsed > self.recovery_max_duration:
                            rospy.logwarn(
                                '[RecoveryMPC] Recovery timeout after %.1fs, forcing re-init.',
                                elapsed,
                            )
                            # 强制重置状态，等待新数据
                            self.reset_recovery_state()
                            self.have_target = False
                            rate.sleep()
                            continue

                    self.execute_recovery()
                    self.publish_recovery_status()

                    rospy.loginfo_throttle(
                        2.0,
                        '[RecoveryMPC] RECOVERY mode=%s, cov_trace=%.2f, elapsed=%.1fs',
                        self.recovery_mode,
                        self.cov_trace(),
                        (rospy.Time.now() - self.recovery_start_time).to_sec() if self.recovery_start_time else 0.0,
                    )
                    rate.sleep()
                    continue

            # --- 正常 CE-NMPC 跟踪 ---
            try:
                vx_cmd, vy_cmd, yaw_cmd, reached = self.solve_mpc()
            except Exception as exc:
                rospy.logwarn_throttle(1.0, '[RecoveryMPC] solver failed: %s', str(exc))
                vx_cmd, vy_cmd, yaw_cmd, reached = 0.0, 0.0, self.last_cmd_yaw, False

            self.last_cmd_u = vx_cmd
            self.last_cmd_v = vy_cmd
            self.last_cmd_yaw = yaw_cmd
            self.publish_cmd(vx_cmd, vy_cmd, yaw_cmd)
            self.publish_recovery_status()

            d = math.hypot(self.target_x - self.x, self.target_y - self.y)
            if reached:
                rospy.loginfo_throttle(2.0, '[RecoveryMPC] target reached within standoff tolerance.')
            else:
                rospy.loginfo_throttle(
                    1.0,
                    '[RecoveryMPC] target=(%.2f, %.2f), dist=%.2f, cmd=(%.2f, %.2f), yaw=%.2f, cov_tr=%.2f'
                    % (self.target_x, self.target_y, d, vx_cmd, vy_cmd, yaw_cmd, self.cov_trace()),
                )
            rate.sleep()


if __name__ == '__main__':
    try:
        RecoveringDynamicTargetMPCTracker().run()
    except rospy.ROSInterruptException:
        pass
