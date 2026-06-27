#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：实现基于 IPOPT 的二维 MPC 轨迹规划器，用于静态目标跟踪控制。
输入：通过 ROS 参数以及目标、里程计、边界和感知话题接收规划输入。
输出：发布目标、速度、偏航、障碍物或规划结果，并维护优化器内部状态。
"""
import math
import casadi as ca
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def wrap_pi(angle):
    """
    功能：将输入角度归一化到 [-pi, pi] 区间。
    输入：`angle` 为该函数执行所需的输入参数。
    输出：返回归一化后的弧度角。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class IPOPTMPC2DPlanner:
    """
    功能：构建并求解二维 IPOPT MPC 问题，输出静态目标跟踪控制命令。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('mpc_2d_planner', anonymous=True)

        # Topics
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.goal_topic = rospy.get_param('~goal_topic', '/target_pose')
        self.rviz_goal_topic = rospy.get_param('~rviz_goal_topic', '/move_base_simple/goal')
        self.vel_topic = rospy.get_param('~vel_topic', '/target_velocity')
        self.yaw_topic = rospy.get_param('~yaw_topic', '/target_yaw')

        # Goal
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.use_goal_yaw = rospy.get_param('~use_goal_yaw', True)
        self.yaw_switch_dist = rospy.get_param('~yaw_switch_dist', 1.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.1)

        # MPC
        self.control_rate = rospy.get_param('~control_rate', 5.0)
        self.dt = rospy.get_param('~mpc_dt', 0.20)
        self.N = rospy.get_param('~horizon_steps', 15)

        # Limits
        self.u_max = rospy.get_param('~u_max', 0.8)
        self.v_max = rospy.get_param('~v_max', 0.6)
        self.r_cmd_max = rospy.get_param('~r_max', 0.6)

        self.du_step_max = rospy.get_param('~du_step_max', 0.25)
        self.dv_step_max = rospy.get_param('~dv_step_max', 0.20)
        self.dyaw_step_max = rospy.get_param('~dyaw_step_max', 0.30)

        # Virtual low-level mapping: cmd -> generalized force
        self.psi_kp = rospy.get_param('~psi_kp', 1.6)
        self.k_tau_u = rospy.get_param('~k_tau_u', 120.0)
        self.k_tau_v = rospy.get_param('~k_tau_v', 120.0)
        self.k_tau_r = rospy.get_param('~k_tau_r', 80.0)
        self.tau_u_max = rospy.get_param('~tau_u_max', 200.0)
        self.tau_v_max = rospy.get_param('~tau_v_max', 200.0)
        self.tau_r_max = rospy.get_param('~tau_r_max', 120.0)

        # Cost
        self.w_pos = rospy.get_param('~w_pos', 4.0)
        self.w_terminal = rospy.get_param('~w_terminal', 30.0)
        self.w_yaw = rospy.get_param('~w_yaw', 1.0)
        self.w_terminal_yaw = rospy.get_param('~w_terminal_yaw', 12.0)
        self.w_ctrl = rospy.get_param('~w_ctrl', 0.2)
        self.w_sway = rospy.get_param('~w_sway', 0.3)
        self.w_du = rospy.get_param('~w_du', 0.6)
        self.w_dyaw = rospy.get_param('~w_dyaw', 0.3)

        # Fossen params (shared yaml)
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

        # Runtime state
        self.have_odom = False
        self.target_received = False
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        self.last_cmd_u = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_yaw = 0.0

        # Warm start for IPOPT
        self.U_init = [0.0] * self.N
        self.V_init = [0.0] * self.N
        self.YAW_init = [0.0] * self.N

        # Build NLP solver once
        self._build_nlp()

        # ROS I/O
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_cb, queue_size=1)
        self.rviz_goal_sub = rospy.Subscriber(self.rviz_goal_topic, PoseStamped, self.goal_cb, queue_size=1)

        self.vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=10)
        self.yaw_pub = rospy.Publisher(self.yaw_topic, Float32, queue_size=10)

        rospy.loginfo('[MPC2D-IPOPT] started. N=%d dt=%.2f' % (self.N, self.dt))

    @staticmethod
    def _clip(val, vmin, vmax):
        """
        功能：对数值进行上下限裁剪，避免控制量或中间变量超界。
        输入：`val`、`vmin`、`vmax` 为该函数执行所需的输入参数。
        输出：返回裁剪后的数值。
        """
        return max(vmin, min(vmax, val))

    def _build_nlp(self):
        # Decision vars: sequence of [u_cmd, v_cmd, yaw_cmd]
        """
        功能：构建 MPC 或优化器所需的非线性规划模型。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        U = ca.SX.sym('U', self.N)
        V = ca.SX.sym('V', self.N)
        Y = ca.SX.sym('Y', self.N)
        W = ca.vertcat(U, V, Y)

        # Params: current state + goal + previous command
        # p = [x,y,psi,u,v,r, gx,gy,gyaw, use_gyaw, yaw_sw_dist, prev_u,prev_v,prev_y]
        P = ca.SX.sym('P', 14)

        x = P[0]
        y = P[1]
        psi = P[2]
        u = P[3]
        v = P[4]
        r = P[5]

        gx = P[6]
        gy = P[7]
        gyaw = P[8]
        use_gyaw = P[9]
        yaw_sw_dist = P[10]

        prev_u_cmd = P[11]
        prev_v_cmd = P[12]
        prev_yaw_cmd = P[13]

        # Mass matrix and inverse
        M = ca.DM([
            [self.m11, 0.0, 0.0],
            [0.0, self.m22, self.m23],
            [0.0, self.m23, self.m33],
        ])
        M_inv = ca.inv(M)

        J = 0
        g = []
        lbg = []
        ubg = []

        for k in range(self.N):
            u_cmd = U[k]
            v_cmd = V[k]
            yaw_cmd = Y[k]

            # yaw loop -> desired yaw rate
            e_psi = ca.atan2(ca.sin(yaw_cmd - psi), ca.cos(yaw_cmd - psi))
            r_cmd = self.psi_kp * e_psi
            r_cmd = ca.fmin(self.r_cmd_max, ca.fmax(-self.r_cmd_max, r_cmd))

            # cmd -> generalized forces
            tau_u = self.k_tau_u * (u_cmd - u)
            tau_v = self.k_tau_v * (v_cmd - v)
            tau_r = self.k_tau_r * (r_cmd - r)
            tau_u = ca.fmin(self.tau_u_max, ca.fmax(-self.tau_u_max, tau_u))
            tau_v = ca.fmin(self.tau_v_max, ca.fmax(-self.tau_v_max, tau_v))
            tau_r = ca.fmin(self.tau_r_max, ca.fmax(-self.tau_r_max, tau_r))

            # C(nu)
            C = ca.SX(3, 3)
            C[0, 2] = -self.m22 * v - self.m23 * r
            C[1, 2] = self.m11 * u
            C[2, 0] = self.m22 * v + self.m23 * r
            C[2, 1] = -self.m11 * u

            # D(nu)
            du = self.d11 + self.n11 * ca.fabs(u)
            dv = self.d22 + self.n22 * ca.fabs(v)
            dr = self.d33 + self.n33 * ca.fabs(r)
            d23 = self.d23 + self.n23 * ca.fabs(r)
            d32 = self.d32 + self.n32 * ca.fabs(v)
            D = ca.SX(3, 3)
            D[0, 0] = du
            D[1, 1] = dv
            D[1, 2] = d23
            D[2, 1] = d32
            D[2, 2] = dr

            nu = ca.vertcat(u, v, r)
            tau = ca.vertcat(tau_u, tau_v, tau_r)

            nu_dot = ca.mtimes(M_inv, (tau - ca.mtimes(C + D, nu)))
            u = u + nu_dot[0] * self.dt
            v = v + nu_dot[1] * self.dt
            r = r + nu_dot[2] * self.dt

            x_dot = ca.cos(psi) * u - ca.sin(psi) * v
            y_dot = ca.sin(psi) * u + ca.cos(psi) * v
            x = x + x_dot * self.dt
            y = y + y_dot * self.dt
            psi = psi + r * self.dt
            psi = ca.atan2(ca.sin(psi), ca.cos(psi))

            dx = gx - x
            dy = gy - y
            dist2 = dx * dx + dy * dy
            dist = ca.sqrt(dist2 + 1e-9)
            yaw_path = ca.atan2(dy, dx)
            yaw_ref = use_gyaw * ca.if_else(dist < yaw_sw_dist, gyaw, yaw_path) + (1.0 - use_gyaw) * yaw_path
            yaw_err = ca.atan2(ca.sin(yaw_ref - psi), ca.cos(yaw_ref - psi))

            du_cmd = u_cmd - prev_u_cmd
            dv_cmd = v_cmd - prev_v_cmd
            dyaw_cmd = ca.atan2(ca.sin(yaw_cmd - prev_yaw_cmd), ca.cos(yaw_cmd - prev_yaw_cmd))

            J = J + self.w_pos * dist2
            J = J + self.w_yaw * yaw_err * yaw_err
            J = J + self.w_ctrl * (u_cmd * u_cmd + v_cmd * v_cmd)
            J = J + self.w_sway * (v * v)
            J = J + self.w_du * (du_cmd * du_cmd + dv_cmd * dv_cmd)
            J = J + self.w_dyaw * (dyaw_cmd * dyaw_cmd)

            prev_u_cmd = u_cmd
            prev_v_cmd = v_cmd
            prev_yaw_cmd = yaw_cmd

            # hard slew constraints
            g.append(du_cmd)
            lbg.append(-self.du_step_max)
            ubg.append(self.du_step_max)
            g.append(dv_cmd)
            lbg.append(-self.dv_step_max)
            ubg.append(self.dv_step_max)
            g.append(dyaw_cmd)
            lbg.append(-self.dyaw_step_max)
            ubg.append(self.dyaw_step_max)

        dx_t = gx - x
        dy_t = gy - y
        J = J + self.w_terminal * (dx_t * dx_t + dy_t * dy_t)
        yaw_t_err = ca.atan2(ca.sin(gyaw - psi), ca.cos(gyaw - psi))
        J = J + use_gyaw * self.w_terminal_yaw * (yaw_t_err * yaw_t_err)

        nlp = {'x': W, 'f': J, 'g': ca.vertcat(*g), 'p': P}

        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': int(rospy.get_param('~ipopt_max_iter', 80)),
            'ipopt.tol': float(rospy.get_param('~ipopt_tol', 1e-3)),
            'ipopt.acceptable_tol': float(rospy.get_param('~ipopt_acceptable_tol', 5e-3)),
            'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbg = lbg
        self.ubg = ubg

        # variable bounds
        self.lbx = []
        self.ubx = []
        for _ in range(self.N):
            self.lbx.append(-self.u_max)
            self.ubx.append(self.u_max)
        for _ in range(self.N):
            self.lbx.append(-self.v_max)
            self.ubx.append(self.v_max)
        for _ in range(self.N):
            self.lbx.append(-math.pi)
            self.ubx.append(math.pi)

        # keep last primal/dual for warm start
        self.last_sol = None
        self.last_lam_x = None
        self.last_lam_g = None

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

    def goal_cb(self, msg):
        """
        功能：处理目标位姿消息并更新规划目标。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y

        q = msg.pose.orientation
        has_orientation = abs(q.w) + abs(q.x) + abs(q.y) + abs(q.z) > 1e-8
        if has_orientation:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.goal_yaw = math.atan2(siny_cosp, cosy_cosp)
        elif not self.target_received and self.have_odom:
            self.goal_yaw = self.psi

        self.target_received = True

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
        dx = self.goal_x - self.x
        dy = self.goal_y - self.y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tolerance:
            yaw_hold = self.goal_yaw if self.use_goal_yaw else self.psi
            self.U_init = [0.0] * self.N
            self.V_init = [0.0] * self.N
            self.YAW_init = [yaw_hold] * self.N
            return 0.0, 0.0, yaw_hold, True

        self._shift_warm_start()

        if self.use_goal_yaw and dist < self.yaw_switch_dist:
            yaw_ref = self.goal_yaw
        else:
            yaw_ref = math.atan2(dy, dx)

        # gentle attraction in warm start
        u_ref = self._clip(0.6 * dx, -self.u_max, self.u_max)
        v_ref = self._clip(0.6 * dy, -self.v_max, self.v_max)
        for k in range(self.N):
            self.U_init[k] = self._clip(0.7 * self.U_init[k] + 0.3 * u_ref, -self.u_max, self.u_max)
            self.V_init[k] = self._clip(0.7 * self.V_init[k] + 0.3 * v_ref, -self.v_max, self.v_max)
            self.YAW_init[k] = wrap_pi(0.7 * self.YAW_init[k] + 0.3 * yaw_ref)

        x0 = self.U_init + self.V_init + self.YAW_init

        p = [
            self.x, self.y, self.psi, self.u, self.v, self.r,
            self.goal_x, self.goal_y, self.goal_yaw,
            1.0 if self.use_goal_yaw else 0.0,
            self.yaw_switch_dist,
            self.last_cmd_u, self.last_cmd_v, self.last_cmd_yaw,
        ]

        args = {
            'x0': x0,
            'lbx': self.lbx,
            'ubx': self.ubx,
            'lbg': self.lbg,
            'ubg': self.ubg,
            'p': p,
        }
        if self.last_lam_x is not None and self.last_lam_g is not None:
            args['lam_x0'] = self.last_lam_x
            args['lam_g0'] = self.last_lam_g

        sol = self.solver(**args)

        w_opt = sol['x'].full().flatten().tolist()
        self.last_lam_x = sol['lam_x']
        self.last_lam_g = sol['lam_g']

        U_opt = w_opt[0:self.N]
        V_opt = w_opt[self.N:2 * self.N]
        Y_opt = [wrap_pi(a) for a in w_opt[2 * self.N:3 * self.N]]

        self.U_init = U_opt
        self.V_init = V_opt
        self.YAW_init = Y_opt

        return U_opt[0], V_opt[0], Y_opt[0], False

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

    def run(self):
        """
        功能：执行 `run` 对应的核心处理逻辑。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if not self.have_odom:
                rospy.logwarn_throttle(2.0, '[MPC2D-IPOPT] waiting for odom...')
                rate.sleep()
                continue
            if not self.target_received:
                rospy.loginfo_throttle(2.0, '[MPC2D-IPOPT] waiting for target pose...')
                rate.sleep()
                continue

            try:
                vx_cmd, vy_cmd, yaw_cmd, reached = self.solve_mpc()
            except Exception as e:
                rospy.logwarn_throttle(1.0, '[MPC2D-IPOPT] solver failed: %s' % str(e))
                vx_cmd, vy_cmd, yaw_cmd, reached = 0.0, 0.0, self.last_cmd_yaw, False

            self.last_cmd_u = vx_cmd
            self.last_cmd_v = vy_cmd
            self.last_cmd_yaw = yaw_cmd

            self.publish_cmd(vx_cmd, vy_cmd, yaw_cmd)

            if reached:
                rospy.loginfo_throttle(2.0, '[MPC2D-IPOPT] goal reached, holding position.')
            else:
                d = math.hypot(self.goal_x - self.x, self.goal_y - self.y)
                rospy.loginfo_throttle(
                    1.0,
                    '[MPC2D-IPOPT] goal=(%.2f, %.2f), dist=%.2f, cmd=(%.2f, %.2f), yaw=%.2f'
                    % (self.goal_x, self.goal_y, d, vx_cmd, vy_cmd, yaw_cmd),
                )

            rate.sleep()


if __name__ == '__main__':
    try:
        IPOPTMPC2DPlanner().run()
    except rospy.ROSInterruptException:
        pass
