#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：使用视线法（LOS）制导律对动态目标执行跟踪控制。
      与 belief-aware MPC 共享相同的 PDAF-EKF 估计器，但控制器采用
      简单的比例制导律，不依赖 NMPC 优化（基线方法 B2）。
输入：通过 ROS 参数以及里程计、belief、方位话题接收数据。
输出：发布目标速度（/target_velocity_xy）与偏航角（/target_yaw）。
"""
import math
import sys
from pathlib import Path

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from utils.sonar_obstacle_avoidance import SonarObstacleAvoidance


def wrap_pi(angle):
    """
    功能：将输入角度归一化到 [-pi, pi] 区间。
    输入：`angle` 为该函数执行所需的输入参数。
    输出：返回归一化后的弧度角。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class LOSTargetTracker:
    """
    功能：基于 LOS 制导律对动态目标执行跟踪，支持自适应站位距离与搜索行为。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('los_target_tracker', anonymous=True)

        # --- 话题接口参数 ---
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.vel_topic = rospy.get_param('~vel_topic', '/target_velocity_xy')
        self.yaw_topic = rospy.get_param('~yaw_topic', '/target_yaw')

        # --- 控制频率 ---
        self.control_rate = float(rospy.get_param('~control_rate', 5.0))

        # --- 跟踪参数 ---
        self.desired_standoff = float(rospy.get_param('~desired_standoff', 3.0))
        self.goal_tolerance = float(rospy.get_param('~goal_tolerance', 0.8))

        # --- LOS 制导增益 ---
        self.los_range_kp = float(rospy.get_param('~los_range_kp', 0.5))
        self.los_min_speed = float(rospy.get_param('~los_min_speed', 0.10))
        self.u_max = float(rospy.get_param('~u_max', 0.8))
        self.v_max = float(rospy.get_param('~v_max', 0.3))

        # --- 自适应站位距离（基于不确定性）---
        self.use_adaptive_standoff = bool(rospy.get_param('~use_adaptive_standoff', True))
        self.exploration_standoff = float(rospy.get_param('~exploration_standoff', 4.5))
        self.unc_low = float(rospy.get_param('~unc_low', 1.0))
        self.unc_high = float(rospy.get_param('~unc_high', 8.0))
        self.sigma_min = float(rospy.get_param('~sigma_min', 0.1))
        self.unc_pos_gain = float(rospy.get_param('~unc_pos_gain', 1.0))
        self.unc_vel_gain = float(rospy.get_param('~unc_vel_gain', 0.4))

        # --- 声呐 FOV 参数（用于偏航对准目标）---
        self.sonar_fov_deg = float(rospy.get_param('~sonar_fov_deg', 130.0))
        self.sonar_max_range = float(rospy.get_param('~sonar_max_range', 30.0))

        # --- 搜索行为参数 ---
        self.search_enable = bool(rospy.get_param('~search_enable', True))
        self.search_forward_speed = float(rospy.get_param('~search_forward_speed', 0.15))
        self.search_yaw_rate = float(rospy.get_param('~search_yaw_rate', 0.25))
        self.bearing_search_enable = bool(rospy.get_param('~bearing_search_enable', True))
        self.bearing_search_timeout = float(rospy.get_param('~bearing_search_timeout', 1.0))
        self.bearing_search_forward_min_scale = float(rospy.get_param('~bearing_search_forward_min_scale', 0.25))
        self.bearing_search_lateral_gain = float(rospy.get_param('~bearing_search_lateral_gain', 0.35))

        # --- 运行时状态 ---
        self.have_odom = False
        self.have_belief = False
        self.latest_bearing = None
        self.latest_bearing_stamp = None

        # 机器人状态
        self.x = 0.0
        self.y = 0.0
        self.psi = 0.0
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0

        # 目标 belief 状态
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

        # 上一周期指令缓存
        self.last_cmd_u = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_yaw = 0.0

        # 搜索状态机
        self.search_yaw_accumulator = 0.0
        self.avoidance = SonarObstacleAvoidance()

        # --- 订阅者 ---
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.belief_sub = rospy.Subscriber(self.belief_topic, Odometry, self.belief_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)

        # --- 发布者 ---
        self.vel_pub = rospy.Publisher(self.vel_topic, Twist, queue_size=10)
        self.yaw_pub = rospy.Publisher(self.yaw_topic, Float32, queue_size=10)

        rospy.loginfo('[LOSTracker] started. belief topic: %s', self.belief_topic)

    # ========================================================================
    #  回调方法
    # ========================================================================

    def odom_cb(self, msg):
        """
        功能：处理机器人里程计消息，更新位姿与速度状态。
        输入：`msg` 为 Odometry 类型的 ROS 消息。
        输出：无显式返回；更新内部状态。
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
        功能：处理目标 belief 消息，提取位置、速度与协方差信息。
        输入：`msg` 为 Odometry 类型的 ROS 消息。
        输出：无显式返回；更新内部状态。
        """
        self.target_x = msg.pose.pose.position.x
        self.target_y = msg.pose.pose.position.y
        self.target_vx = msg.twist.twist.linear.x
        self.target_vy = msg.twist.twist.linear.y

        cov = msg.pose.covariance
        self.target_cov_xx = float(cov[0])
        self.target_cov_xy = float(cov[1])
        self.target_cov_yy = float(cov[7])
        self.target_cov_vxvx = float(cov[21])
        self.target_cov_vxvy = float(cov[22])
        self.target_cov_vyvy = float(cov[28])
        self.have_belief = True

    def bearing_cb(self, msg):
        """
        功能：缓存最新的声学方位角测量。
        输入：`msg` 为 Float32 类型的 ROS 消息。
        输出：无显式返回；更新内部状态。
        """
        self.latest_bearing = msg.data
        self.latest_bearing_stamp = rospy.Time.now()

    # ========================================================================
    #  不确定性计算
    # ========================================================================

    def belief_scalar_uncertainty(self):
        """
        功能：从 belief 协方差中提取标量不确定度，用于自适应站位距离。
        输入：无显式输入参数；依赖对象缓存的协方差状态。
        输出：返回标量不确定度 sigma。
        """
        pos_unc = max(0.0, self.target_cov_xx + self.target_cov_yy)
        vel_unc = max(0.0, self.target_cov_vxvx + self.target_cov_vyvy)
        return self.unc_pos_gain * pos_unc + self.unc_vel_gain * vel_unc

    # ========================================================================
    #  LOS 制导计算
    # ========================================================================

    def compute_los_command(self):
        """
        功能：根据目标相对位置计算 LOS 制导指令（前向速度 + 偏航角）。
        输入：无显式输入参数；依赖对象缓存的机器人与目标状态。
        输出：返回 (u_cmd, v_cmd, yaw_cmd) 三元组。
        """
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)

        # 在目标极近时保持上一周期偏航角
        if dist < self.goal_tolerance:
            return 0.0, 0.0, self.last_cmd_yaw

        # LOS 制导：偏航角指向目标
        psi_desired = math.atan2(dy, dx)

        # 自适应站位距离（与 proposed 方法相同的 alpha 混合）
        if self.use_adaptive_standoff:
            sigma = max(self.sigma_min, self.belief_scalar_uncertainty())
            span = max(self.unc_high - self.unc_low, 1e-6)
            alpha = max(0.0, min(1.0, (sigma - self.unc_low) / span))
            standoff = (1.0 - alpha) * self.desired_standoff + alpha * self.exploration_standoff
        else:
            standoff = self.desired_standoff

        # 前向速度与站位误差成比例
        range_error = dist - standoff
        u_cmd = self.los_range_kp * range_error
        u_cmd = max(-self.u_max, min(self.u_max, u_cmd))

        # 确保最小前向速度以维持机动能力
        if abs(range_error) > self.goal_tolerance and abs(u_cmd) < self.los_min_speed:
            u_cmd = self.los_min_speed if range_error > 0 else -self.los_min_speed

        # 侧向速度用于辅助声呐 FOV 对准（如有目标横向偏移）
        x_body = math.cos(self.psi) * dx + math.sin(self.psi) * dy
        y_body = -math.sin(self.psi) * dx + math.cos(self.psi) * dy
        v_cmd = 0.15 * y_body / max(dist, 1e-3)
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))

        return u_cmd, v_cmd, psi_desired

    # ========================================================================
    #  搜索行为
    # ========================================================================

    def is_bearing_fresh(self):
        """
        功能：检查最近一次声学方位测量是否在有效期内。
        输入：无显式输入参数；依赖对象缓存的 bearing 时间戳。
        输出：返回 True 如果 bearing 数据新鲜。
        """
        if self.latest_bearing_stamp is None:
            return False
        elapsed = (rospy.Time.now() - self.latest_bearing_stamp).to_sec()
        return elapsed < self.bearing_search_timeout

    def publish_search_cmd(self):
        """
        功能：发布旋转搜索指令（无 belief 且无 bearing 时使用）。
        输入：无显式输入参数。
        输出：无显式返回；发布速度与偏航角指令。
        """
        self.search_yaw_accumulator += self.search_yaw_rate / self.control_rate
        yaw_cmd = wrap_pi(self.last_cmd_yaw + self.search_yaw_rate / self.control_rate)
        self.publish_cmd(self.search_forward_speed, 0.0, yaw_cmd)
        self.last_cmd_yaw = yaw_cmd

    def publish_bearing_search_cmd(self):
        """
        功能：发布方位制导搜索指令（有 bearing 但无 belief 时使用）。
        输入：无显式输入参数。
        输出：无显式返回；发布速度与偏航角指令。
        """
        if self.latest_bearing is None:
            self.publish_search_cmd()
            return

        # 偏航角指向声学方位方向（世界系）
        yaw_cmd = wrap_pi(self.psi + self.latest_bearing)

        # 前向速度按方位搜索参数缩放
        u_cmd = max(
            self.bearing_search_forward_min_scale * self.u_max,
            self.search_forward_speed,
        )

        # 侧向速度用于对准声学方位
        yaw_err = wrap_pi(yaw_cmd - self.last_cmd_yaw)
        v_cmd = self.bearing_search_lateral_gain * yaw_err
        v_cmd = max(-self.v_max, min(self.v_max, v_cmd))

        self.publish_cmd(u_cmd, v_cmd, yaw_cmd)
        self.last_cmd_yaw = yaw_cmd

    # ========================================================================
    #  指令发布
    # ========================================================================

    def publish_cmd(self, vx_body, vy_body, yaw):
        """
        功能：发布求解得到的速度与偏航控制命令。
        输入：`vx_body` 前向速度、`vy_body` 侧向速度、`yaw` 偏航角。
        输出：无显式返回；通过发布者输出控制指令。
        """
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
    #  主循环
    # ========================================================================

    def run(self):
        """
        功能：执行 LOS 跟踪器的主控制循环。
        输入：无显式输入参数；依赖对象内部状态与订阅数据。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if not self.have_odom:
                rospy.logwarn_throttle(2.0, '[LOSTracker] waiting for odom...')
                rate.sleep()
                continue

            # 无 belief 时进入搜索模式
            if not self.have_belief:
                if self.bearing_search_enable and self.is_bearing_fresh():
                    rospy.loginfo_throttle(2.0, '[LOSTracker] bearing-guided search (no belief)')
                    self.publish_bearing_search_cmd()
                elif self.search_enable:
                    rospy.loginfo_throttle(2.0, '[LOSTracker] rotation search (no belief, no bearing)')
                    self.publish_search_cmd()
                else:
                    self.publish_cmd(0.0, 0.0, self.last_cmd_yaw)
                rate.sleep()
                continue

            # 正常 LOS 跟踪
            u_cmd, v_cmd, yaw_cmd = self.compute_los_command()
            self.last_cmd_u = u_cmd
            self.last_cmd_v = v_cmd
            self.last_cmd_yaw = yaw_cmd
            self.publish_cmd(u_cmd, v_cmd, yaw_cmd)

            d = math.hypot(self.target_x - self.x, self.target_y - self.y)
            rospy.loginfo_throttle(
                1.0,
                '[LOSTracker] target=(%.2f, %.2f), dist=%.2f, cmd=(%.2f, %.2f), yaw=%.2f'
                % (self.target_x, self.target_y, d, u_cmd, v_cmd, yaw_cmd),
            )
            rate.sleep()


if __name__ == '__main__':
    try:
        LOSTargetTracker().run()
    except rospy.ROSInterruptException:
        pass
