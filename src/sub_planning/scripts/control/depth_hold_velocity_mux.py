#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：在平面速度指令基础上叠加定深控制，向下游控制链路输出完整三轴速度指令。
输入：通过 ROS 参数、状态话题、里程计话题和控制输入话题接收运行配置与控制量。
输出：发布速度、姿态或位置控制相关消息，并维护控制器内部状态。
"""
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry


class DepthHoldVelocityMux:
    """
    功能：将平面速度命令与定深控制输出进行融合，生成最终的三轴速度指令。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('depth_hold_velocity_mux', anonymous=True)

        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.input_vel_topic = rospy.get_param('~input_vel_topic', '/target_velocity_xy')
        self.output_vel_topic = rospy.get_param('~output_vel_topic', '/target_velocity')

        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.timeout = rospy.get_param('~timeout', 0.5)
        self.depth_kp = rospy.get_param('~depth_kp', 0.9)
        self.depth_kd = rospy.get_param('~depth_kd', 0.6)
        self.max_abs_vz = rospy.get_param('~max_abs_vz', 0.25)
        self.depth_deadband = rospy.get_param('~depth_deadband', 0.03)
        self.vz_deadband = rospy.get_param('~vz_deadband', 0.02)
        self.reset_depth_on_rearm = rospy.get_param('~reset_depth_on_rearm', True)

        self.current_state = State()
        self.have_odom = False
        self.current_z = 0.0
        self.current_vz_enu = 0.0
        self.hold_z = None
        self.last_armed = False

        self.planar_cmd = Twist()
        self.last_cmd_time = rospy.Time(0)

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.input_sub = rospy.Subscriber(self.input_vel_topic, Twist, self.input_vel_cb, queue_size=1)
        self.output_pub = rospy.Publisher(self.output_vel_topic, Twist, queue_size=10)

        period = 1.0 / max(self.control_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('Depth-hold mux started. input=%s output=%s', self.input_vel_topic, self.output_vel_topic)

    @staticmethod
    def clip(value, vmin, vmax):
        """
        功能：对数值进行上下限裁剪。
        输入：`value`、`vmin`、`vmax` 为该函数执行所需的输入参数。
        输出：返回裁剪后的数值。
        """
        return max(vmin, min(vmax, value))

    def capture_hold_depth(self, reason):
        """
        功能：锁存当前深度作为定深参考。
        输入：`reason` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        if not self.have_odom:
            return
        self.hold_z = self.current_z
        rospy.loginfo('Depth reference latched from %s: z_ref=%.3f m', reason, self.hold_z)

    def state_cb(self, msg):
        """
        功能：执行 `state_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            if self.have_odom:
                self.capture_hold_depth('arming event')
            else:
                rospy.logwarn('Armed before odom available. Depth reference will be latched on first odom.')

        if armed_falling and self.reset_depth_on_rearm:
            self.hold_z = None

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.current_z = msg.pose.pose.position.z
        self.current_vz_enu = msg.twist.twist.linear.z
        self.have_odom = True

        if self.current_state.armed and self.hold_z is None:
            self.capture_hold_depth('first odom after arming')

    def input_vel_cb(self, msg):
        """
        功能：执行 `input_vel_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.planar_cmd = msg
        self.last_cmd_time = rospy.Time.now()

    def compute_vz_cmd(self):
        """
        功能：根据定深误差计算垂向速度命令。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回垂向速度命令及对应的深度误差。
        """
        if self.hold_z is None or not self.have_odom:
            return 0.0, 0.0

        # Odom z is ENU up-positive. BODY_NED velocity.z is down-positive.
        z_error_enu = self.hold_z - self.current_z
        if abs(z_error_enu) < self.depth_deadband and abs(self.current_vz_enu) < self.vz_deadband:
            return 0.0, z_error_enu

        vz_cmd = self.depth_kp * z_error_enu + self.depth_kd * self.current_vz_enu
        vz_cmd = self.clip(vz_cmd, -self.max_abs_vz, self.max_abs_vz)
        return vz_cmd, z_error_enu

    def timer_cb(self, _event):
        """
        功能：执行 `timer_cb` 对应的核心处理逻辑。
        输入：`_event` 为 ROS 定时器事件；运行时还依赖对象内部状态与缓存数据。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if not self.have_odom:
            rospy.logwarn_throttle(5.0, 'Depth-hold mux waiting for odom...')
            return

        now = rospy.Time.now()
        out = Twist()
        if (now - self.last_cmd_time).to_sec() <= self.timeout:
            out.linear.x = self.planar_cmd.linear.x
            out.linear.y = self.planar_cmd.linear.y
        else:
            rospy.logwarn_throttle(5.0, 'Planar velocity input timeout. Publishing zero vx, vy.')

        vz_cmd, z_error_enu = self.compute_vz_cmd()
        out.linear.z = vz_cmd
        self.output_pub.publish(out)

        rospy.loginfo_throttle(
            2.0,
            'Depth mux: in_xy=(%.2f, %.2f), out_z=%.2f, z_ref=%.2f, z=%.2f, z_err=%.2f'
            % (
                out.linear.x,
                out.linear.y,
                out.linear.z,
                self.hold_z if self.hold_z is not None else float('nan'),
                self.current_z,
                z_error_enu,
            ),
        )

    def run(self):
        """
        功能：执行 `run` 对应的核心处理逻辑。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        rospy.spin()


if __name__ == '__main__':
    try:
        node = DepthHoldVelocityMux()
        node.run()
    except rospy.ROSInterruptException:
        pass
