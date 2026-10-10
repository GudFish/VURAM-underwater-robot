#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：接收高层速度与偏航命令，并将其转换为 MAVROS 原始速度和姿态设定值。
输入：通过 ROS 参数、状态话题、里程计话题和控制输入话题接收运行配置与控制量。
输出：发布速度、姿态或位置控制相关消息，并维护控制器内部状态。
"""
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rospy
import math
import tf
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Quaternion, Twist
from std_msgs.msg import Float32

class CombinedPlanner:
    """
    功能：桥接高层速度与偏航命令到 MAVROS 原始控制接口。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('combined_planner', anonymous=True)

        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/target_velocity')
        self.target_yaw_topic = rospy.get_param('~target_yaw_topic', '/target_yaw')
        self.setpoint_topic = rospy.get_param('~setpoint_topic', '/mavros/setpoint_raw/local')
        self.attitude_topic = rospy.get_param('~attitude_topic', '/mavros/setpoint_raw/attitude')
        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.timeout = rospy.get_param('~timeout', 0.5)
        
        # ---- 订阅 ----
        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb)
        self.cmd_vel_sub = rospy.Subscriber(self.cmd_vel_topic, Twist, self.cmd_vel_cb)
        self.target_yaw_sub = rospy.Subscriber(self.target_yaw_topic, Float32, self.target_yaw_cb)

        # ---- 发布 ----
        self.vel_pub = rospy.Publisher(self.setpoint_topic, PositionTarget, queue_size=10)
        self.att_pub = rospy.Publisher(self.attitude_topic, AttitudeTarget, queue_size=10)

        # ---- 状态变量 ----
        self.current_state = State()
        self.current_odom = Odometry()
        
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw = math.pi/2
        
        # 记录最后接收时间
        self.last_vel_time = rospy.Time.now()
        self.last_yaw_time = rospy.Time.now()
        
        self.vel_received = False
        self.yaw_received = False

        # 定时器：可配置频率
        period = 1.0 / max(self.control_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.control_loop)

        rospy.loginfo(
            "Combined Planner waiting for external velocity/yaw commands at %.1f Hz...",
            self.control_rate,
        )

    def state_cb(self, msg):
        """
        功能：执行 `state_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.current_state = msg

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.current_odom = msg

    def cmd_vel_cb(self, msg):
        """
        功能：执行 `cmd_vel_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_vz = msg.linear.z
        self.last_vel_time = rospy.Time.now()
        self.vel_received = True

    def target_yaw_cb(self, msg):
        """
        功能：执行 `target_yaw_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.target_yaw = msg.data
        self.last_yaw_time = rospy.Time.now()
        self.yaw_received = True

    def control_loop(self, event):
        # 1. 安全检查
        """
        功能：执行 `control_loop` 对应的核心处理逻辑。
        输入：`event` 为 ROS 定时器事件；运行时还依赖对象内部状态与缓存数据。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if not self.current_state.connected:
            rospy.logwarn_throttle(5, "Waiting for connection...")
            return
        if self.current_state.mode != "GUIDED" and self.current_state.mode != "ALT_HOLD":
            rospy.logwarn_throttle(5, "Waiting for GUIDED mode, or Depth Hold mode...")
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(5, "Waiting for arming...")
            return
        
        # 2. 超时检查
        now = rospy.Time.now()
        
        # 速度超时：超过1秒未更新则置零
        if (now - self.last_vel_time).to_sec() > self.timeout:
            self.target_vx = 0.0
            self.target_vy = 0.0
            self.target_vz = 0.0
            rospy.logwarn_throttle(5, "Velocity command timeout! Setting velocity to zero.")
        
        # 偏航角：即使超时也保持最后一次接收的值，不做额外处理

        # 3. 构建并发布速度指令 (SET_POSITION_TARGET_LOCAL_NED)
        vel_msg = PositionTarget()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        
        vel_msg.type_mask = (PositionTarget.IGNORE_PX |
                             PositionTarget.IGNORE_PY |
                             PositionTarget.IGNORE_PZ |
                             PositionTarget.IGNORE_AFX |
                             PositionTarget.IGNORE_AFY |
                             PositionTarget.IGNORE_AFZ |
                             PositionTarget.IGNORE_YAW |
                             PositionTarget.IGNORE_YAW_RATE)
        
        vel_msg.velocity.x = self.target_vx
        vel_msg.velocity.y = self.target_vy
        vel_msg.velocity.z = self.target_vz
        self.vel_pub.publish(vel_msg)

        # 4. 构建并发布偏航指令 (SET_ATTITUDE_TARGET)
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        att_msg.type_mask = (AttitudeTarget.IGNORE_YAW_RATE | 
                             AttitudeTarget.IGNORE_PITCH_RATE | 
                             AttitudeTarget.IGNORE_ROLL_RATE)
        
        att_msg.body_rate = Vector3(0.0, 0.0, 0.0)
        quat = tf.transformations.quaternion_from_euler(0, 0, self.target_yaw)
        att_msg.orientation = Quaternion(*quat)
        att_msg.thrust = 0.0
        
        self.att_pub.publish(att_msg)

        rospy.loginfo_throttle(2, f"CMD: V_body=({self.target_vx:.2f}, {self.target_vy:.2f}, {self.target_vz:.2f}), Yaw={self.target_yaw:.2f}")

    def run(self):
        """
        功能：执行 `run` 对应的核心处理逻辑。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = CombinedPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
