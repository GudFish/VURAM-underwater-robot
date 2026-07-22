#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：仿真多基阵 TDOA 声学测量，为目标方位估计器提供测试数据。
输入：通过 ROS 参数以及里程计、目标、图像或场景话题接收传感仿真与检测输入。
输出：发布仿真量测、候选目标或图像数据，并维护传感器内部缓存。
"""
import math
import random

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class AcousticTDOASensorSim:
    """
    功能：根据机器人与目标相对位置生成 TDOA 声学测量。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('acoustic_tdoa_sensor_sim', anonymous=True)

        # --- 随机种子（用于场景复现）---
        random_seed = rospy.get_param('~random_seed', -1)
        if random_seed >= 0:
            random.seed(int(random_seed))
            rospy.loginfo('[TDOASim] random seed set to %d', int(random_seed))

        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/dynamic_target/pose')
        self.tdoa_topic = rospy.get_param('~tdoa_topic', '/target_acoustic_tdoa')

        self.measurement_rate = float(rospy.get_param('~measurement_rate', 2.0))
        self.sound_speed = float(rospy.get_param('~sound_speed', 1500.0))
        self.tdoa_noise_std = float(rospy.get_param('~tdoa_noise_std', 2.0e-5))
        self.max_range = float(rospy.get_param('~max_range', 80.0))
        self.drop_prob = float(rospy.get_param('~drop_prob', 0.15))
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)
        self.beacon_ref = int(rospy.get_param('~beacon_ref', 0))

        self.beacons_body = self.load_beacons()
        self.beacon_ref = max(0, min(len(self.beacons_body) - 1, self.beacon_ref))

        self.have_robot = False
        self.have_target = False
        self.current_state = State()
        self.last_armed = False
        self.measurement_enabled = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.target_sub = rospy.Subscriber(self.target_pose_topic, PoseStamped, self.target_cb, queue_size=1)
        self.tdoa_pub = rospy.Publisher(self.tdoa_topic, Float32MultiArray, queue_size=10)

        period = 1.0 / max(self.measurement_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('[TDOASim] started. waiting for arm before publishing %s', self.tdoa_topic)

    def load_beacons(self):
        """
        功能：加载并转换阵列基元坐标配置。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        default = [[0.65, 0.60], [-0.65, 0.60], [0.65, -0.60], [-0.65, -0.60]]
        beacons = rospy.get_param('~beacons_body', default)
        if len(beacons) != 4:
            rospy.logwarn('[TDOASim] beacons_body must contain 4 [x,y] entries. Using default rectangle.')
            beacons = default
        out = []
        for item in beacons:
            out.append((float(item[0]), float(item[1])))
        return out

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
            self.measurement_enabled = True
            rospy.loginfo('[TDOASim] vehicle armed. TDOA simulation enabled.')
        if armed_falling and self.reset_on_disarm:
            self.measurement_enabled = False
            rospy.loginfo('[TDOASim] vehicle disarmed. TDOA simulation disabled.')
        self.last_armed = msg.armed

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_robot = True

    def target_cb(self, msg):
        """
        功能：处理目标位姿消息并更新目标状态缓存。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.have_target = True

    def body_to_world(self, bx, by):
        """
        功能：将机体系二维点转换到世界坐标系。
        输入：`bx`、`by` 为该函数执行所需的输入参数。
        输出：返回转换后的世界坐标分量。
        """
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return self.robot_x + c * bx - s * by, self.robot_y + s * bx + c * by

    def timer_cb(self, _event):
        """
        功能：执行 `timer_cb` 对应的核心处理逻辑。
        输入：`_event` 为 ROS 定时器事件；运行时还依赖对象内部状态与缓存数据。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if not self.measurement_enabled:
            rospy.logwarn_throttle(2.0, '[TDOASim] waiting for vehicle arm...')
            return
        if not self.have_robot or not self.have_target:
            return
        if random.random() < self.drop_prob:
            return

        arrivals = []
        for bx, by in self.beacons_body:
            wx, wy = self.body_to_world(bx, by)
            distance = math.hypot(self.target_x - wx, self.target_y - wy)
            if distance > self.max_range:
                return
            arrivals.append(distance / max(self.sound_speed, 1e-6) + random.gauss(0.0, self.tdoa_noise_std))

        ref_time = arrivals[self.beacon_ref]
        msg = Float32MultiArray()
        msg.data = [arrivals[i] - ref_time for i in range(len(arrivals)) if i != self.beacon_ref]
        self.tdoa_pub.publish(msg)


if __name__ == '__main__':
    try:
        AcousticTDOASensorSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
