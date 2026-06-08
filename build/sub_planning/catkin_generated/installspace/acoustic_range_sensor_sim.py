#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class AcousticRangeSensorSim:
    def __init__(self):
        rospy.init_node('acoustic_range_sensor_sim', anonymous=True)

        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/dynamic_target/pose')
        self.left_range_topic = rospy.get_param('~left_range_topic', '/target_acoustic_range_left')
        self.right_range_topic = rospy.get_param('~right_range_topic', '/target_acoustic_range_right')

        self.measurement_rate = rospy.get_param('~measurement_rate', 1.0)
        self.range_noise_std = rospy.get_param('~range_noise_std', 0.4)
        self.left_range_noise_std = rospy.get_param('~left_range_noise_std', self.range_noise_std)
        self.right_range_noise_std = rospy.get_param('~right_range_noise_std', self.range_noise_std)
        self.left_beacon_x = rospy.get_param('~left_beacon_x', 0.0)
        self.left_beacon_y = rospy.get_param('~left_beacon_y', 0.65)
        self.right_beacon_x = rospy.get_param('~right_beacon_x', 0.0)
        self.right_beacon_y = rospy.get_param('~right_beacon_y', -0.65)
        self.max_range = rospy.get_param('~max_range', 80.0)
        self.drop_prob = rospy.get_param('~drop_prob', 0.15)
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)

        self.have_robot = False
        self.have_target = False
        self.current_state = State()
        self.last_armed = False
        self.range_enabled = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.target_sub = rospy.Subscriber(self.target_pose_topic, PoseStamped, self.target_cb, queue_size=1)
        self.left_range_pub = rospy.Publisher(self.left_range_topic, Float32, queue_size=10)
        self.right_range_pub = rospy.Publisher(self.right_range_topic, Float32, queue_size=10)

        period = 1.0 / max(self.measurement_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo(
            '[AcousticRangeSim] started. waiting for arm before publishing ranges on: %s and %s',
            self.left_range_topic,
            self.right_range_topic,
        )

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            self.range_enabled = True
            rospy.loginfo('[AcousticRangeSim] vehicle armed. dual acoustic range simulation enabled.')

        if armed_falling and self.reset_on_disarm:
            self.range_enabled = False
            rospy.loginfo('[AcousticRangeSim] vehicle disarmed. dual acoustic range simulation disabled.')

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_robot = True

    def target_cb(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.have_target = True

    def body_to_world(self, bx, by):
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return self.robot_x + c * bx - s * by, self.robot_y + s * bx + c * by

    def timer_cb(self, _event):
        if not self.range_enabled:
            rospy.logwarn_throttle(2.0, '[AcousticRangeSim] waiting for vehicle arm...')
            return
        if not self.have_robot or not self.have_target:
            return
        if random.random() < self.drop_prob:
            return

        left_x, left_y = self.body_to_world(self.left_beacon_x, self.left_beacon_y)
        right_x, right_y = self.body_to_world(self.right_beacon_x, self.right_beacon_y)
        left_true = math.hypot(self.target_x - left_x, self.target_y - left_y)
        right_true = math.hypot(self.target_x - right_x, self.target_y - right_y)
        if left_true > self.max_range or right_true > self.max_range:
            return

        left_msg = Float32()
        right_msg = Float32()
        left_msg.data = max(0.0, left_true + random.gauss(0.0, self.left_range_noise_std))
        right_msg.data = max(0.0, right_true + random.gauss(0.0, self.right_range_noise_std))
        self.left_range_pub.publish(left_msg)
        self.right_range_pub.publish(right_msg)


if __name__ == '__main__':
    try:
        node = AcousticRangeSensorSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
