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
        self.range_topic = rospy.get_param('~range_topic', '/target_acoustic_range')

        self.measurement_rate = rospy.get_param('~measurement_rate', 1.0)
        self.range_noise_std = rospy.get_param('~range_noise_std', 0.4)
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
        self.target_x = 0.0
        self.target_y = 0.0

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.target_sub = rospy.Subscriber(self.target_pose_topic, PoseStamped, self.target_cb, queue_size=1)
        self.range_pub = rospy.Publisher(self.range_topic, Float32, queue_size=10)

        period = 1.0 / max(self.measurement_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('[AcousticRangeSim] started. waiting for arm before publishing range on: %s', self.range_topic)

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            self.range_enabled = True
            rospy.loginfo('[AcousticRangeSim] vehicle armed. acoustic range simulation enabled.')

        if armed_falling and self.reset_on_disarm:
            self.range_enabled = False
            rospy.loginfo('[AcousticRangeSim] vehicle disarmed. acoustic range simulation disabled.')

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.have_robot = True

    def target_cb(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.have_target = True

    def timer_cb(self, _event):
        if not self.range_enabled:
            rospy.logwarn_throttle(2.0, '[AcousticRangeSim] waiting for vehicle arm...')
            return
        if not self.have_robot or not self.have_target:
            return

        true_range = math.hypot(self.target_x - self.robot_x, self.target_y - self.robot_y)
        if true_range > self.max_range:
            return
        if random.random() < self.drop_prob:
            return

        noisy_range = true_range + random.gauss(0.0, self.range_noise_std)
        msg = Float32()
        msg.data = max(0.0, noisy_range)
        self.range_pub.publish(msg)


if __name__ == '__main__':
    try:
        node = AcousticRangeSensorSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
