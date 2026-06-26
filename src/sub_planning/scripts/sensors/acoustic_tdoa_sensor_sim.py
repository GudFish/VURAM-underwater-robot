#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class AcousticTDOASensorSim:
    def __init__(self):
        rospy.init_node('acoustic_tdoa_sensor_sim', anonymous=True)

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
