#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models.range_only_ekf import RangeOnlyCVEKF2D


class TargetStateEstimatorEKF:
    def __init__(self):
        rospy.init_node('target_state_estimator_ekf', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.range_topic = rospy.get_param('~range_topic', '/target_acoustic_range')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/target_estimate/pose')
        self.target_twist_topic = rospy.get_param('~target_twist_topic', '/target_estimate/twist')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')

        self.filter_rate = float(rospy.get_param('~filter_rate', 10.0))
        self.range_noise_std = float(rospy.get_param('~range_noise_std', 0.4))
        self.process_noise_diag = [float(v) for v in rospy.get_param('~process_noise_diag', [0.5, 0.5, 0.8, 0.8])]
        self.init_mean = [float(v) for v in rospy.get_param('~init_mean', [8.0, 0.0, 0.4, 0.0])]
        self.init_cov_diag = [float(v) for v in rospy.get_param('~init_cov_diag', [9.0, 9.0, 1.0, 1.0])]
        self.auto_init_range_angle = float(rospy.get_param('~auto_init_range_angle', 0.35))
        self.use_prior_init = rospy.get_param('~use_prior_init', False)
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)

        self.current_state = State()
        self.last_armed = False
        self.have_robot = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.last_stamp = None
        self.pending_range = None

        self.init_covariance = np.diag(np.array(self.init_cov_diag, dtype=float))
        self.init_mean_np = np.array(self.init_mean, dtype=float).reshape(4)
        self.ekf = self.create_filter()

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.range_sub = rospy.Subscriber(self.range_topic, Float32, self.range_cb, queue_size=5)
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher(self.target_twist_topic, TwistStamped, queue_size=10)
        self.belief_pub = rospy.Publisher(self.belief_topic, Odometry, queue_size=10)

        period = 1.0 / max(self.filter_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('[TargetEKF] started. estimate pose topic: %s', self.target_pose_topic)

    def create_filter(self):
        return RangeOnlyCVEKF2D(
            process_noise_diag=self.process_noise_diag,
            range_noise_var=float(self.range_noise_std) ** 2,
        )

    def ekf_initialized(self):
        return self.ekf.initialized

    def reset_estimator(self, reason):
        self.pending_range = None
        self.last_stamp = None
        self.ekf = self.create_filter()
        rospy.loginfo('[TargetEKF] estimator reset: %s', reason)

    def initialize_filter(self, state, log_prefix):
        self.ekf.set_state(state, self.init_covariance)
        rospy.loginfo(
            '%s initialized: (%.2f, %.2f, %.2f, %.2f)',
            log_prefix,
            state[0],
            state[1],
            state[2],
            state[3],
        )

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            self.reset_estimator('armed, waiting for first acoustic range')

        if armed_falling and self.reset_on_disarm:
            self.reset_estimator('disarmed')

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.have_robot = True
        if self.use_prior_init and self.current_state.armed and not self.ekf_initialized():
            prior_state = self.init_mean_np.copy()
            prior_state[0] += self.robot_x
            prior_state[1] += self.robot_y
            self.initialize_filter(prior_state, '[TargetEKF] relative prior')

    def range_cb(self, msg):
        if not self.current_state.armed:
            return
        self.pending_range = float(msg.data)
        if not self.ekf_initialized() and self.have_robot:
            heading = self.auto_init_range_angle
            init_state = np.array(
                [
                    self.robot_x + self.pending_range * math.cos(heading),
                    self.robot_y + self.pending_range * math.sin(heading),
                    self.init_mean_np[2],
                    self.init_mean_np[3],
                ],
                dtype=float,
            )
            self.initialize_filter(init_state, '[TargetEKF] first range')

    def timer_cb(self, event):
        if not self.have_robot:
            rospy.logwarn_throttle(2.0, '[TargetEKF] waiting for robot odom...')
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(2.0, '[TargetEKF] waiting for vehicle arm...')
            return
        if not self.ekf_initialized():
            rospy.logwarn_throttle(2.0, '[TargetEKF] waiting for first range measurement...')
            return

        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        if self.last_stamp is None:
            dt = 1.0 / max(self.filter_rate, 1e-3)
        else:
            dt = max((now - self.last_stamp).to_sec(), 1e-3)
        self.last_stamp = now

        self.ekf.predict(dt)
        innovation = 0.0
        if self.pending_range is not None:
            _, _, innovation = self.ekf.update((self.robot_x, self.robot_y), self.pending_range)
            self.pending_range = None

        state = self.ekf.x.copy()
        belief_cov = self.ekf.P.copy()
        vx = state[2]
        vy = state[3]
        yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-4 else 0.0

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = state[0]
        pose_msg.pose.position.y = state[1]
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(0.5 * yaw)
        pose_msg.pose.orientation.w = math.cos(0.5 * yaw)
        self.pose_pub.publish(pose_msg)

        twist_msg = TwistStamped()
        twist_msg.header = pose_msg.header
        twist_msg.twist.linear.x = vx
        twist_msg.twist.linear.y = vy
        twist_msg.twist.angular.z = 0.0
        self.twist_pub.publish(twist_msg)

        belief_msg = Odometry()
        belief_msg.header = pose_msg.header
        belief_msg.child_frame_id = 'target_belief'
        belief_msg.pose.pose = pose_msg.pose
        belief_msg.twist.twist = twist_msg.twist
        belief_msg.pose.covariance[0] = belief_cov[0, 0]
        belief_msg.pose.covariance[1] = belief_cov[0, 1]
        belief_msg.pose.covariance[6] = belief_cov[1, 0]
        belief_msg.pose.covariance[7] = belief_cov[1, 1]
        belief_msg.twist.covariance[0] = belief_cov[2, 2]
        belief_msg.twist.covariance[1] = belief_cov[2, 3]
        belief_msg.twist.covariance[6] = belief_cov[3, 2]
        belief_msg.twist.covariance[7] = belief_cov[3, 3]
        self.belief_pub.publish(belief_msg)

        pos_trace = belief_cov[0, 0] + belief_cov[1, 1]
        rospy.loginfo_throttle(
            2.0,
            '[TargetEKF] est=(%.2f, %.2f), vel=(%.2f, %.2f), trP=%.3f, innovation=%.3f',
            state[0],
            state[1],
            vx,
            vy,
            pos_trace,
            innovation,
        )


if __name__ == '__main__':
    try:
        TargetStateEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
