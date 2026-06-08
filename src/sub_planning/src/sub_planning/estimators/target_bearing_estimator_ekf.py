#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from sub_planning.models.bearing_only_ekf import BearingOnlyCVEKF2D


class TargetBearingEstimatorEKF:
    def __init__(self):
        rospy.init_node('target_bearing_estimator_ekf', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/target_estimate/pose')
        self.target_twist_topic = rospy.get_param('~target_twist_topic', '/target_estimate/twist')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')

        self.filter_rate = float(rospy.get_param('~filter_rate', 10.0))
        self.bearing_noise_std = float(rospy.get_param('~bearing_noise_std', 0.08))
        self.min_bearing_range = float(rospy.get_param('~min_bearing_range', 0.25))
        self.process_noise_diag = [float(v) for v in rospy.get_param('~process_noise_diag', [0.4, 0.4, 0.8, 0.8])]
        self.init_mean = [float(v) for v in rospy.get_param('~init_mean', [8.0, 0.0, 0.3, 0.0])]
        self.init_cov_diag = [float(v) for v in rospy.get_param('~init_cov_diag', [36.0, 9.0, 2.0, 2.0])]
        self.init_range = float(rospy.get_param('~init_range', 8.0))
        self.init_range_var = float(rospy.get_param('~init_range_var', 36.0))
        self.init_cross_range_var = float(rospy.get_param('~init_cross_range_var', 4.0))
        self.use_prior_init = rospy.get_param('~use_prior_init', False)
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)

        self.current_state = State()
        self.last_armed = False
        self.have_robot = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_stamp = None
        self.pending_bearing = None

        self.init_covariance = np.diag(np.array(self.init_cov_diag, dtype=float))
        self.init_mean_np = np.array(self.init_mean, dtype=float).reshape(4)
        self.ekf = self.create_filter()

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher(self.target_twist_topic, TwistStamped, queue_size=10)
        self.belief_pub = rospy.Publisher(self.belief_topic, Odometry, queue_size=10)

        period = 1.0 / max(self.filter_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('[TargetBearingEKF] started. bearing=%s estimate=%s', self.bearing_topic, self.target_pose_topic)

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def create_filter(self):
        return BearingOnlyCVEKF2D(
            process_noise_diag=self.process_noise_diag,
            bearing_noise_var=self.bearing_noise_std ** 2,
            min_range=self.min_bearing_range,
        )

    def ekf_initialized(self):
        return self.ekf.initialized

    def reset_estimator(self, reason):
        self.last_stamp = None
        self.pending_bearing = None
        self.ekf = self.create_filter()
        rospy.loginfo('[TargetBearingEKF] estimator reset: %s', reason)

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            self.reset_estimator('armed, waiting for first TDOA bearing')

        if armed_falling and self.reset_on_disarm:
            self.reset_estimator('disarmed')

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_robot = True
        if self.use_prior_init and self.current_state.armed and not self.ekf_initialized():
            self.initialize_from_prior()

    def rotate_body_to_world(self, bx, by):
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return c * bx - s * by, s * bx + c * by

    def initialize_from_prior(self):
        rel_x, rel_y = self.rotate_body_to_world(self.init_mean_np[0], self.init_mean_np[1])
        vel_x, vel_y = self.rotate_body_to_world(self.init_mean_np[2], self.init_mean_np[3])
        state = np.array([self.robot_x + rel_x, self.robot_y + rel_y, vel_x, vel_y], dtype=float)
        self.ekf.set_state(state, self.init_covariance)
        rospy.loginfo('[TargetBearingEKF] relative prior initialized: (%.2f, %.2f, %.2f, %.2f)', state[0], state[1], state[2], state[3])

    def initialize_from_bearing(self, bearing):
        world_bearing = self.wrap_pi(self.robot_yaw + float(bearing))
        ux = math.cos(world_bearing)
        uy = math.sin(world_bearing)
        state = np.array(
            [
                self.robot_x + self.init_range * ux,
                self.robot_y + self.init_range * uy,
                self.init_mean_np[2],
                self.init_mean_np[3],
            ],
            dtype=float,
        )

        along = np.array([ux, uy], dtype=float)
        cross = np.array([-uy, ux], dtype=float)
        pos_cov = self.init_range_var * np.outer(along, along) + self.init_cross_range_var * np.outer(cross, cross)
        cov = np.array(self.init_covariance, dtype=float)
        cov[0:2, 0:2] = pos_cov
        self.ekf.set_state(state, cov)
        rospy.loginfo(
            '[TargetBearingEKF] first bearing initialized: bearing=%.2f range=%.2f state=(%.2f, %.2f, %.2f, %.2f)',
            bearing,
            self.init_range,
            state[0],
            state[1],
            state[2],
            state[3],
        )

    def bearing_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return

        bearing = self.wrap_pi(float(msg.data))
        if not self.ekf_initialized():
            self.initialize_from_bearing(bearing)

        self.pending_bearing = {
            'stamp': rospy.Time.now(),
            'bearing': bearing,
            'robot_xy': (self.robot_x, self.robot_y),
            'robot_yaw': self.robot_yaw,
        }

    def timer_cb(self, event):
        if not self.have_robot:
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for robot odom...')
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for vehicle arm...')
            return
        if not self.ekf_initialized():
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for first bearing measurement...')
            return

        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        if self.last_stamp is None:
            dt = 1.0 / max(self.filter_rate, 1e-3)
        else:
            dt = max((now - self.last_stamp).to_sec(), 1e-3)
        self.last_stamp = now

        self.ekf.predict(dt)
        innovation = 0.0
        nis = 0.0
        if self.pending_bearing is not None:
            obs = self.pending_bearing
            _, _, innovation = self.ekf.update(obs['robot_xy'], obs['robot_yaw'], obs['bearing'])
            _, _, nis = self.ekf.measurement_stats(obs['robot_xy'], obs['robot_yaw'], obs['bearing'])
            self.pending_bearing = None

        self.publish_estimate(now, innovation, nis)

    def publish_estimate(self, stamp, innovation, nis):
        state = self.ekf.x.copy()
        belief_cov = self.ekf.P.copy()
        vx = state[2]
        vy = state[3]
        yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-4 else 0.0

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
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
            '[TargetBearingEKF] est=(%.2f, %.2f), vel=(%.2f, %.2f), trP=%.3f, bearing_innov=%.3f, nis=%.2f',
            state[0],
            state[1],
            vx,
            vy,
            pos_trace,
            innovation,
            nis,
        )


if __name__ == '__main__':
    try:
        TargetBearingEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
