#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


class TargetBearingWindowLS:
    def __init__(self):
        rospy.init_node('target_bearing_window_ls', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/target_estimate/pose')
        self.target_twist_topic = rospy.get_param('~target_twist_topic', '/target_estimate/twist')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')

        self.filter_rate = float(rospy.get_param('~filter_rate', 10.0))
        self.window_size = int(rospy.get_param('~window_size', 16))
        self.min_samples = int(rospy.get_param('~min_samples', 5))
        self.max_sample_age = float(rospy.get_param('~max_sample_age', 12.0))
        self.bearing_noise_std = float(rospy.get_param('~bearing_noise_std', 0.10))
        self.lm_max_iter = int(rospy.get_param('~lm_max_iter', 12))
        self.lm_tol = float(rospy.get_param('~lm_tol', 1e-5))
        self.lm_lambda = float(rospy.get_param('~lm_lambda', 1e-3))
        self.robust_huber_delta = float(rospy.get_param('~robust_huber_delta', 0.25))
        self.reg_position = float(rospy.get_param('~reg_position', 1e-6))
        self.reg_velocity = float(rospy.get_param('~reg_velocity', 1e-4))
        self.max_speed = float(rospy.get_param('~max_speed', 2.0))
        self.max_distance = float(rospy.get_param('~max_distance', 100.0))
        self.covariance_cap = float(rospy.get_param('~covariance_cap', 1e4))
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)

        self.current_state = State()
        self.last_armed = False
        self.have_robot = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.samples = []
        self.last_solution = None

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher(self.target_twist_topic, TwistStamped, queue_size=10)
        self.belief_pub = rospy.Publisher(self.belief_topic, Odometry, queue_size=10)

        period = 1.0 / max(self.filter_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('[TargetBearingWindowLS] started. bearing=%s estimate=%s', self.bearing_topic, self.target_pose_topic)

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            self.reset_estimator('armed, collecting bearing window')
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

    def reset_estimator(self, reason):
        self.samples = []
        self.last_solution = None
        rospy.loginfo('[TargetBearingWindowLS] reset: %s', reason)

    def bearing_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return

        bearing_body = self.wrap_pi(float(msg.data))
        bearing_world = self.wrap_pi(self.robot_yaw + bearing_body)
        self.samples.append(
            {
                'stamp': rospy.Time.now().to_sec(),
                'bearing': bearing_world,
                'robot_x': self.robot_x,
                'robot_y': self.robot_y,
                'robot_yaw': self.robot_yaw,
            }
        )
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]

    def prune_samples(self, now_sec):
        self.samples = [s for s in self.samples if now_sec - s['stamp'] <= self.max_sample_age]
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]

    def build_linear_initial_guess(self, now_sec):
        rows = []
        rhs = []
        weight = 1.0 / max(self.bearing_noise_std, 1e-6)
        for sample in self.samples:
            dt = sample['stamp'] - now_sec
            b = sample['bearing']
            nx = -math.sin(b)
            ny = math.cos(b)
            rows.append([weight * nx, weight * ny, weight * dt * nx, weight * dt * ny])
            rhs.append(weight * (nx * sample['robot_x'] + ny * sample['robot_y']))

        if self.last_solution is not None:
            dt_last = now_sec - self.last_solution['stamp']
            predicted = self.last_solution['state'].copy()
            predicted[0] += predicted[2] * dt_last
            predicted[1] += predicted[3] * dt_last
            pos_w = math.sqrt(max(self.reg_position, 0.0))
            vel_w = math.sqrt(max(self.reg_velocity, 0.0))
            for idx in range(2):
                row = [0.0, 0.0, 0.0, 0.0]
                row[idx] = pos_w
                rows.append(row)
                rhs.append(pos_w * predicted[idx])
            for idx in range(2, 4):
                row = [0.0, 0.0, 0.0, 0.0]
                row[idx] = vel_w
                rows.append(row)
                rhs.append(vel_w * predicted[idx])
        else:
            vel_w = math.sqrt(max(self.reg_velocity, 0.0))
            rows.append([0.0, 0.0, vel_w, 0.0])
            rhs.append(0.0)
            rows.append([0.0, 0.0, 0.0, vel_w])
            rhs.append(0.0)

        a = np.array(rows, dtype=float)
        b_vec = np.array(rhs, dtype=float)
        try:
            theta, _, _, _ = np.linalg.lstsq(a, b_vec, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return theta

    def residual_and_jacobian(self, theta, now_sec):
        residuals = []
        jac_rows = []
        weights = []
        base_w = 1.0 / max(self.bearing_noise_std, 1e-6)
        for sample in self.samples:
            dt = sample['stamp'] - now_sec
            px = theta[0] + theta[2] * dt
            py = theta[1] + theta[3] * dt
            dx = px - sample['robot_x']
            dy = py - sample['robot_y']
            r2 = max(dx * dx + dy * dy, 1e-6)
            pred = math.atan2(dy, dx)
            residual = self.wrap_pi(pred - sample['bearing'])

            robust = 1.0
            if self.robust_huber_delta > 0.0 and abs(residual) > self.robust_huber_delta:
                robust = self.robust_huber_delta / abs(residual)
            w = base_w * math.sqrt(robust)

            dh_dx = -dy / r2
            dh_dy = dx / r2
            residuals.append(w * residual)
            jac_rows.append([w * dh_dx, w * dh_dy, w * dt * dh_dx, w * dt * dh_dy])
            weights.append(w)

        return np.array(residuals, dtype=float), np.array(jac_rows, dtype=float), np.array(weights, dtype=float)

    def solve_window(self, now_sec):
        theta = self.build_linear_initial_guess(now_sec)
        if theta is None or not np.all(np.isfinite(theta)):
            return None, None, None

        lm_lambda = max(self.lm_lambda, 1e-9)
        best_theta = theta.copy()
        residuals, jac, _ = self.residual_and_jacobian(theta, now_sec)
        best_cost = 0.5 * float(residuals.dot(residuals))

        for _ in range(max(self.lm_max_iter, 1)):
            residuals, jac, _ = self.residual_and_jacobian(theta, now_sec)
            reg = np.diag([self.reg_position, self.reg_position, self.reg_velocity, self.reg_velocity])
            hessian = jac.T.dot(jac) + reg + lm_lambda * np.eye(4)
            gradient = jac.T.dot(residuals)
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break

            if np.linalg.norm(step) < self.lm_tol:
                break

            candidate = theta + step
            self.apply_state_limits(candidate)
            cand_residuals, _, _ = self.residual_and_jacobian(candidate, now_sec)
            cand_cost = 0.5 * float(cand_residuals.dot(cand_residuals))
            if math.isfinite(cand_cost) and cand_cost < best_cost:
                theta = candidate
                best_theta = candidate.copy()
                best_cost = cand_cost
                lm_lambda = max(lm_lambda * 0.5, 1e-9)
            else:
                lm_lambda = min(lm_lambda * 5.0, 1e6)

        self.apply_state_limits(best_theta)
        residuals, jac, _ = self.residual_and_jacobian(best_theta, now_sec)
        covariance = self.estimate_covariance(jac)
        rms = math.sqrt(float(residuals.dot(residuals)) / max(len(residuals), 1)) * max(self.bearing_noise_std, 1e-6)
        return best_theta, covariance, rms

    def apply_state_limits(self, theta):
        speed = math.hypot(theta[2], theta[3])
        if speed > self.max_speed > 0.0:
            scale = self.max_speed / max(speed, 1e-9)
            theta[2] *= scale
            theta[3] *= scale

        rel_x = theta[0] - self.robot_x
        rel_y = theta[1] - self.robot_y
        dist = math.hypot(rel_x, rel_y)
        if dist > self.max_distance > 0.0:
            scale = self.max_distance / max(dist, 1e-9)
            theta[0] = self.robot_x + rel_x * scale
            theta[1] = self.robot_y + rel_y * scale

    def estimate_covariance(self, jac):
        reg = np.diag([self.reg_position, self.reg_position, self.reg_velocity, self.reg_velocity])
        hessian = jac.T.dot(jac) + reg
        try:
            cov = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hessian)
        cov = 0.5 * (cov + cov.T)
        return np.clip(cov, -self.covariance_cap, self.covariance_cap)

    def timer_cb(self, event):
        if not self.have_robot:
            rospy.logwarn_throttle(2.0, '[TargetBearingWindowLS] waiting for robot odom...')
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(2.0, '[TargetBearingWindowLS] waiting for vehicle arm...')
            return

        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        now_sec = now.to_sec()
        self.prune_samples(now_sec)
        if len(self.samples) < self.min_samples:
            rospy.logwarn_throttle(
                2.0,
                '[TargetBearingWindowLS] waiting for bearing samples: %d/%d',
                len(self.samples),
                self.min_samples,
            )
            return

        state, covariance, rms = self.solve_window(now_sec)
        if state is None or covariance is None:
            rospy.logwarn_throttle(1.0, '[TargetBearingWindowLS] window least-squares failed.')
            return

        self.last_solution = {'stamp': now_sec, 'state': state.copy(), 'covariance': covariance.copy()}
        self.publish_estimate(now, state, covariance, rms)

    def publish_estimate(self, stamp, state, covariance, rms):
        vx = float(state[2])
        vy = float(state[3])
        yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > 1e-4 else 0.0

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(state[0])
        pose_msg.pose.position.y = float(state[1])
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
        belief_msg.pose.covariance[0] = covariance[0, 0]
        belief_msg.pose.covariance[1] = covariance[0, 1]
        belief_msg.pose.covariance[6] = covariance[1, 0]
        belief_msg.pose.covariance[7] = covariance[1, 1]
        belief_msg.twist.covariance[0] = covariance[2, 2]
        belief_msg.twist.covariance[1] = covariance[2, 3]
        belief_msg.twist.covariance[6] = covariance[3, 2]
        belief_msg.twist.covariance[7] = covariance[3, 3]
        self.belief_pub.publish(belief_msg)

        pos_trace = covariance[0, 0] + covariance[1, 1]
        rospy.loginfo_throttle(
            2.0,
            '[TargetBearingWindowLS] est=(%.2f, %.2f), vel=(%.2f, %.2f), samples=%d, trP=%.3f, bearing_rms=%.3f',
            state[0],
            state[1],
            vx,
            vy,
            len(self.samples),
            pos_trace,
            rms,
        )


if __name__ == '__main__':
    try:
        TargetBearingWindowLS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
