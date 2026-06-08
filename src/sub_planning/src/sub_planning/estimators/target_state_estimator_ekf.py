#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from sub_planning.models.range_only_ekf import RangeOnlyCVEKF2D


class TargetStateEstimatorEKF:
    def __init__(self):
        rospy.init_node('target_state_estimator_ekf', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.left_range_topic = rospy.get_param('~left_range_topic', '/target_acoustic_range_left')
        self.right_range_topic = rospy.get_param('~right_range_topic', '/target_acoustic_range_right')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/target_estimate/pose')
        self.target_twist_topic = rospy.get_param('~target_twist_topic', '/target_estimate/twist')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')

        self.filter_rate = float(rospy.get_param('~filter_rate', 10.0))
        self.range_noise_std = float(rospy.get_param('~range_noise_std', 0.4))
        self.left_range_noise_std = float(rospy.get_param('~left_range_noise_std', self.range_noise_std))
        self.right_range_noise_std = float(rospy.get_param('~right_range_noise_std', self.range_noise_std))
        self.left_beacon_x = float(rospy.get_param('~left_beacon_x', 0.0))
        self.left_beacon_y = float(rospy.get_param('~left_beacon_y', 0.65))
        self.right_beacon_x = float(rospy.get_param('~right_beacon_x', 0.0))
        self.right_beacon_y = float(rospy.get_param('~right_beacon_y', -0.65))
        self.process_noise_diag = [float(v) for v in rospy.get_param('~process_noise_diag', [0.5, 0.5, 0.8, 0.8])]
        self.init_mean = [float(v) for v in rospy.get_param('~init_mean', [8.0, 0.0, 0.4, 0.0])]
        self.init_cov_diag = [float(v) for v in rospy.get_param('~init_cov_diag', [9.0, 9.0, 1.0, 1.0])]
        self.auto_init_range_angle = float(rospy.get_param('~auto_init_range_angle', 0.35))
        self.use_prior_init = rospy.get_param('~use_prior_init', False)
        self.reset_on_disarm = rospy.get_param('~reset_on_disarm', True)
        self.init_pair_count = int(rospy.get_param('~init_pair_count', 10))
        self.range_pair_tolerance = float(rospy.get_param('~range_pair_tolerance', 0.15))
        self.init_fit_max_iter = int(rospy.get_param('~init_fit_max_iter', 20))
        self.init_fit_tol = float(rospy.get_param('~init_fit_tol', 1e-3))
        self.init_fit_lambda = float(rospy.get_param('~init_fit_lambda', 1e-2))

        self.current_state = State()
        self.last_armed = False
        self.have_robot = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.last_stamp = None

        self.latest_left_obs = None
        self.latest_right_obs = None
        self.pending_pair = None
        self.init_pair_buffer = []

        self.init_covariance = np.diag(np.array(self.init_cov_diag, dtype=float))
        self.init_mean_np = np.array(self.init_mean, dtype=float).reshape(4)
        self.ekf = self.create_filter()

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.left_range_sub = rospy.Subscriber(self.left_range_topic, Float32, self.left_range_cb, queue_size=5)
        self.right_range_sub = rospy.Subscriber(self.right_range_topic, Float32, self.right_range_cb, queue_size=5)
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
        self.last_stamp = None
        self.latest_left_obs = None
        self.latest_right_obs = None
        self.pending_pair = None
        self.init_pair_buffer = []
        self.ekf = self.create_filter()
        rospy.loginfo('[TargetEKF] estimator reset: %s', reason)

    def initialize_filter(self, state, log_prefix):
        self.ekf.set_state(state, self.init_covariance)
        self.pending_pair = None
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
            self.reset_estimator('armed, collecting dual acoustic ranges for batch initialization')

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
            prior_state = self.init_mean_np.copy()
            prior_state[0] += self.robot_x
            prior_state[1] += self.robot_y
            self.initialize_filter(prior_state, '[TargetEKF] relative prior')

    def body_to_world(self, bx, by):
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return self.robot_x + c * bx - s * by, self.robot_y + s * bx + c * by

    def wrap_pi(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def make_observation(self, measured_range, beacon_x, beacon_y):
        observer_xy = self.body_to_world(beacon_x, beacon_y)
        return {
            'stamp': rospy.Time.now().to_sec(),
            'range': float(measured_range),
            'observer_xy': observer_xy,
            'robot_xy': (self.robot_x, self.robot_y),
        }

    def left_range_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return
        self.latest_left_obs = self.make_observation(msg.data, self.left_beacon_x, self.left_beacon_y)
        self.try_form_range_pair()

    def right_range_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return
        self.latest_right_obs = self.make_observation(msg.data, self.right_beacon_x, self.right_beacon_y)
        self.try_form_range_pair()

    def try_form_range_pair(self):
        if self.latest_left_obs is None or self.latest_right_obs is None:
            return

        dt = abs(self.latest_left_obs['stamp'] - self.latest_right_obs['stamp'])
        if dt > self.range_pair_tolerance:
            if self.latest_left_obs['stamp'] < self.latest_right_obs['stamp']:
                self.latest_left_obs = None
            else:
                self.latest_right_obs = None
            rospy.logwarn_throttle(2.0, '[TargetEKF] dual range pair out of sync, dropping older sample.')
            return

        pair = {
            'stamp': max(self.latest_left_obs['stamp'], self.latest_right_obs['stamp']),
            'left_range': self.latest_left_obs['range'],
            'right_range': self.latest_right_obs['range'],
            'left_xy': self.latest_left_obs['observer_xy'],
            'right_xy': self.latest_right_obs['observer_xy'],
            'robot_xy': self.latest_right_obs['robot_xy'],
        }
        self.latest_left_obs = None
        self.latest_right_obs = None

        if self.ekf_initialized():
            self.pending_pair = pair
            return

        self.init_pair_buffer.append(pair)
        if len(self.init_pair_buffer) > self.init_pair_count:
            self.init_pair_buffer = self.init_pair_buffer[-self.init_pair_count:]
        rospy.loginfo_throttle(
            1.0,
            '[TargetEKF] initialization buffer: %d/%d dual range pairs',
            len(self.init_pair_buffer),
            self.init_pair_count,
        )
        if len(self.init_pair_buffer) >= self.init_pair_count:
            self.try_initialize_from_batch()

    def circle_candidates(self, observer_a, observer_b, range_a, range_b):
        ax, ay = observer_a
        bx, by = observer_b
        dx = bx - ax
        dy = by - ay
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return []

        ex = np.array([dx / d, dy / d], dtype=float)
        a = (range_a * range_a - range_b * range_b + d * d) / (2.0 * d)
        base = np.array([ax, ay], dtype=float) + a * ex
        h_sq = range_a * range_a - a * a
        if h_sq <= 1e-9:
            return [base]

        h = math.sqrt(max(h_sq, 0.0))
        ey = np.array([-ex[1], ex[0]], dtype=float)
        return [base + h * ey, base - h * ey]

    def select_candidate(self, candidates, pair):
        if not candidates:
            mean_range = 0.5 * (pair['left_range'] + pair['right_range'])
            return np.array(
                [
                    pair['robot_xy'][0] + mean_range * math.cos(self.auto_init_range_angle),
                    pair['robot_xy'][1] + mean_range * math.sin(self.auto_init_range_angle),
                ],
                dtype=float,
            )

        best = None
        best_cost = None
        for cand in candidates:
            heading = math.atan2(cand[1] - pair['robot_xy'][1], cand[0] - pair['robot_xy'][0])
            cost = self.wrap_pi(heading - self.auto_init_range_angle) ** 2
            if best_cost is None or cost < best_cost:
                best = np.array(cand, dtype=float)
                best_cost = cost
        return best

    def build_initial_guess(self, pairs):
        first_pair = pairs[0]
        last_pair = pairs[-1]
        first_cands = self.circle_candidates(first_pair['left_xy'], first_pair['right_xy'], first_pair['left_range'], first_pair['right_range'])
        last_cands = self.circle_candidates(last_pair['left_xy'], last_pair['right_xy'], last_pair['left_range'], last_pair['right_range'])

        if not first_cands:
            first_pos = self.select_candidate([], first_pair)
            first_cands = [first_pos]
        if not last_cands:
            last_pos = self.select_candidate([], last_pair)
            last_cands = [last_pos]

        best_theta = None
        best_cost = None
        dt_total = max(last_pair['stamp'] - first_pair['stamp'], 1e-3)
        v_prior = self.init_mean_np[2:4]
        for first_pos in first_cands:
            for last_pos in last_cands:
                vel = (last_pos - first_pos) / dt_total
                heading_cost = self.wrap_pi(
                    math.atan2(last_pos[1] - last_pair['robot_xy'][1], last_pos[0] - last_pair['robot_xy'][0]) - self.auto_init_range_angle
                ) ** 2
                speed_cost = float(np.sum((vel - v_prior) ** 2))
                cost = heading_cost + 0.2 * speed_cost
                if best_cost is None or cost < best_cost:
                    best_theta = np.array([last_pos[0], last_pos[1], vel[0], vel[1]], dtype=float)
                    best_cost = cost

        if best_theta is None:
            fallback_pos = self.select_candidate([], last_pair)
            best_theta = np.array([fallback_pos[0], fallback_pos[1], self.init_mean_np[2], self.init_mean_np[3]], dtype=float)
        return best_theta

    def residual_and_jacobian(self, theta, pairs, t_ref):
        x_ref, y_ref, vx, vy = theta.tolist()
        residuals = []
        jac_rows = []
        for pair in pairs:
            dt = pair['stamp'] - t_ref
            px = x_ref + vx * dt
            py = y_ref + vy * dt
            for observer_xy, measured_range in ((pair['left_xy'], pair['left_range']), (pair['right_xy'], pair['right_range'])):
                dx = px - observer_xy[0]
                dy = py - observer_xy[1]
                dist = max(math.hypot(dx, dy), 1e-6)
                residuals.append(dist - measured_range)
                grad_px = dx / dist
                grad_py = dy / dist
                jac_rows.append([grad_px, grad_py, dt * grad_px, dt * grad_py])
        return np.array(residuals, dtype=float), np.array(jac_rows, dtype=float)

    def solve_batch_least_squares(self, pairs):
        if len(pairs) < 2:
            return None, None

        theta = self.build_initial_guess(pairs)
        t_ref = pairs[-1]['stamp']
        lm_lambda = max(self.init_fit_lambda, 1e-8)
        best_theta = theta.copy()
        residuals, _ = self.residual_and_jacobian(theta, pairs, t_ref)
        best_cost = 0.5 * float(residuals.dot(residuals))

        for _ in range(max(self.init_fit_max_iter, 1)):
            residuals, jac = self.residual_and_jacobian(theta, pairs, t_ref)
            hessian = jac.T.dot(jac) + lm_lambda * np.eye(4)
            gradient = jac.T.dot(residuals)
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return best_theta, best_cost

            if np.linalg.norm(step) < self.init_fit_tol:
                break

            candidate = theta + step
            cand_res, _ = self.residual_and_jacobian(candidate, pairs, t_ref)
            cand_cost = 0.5 * float(cand_res.dot(cand_res))
            if math.isfinite(cand_cost) and cand_cost < best_cost:
                theta = candidate
                best_theta = candidate.copy()
                best_cost = cand_cost
                lm_lambda = max(lm_lambda * 0.5, 1e-8)
            else:
                lm_lambda = min(lm_lambda * 5.0, 1e6)

        return best_theta, best_cost

    def try_initialize_from_batch(self):
        pairs = self.init_pair_buffer[-self.init_pair_count:]
        theta, cost = self.solve_batch_least_squares(pairs)
        if theta is None or not np.all(np.isfinite(theta)):
            rospy.logwarn_throttle(2.0, '[TargetEKF] batch least-squares initialization failed, waiting for more pairs.')
            return

        self.initialize_filter(theta, '[TargetEKF] batch LS init')
        rospy.loginfo(
            '[TargetEKF] batch LS residual cost: %.4f using %d dual range pairs.',
            0.0 if cost is None else cost,
            len(pairs),
        )

    def timer_cb(self, event):
        if not self.have_robot:
            rospy.logwarn_throttle(2.0, '[TargetEKF] waiting for robot odom...')
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(2.0, '[TargetEKF] waiting for vehicle arm...')
            return
        if not self.ekf_initialized():
            rospy.logwarn_throttle(
                2.0,
                '[TargetEKF] waiting for %d dual range pairs for batch initialization... current=%d',
                self.init_pair_count,
                len(self.init_pair_buffer),
            )
            return

        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        if self.last_stamp is None:
            dt = 1.0 / max(self.filter_rate, 1e-3)
        else:
            dt = max((now - self.last_stamp).to_sec(), 1e-3)
        self.last_stamp = now

        self.ekf.predict(dt)
        innovation_norm = 0.0
        if self.pending_pair is not None:
            _, _, innovation = self.ekf.update_multi(
                [self.pending_pair['left_xy'], self.pending_pair['right_xy']],
                [self.pending_pair['left_range'], self.pending_pair['right_range']],
                [self.left_range_noise_std ** 2, self.right_range_noise_std ** 2],
            )
            innovation_norm = float(np.linalg.norm(innovation))
            self.pending_pair = None

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
            '[TargetEKF] est=(%.2f, %.2f), vel=(%.2f, %.2f), trP=%.3f, dual_innovation=%.3f',
            state[0],
            state[1],
            vx,
            vy,
            pos_trace,
            innovation_norm,
        )


if __name__ == '__main__':
    try:
        TargetStateEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
