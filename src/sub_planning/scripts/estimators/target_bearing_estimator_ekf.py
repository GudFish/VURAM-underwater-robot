#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import math
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from models.bearing_only_ekf import BearingOnlyCVEKF2D


class TargetBearingEstimatorEKF:
    def __init__(self):
        rospy.init_node('target_bearing_estimator_ekf', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.sonar_measurement_topic = rospy.get_param('~sonar_measurement_topic', '/forward_sonar/measurement')
        self.sonar_candidates_topic = rospy.get_param('~sonar_candidates_topic', '/forward_sonar/candidates')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/target_estimate/pose')
        self.target_twist_topic = rospy.get_param('~target_twist_topic', '/target_estimate/twist')
        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')
        self.pdaf_debug_topic = rospy.get_param('~pdaf_debug_topic', '/target_estimate/pdaf_debug')

        self.filter_rate = float(rospy.get_param('~filter_rate', 10.0))
        self.bearing_noise_std = float(rospy.get_param('~bearing_noise_std', 0.08))
        self.sonar_range_noise_std = float(rospy.get_param('~sonar_range_noise_std', 0.75))
        self.sonar_bearing_noise_std = float(rospy.get_param('~sonar_bearing_noise_std', 0.08))
        self.sonar_min_score = float(rospy.get_param('~sonar_min_score', 0.15))
        self.sonar_max_uncertainty = float(rospy.get_param('~sonar_max_uncertainty', 0.75))
        self.sonar_uncertainty_noise_scale = float(rospy.get_param('~sonar_uncertainty_noise_scale', 6.0))
        self.sonar_measurement_timeout = float(rospy.get_param('~sonar_measurement_timeout', 0.5))
        self.sonar_candidate_timeout = float(rospy.get_param('~sonar_candidate_timeout', 0.5))
        self.acoustic_measurement_timeout = float(rospy.get_param('~acoustic_measurement_timeout', 0.5))
        self.pda_detection_probability = float(rospy.get_param('~pda_detection_probability', 0.85))
        self.pda_missed_weight = float(rospy.get_param('~pda_missed_weight', 0.20))
        self.pda_gate_chi2 = float(rospy.get_param('~pda_gate_chi2', 9.21))
        self.pda_min_beta_sum = float(rospy.get_param('~pda_min_beta_sum', 0.05))
        self.pda_max_candidates = int(rospy.get_param('~pda_max_candidates', 12))
        self.pda_range_weight_min = float(rospy.get_param('~pda_range_weight_min', 0.35))
        self.pda_range_weight_power = float(rospy.get_param('~pda_range_weight_power', 1.0))
        self.bearing_candidate_gate_deg = float(rospy.get_param('~bearing_candidate_gate_deg', 12.0))
        self.max_valid_target_range = float(rospy.get_param('~max_valid_target_range', 45.0))
        self.max_target_speed = float(rospy.get_param('~max_target_speed', 2.0))
        self.lost_track_missed_updates = int(rospy.get_param('~lost_track_missed_updates', 5))
        self.reacquire_when_out_of_range = rospy.get_param('~reacquire_when_out_of_range', True)
        self.prefer_sonar_init = rospy.get_param('~prefer_sonar_init', True)
        self.allow_bearing_only_init = rospy.get_param('~allow_bearing_only_init', True)
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
        self.pending_sonar = None
        self.pending_sonar_candidates = None
        self.consecutive_pdaf_misses = 0
        self.track_lost = False

        self.init_covariance = np.diag(np.array(self.init_cov_diag, dtype=float))
        self.init_mean_np = np.array(self.init_mean, dtype=float).reshape(4)
        self.ekf = self.create_filter()

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.sonar_sub = rospy.Subscriber(self.sonar_measurement_topic, Float32MultiArray, self.sonar_cb, queue_size=10)
        self.sonar_candidates_sub = rospy.Subscriber(self.sonar_candidates_topic, Float32MultiArray, self.sonar_candidates_cb, queue_size=10)
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher(self.target_twist_topic, TwistStamped, queue_size=10)
        self.belief_pub = rospy.Publisher(self.belief_topic, Odometry, queue_size=10)
        self.pdaf_debug_pub = rospy.Publisher(self.pdaf_debug_topic, Float32MultiArray, queue_size=10)

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
        self.pending_sonar = None
        self.pending_sonar_candidates = None
        self.consecutive_pdaf_misses = 0
        self.track_lost = False
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
        self.consecutive_pdaf_misses = 0
        self.track_lost = False
        rospy.loginfo(
            '[TargetBearingEKF] first bearing initialized: bearing=%.2f range=%.2f state=(%.2f, %.2f, %.2f, %.2f)',
            bearing,
            self.init_range,
            state[0],
            state[1],
            state[2],
            state[3],
        )

    def initialize_from_sonar(self, measured_range, measured_bearing):
        world_bearing = self.wrap_pi(self.robot_yaw + float(measured_bearing))
        ux = math.cos(world_bearing)
        uy = math.sin(world_bearing)
        range_value = max(float(measured_range), self.min_bearing_range)
        state = np.array(
            [
                self.robot_x + range_value * ux,
                self.robot_y + range_value * uy,
                self.init_mean_np[2],
                self.init_mean_np[3],
            ],
            dtype=float,
        )

        along = np.array([ux, uy], dtype=float)
        cross = np.array([-uy, ux], dtype=float)
        range_var = max(self.sonar_range_noise_std ** 2, 0.25)
        cross_var = max((range_value * self.sonar_bearing_noise_std) ** 2, 0.25)
        pos_cov = range_var * np.outer(along, along) + cross_var * np.outer(cross, cross)
        cov = np.array(self.init_covariance, dtype=float)
        cov[0:2, 0:2] = pos_cov
        self.ekf.set_state(state, cov)
        self.consecutive_pdaf_misses = 0
        self.track_lost = False
        rospy.loginfo(
            '[TargetBearingEKF] sonar initialized: range=%.2f bearing=%.2f state=(%.2f, %.2f, %.2f, %.2f)',
            range_value,
            measured_bearing,
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
            if self.prefer_sonar_init and self.try_initialize_from_candidates(bearing):
                pass
            elif self.prefer_sonar_init and self.pending_sonar is not None and self.measurement_is_fresh(
                    self.pending_sonar['stamp'],
                    self.sonar_measurement_timeout,
                ):
                sonar = self.pending_sonar
                self.initialize_from_sonar(sonar['range'], sonar['bearing'])
                self.pending_sonar = None
            elif self.allow_bearing_only_init:
                self.initialize_from_bearing(bearing)

        self.pending_bearing = {
            'stamp': rospy.Time.now(),
            'bearing': bearing,
            'robot_xy': (self.robot_x, self.robot_y),
            'robot_yaw': self.robot_yaw,
        }

    def sonar_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return
        if len(msg.data) < 3:
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] sonar measurement requires [range, bearing, score].')
            return

        measured_range = float(msg.data[0])
        measured_bearing = self.wrap_pi(float(msg.data[1]))
        score = float(msg.data[2])
        uncertainty = float(msg.data[3]) if len(msg.data) >= 4 else 0.0
        uncertainty = max(0.0, min(1.0, uncertainty))
        stamp = rospy.Time.now()

        self.pending_sonar = {
            'stamp': stamp,
            'range': measured_range,
            'bearing': measured_bearing,
            'score': score,
            'uncertainty': uncertainty,
            'robot_xy': (self.robot_x, self.robot_y),
            'robot_yaw': self.robot_yaw,
        }

        if not self.ekf_initialized() and score >= self.sonar_min_score and uncertainty <= self.sonar_max_uncertainty:
            self.initialize_from_sonar(measured_range, measured_bearing)
            self.pending_sonar = None

    def sonar_candidates_cb(self, msg):
        if not self.current_state.armed or not self.have_robot:
            return
        if len(msg.data) < 1:
            return

        count = max(0, int(round(float(msg.data[0]))))
        candidates = []
        stride = 8
        max_count = min(count, self.pda_max_candidates)
        for idx in range(max_count):
            base = 1 + idx * stride
            if base + stride > len(msg.data):
                break
            measured_range = float(msg.data[base])
            measured_bearing = self.wrap_pi(float(msg.data[base + 1]))
            score = max(0.0, min(1.0, float(msg.data[base + 2])))
            uncertainty = max(0.0, min(1.0, float(msg.data[base + 3])))
            if score < self.sonar_min_score or uncertainty > self.sonar_max_uncertainty:
                continue
            candidates.append(
                {
                    'range': measured_range,
                    'bearing': measured_bearing,
                    'score': score,
                    'uncertainty': uncertainty,
                    'row': float(msg.data[base + 4]),
                    'col': float(msg.data[base + 5]),
                    'area': float(msg.data[base + 6]),
                    'threshold': float(msg.data[base + 7]),
                }
            )

        self.pending_sonar_candidates = {
            'stamp': rospy.Time.now(),
            'candidates': candidates,
            'robot_xy': (self.robot_x, self.robot_y),
            'robot_yaw': self.robot_yaw,
        }

        if not self.ekf_initialized() and self.pending_bearing is not None:
            if self.measurement_is_fresh(self.pending_bearing['stamp'], self.acoustic_measurement_timeout):
                self.try_initialize_from_candidates(self.pending_bearing['bearing'])

    def measurement_is_fresh(self, stamp, timeout):
        if stamp is None:
            return False
        age = (rospy.Time.now() - stamp).to_sec()
        return age <= max(timeout, 0.0)

    def candidate_bearing_gate(self, candidates, bearing):
        if bearing is None:
            return list(candidates)
        gate = math.radians(self.bearing_candidate_gate_deg)
        return [item for item in candidates if abs(self.wrap_pi(item['bearing'] - bearing)) <= gate]

    def try_initialize_from_candidates(self, bearing):
        if self.pending_sonar_candidates is None:
            return False
        if not self.measurement_is_fresh(self.pending_sonar_candidates['stamp'], self.sonar_candidate_timeout):
            self.pending_sonar_candidates = None
            return False

        candidates = self.candidate_bearing_gate(self.pending_sonar_candidates['candidates'], bearing)
        if not candidates:
            return False

        weights = np.array([max(item['score'], 1.0e-3) for item in candidates], dtype=float)
        weights /= max(float(np.sum(weights)), 1.0e-9)
        world_points = []
        for item in candidates:
            world_bearing = self.wrap_pi(self.robot_yaw + item['bearing'])
            world_points.append(
                [
                    self.robot_x + item['range'] * math.cos(world_bearing),
                    self.robot_y + item['range'] * math.sin(world_bearing),
                ]
            )
        world_points = np.array(world_points, dtype=float)
        pos_mean = weights.dot(world_points)
        state = np.array(
            [
                pos_mean[0],
                pos_mean[1],
                self.init_mean_np[2],
                self.init_mean_np[3],
            ],
            dtype=float,
        )

        diffs = world_points - pos_mean
        pos_cov = np.zeros((2, 2), dtype=float)
        for weight, diff in zip(weights, diffs):
            pos_cov += weight * np.outer(diff, diff)

        best_range = max(float(np.sum(weights * np.array([item['range'] for item in candidates]))), self.min_bearing_range)
        mean_bearing = math.atan2(pos_mean[1] - self.robot_y, pos_mean[0] - self.robot_x)
        along = np.array([math.cos(mean_bearing), math.sin(mean_bearing)], dtype=float)
        cross = np.array([-along[1], along[0]], dtype=float)
        range_var = max(self.sonar_range_noise_std ** 2, 0.25)
        cross_var = max((best_range * self.sonar_bearing_noise_std) ** 2, 0.25)
        pos_cov += range_var * np.outer(along, along) + cross_var * np.outer(cross, cross)

        cov = np.array(self.init_covariance, dtype=float)
        cov[0:2, 0:2] = pos_cov
        self.ekf.set_state(state, cov)
        self.consecutive_pdaf_misses = 0
        self.track_lost = False
        self.pending_sonar_candidates = None
        rospy.loginfo(
            '[TargetBearingEKF] PDA candidates initialized: count=%d bearing=%.2f state=(%.2f, %.2f, %.2f, %.2f)',
            len(candidates),
            bearing,
            state[0],
            state[1],
            state[2],
            state[3],
        )
        return True

    def timer_cb(self, event):
        if not self.have_robot:
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for robot odom...')
            return
        if not self.current_state.armed:
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for vehicle arm...')
            return
        if not self.ekf_initialized():
            rospy.logwarn_throttle(2.0, '[TargetBearingEKF] waiting for first acoustic or sonar measurement...')
            return

        now = event.current_real if event.current_real != rospy.Time() else rospy.Time.now()
        if self.last_stamp is None:
            dt = 1.0 / max(self.filter_rate, 1e-3)
        else:
            dt = max((now - self.last_stamp).to_sec(), 1e-3)
        self.last_stamp = now

        self.ekf.predict(dt)
        self.constrain_ekf_state()
        acoustic_innovation = 0.0
        acoustic_nis = 0.0
        sonar_range_innovation = 0.0
        sonar_bearing_innovation = 0.0
        sonar_nis = 0.0
        pdaf_stats = {
            'num_candidates': 0,
            'num_gated': 0,
            'beta0': 1.0,
            'best_beta': 0.0,
            'weighted_range': 0.0,
            'weighted_bearing': 0.0,
            'best_nis': 0.0,
            'track_lost': float(self.track_lost),
            'relative_range': self.current_relative_range(),
            'missed_updates': float(self.consecutive_pdaf_misses),
        }

        if self.pending_sonar_candidates is not None:
            if not self.measurement_is_fresh(self.pending_sonar_candidates['stamp'], self.sonar_candidate_timeout):
                self.pending_sonar_candidates = None
            else:
                obs = self.pending_sonar_candidates
                acoustic_gate = None
                if self.pending_bearing is not None and self.measurement_is_fresh(
                    self.pending_bearing['stamp'],
                    self.acoustic_measurement_timeout,
                ):
                    acoustic_gate = self.pending_bearing['bearing']

                candidates = self.candidate_bearing_gate(obs['candidates'], acoustic_gate)
                if self.should_reacquire_from_candidates(candidates, acoustic_gate):
                    self.try_initialize_from_candidates(acoustic_gate)
                    pdaf_stats['num_candidates'] = len(obs['candidates'])
                    pdaf_stats['num_gated'] = len(candidates)
                    pdaf_stats['beta0'] = 0.0
                    pdaf_stats['best_beta'] = 1.0 if candidates else 0.0
                    pdaf_stats['relative_range'] = self.current_relative_range()
                    pdaf_stats['missed_updates'] = float(self.consecutive_pdaf_misses)
                    pdaf_stats['track_lost'] = float(self.track_lost)
                    self.pending_sonar_candidates = None
                    candidates = []
                if candidates:
                    _, _, pdaf_stats = self.ekf.update_range_bearing_pdaf(
                        obs['robot_xy'],
                        obs['robot_yaw'],
                        candidates,
                        self.sonar_range_noise_std ** 2,
                        self.sonar_bearing_noise_std ** 2,
                        detection_probability=self.pda_detection_probability,
                        missed_weight=self.pda_missed_weight,
                        gate_chi2=self.pda_gate_chi2,
                        min_beta_sum=self.pda_min_beta_sum,
                        range_weight_min=self.pda_range_weight_min,
                        range_weight_power=self.pda_range_weight_power,
                    )
                    innovation = pdaf_stats.get('innovation', np.zeros(2, dtype=float))
                    sonar_range_innovation = float(innovation[0])
                    sonar_bearing_innovation = float(innovation[1])
                    sonar_nis = float(pdaf_stats.get('best_nis', 0.0))
                    if pdaf_stats.get('updated', False):
                        self.consecutive_pdaf_misses = 0
                        self.track_lost = False
                    else:
                        self.register_pdaf_miss()
                else:
                    pdaf_stats['num_candidates'] = len(obs['candidates'])
                    if obs['candidates']:
                        self.register_pdaf_miss()
                self.pending_sonar_candidates = None
        elif self.pending_sonar is not None:
            if not self.measurement_is_fresh(self.pending_sonar['stamp'], self.sonar_measurement_timeout):
                self.pending_sonar = None
            elif (
                    self.pending_sonar['score'] >= self.sonar_min_score
                    and self.pending_sonar.get('uncertainty', 0.0) <= self.sonar_max_uncertainty
                ):
                    obs = self.pending_sonar
                    noise_scale = 1.0 + self.sonar_uncertainty_noise_scale * obs.get('uncertainty', 0.0) ** 2
                    _, _, innovation = self.ekf.update_range_bearing(
                        obs['robot_xy'],
                        obs['robot_yaw'],
                        obs['range'],
                        obs['bearing'],
                        self.sonar_range_noise_std ** 2 * noise_scale,
                        self.sonar_bearing_noise_std ** 2 * noise_scale,
                    )
                    _, _, sonar_nis = self.ekf.range_bearing_stats(
                        obs['robot_xy'],
                        obs['robot_yaw'],
                        obs['range'],
                        obs['bearing'],
                        self.sonar_range_noise_std ** 2 * noise_scale,
                        self.sonar_bearing_noise_std ** 2 * noise_scale,
                    )
                    sonar_range_innovation = float(innovation[0])
                    sonar_bearing_innovation = float(innovation[1])
                    self.pending_sonar = None
                    self.consecutive_pdaf_misses = 0
                    self.track_lost = False

        self.constrain_ekf_state()
        pdaf_stats['track_lost'] = float(self.track_lost)
        pdaf_stats['relative_range'] = self.current_relative_range()
        pdaf_stats['missed_updates'] = float(self.consecutive_pdaf_misses)
        self.publish_pdaf_debug(
            pdaf_stats,
            -999.0
            if self.pending_bearing is None
            or not self.measurement_is_fresh(self.pending_bearing['stamp'], self.acoustic_measurement_timeout)
            else self.pending_bearing['bearing'],
        )

        if self.pending_bearing is not None:
            if not self.measurement_is_fresh(self.pending_bearing['stamp'], self.acoustic_measurement_timeout):
                self.pending_bearing = None
            else:
                obs = self.pending_bearing
                _, _, acoustic_innovation = self.ekf.update(obs['robot_xy'], obs['robot_yaw'], obs['bearing'])
                _, _, acoustic_nis = self.ekf.measurement_stats(obs['robot_xy'], obs['robot_yaw'], obs['bearing'])
                self.constrain_ekf_state()
                self.pending_bearing = None

        self.publish_estimate(
            now,
            acoustic_innovation,
            acoustic_nis,
            sonar_range_innovation,
            sonar_bearing_innovation,
            sonar_nis,
        )

    def publish_pdaf_debug(self, stats, acoustic_bearing):
        msg = Float32MultiArray()
        msg.data = [
            float(stats.get('num_candidates', 0)),
            float(stats.get('num_gated', 0)),
            float(stats.get('beta0', 1.0)),
            float(stats.get('best_beta', 0.0)),
            float(stats.get('weighted_range', 0.0)),
            float(stats.get('weighted_bearing', 0.0)),
            float(stats.get('best_nis', 0.0)),
            float(acoustic_bearing),
            float(stats.get('track_lost', 0.0)),
            float(stats.get('relative_range', 0.0)),
            float(stats.get('missed_updates', 0.0)),
        ]
        self.pdaf_debug_pub.publish(msg)

    def current_relative_range(self):
        if not self.ekf_initialized():
            return 0.0
        dx = self.ekf.x[0] - self.robot_x
        dy = self.ekf.x[1] - self.robot_y
        return float(math.hypot(dx, dy))

    def register_pdaf_miss(self):
        self.consecutive_pdaf_misses += 1
        if (
            self.consecutive_pdaf_misses >= max(self.lost_track_missed_updates, 1)
            and self.current_relative_range() > self.max_valid_target_range
        ):
            self.track_lost = True

    def should_reacquire_from_candidates(self, candidates, acoustic_gate):
        if not self.reacquire_when_out_of_range:
            return False
        if acoustic_gate is None or not candidates:
            return False
        if not self.track_lost and self.current_relative_range() <= self.max_valid_target_range:
            return False
        return True

    def constrain_ekf_state(self):
        if not self.ekf_initialized():
            return

        speed = math.hypot(self.ekf.x[2], self.ekf.x[3])
        if self.max_target_speed > 0.0 and speed > self.max_target_speed:
            scale = self.max_target_speed / max(speed, 1.0e-9)
            self.ekf.x[2] *= scale
            self.ekf.x[3] *= scale

        rel_x = self.ekf.x[0] - self.robot_x
        rel_y = self.ekf.x[1] - self.robot_y
        rel_range = math.hypot(rel_x, rel_y)
        if self.max_valid_target_range > 0.0 and rel_range > self.max_valid_target_range:
            scale = self.max_valid_target_range / max(rel_range, 1.0e-9)
            self.ekf.x[0] = self.robot_x + rel_x * scale
            self.ekf.x[1] = self.robot_y + rel_y * scale
            self.track_lost = True

        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)

    def publish_estimate(self, stamp, acoustic_innovation, acoustic_nis, sonar_range_innovation, sonar_bearing_innovation, sonar_nis):
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
            '[TargetBearingEKF] est=(%.2f, %.2f), vel=(%.2f, %.2f), trP=%.3f, acoustic_innov=%.3f, acoustic_nis=%.2f, sonar_dr=%.3f, sonar_db=%.3f, sonar_nis=%.2f',
            state[0],
            state[1],
            vx,
            vy,
            pos_trace,
            acoustic_innovation,
            acoustic_nis,
            sonar_range_innovation,
            sonar_bearing_innovation,
            sonar_nis,
        )


if __name__ == '__main__':
    try:
        TargetBearingEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
