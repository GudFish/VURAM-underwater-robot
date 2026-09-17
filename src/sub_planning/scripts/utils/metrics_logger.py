#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Record the compact, paper-facing metrics for target-tracking experiments."""
import csv
import math
import os
import time

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


class MetricsLogger:
    """Sample common simulation topics after arming and write one CSV per run."""

    def __init__(self):
        rospy.init_node('metrics_logger', anonymous=True)

        self.output_dir = rospy.get_param('~output_dir', os.path.expanduser('~/sub_planning_metrics'))
        self.method_name = rospy.get_param('~method', 'unknown')
        self.random_seed = int(rospy.get_param('~random_seed', -1))
        self.record_rate = float(rospy.get_param('~record_rate', 10.0))
        self.flush_interval = int(rospy.get_param('~flush_interval', 50))
        self.desired_standoff = float(rospy.get_param('~desired_standoff', 3.0))
        self.standoff_tolerance = float(rospy.get_param('~standoff_tolerance', 0.8))
        self.sonar_fov_deg = float(rospy.get_param('~sonar_fov_deg', 130.0))
        self.sonar_min_range = float(rospy.get_param('~sonar_min_range', 0.2))
        self.sonar_max_range = float(rospy.get_param('~sonar_max_range', 30.0))
        os.makedirs(self.output_dir, exist_ok=True)

        self.record_start_time = None
        self.armed = False
        self.gt_x = self.gt_y = float('nan')
        self.est_x = self.est_y = float('nan')
        self.cov_xx = self.cov_yy = float('nan')
        self.robot_x = self.robot_y = self.robot_psi = float('nan')
        self.cmd_u = self.cmd_v = float('nan')
        self.solver_time_ms = float('nan')
        self.solver_feasible = float('nan')
        self.recovery_active = float('nan')
        self.recovery_mode = float('nan')
        self.recovery_elapsed_s = float('nan')
        self.recovery_count = float('nan')
        self.obstacles = []
        self.min_obstacle_clearance = float('nan')
        self.near_collision_count = 0
        self.collision_count = 0
        self._near_active = False
        self._collision_active = False
        self.record_count = 0

        rospy.Subscriber('/mavros/state', State, self._state_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~target_gt_topic', '/dynamic_target/pose'), PoseStamped,
                         self._target_gt_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~target_est_pose_topic', '/target_estimate/pose'), PoseStamped,
                         self._target_est_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~belief_topic', '/target_estimate/belief'), Odometry,
                         self._belief_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~odom_topic', '/mavros/local_position/odom'), Odometry,
                         self._odom_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~vel_cmd_topic', '/target_velocity_xy'), Twist,
                         self._vel_cmd_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~solver_time_topic', '/planner/solver_time'), Float32MultiArray,
                         self._solver_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~recovery_status_topic', '/planner/recovery_status'), Float32MultiArray,
                         self._recovery_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param('~obstacle_gt_topic', '/dynamic_obstacles/markers'), MarkerArray,
                         self._obstacle_cb, queue_size=2)

        # 可选精确文件名（回放恢复工具用，如与 bag 同名，仅扩展名不同）。
        # 未设置时保持默认命名 {method}_seed{seed}_{时间戳}.csv / {method}_{时间戳}.csv
        csv_name = rospy.get_param('~csv_name', '')
        if not csv_name:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if self.random_seed >= 0:
                csv_name = '{0}_seed{1}_{2}.csv'.format(self.method_name, self.random_seed, timestamp)
            else:
                csv_name = '{}_{}.csv'.format(self.method_name, timestamp)
        elif not csv_name.endswith('.csv'):
            csv_name += '.csv'
        self.csv_path = os.path.join(self.output_dir, csv_name)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            't_sec',
            # Trajectory support (not included in the scalar summary table).
            'gt_x', 'gt_y', 'robot_x', 'robot_y', 'robot_psi',
            'estimation_pos_error_m', 'cov_trace_m2',
            'gt_standoff_m', 'standoff_error_m', 'within_standoff_tolerance',
            'control_effort_sq', 'target_in_sonar_fov',
            'solver_time_ms', 'solver_feasible',
            'recovery_active', 'recovery_mode', 'recovery_elapsed_s', 'recovery_count',
            'min_obstacle_clearance_m', 'near_collision_count', 'collision_count',
        ])
        self.csv_file.flush()
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.record_rate, 0.1)), self._timer_cb)
        rospy.loginfo('[MetricsLogger] recording compact metrics to %s', self.csv_path)

    def _state_cb(self, msg):
        if not self.armed and msg.armed:
            self.record_start_time = rospy.Time.now().to_sec()
            rospy.loginfo('[MetricsLogger] vehicle armed; recording started')
        self.armed = msg.armed

    def _target_gt_cb(self, msg):
        self.gt_x = msg.pose.position.x
        self.gt_y = msg.pose.position.y

    def _target_est_cb(self, msg):
        self.est_x = msg.pose.position.x
        self.est_y = msg.pose.position.y

    def _belief_cb(self, msg):
        self.cov_xx = float(msg.pose.covariance[0])
        self.cov_yy = float(msg.pose.covariance[7])

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_psi = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _vel_cmd_cb(self, msg):
        self.cmd_u = msg.linear.x
        self.cmd_v = msg.linear.y

    def _solver_cb(self, msg):
        if len(msg.data) >= 1:
            self.solver_time_ms = float(msg.data[0])
        if len(msg.data) >= 3:
            self.solver_feasible = 1.0 if float(msg.data[2]) > 0.5 else 0.0

    def _recovery_cb(self, msg):
        if len(msg.data) >= 1:
            self.recovery_active = float(msg.data[0])
        if len(msg.data) >= 2:
            self.recovery_mode = float(msg.data[1])
        if len(msg.data) >= 3:
            self.recovery_elapsed_s = float(msg.data[2])
        if len(msg.data) >= 4:
            self.recovery_count = float(msg.data[3])

    def _obstacle_cb(self, msg):
        self.obstacles = [marker for marker in msg.markers if marker.action != Marker.DELETE]

    def _update_obstacle_metrics(self):
        if not self._finite(self.robot_x, self.robot_y) or not self.obstacles:
            return float('nan')
        clearances = []
        for marker in self.obstacles:
            radius = 0.5 * max(float(marker.scale.x), float(marker.scale.y))
            clearances.append(math.hypot(self.robot_x - marker.pose.position.x, self.robot_y - marker.pose.position.y) - radius - 0.45)
        minimum = min(clearances)
        near = minimum < 1.5
        collision = minimum < 0.0
        if near and not self._near_active:
            self.near_collision_count += 1
        if collision and not self._collision_active:
            self.collision_count += 1
        self._near_active, self._collision_active = near, collision
        self.min_obstacle_clearance = minimum
        return minimum

    @staticmethod
    def _finite(*values):
        return all(math.isfinite(value) for value in values)

    def _target_in_sonar_fov(self):
        if not self._finite(self.gt_x, self.gt_y, self.robot_x, self.robot_y, self.robot_psi):
            return float('nan')
        dx = self.gt_x - self.robot_x
        dy = self.gt_y - self.robot_y
        distance = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) - self.robot_psi
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        in_fov = abs(bearing) <= math.radians(self.sonar_fov_deg) * 0.5
        in_range = self.sonar_min_range <= distance <= self.sonar_max_range
        return 1.0 if in_fov and in_range else 0.0

    def _timer_cb(self, _event):
        if self.record_start_time is None:
            return
        now = rospy.Time.now().to_sec()
        estimation_error = float('nan')
        if self._finite(self.est_x, self.est_y, self.gt_x, self.gt_y):
            estimation_error = math.hypot(self.est_x - self.gt_x, self.est_y - self.gt_y)
        cov_trace = self.cov_xx + self.cov_yy if self._finite(self.cov_xx, self.cov_yy) else float('nan')
        gt_standoff = float('nan')
        standoff_error = float('nan')
        within_tolerance = float('nan')
        if self._finite(self.gt_x, self.gt_y, self.robot_x, self.robot_y):
            gt_standoff = math.hypot(self.gt_x - self.robot_x, self.gt_y - self.robot_y)
            standoff_error = gt_standoff - self.desired_standoff
            within_tolerance = 1.0 if abs(standoff_error) <= self.standoff_tolerance else 0.0
        control_effort_sq = float('nan')
        if self._finite(self.cmd_u, self.cmd_v):
            control_effort_sq = self.cmd_u * self.cmd_u + self.cmd_v * self.cmd_v
        minimum_clearance = self._update_obstacle_metrics()
        self.writer.writerow([
            now - self.record_start_time,
            self.gt_x, self.gt_y, self.robot_x, self.robot_y, self.robot_psi,
            estimation_error, cov_trace,
            gt_standoff, standoff_error, within_tolerance,
            control_effort_sq, self._target_in_sonar_fov(),
            self.solver_time_ms, self.solver_feasible,
            self.recovery_active, self.recovery_mode, self.recovery_elapsed_s, self.recovery_count,
            minimum_clearance, self.near_collision_count, self.collision_count,
        ])
        self.record_count += 1
        if self.record_count % self.flush_interval == 0:
            self.csv_file.flush()

    def shutdown(self):
        self.csv_file.flush()
        self.csv_file.close()
        rospy.loginfo('[MetricsLogger] saved %d records to %s', self.record_count, self.csv_path)


if __name__ == '__main__':
    logger = MetricsLogger()
    rospy.on_shutdown(logger.shutdown)
    rospy.spin()
