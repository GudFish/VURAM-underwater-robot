#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray


class TDOABearingEstimator:
    def __init__(self):
        rospy.init_node('tdoa_bearing_estimator', anonymous=True)

        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.tdoa_topic = rospy.get_param('~tdoa_topic', '/target_acoustic_tdoa')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.debug_topic = rospy.get_param('~debug_topic', '/target_acoustic_bearing_debug')
        self.sound_speed = float(rospy.get_param('~sound_speed', 1500.0))
        self.beacon_ref = int(rospy.get_param('~beacon_ref', 0))
        self.min_unit_norm = float(rospy.get_param('~min_unit_norm', 1e-4))
        self.residual_warn = float(rospy.get_param('~residual_warn', 0.25))
        self.beacons_body = self.load_beacons()
        self.beacon_ref = max(0, min(len(self.beacons_body) - 1, self.beacon_ref))

        self.have_odom = False
        self.robot_yaw = 0.0

        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.tdoa_sub = rospy.Subscriber(self.tdoa_topic, Float32MultiArray, self.tdoa_cb, queue_size=10)
        self.bearing_pub = rospy.Publisher(self.bearing_topic, Float32, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, Float32MultiArray, queue_size=10)
        rospy.loginfo('[TDOABearing] started. tdoa=%s bearing=%s', self.tdoa_topic, self.bearing_topic)

    def load_beacons(self):
        default = [[0.65, 0.60], [-0.65, 0.60], [0.65, -0.60], [-0.65, -0.60]]
        beacons = rospy.get_param('~beacons_body', default)
        if len(beacons) != 4:
            rospy.logwarn('[TDOABearing] beacons_body must contain 4 [x,y] entries. Using default rectangle.')
            beacons = default
        return np.array([[float(v[0]), float(v[1])] for v in beacons], dtype=float)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_odom = True

    def tdoa_cb(self, msg):
        if len(msg.data) != len(self.beacons_body) - 1:
            rospy.logwarn_throttle(2.0, '[TDOABearing] expected %d TDOA values, got %d', len(self.beacons_body) - 1, len(msg.data))
            return

        ref = self.beacons_body[self.beacon_ref]
        rows = []
        y = []
        tdoa_idx = 0
        for idx, beacon in enumerate(self.beacons_body):
            if idx == self.beacon_ref:
                continue
            rows.append(beacon - ref)
            y.append(-self.sound_speed * float(msg.data[tdoa_idx]))
            tdoa_idx += 1

        a = np.vstack(rows)
        b = np.array(y, dtype=float)
        try:
            direction, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            rospy.logwarn_throttle(2.0, '[TDOABearing] least-squares failed.')
            return

        norm = float(np.linalg.norm(direction))
        if norm < self.min_unit_norm:
            rospy.logwarn_throttle(2.0, '[TDOABearing] degenerate direction estimate.')
            return
        unit = direction / norm
        bearing = math.atan2(unit[1], unit[0])
        pred = a.dot(unit)
        residual_norm = float(np.linalg.norm(pred - b) / max(np.linalg.norm(b), 1e-6))
        if residual_norm > self.residual_warn:
            rospy.logwarn_throttle(2.0, '[TDOABearing] high LS residual ratio: %.3f', residual_norm)

        bearing_msg = Float32()
        bearing_msg.data = bearing
        self.bearing_pub.publish(bearing_msg)

        debug = Float32MultiArray()
        debug.data = [bearing, residual_norm, float(unit[0]), float(unit[1]), norm]
        self.debug_pub.publish(debug)


if __name__ == '__main__':
    try:
        TDOABearingEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
