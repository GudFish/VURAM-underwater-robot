#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class TargetBeliefViz:
    def __init__(self):
        rospy.init_node('target_belief_viz', anonymous=True)

        self.belief_topic = rospy.get_param('~belief_topic', '/target_estimate/belief')
        self.marker_topic = rospy.get_param('~marker_topic', '/target_estimate/belief_marker')
        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.ellipse_sigma = float(rospy.get_param('~ellipse_sigma', 2.0))
        self.ellipse_segments = max(12, int(rospy.get_param('~ellipse_segments', 48)))
        self.ellipse_line_width = float(rospy.get_param('~ellipse_line_width', 0.08))
        self.text_height = float(rospy.get_param('~text_height', 0.7))
        self.text_z_offset = float(rospy.get_param('~text_z_offset', 0.9))
        self.show_trace = rospy.get_param('~show_trace', True)
        self.show_pos_std = rospy.get_param('~show_pos_std', True)

        self.latest_belief = None

        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=10)
        rospy.Subscriber(self.belief_topic, Odometry, self.belief_cb, queue_size=1)

    def belief_cb(self, msg):
        self.latest_belief = msg
        self.publish_markers()

    def publish_markers(self):
        if self.latest_belief is None:
            return

        belief = self.latest_belief
        stamp = belief.header.stamp if belief.header.stamp != rospy.Time() else rospy.Time.now()
        frame_id = belief.header.frame_id if belief.header.frame_id else self.frame_id

        px = belief.pose.pose.position.x
        py = belief.pose.pose.position.y
        pxx = belief.pose.covariance[0]
        pxy = belief.pose.covariance[1]
        pyx = belief.pose.covariance[6]
        pyy = belief.pose.covariance[7]
        cov_xy = np.array([[pxx, pxy], [pyx, pyy]], dtype=float)
        cov_xy = 0.5 * (cov_xy + cov_xy.T)

        ellipse_marker = Marker()
        ellipse_marker.header.stamp = stamp
        ellipse_marker.header.frame_id = frame_id
        ellipse_marker.ns = 'target_belief'
        ellipse_marker.id = 0
        ellipse_marker.type = Marker.LINE_STRIP
        ellipse_marker.action = Marker.ADD
        ellipse_marker.pose.orientation.w = 1.0
        ellipse_marker.scale.x = self.ellipse_line_width
        ellipse_marker.color.r = 1.0
        ellipse_marker.color.g = 0.85
        ellipse_marker.color.b = 0.1
        ellipse_marker.color.a = 0.95
        ellipse_marker.lifetime = rospy.Duration(0.0)
        ellipse_marker.points = self.build_ellipse_points(px, py, cov_xy)
        self.marker_pub.publish(ellipse_marker)

        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = frame_id
        text_marker.ns = 'target_belief'
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = px
        text_marker.pose.position.y = py
        text_marker.pose.position.z = self.text_z_offset
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = self.text_height
        text_marker.color.r = 0.0
        text_marker.color.g = 1.0
        text_marker.color.b = 0.4
        text_marker.color.a = 0.95
        text_marker.lifetime = rospy.Duration(0.0)
        trace_pos = max(0.0, cov_xy[0, 0] + cov_xy[1, 1])
        lines = []
        if self.show_trace:
            lines.append('trP={:.3f}'.format(trace_pos))
        if self.show_pos_std:
            lines.append('std={:.3f}'.format(math.sqrt(trace_pos)))
        text_marker.text = '\n'.join(lines) if lines else 'belief'
        self.marker_pub.publish(text_marker)

    def build_ellipse_points(self, cx, cy, cov_xy):
        eigvals, eigvecs = np.linalg.eigh(cov_xy)
        eigvals = np.maximum(eigvals, 1e-9)
        axes = self.ellipse_sigma * np.sqrt(eigvals)

        points = []
        for idx in range(self.ellipse_segments + 1):
            theta = 2.0 * math.pi * float(idx) / float(self.ellipse_segments)
            local = np.array([axes[0] * math.cos(theta), axes[1] * math.sin(theta)], dtype=float)
            xy = eigvecs.dot(local)
            pt = Point()
            pt.x = cx + xy[0]
            pt.y = cy + xy[1]
            pt.z = 0.0
            points.append(pt)
        return points


if __name__ == '__main__':
    try:
        TargetBeliefViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
