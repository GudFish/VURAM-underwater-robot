#!/usr/bin/env python3
"""Display the same 3-D sonar aperture and installation pose used by the renderer."""
import math
import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


def quaternion_product(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return np.array([w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u,
                     w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s])


class ForwardSonarRangeViz:
    def __init__(self):
        rospy.init_node('forward_sonar_range_viz', anonymous=True)
        def shared(name, default):
            return rospy.get_param('~' + name, rospy.get_param('/forward_sonar/geometry/' + name, default))
        self.fov = math.radians(float(shared('fov_deg', 130.)))
        self.max_range = float(shared('max_range', 30.))
        self.min_range = float(shared('min_range', .2))
        self.vertical_fov = math.radians(float(rospy.get_param('/forward_sonar_sim/vertical_fov_deg', 20.)))
        self.xyz = np.array(rospy.get_param('/forward_sonar_sim/sensor_xyz', [0., 0., 0.]), dtype=float)
        r, p, y = np.radians(rospy.get_param('/forward_sonar_sim/sensor_rpy_deg', [0., 0., 0.])) * .5
        self.mount_q = np.array([math.sin(r)*math.cos(p)*math.cos(y)-math.cos(r)*math.sin(p)*math.sin(y),
                                 math.cos(r)*math.sin(p)*math.cos(y)+math.sin(r)*math.cos(p)*math.sin(y),
                                 math.cos(r)*math.cos(p)*math.sin(y)-math.sin(r)*math.sin(p)*math.cos(y),
                                 math.cos(r)*math.cos(p)*math.cos(y)+math.sin(r)*math.sin(p)*math.sin(y)])
        self.pose = None
        self.frame = rospy.get_param('~frame_id', 'map')
        self.sub = rospy.Subscriber(rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom'),
                                    Odometry, self.odom_cb, queue_size=1)
        self.pub = rospy.Publisher(rospy.get_param('~marker_topic', '/forward_sonar/markers'), MarkerArray, queue_size=1)
        rate = float(rospy.get_param('~publish_rate', 10.))
        self.timer = rospy.Timer(rospy.Duration(1. / max(rate, .01)), self.tick)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        robot_q = np.array([q.x, q.y, q.z, q.w])
        norm = np.linalg.norm(robot_q)
        if not np.isfinite(norm) or norm < 1e-12:
            return
        robot_q /= norm
        offset = quaternion_product(quaternion_product(robot_q, np.r_[self.xyz, 0.]),
                                    robot_q * np.array([-1., -1., -1., 1.]))[:3]
        orientation = quaternion_product(robot_q, self.mount_q)
        pose = Pose()
        pose.position.x = msg.pose.pose.position.x + offset[0]
        pose.position.y = msg.pose.pose.position.y + offset[1]
        pose.position.z = msg.pose.pose.position.z + offset[2]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
        self.pose = pose
        self.frame = msg.header.frame_id or self.frame

    def tick(self, _):
        if self.pose is None:
            return
        def point(angle, elevation=0., radius=None):
            radius = self.max_range if radius is None else radius
            return Point(radius*math.cos(elevation)*math.cos(angle),
                         radius*math.cos(elevation)*math.sin(angle), radius*math.sin(elevation))
        def marker(identifier, kind, alpha):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.frame, rospy.Time.now()
            m.ns, m.id, m.type, m.action = 'forward_sonar', identifier, kind, Marker.ADD
            m.pose = self.pose
            m.color.g, m.color.b, m.color.a = .75, 1., alpha
            return m
        angles = np.linspace(-self.fov/2, self.fov/2, 65)
        fan = marker(0, Marker.TRIANGLE_LIST, .15)
        fan.scale.x = fan.scale.y = fan.scale.z = 1.
        for a, b in zip(angles[:-1], angles[1:]):
            fan.points.extend([point(a, radius=self.min_range), point(a), point(b)])
            fan.points.extend([point(a, radius=self.min_range), point(b), point(b, radius=self.min_range)])
        outline = marker(1, Marker.LINE_STRIP, .9)
        outline.scale.x = .06
        outline.points = [point(angles[0], radius=self.min_range)] + [point(a) for a in angles]
        outline.points += [point(a, radius=self.min_range) for a in angles[::-1]]
        outline.points += [point(angles[0])]
        volume = marker(2, Marker.LINE_LIST, .55)
        volume.scale.x = .035
        for elevation in [-self.vertical_fov/2, self.vertical_fov/2]:
            for a, b in zip(angles[:-1], angles[1:]):
                volume.points.extend([point(a, elevation), point(b, elevation)])
            for a in [angles[0], angles[-1]]:
                volume.points.extend([Point(), point(a, elevation)])
        self.pub.publish(MarkerArray(markers=[fan, outline, volume]))


if __name__ == '__main__':
    try:
        ForwardSonarRangeViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
