#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：在 RViz 中发布前视声呐扇形量程可视化标记。
输入：通过 ROS 参数以及状态、边界、图像或置信度话题接收可视化输入。
输出：发布 RViz 标记、TF 或显示结果，辅助观察系统运行状态。
"""
import math

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


class ForwardSonarRangeViz:
    """
    功能：构建并发布前视声呐扇区范围的 RViz 标记。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('forward_sonar_range_viz', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.marker_topic = rospy.get_param('~marker_topic', '/forward_sonar/markers')
        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.publish_rate = float(rospy.get_param('~publish_rate', 8.0))
        self.fov_deg = float(rospy.get_param('~fov_deg', 90.0))
        self.max_range = float(rospy.get_param('~max_range', 35.0))
        self.min_range = float(rospy.get_param('~min_range', 0.8))

        self.have_robot = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0

        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=2)

        period = 1.0 / max(self.publish_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo(
            '[ForwardSonarRangeViz] publishing markers=%s fov=%.1f deg range=%.1f m',
            self.marker_topic,
            self.fov_deg,
            self.max_range,
        )

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id
        self.have_robot = True

    def body_to_world(self, bx, by):
        """
        功能：将机体系二维点转换到世界坐标系。
        输入：`bx`、`by` 为该函数执行所需的输入参数。
        输出：返回转换后的世界坐标分量。
        """
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return self.robot_x + c * bx - s * by, self.robot_y + s * bx + c * by

    def make_point(self, x, y, z=None):
        """
        功能：创建并返回一个 `geometry_msgs/Point` 对象。
        输入：`x`、`y`、`z` 为该函数执行所需的输入参数。
        输出：返回构造好的点消息对象。
        """
        point = Point()
        point.x = x
        point.y = y
        point.z = self.robot_z if z is None else z
        return point

    def publish_markers(self, stamp):
        """
        功能：构建并发布当前可视化 Marker 集合。
        输入：`stamp` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        marker_array = MarkerArray()
        half_fov = math.radians(self.fov_deg) * 0.5

        fan = Marker()
        fan.header.frame_id = self.frame_id
        fan.header.stamp = stamp
        fan.ns = 'forward_sonar'
        fan.id = 0
        fan.type = Marker.TRIANGLE_LIST
        fan.action = Marker.ADD
        fan.pose.orientation.w = 1.0
        fan.scale.x = fan.scale.y = fan.scale.z = 1.0
        fan.color.r = 0.0
        fan.color.g = 0.65
        fan.color.b = 1.0
        fan.color.a = 0.18
        steps = 36
        origin = self.make_point(self.robot_x, self.robot_y)
        prev_angle = -half_fov
        for idx in range(1, steps + 1):
            angle = -half_fov + idx * (2.0 * half_fov / steps)
            p0x, p0y = self.body_to_world(self.max_range * math.cos(prev_angle), self.max_range * math.sin(prev_angle))
            p1x, p1y = self.body_to_world(self.max_range * math.cos(angle), self.max_range * math.sin(angle))
            fan.points.extend([origin, self.make_point(p0x, p0y), self.make_point(p1x, p1y)])
            prev_angle = angle
        marker_array.markers.append(fan)

        outline = Marker()
        outline.header.frame_id = self.frame_id
        outline.header.stamp = stamp
        outline.ns = 'forward_sonar'
        outline.id = 1
        outline.type = Marker.LINE_STRIP
        outline.action = Marker.ADD
        outline.pose.orientation.w = 1.0
        outline.scale.x = 0.12
        outline.color.r = 0.0
        outline.color.g = 0.85
        outline.color.b = 1.0
        outline.color.a = 0.95
        right_x, right_y = self.body_to_world(self.max_range * math.cos(-half_fov), self.max_range * math.sin(-half_fov))
        left_x, left_y = self.body_to_world(self.max_range * math.cos(half_fov), self.max_range * math.sin(half_fov))
        outline.points.append(self.make_point(right_x, right_y))
        for idx in range(steps + 1):
            angle = -half_fov + idx * (2.0 * half_fov / steps)
            px, py = self.body_to_world(self.max_range * math.cos(angle), self.max_range * math.sin(angle))
            outline.points.append(self.make_point(px, py))
        outline.points.append(self.make_point(left_x, left_y))
        outline.points.append(self.make_point(self.robot_x, self.robot_y))
        outline.points.append(self.make_point(right_x, right_y))
        marker_array.markers.append(outline)

        self.marker_pub.publish(marker_array)

    def timer_cb(self, _event):
        """
        功能：执行 `timer_cb` 对应的核心处理逻辑。
        输入：`_event` 为 ROS 定时器事件；运行时还依赖对象内部状态与缓存数据。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if not self.have_robot:
            rospy.logwarn_throttle(3.0, '[ForwardSonarRangeViz] waiting for robot odometry: %s', self.robot_odom_topic)
            return
        self.publish_markers(rospy.Time.now())


if __name__ == '__main__':
    try:
        ForwardSonarRangeViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
