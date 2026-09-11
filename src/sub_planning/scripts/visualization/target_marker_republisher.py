#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
target_marker_republisher.py
回放 rosbag 时重发布目标网格 marker。

背景：目标 AUV 的 3D 模型由仿真节点 dynamic_target_generator.py 通过
/forward_sonar/marker 发布（MESH_RESOURCE，AUV.STL），该节点在回放时
不运行，且该话题未被 bag_record.launch 录制；bag 中只有 /dynamic_target/pose。
本节点订阅回放出的 /dynamic_target/pose，按与仿真完全相同的参数
（网格资源、缩放、颜色、ns/id）重发布 /dynamic_target/marker，
使 RViz 的 "Dynamic Target AUV" 显示在回放时可见。

话题参数：
  ~pose_topic    – 目标位姿话题（默认 /dynamic_target/pose）
  ~marker_topic  – 网格 marker 输出话题（默认 /dynamic_target/marker）
  ~mesh_resource – 网格资源（默认 package://sub_planning/meshes/AUV.STL）
  ~mesh_scale    – 网格缩放（默认 1.0，与仿真一致）
  ~mesh_color    – RGBA 颜色（默认 [1.0, 0.84, 0.0, 0.92]，与仿真一致）
"""
import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class TargetMarkerRepublisher(object):
    def __init__(self):
        self.mesh_resource = rospy.get_param(
            '~mesh_resource', 'package://sub_planning/meshes/AUV.STL')
        self.mesh_scale = float(rospy.get_param('~mesh_scale', 1.0))
        self.mesh_color = rospy.get_param(
            '~mesh_color', [1.0, 0.84, 0.0, 0.92])
        self.pose_topic = rospy.get_param('~pose_topic', '/dynamic_target/pose')
        self.marker_topic = rospy.get_param('~marker_topic', '/dynamic_target/marker')
        self.ns = rospy.get_param('~ns', 'dynamic_target')

        # 回放高倍速时限制重发布频率，避免无谓的重复渲染开销
        self.min_interval = rospy.Duration(rospy.get_param('~min_interval', 0.05))
        self.last_publish = None

        self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.pose_cb, queue_size=1)
        self.pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)
        rospy.loginfo('[target_marker_republisher] %s -> %s', self.pose_topic, self.marker_topic)

    def pose_cb(self, msg):
        """
        功能：接收目标位姿，按仿真同款参数发布网格 marker。
        输入：`msg` 为 /dynamic_target/pose 的 PoseStamped 消息。
        输出：无显式返回；发布 /dynamic_target/marker。
        """
        now = rospy.Time.now()
        if self.last_publish is not None and now - self.last_publish < self.min_interval:
            return
        self.last_publish = now

        marker = Marker()
        marker.header = msg.header
        marker.ns = self.ns
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = msg.pose
        marker.scale.x = self.mesh_scale
        marker.scale.y = self.mesh_scale
        marker.scale.z = self.mesh_scale
        marker.mesh_resource = self.mesh_resource
        marker.mesh_use_embedded_materials = False
        color = list(self.mesh_color)
        while len(color) < 4:
            color.append(1.0)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        self.pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node('target_marker_republisher')
    TargetMarkerRepublisher()
    rospy.spin()
