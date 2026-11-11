"""Odometry snapshots shared by PD and learned-policy controllers."""
import time

import numpy as np
import rospy
from nav_msgs.msg import Odometry

from utils.math3d import world_to_body


class PoseState(object):
    """MAVROS odometry snapshot: s[13] = [v_body6, quat_wxyz4, pos3].

    /mavros/local_position/odom 与 DVL odom 的 twist 均为机体系（FLU 轴，
    x 前 / y 左 / z 上），默认直接使用；~twist_in_world=true 时按四元数
    转到机体系（Gazebo pose_gt 场景）。
    """

    def __init__(self, odom_topic, twist_in_world=False, on_sample=None):
        self.twist_in_world = twist_in_world
        self.on_sample = on_sample
        self.pose_msg = None
        self._snapshot = None
        self.sub = rospy.Subscriber(odom_topic, Odometry, self._cb,
                                    queue_size=5)

    def _cb(self, msg):
        received = time.monotonic()
        if self.on_sample is not None:
            self.on_sample(msg.header.stamp.to_sec(), received)
        self._snapshot = (msg, received)
        self.pose_msg = msg

    def step(self):
        """Snapshot at the control tick; None until the first pose arrives."""
        snapshot = self._snapshot
        if snapshot is None:
            return None
        pm, received = snapshot
        pos = np.array([pm.pose.pose.position.x,
                        pm.pose.pose.position.y,
                        pm.pose.pose.position.z])
        q = pm.pose.pose.orientation
        q_wxyz = np.array([q.w, q.x, q.y, q.z])   # message xyzw -> state wxyz
        v = np.array([pm.twist.twist.linear.x,
                      pm.twist.twist.linear.y,
                      pm.twist.twist.linear.z])
        w = np.array([pm.twist.twist.angular.x,
                      pm.twist.twist.angular.y,
                      pm.twist.twist.angular.z])
        if self.twist_in_world:
            v = world_to_body(v, q_wxyz)
            w = world_to_body(w, q_wxyz)
        return {'state': np.concatenate([v, w, q_wxyz, pos]),
                'time': pm.header.stamp.to_sec(), 'received_monotonic': received}

