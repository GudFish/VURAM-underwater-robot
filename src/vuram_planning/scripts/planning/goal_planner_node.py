#!/usr/bin/env python3
"""RViz 2D Nav Goal 目标点轨迹规划节点。

订阅 RViz "2D Nav Goal" 工具发布的 /move_base_simple/goal，从机器人当前
位姿+速度规划一条三次 Hermite 平滑轨迹（时间轴由期望速度近似确定），
以 20 Hz 发布瞬时参考（nav_msgs/Odometry：位置+姿态+速度+w_z）供策略
节点跟踪（ref_source:=external 模式）。到达终点后持续发布恒定的终点
参考（无限期保持）；运动中收到新目标时从当前位姿+速度无缝重规划。
深度 z 保持规划时刻的当前深度（忽略目标 z）。

注意：规划器死亡时策略会冻结在最后一条参考消息上（收敛到该点保持）；
推进器链路的 1 s 看门狗仍兜底 PWM。
"""
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from planning.trajectory import make_trajectory, yaw_to_quat


class GoalPlannerNode(object):
    """目标点轨迹规划：goal + odom -> Hermite 轨迹 -> 20 Hz 参考发布。"""

    def __init__(self):
        rospy.init_node('goal_planner', anonymous=False)

        # ---- ROS 参数 ----
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        self.odom_topic = rospy.get_param('~odom_topic',
                                          '/mavros/local_position/odom')
        self.ref_topic = rospy.get_param('~ref_topic', '/planning/reference_odom')
        self.desired_speed = float(rospy.get_param('~desired_speed', 0.3))
        self.publish_rate = float(rospy.get_param('~publish_rate', 20.0))
        self.viz_topic = rospy.get_param('~viz_topic', '/planning/path_array')
        self.viz_frame_id = rospy.get_param('~viz_frame_id', 'map')
        self.min_duration = float(rospy.get_param('~min_duration', 0.5))
        self.max_duration = float(rospy.get_param('~max_duration', 180.0))
        self.hold_dist = float(rospy.get_param('~hold_dist', 0.05))
        # mavros 与 DVL 的 odom twist 均为机体系；置 true 表示已是世界系
        self.twist_in_world = rospy.get_param('~twist_in_world', False)
        self.path_sample_dt = float(rospy.get_param('~path_sample_dt', 0.5))
        self.viz_rate = float(rospy.get_param('~viz_rate', 1.0))

        # ---- 状态 ----
        self._odom_msg = None
        self._traj = None
        self._traj_t0 = None
        self._viz_ma = MarkerArray()
        # 未设目标时的保持参考锚点：固定在首个 odom 位姿上（不跟随 odom
        # 漂移——否则 e_pos 恒为零，策略永远没有回拉力，车辆随波漂走）
        self._hold_pose = None

        # ---- 订阅 / 发布 ----
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped,
                                         self._goal_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry,
                                         self._odom_cb, queue_size=5)
        self.ref_pub = rospy.Publisher(self.ref_topic, Odometry, queue_size=1)
        self.viz_pub = rospy.Publisher(self.viz_topic, MarkerArray,
                                       queue_size=5)

        # ---- 定时器 ----
        self.tick_timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate),
                                      self._tick)
        self.viz_timer = rospy.Timer(rospy.Duration(1.0 / self.viz_rate),
                                     self._viz_tick)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        """缓存最新里程计；首个 odom 同时锁定保持参考锚点。"""
        self._odom_msg = msg
        if self._hold_pose is None:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self._hold_pose = (
                np.array([p.x, p.y, p.z]),
                np.array([q.x, q.y, q.z, q.w]))

    def _goal_cb(self, msg):
        """新目标：从当前位姿+速度无缝重规划（eval(0) = 当前状态）。"""
        if self._odom_msg is None:
            rospy.logwarn('goal received before first odom; ignored')
            return
        odom = self._odom_msg
        pos = np.array([odom.pose.pose.position.x,
                        odom.pose.pose.position.y,
                        odom.pose.pose.position.z])
        q = odom.pose.pose.orientation
        start_yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        v_body = np.array([odom.twist.twist.linear.x,
                           odom.twist.twist.linear.y,
                           odom.twist.twist.linear.z])
        if self.twist_in_world:
            v_world = v_body
        else:
            R = tf_trans.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
            v_world = R @ v_body

        # 2D 目标：z 保持规划时刻的当前深度（忽略目标 z）
        goal_pos = np.array([msg.pose.position.x,
                             msg.pose.position.y,
                             pos[2]])
        self._traj = make_trajectory(
            pos, v_world, goal_pos, start_yaw,
            v_des=self.desired_speed,
            t_min=self.min_duration, t_max=self.max_duration,
            hold_dist=self.hold_dist)
        self._traj_t0 = rospy.Time.now()
        self._build_viz()
        self.viz_pub.publish(self._viz_ma)
        rospy.loginfo('new goal (%.2f, %.2f): dist=%.2f m, T=%.2f s, '
                      'start speed=%.2f m/s',
                      goal_pos[0], goal_pos[1],
                      float(np.linalg.norm(goal_pos - pos)),
                      self._traj.T,
                      float(np.linalg.norm(v_world)))

    # ------------------------------------------------------------------
    # 参考发布
    # ------------------------------------------------------------------

    def _tick(self, _event):
        """20 Hz：发布瞬时参考（无 odom 不发布；无目标发保持参考）。"""
        if self._odom_msg is None:
            return
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.viz_frame_id

        if self._traj is None:
            # 尚无目标：保持参考 = 锚定位姿（首个 odom 处，固定不动）+ 零速度
            # （若回显当前 odom，参考会跟着车辆漂移，e_pos 恒零、无回拉力）
            hp_pos, hp_quat = self._hold_pose
            msg.pose.pose.position.x = hp_pos[0]
            msg.pose.pose.position.y = hp_pos[1]
            msg.pose.pose.position.z = hp_pos[2]
            msg.pose.pose.orientation.x = hp_quat[0]
            msg.pose.pose.orientation.y = hp_quat[1]
            msg.pose.pose.orientation.z = hp_quat[2]
            msg.pose.pose.orientation.w = hp_quat[3]
            # twist 全零（保持）
        else:
            t = (rospy.Time.now() - self._traj_t0).to_sec()
            pos, rpy, vel, w_z = self._traj.eval(t)
            msg.pose.pose.position.x = pos[0]
            msg.pose.pose.position.y = pos[1]
            msg.pose.pose.position.z = pos[2]
            q = yaw_to_quat(rpy[2])
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            msg.twist.twist.linear.x = vel[0]
            msg.twist.twist.linear.y = vel[1]
            msg.twist.twist.linear.z = vel[2]
            msg.twist.twist.angular.z = w_z
        self.ref_pub.publish(msg)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def _build_viz(self):
        """重建路径/目标 marker（轨迹变化时调用）。"""
        ma = MarkerArray()
        path = Marker()
        path.header.frame_id = self.viz_frame_id
        path.header.stamp = rospy.Time.now()
        path.ns = 'planning'
        path.id = 0
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.scale.x = 0.03
        # 零范数四元数会被 RViz 静默丢弃，必须置单位四元数
        path.pose.orientation.w = 1.0
        path.color.r, path.color.g, path.color.b, path.color.a = \
            0.1, 0.8, 0.9, 0.9
        for p in self._traj.sample_positions(self.path_sample_dt):
            pt = Point()
            pt.x, pt.y, pt.z = p[0], p[1], p[2]
            path.points.append(pt)
        ma.markers.append(path)

        goal_marker = Marker()
        goal_marker.header.frame_id = self.viz_frame_id
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = 'planning'
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.scale.x = goal_marker.scale.y = goal_marker.scale.z = 0.25
        goal_marker.pose.orientation.w = 1.0
        goal_marker.pose.position.x = self._traj.goal_pos[0]
        goal_marker.pose.position.y = self._traj.goal_pos[1]
        goal_marker.pose.position.z = self._traj.goal_pos[2]
        goal_marker.color.r, goal_marker.color.g, goal_marker.color.b, \
            goal_marker.color.a = 1.0, 0.9, 0.1, 0.8
        ma.markers.append(goal_marker)
        self._viz_ma = ma

    def _viz_tick(self, _event):
        if self._traj is not None and self._viz_ma.markers:
            self._viz_ma.markers[0].header.stamp = rospy.Time.now()
            self._viz_ma.markers[1].header.stamp = rospy.Time.now()
            self.viz_pub.publish(self._viz_ma)


if __name__ == '__main__':
    try:
        node = GoalPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
