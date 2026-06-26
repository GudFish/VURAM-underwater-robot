#!/usr/bin/env python3
import rospy
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry  # 备用：也支持 Odometry 话题

class TFBroadcasterNode:
    def __init__(self):
        rospy.init_node('mavros_tf_broadcaster')
        self.br = tf2_ros.TransformBroadcaster()
        
        # --- 修正四元数配置 ---
        # 根据你的颠倒情况，选择下面其中一种修正（注释/取消注释）
        # 修正1：绕 X 轴旋转 180°（最常用，适用于 ENU ↔ NED 或上下颠倒）
        # self.correction_quat = tf_trans.quaternion_from_euler(3.14159, 0.0, 0.0)   # (pi, 0, 0)
        # 修正2：绕 Y 轴旋转 180°（如果前后颠倒且上下颠倒）
        # self.correction_quat = tf_trans.quaternion_from_euler(0.0, 3.14159, 0.0)
        # 修正3：绕 Z 轴旋转 180°（仅水平调头，不解决上下颠倒，一般不用）
        self.correction_quat = tf_trans.quaternion_from_euler(0.0, 0.0, 3.14159/2)
        # 修正4：无修正（保持原始朝向）
        # self.correction_quat = [0.0, 0.0, 0.0, 1.0]
        
        # 订阅 MAVROS 位姿话题
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', 
                                         PoseStamped, 
                                         self.pose_callback)
        # 可选：如果你使用的是 Odometry 话题，可以同时订阅它（例如 /mavros/local_position/odom）
        # self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_callback)

    def quat_multiply(self, q1, q2):
        """四元数乘法：q1 * q2，返回 [x, y, z, w]"""
        return tf_trans.quaternion_multiply(q1, q2)

    def pose_callback(self, msg):
        # 接收到的原始四元数 (x, y, z, w)
        orig_quat = [msg.pose.orientation.x,
                     msg.pose.orientation.y,
                     msg.pose.orientation.z,
                     msg.pose.orientation.w]
        
        # 应用修正：最终四元数 = 修正四元数 * 原始四元数
        # 注意顺序：先修正，再应用原始旋转（通常修正代表坐标系的固定差异）
        corrected_quat = self.quat_multiply(orig_quat, self.correction_quat)
        
        # 构建 TransformStamped 消息
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"          # 父坐标系（世界系）
        t.child_frame_id = "base_link"     # 必须与 URDF 中的根 link 名称完全一致
        
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        
        t.transform.rotation.x = corrected_quat[0]
        t.transform.rotation.y = corrected_quat[1]
        t.transform.rotation.z = corrected_quat[2]
        t.transform.rotation.w = corrected_quat[3]
        
        self.br.sendTransform(t)

    # 可选：如果使用 Odometry 话题
    def odom_callback(self, msg):
        # 类似 pose_callback，从 msg.pose.pose 中提取位姿
        pass

if __name__ == '__main__':
    try:
        node = TFBroadcasterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass