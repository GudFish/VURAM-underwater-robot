#!/usr/bin/env python3
"""
功能：广播 map 到 base_link 的 TF 变换，并支持对姿态方向做固定修正。
输入：通过 ROS 参数以及状态、边界、图像或置信度话题接收可视化输入。
输出：发布 RViz 标记、TF 或显示结果，辅助观察系统运行状态。
"""
import rospy
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry  # 备用：也支持 Odometry 话题

class TFBroadcasterNode:
    """
    功能：根据位姿消息广播 TF 变换并执行固定姿态修正。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
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
        
        # 订阅 MAVROS 位姿话题（在线运行时可用）
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose',
                                         PoseStamped,
                                         self.pose_callback)
        # 同时订阅 Odometry 话题（bag 回放时 /mavros/local_position/pose
        # 可能未被记录，而 odom 包含相同的位姿信息）
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom',
                                         Odometry,
                                         self.odom_callback)

    def quat_multiply(self, q1, q2):
        """
        功能：执行两个四元数的乘法。
        输入：`q1`、`q2` 为该函数执行所需的输入参数。
        输出：返回四元数乘法结果。
        """
        return tf_trans.quaternion_multiply(q1, q2)

    def broadcast_pose(self, pose):
        """
        功能：将位姿（Pose 类型）广播为 map -> base_link 的 TF 变换。
        输入：`pose` 为 geometry_msgs/Pose 类型的位姿。
        输出：无显式返回；发送 TF 变换。
        """
        orig_quat = [pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z,
                     pose.orientation.w]

        # 应用修正：最终四元数 = 修正四元数 * 原始四元数
        # 注意顺序：先修正，再应用原始旋转（通常修正代表坐标系的固定差异）
        corrected_quat = self.quat_multiply(orig_quat, self.correction_quat)

        # 构建 TransformStamped 消息
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"          # 父坐标系（世界系）
        t.child_frame_id = "base_link"     # 必须与 URDF 中的根 link 名称完全一致

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z

        t.transform.rotation.x = corrected_quat[0]
        t.transform.rotation.y = corrected_quat[1]
        t.transform.rotation.z = corrected_quat[2]
        t.transform.rotation.w = corrected_quat[3]

        self.br.sendTransform(t)

    def pose_callback(self, msg):
        """
        功能：处理位姿消息并广播对应 TF。
        输入：`msg` 为 PoseStamped 类型的 ROS 消息。
        输出：无显式返回；发送 TF 变换。
        """
        self.broadcast_pose(msg.pose)

    def odom_callback(self, msg):
        """
        功能：处理里程计消息并广播对应 TF（bag 回放时同样生效）。
        输入：`msg` 为 Odometry 类型的 ROS 消息。
        输出：无显式返回；发送 TF 变换。
        """
        self.broadcast_pose(msg.pose.pose)

if __name__ == '__main__':
    try:
        node = TFBroadcasterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass