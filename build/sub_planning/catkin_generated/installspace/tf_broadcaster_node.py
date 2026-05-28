#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, PoseStamped
import PyKDL

class TFBroadcasterNode:
    def __init__(self):
        rospy.init_node('mavros_tf_broadcaster')
        # 创建一个tf2广播器
        self.br = tf2_ros.TransformBroadcaster()
        # 订阅MAVROS的位姿话题
        self.pose_sub = rospy.Subscriber('/mavros/local_position/pose', 
                                         PoseStamped, 
                                         self.pose_callback)

    def pose_callback(self, msg):
        # 构建TransformStamped消息
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        # 父坐标系设为世界系，根据您的系统，可以是 'map' 或 'odom'
        t.header.frame_id = "map"
        # 子坐标系设为机器人的根坐标系，必须与URDF中的base_link名称完全一致
        t.child_frame_id = "base_link" 

        # 填充平移部分 (position)
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z

        # 填充旋转部分 (orientation)
        t.transform.rotation.x = msg.pose.orientation.x
        t.transform.rotation.y = msg.pose.orientation.y
        t.transform.rotation.z = msg.pose.orientation.z
        t.transform.rotation.w = msg.pose.orientation.w

        # 发布这个变换
        self.br.sendTransform(t)

if __name__ == '__main__':
    try:
        TFBroadcasterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass