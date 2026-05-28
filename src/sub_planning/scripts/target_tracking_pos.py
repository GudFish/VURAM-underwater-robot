#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

class TargetRelay:
    def __init__(self):
        rospy.init_node('target_tracking_pos')
        
        # 订阅动态目标位置
        self.sub = rospy.Subscriber('/dynamic_target/pose', PoseStamped, self.callback)
        # 发布到目标话题
        self.pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
        
        rospy.loginfo("目标转发节点已启动，等待 /dynamic_target/pose 消息...")
    
    def callback(self, msg):
        # 直接转发
        self.pub.publish(msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = TargetRelay()
        node.run()
    except rospy.ROSInterruptException:
        pass