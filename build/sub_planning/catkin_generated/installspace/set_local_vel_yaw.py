#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Quaternion, Twist
from std_msgs.msg import Float32

class CombinedPlanner:
    def __init__(self):
        rospy.init_node('combined_planner', anonymous=True)
        
        # ---- 订阅：飞控状态与本地位置 ----
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # ---- 订阅：外部规划器给出的目标值 ----
        # 期望速度 (m/s)，使用 geometry_msgs/Twist
        self.cmd_vel_sub = rospy.Subscriber('/target_velocity', Twist, self.cmd_vel_cb)
        # 期望偏航角 (rad)，使用 std_msgs/Float32
        self.target_yaw_sub = rospy.Subscriber('/target_yaw', Float32, self.target_yaw_cb)

        # ---- 发布：两个独立的控制通道 ----
        self.vel_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self.att_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

        # ---- 内部状态变量 ----
        self.current_state = State()
        self.current_odom = Odometry()
        
        # 存储外部规划器给出的最新目标值
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_vz = 0.0
        self.target_yaw = 0.0
        
        # 标记是否收到过有效指令（可选）
        self.vel_received = False
        self.yaw_received = False

        # ---- 定时器：10Hz 控制循环 ----
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("Combined Planner waiting for external velocity/yaw commands...")

    def state_cb(self, msg):
        self.current_state = msg

    def odom_cb(self, msg):
        self.current_odom = msg

    def cmd_vel_cb(self, msg):
        """接收外部速度指令 (机体坐标系)"""
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_vz = msg.linear.z
        self.vel_received = True

    def target_yaw_cb(self, msg):
        """接收外部偏航角指令 (rad)"""
        self.target_yaw = msg.data
        self.yaw_received = True

    def control_loop(self, event):
        # 1. 安全检查
        if not self.current_state.connected or self.current_state.mode != "GUIDED" or not self.current_state.armed:
            rospy.logwarn_throttle(5, "Waiting for connection, GUIDED mode, or arming...")
            return

        # 2. 使用外部规划器提供的最新目标值
        #    如果没有收到过指令，可以选择不发送或发送零值
        if not self.vel_received or not self.yaw_received:
            rospy.logwarn_throttle(10, "Waiting for external velocity/yaw commands...")
            # 即使未收到，也可发送零指令，避免飞控超时退出GUIDED模式
            # 这里我们继续发送零值，以便飞控保持待命状态

        # 3. 构建并发布速度指令 (SET_POSITION_TARGET_LOCAL_NED)
        vel_msg = PositionTarget()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        
        vel_msg.type_mask = (PositionTarget.IGNORE_PX |
                             PositionTarget.IGNORE_PY |
                             PositionTarget.IGNORE_PZ |
                             PositionTarget.IGNORE_AFX |
                             PositionTarget.IGNORE_AFY |
                             PositionTarget.IGNORE_AFZ |
                             PositionTarget.IGNORE_YAW |
                             PositionTarget.IGNORE_YAW_RATE)
        
        vel_msg.velocity.x = self.target_vx
        vel_msg.velocity.y = self.target_vy
        vel_msg.velocity.z = self.target_vz
        self.vel_pub.publish(vel_msg)

        # 4. 构建并发布偏航指令 (SET_ATTITUDE_TARGET)
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 根据实际测试，该掩码组合能让姿态控制生效
        att_msg.type_mask = (AttitudeTarget.IGNORE_YAW_RATE | 
                             AttitudeTarget.IGNORE_PITCH_RATE | 
                             AttitudeTarget.IGNORE_ROLL_RATE)
        
        att_msg.body_rate = Vector3(0.0, 0.0, 0.0)
        quat = tf.transformations.quaternion_from_euler(0, 0, self.target_yaw)
        att_msg.orientation = Quaternion(*quat)
        att_msg.thrust = 0.0
        
        self.att_pub.publish(att_msg)

        rospy.loginfo_throttle(2, f"CMD: V_body=({self.target_vx:.2f}, {self.target_vy:.2f}, {self.target_vz:.2f}), Yaw={self.target_yaw:.2f}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = CombinedPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass