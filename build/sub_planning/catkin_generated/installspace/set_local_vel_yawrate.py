#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
from mavros_msgs.msg import State, PositionTarget, AttitudeTarget
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3, Quaternion

class CombinedPlanner:
    def __init__(self):
        rospy.init_node('combined_planner', anonymous=True)
        
        # ---- 订阅 ----
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_cb)
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)

        # ---- 发布：两个独立的通道 ----
        # 通道1：控制速度 (通过 SET_POSITION_TARGET_LOCAL_NED)
        self.vel_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        # 通道2：控制偏航率 (通过 SET_ATTITUDE_TARGET)
        self.att_pub = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=10)

        # ---- 状态变量 ----
        self.current_state = State()
        self.current_odom = Odometry()
        
        # ---- 定时器：10Hz 规划循环 ----
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("Combined Planner (Velocity + YawRate) started. Waiting for GUIDED mode...")

    def state_cb(self, msg):
        self.current_state = msg

    def odom_cb(self, msg):
        self.current_odom = msg

    def control_loop(self, event):
        # 1. 安全检查
        if not self.current_state.connected or self.current_state.mode != "GUIDED" or not self.current_state.armed:
            rospy.logwarn_throttle(5, "Waiting for connection, GUIDED mode, or arming...")
            return

        # 2. 核心规划逻辑 (10Hz)
        #    在这里设定你的期望速度(m/s) 和 偏航角速度(rad/s)
        t = rospy.Time.now().to_sec()
        
        # 2.1 速度规划 (例如，在机体坐标系下)
        vx_body = 0.3  # 前进速度 0.3 m/s
        vy_body = 0.2  # 横向速度
        vz_body = 0.0  # 垂向速度

        # 3. 构建并发布速度指令 (SET_POSITION_TARGET_LOCAL_NED)
        vel_msg = PositionTarget()
        vel_msg.header.stamp = rospy.Time.now()
        vel_msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        
        # 关键type_mask: 只控制速度，忽略其他
        # 这个掩码告诉飞控：忽略位置、加速度、偏航、偏航率
        vel_msg.type_mask = (PositionTarget.IGNORE_PX |
                             PositionTarget.IGNORE_PY |
                             PositionTarget.IGNORE_PZ |
                             PositionTarget.IGNORE_AFX |
                             PositionTarget.IGNORE_AFY |
                             PositionTarget.IGNORE_AFZ |
                             PositionTarget.IGNORE_YAW_RATE)
        
        vel_msg.velocity.x = vx_body
        vel_msg.velocity.y = vy_body
        vel_msg.velocity.z = vz_body
        vel_msg.yaw = 1
        self.vel_pub.publish(vel_msg)

        # 4. 构建并发布偏航率指令 (SET_ATTITUDE_TARGET)
        att_msg = AttitudeTarget()
        att_msg.header.stamp = rospy.Time.now()
        
        # 关键type_mask: 只控制角速度，忽略姿态和推力
        # 这个掩码告诉飞控：忽略四元数姿态、忽略推力
        att_msg.type_mask = 0
        
        # 只设置偏航率，其他角速度为0
        att_msg.body_rate = Vector3(0.0, 0.0, 0.0)

        target_yaw = 1
        quat = tf.transformations.quaternion_from_euler(0, 0, target_yaw)
        att_msg.orientation = Quaternion(*quat)
        
        # 推力设为0，因为已被忽略
        att_msg.thrust = 0.0
        
        # self.att_pub.publish(att_msg)

        rospy.loginfo_throttle(2, f"CMD: V_body=({vx_body:.2f}, {vy_body:.2f}, {vz_body:.2f}), Yaw={target_yaw:.2f}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        planner = CombinedPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass