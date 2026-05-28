#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import tf
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float32

class PositionController:
    def __init__(self):
        rospy.init_node('position_controller', anonymous=True)

        # 订阅 RViz 发布的 2D Nav Goal 目标点
        self.sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.target_pose_cb)

        # ---- 订阅：获取ArduSub状态 ----
        rospy.Subscriber('/mavros/local_position/odom', Odometry, self.odom_cb)
        # ---- 订阅：获取外部目标位姿 ----
        rospy.Subscriber('/target_pose', PoseStamped, self.target_pose_cb)

        # ---- 发布：发布速度与偏航角指令 ----
        self.vel_pub = rospy.Publisher('/target_velocity', Twist, queue_size=10)
        self.yaw_pub = rospy.Publisher('/target_yaw', Float32, queue_size=10)

        # ---- 状态变量 ----
        self.current_pose = None          # [x, y, z, yaw] (ENU)
        self.current_twist = None
        self.target_received = False

        # ---- 目标点 (ENU) ----
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = math.pi/2

        # ---- 渐进偏航控制状态 ----
        self.current_target_yaw = 0.0     # 当前正在执行的渐进目标偏航角

        # ---- PID 参数 ----
        self.kp_x = rospy.get_param('~kp_x', 0.5)
        self.ki_x = rospy.get_param('~ki_x', 0.01)
        self.kd_x = rospy.get_param('~kd_x', 0.1)
        self.kp_y = rospy.get_param('~kp_y', 0.5)
        self.ki_y = rospy.get_param('~ki_y', 0.01)
        self.kd_y = rospy.get_param('~kd_y', 0.1)
        self.kp_z = rospy.get_param('~kp_z', 1.0)
        self.ki_z = rospy.get_param('~ki_z', 0.02)
        self.kd_z = rospy.get_param('~kd_z', 0.1)
        self.kp_yaw = rospy.get_param('~kp_yaw', 0.8)
        self.ki_yaw = rospy.get_param('~ki_yaw', 0.01)
        self.kd_yaw = rospy.get_param('~kd_yaw', 0.1)

        # ---- PID 状态 ----
        self.integral_x = 0.0; self.prev_error_x = 0.0
        self.integral_y = 0.0; self.prev_error_y = 0.0
        self.integral_z = 0.0; self.prev_error_z = 0.0
        self.integral_yaw = 0.0; self.prev_error_yaw = 0.0

        # ---- 控制参数 ----
        self.max_vel_x = rospy.get_param('~max_vel_x', 0.5)
        self.max_vel_y = rospy.get_param('~max_vel_y', 0.4)
        self.max_vel_z = rospy.get_param('~max_vel_z', 0.3)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 0.3)   # 最大偏航角速度 (rad/s)
        self.dt = rospy.get_param('~pos_ctrl_dt', 0.1)

        self.timer = rospy.Timer(rospy.Duration(self.dt), self.control_loop)
        rospy.loginfo("Position Controller started (Body-frame velocity output).")

    def odom_cb(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw = euler[2]

        self.current_pose = np.array([px, py, pz, yaw])
        self.current_twist = np.array([vx, vy, vz, 0.0])

    def target_pose_cb(self, msg):
        """收到目标位姿时，更新最终目标，并初始化渐进目标为当前实际偏航角"""
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        q = msg.pose.orientation
        euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.target_yaw = euler[2]
        # 若当前已有位姿，将渐进目标初始化为当前实际偏航角，避免跳变
        if self.current_pose is not None:
            self.current_target_yaw = self.current_pose[3]
        else:
            self.current_target_yaw = self.target_yaw   # 后备
        self.target_received = True
        rospy.loginfo(f"Target received: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), final yaw={self.target_yaw:.2f}")

    def control_loop(self, event):
        if self.current_pose is None:
            rospy.logwarn_throttle(5, "No odometry yet.")
            return
        if not self.target_received:
            rospy.loginfo_throttle(5, "No target pose received yet. Not publishing commands.")
            return

        # ---- 位置误差 (ENU) ----
        error_x = self.target_x - self.current_pose[0]
        error_y = self.target_y - self.current_pose[1]
        error_z = self.target_z - self.current_pose[2]

        # ---- 偏航误差：使用渐进目标 current_target_yaw ----
        error_yaw = self.target_yaw - self.current_target_yaw
        # 规范化误差到 [-pi, pi]
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

        # ---- PID 计算期望速度 (ENU) ----
        cmd_vx_enu = self.pid_control('x', error_x, self.dt)
        cmd_vy_enu = self.pid_control('y', error_y, self.dt)
        cmd_vz_enu = self.pid_control('z', error_z, self.dt)

        # ---- 坐标系旋转：ENU速度 -> 机体速度 ----
        yaw = self.current_pose[3]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_body =  cmd_vx_enu * cos_yaw + cmd_vy_enu * sin_yaw
        vy_body = -cmd_vx_enu * sin_yaw + cmd_vy_enu * cos_yaw
        vz_body = cmd_vz_enu

        # ---- 偏航角速度 PID 控制（隐式积分得到渐进目标）----
        cmd_yaw_rate = self.pid_control('yaw', error_yaw, self.dt)
        cmd_yaw_rate = np.clip(cmd_yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        # 积分更新 current_target_yaw
        self.current_target_yaw += cmd_yaw_rate * self.dt
        # 规范化角度到 [-pi, pi]
        self.current_target_yaw = math.atan2(math.sin(self.current_target_yaw), math.cos(self.current_target_yaw))

        # ---- 发布指令 ----
        vel_msg = Twist()
        vel_msg.linear.x = vx_body
        vel_msg.linear.y = vy_body
        vel_msg.linear.z = vz_body
        self.vel_pub.publish(vel_msg)

        yaw_msg = Float32()
        yaw_msg.data = self.current_target_yaw   # 发布渐进目标，而非最终目标
        self.yaw_pub.publish(yaw_msg)

        rospy.loginfo_throttle(2, f"Target ENU: ({self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}), "
                              f"Current ENU: ({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f}, {self.current_pose[2]:.2f}), "
                              f"Body Vel: ({vx_body:.2f}, {vy_body:.2f}, {vz_body:.2f}), "
                              f"Current Yaw Cmd: {self.current_target_yaw:.2f}, Final Yaw: {self.target_yaw:.2f}")

    def pid_control(self, axis, error, dt):
        if axis == 'x':
            kp, ki, kd = self.kp_x, self.ki_x, self.kd_x
            integral = self.integral_x
            prev_error = self.prev_error_x
        elif axis == 'y':
            kp, ki, kd = self.kp_y, self.ki_y, self.kd_y
            integral = self.integral_y
            prev_error = self.prev_error_y
        elif axis == 'z':
            kp, ki, kd = self.kp_z, self.ki_z, self.kd_z
            integral = self.integral_z
            prev_error = self.prev_error_z
        elif axis == 'yaw':
            kp, ki, kd = self.kp_yaw, self.ki_yaw, self.kd_yaw
            integral = self.integral_yaw
            prev_error = self.prev_error_yaw
        else:
            return 0.0

        integral += error * dt
        derivative = (error - prev_error) / dt if dt > 0 else 0.0
        output = kp * error + ki * integral + kd * derivative

        if axis == 'x':
            self.integral_x = integral
            self.prev_error_x = error
        elif axis == 'y':
            self.integral_y = integral
            self.prev_error_y = error
        elif axis == 'z':
            self.integral_z = integral
            self.prev_error_z = error
        elif axis == 'yaw':
            self.integral_yaw = integral
            self.prev_error_yaw = error

        # 限幅
        if axis == 'x':
            output = np.clip(output, -self.max_vel_x, self.max_vel_x)
        elif axis == 'y':
            output = np.clip(output, -self.max_vel_y, self.max_vel_y)
        elif axis == 'z':
            output = np.clip(output, -self.max_vel_z, self.max_vel_z)
        else:
            output = np.clip(output, -self.max_yaw_rate, self.max_yaw_rate)
        return output

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = PositionController()
        controller.run()
    except rospy.ROSInterruptException:
        pass