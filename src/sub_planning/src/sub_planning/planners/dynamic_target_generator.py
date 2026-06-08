#!/usr/bin/env python3
import math

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from shapely.geometry import Point, Polygon

class DynamicTargetGenerator:
    def __init__(self):
        rospy.init_node('dynamic_target_generator')
        
        # --- 等待边界参数加载完成 ---
        timeout = rospy.Duration(60.0)  # 最长等待30秒
        start_time = rospy.Time.now()
        boundary_points = []
        
        while not rospy.is_shutdown():
            if rospy.has_param('/boundary_loader/enu_points'):
                boundary_points = rospy.get_param('/boundary_loader/enu_points')
                if len(boundary_points) >= 3:
                    rospy.loginfo("成功加载边界参数，共 %d 个顶点", len(boundary_points))
                    break
            if rospy.Time.now() - start_time > timeout:
                rospy.logerr("等待边界参数超时（60秒），节点退出")
                rospy.signal_shutdown("边界参数未加载")
                return
            rospy.loginfo_throttle(5, "等待边界参数 /boundary_loader/enu_points 加载...")
            rospy.sleep(2)
        
        if rospy.is_shutdown():
            return
        
        # 构建shapely多边形
        self.boundary_polygon = Polygon(boundary_points)
        if not self.boundary_polygon.is_valid:
            self.boundary_polygon = self.boundary_polygon.buffer(0)  # 尝试修复无效多边形
        
        # 随机初始化目标位置在多边形内部
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        self.pos = np.array([0.0, 0.0])
        while not self.boundary_polygon.contains(Point(self.pos)):
            self.pos = np.array([
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y)
            ])
        
        # 运动参数
        # 使用平滑的“速度 + 航向角 + 转率”模型，避免布朗式抖动轨迹。
        self.speed_min = 0.25
        self.speed_max = 0.75
        self.speed = np.random.uniform(0.35, 0.55)
        self.heading = np.random.uniform(-math.pi, math.pi)
        self.turn_rate = 0.0
        self.speed_tau = 2.5
        self.turn_tau = 4.0
        self.speed_noise_std = 0.06
        self.turn_noise_std = 0.08
        self.max_turn_rate = 0.32
        self.boundary_margin = 2.0
        self.maneuver_mode = 'straight'
        self.mode_change_time = rospy.Time.now()
        self.mode_duration = rospy.Duration(np.random.uniform(4.0, 8.0))
        self.mode_target_speed = self.speed
        self.mode_target_turn_rate = 0.0
        self.velocity = np.array(
            [self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)],
            dtype=float,
        )
        
        # ROS发布者
        self.pose_pub = rospy.Publisher('/dynamic_target/pose', PoseStamped, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("动态目标生成器已启动，边界内初始位置: (%.2f, %.2f)", self.pos[0], self.pos[1])
    
    def is_inside(self, point):
        """检查点是否在边界内部"""
        return self.boundary_polygon.contains(Point(point))

    @staticmethod
    def wrap_pi(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_to_boundary(self, point):
        return self.boundary_polygon.exterior.distance(Point(point))

    def pick_new_mode(self):
        modes = ['straight', 'turn_left', 'turn_right', 'accelerate', 'decelerate']
        weights = np.array([0.52, 0.12, 0.12, 0.14, 0.10], dtype=float)
        self.maneuver_mode = np.random.choice(modes, p=weights / np.sum(weights))
        self.mode_change_time = rospy.Time.now()
        self.mode_duration = rospy.Duration(np.random.uniform(4.0, 8.0))

        if self.maneuver_mode == 'straight':
            self.mode_target_turn_rate = 0.0
            self.mode_target_speed = np.clip(self.speed + np.random.uniform(-0.03, 0.06), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'turn_left':
            self.mode_target_turn_rate = np.random.uniform(0.10, 0.22)
            self.mode_target_speed = np.clip(self.speed + np.random.uniform(-0.04, 0.02), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'turn_right':
            self.mode_target_turn_rate = -np.random.uniform(0.10, 0.22)
            self.mode_target_speed = np.clip(self.speed + np.random.uniform(-0.04, 0.02), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'accelerate':
            self.mode_target_turn_rate = np.random.uniform(-0.05, 0.05)
            self.mode_target_speed = np.random.uniform(max(self.speed, 0.50), self.speed_max)
        else:
            self.mode_target_turn_rate = np.random.uniform(-0.05, 0.05)
            self.mode_target_speed = np.random.uniform(self.speed_min, min(self.speed, 0.40))
    
    def update_position(self, dt):
        """更新位置，使用分模式机动 + 平滑连续转弯/变速模型。"""
        center = np.array(self.boundary_polygon.centroid.coords[0], dtype=float)
        to_center = center - self.pos
        center_heading = math.atan2(to_center[1], to_center[0])

        dist_to_boundary = self.distance_to_boundary(self.pos)
        boundary_gain = np.clip(
            (self.boundary_margin - dist_to_boundary) / max(self.boundary_margin, 1e-6),
            0.0,
            1.0,
        )
        if rospy.Time.now() - self.mode_change_time > self.mode_duration:
            self.pick_new_mode()

        if boundary_gain > 0.05:
            heading_error = self.wrap_pi(center_heading - self.heading)
            desired_turn_rate = np.clip(1.4 * heading_error / max(self.turn_tau, 1e-3), -self.max_turn_rate, self.max_turn_rate)
            desired_speed = max(self.speed_min, 0.42 - 0.12 * boundary_gain)
        else:
            desired_turn_rate = self.mode_target_turn_rate
            desired_speed = self.mode_target_speed
            heading_error = 0.0

        self.turn_rate += (-(self.turn_rate - desired_turn_rate) / max(self.turn_tau, 1e-3)) * dt
        self.turn_rate += np.random.normal(0.0, self.turn_noise_std) * math.sqrt(dt)
        self.turn_rate = np.clip(self.turn_rate, -self.max_turn_rate, self.max_turn_rate)

        self.speed += ((desired_speed - self.speed) / max(self.speed_tau, 1e-3)) * dt
        self.speed += np.random.normal(0.0, self.speed_noise_std) * math.sqrt(dt)
        self.speed = np.clip(self.speed, self.speed_min, self.speed_max)

        self.heading = self.wrap_pi(self.heading + self.turn_rate * dt)
        self.velocity = np.array(
            [self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)],
            dtype=float,
        )
        new_pos = self.pos + self.velocity * dt

        # 边界检查与柔性回退
        if not self.is_inside(new_pos):
            # 撞边后直接朝场地中心重新定向，避免沿边界切向顺移。
            self.heading = self.wrap_pi(center_heading + np.random.uniform(-0.12, 0.12))
            self.turn_rate = 0.0
            self.speed = max(self.speed_min, 0.75 * self.speed)
            self.velocity = np.array(
                [self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)],
                dtype=float,
            )
            new_pos = self.pos + self.velocity * dt
            if not self.is_inside(new_pos):
                norm = max(np.linalg.norm(to_center), 1e-6)
                new_pos = self.pos + 0.35 * to_center / norm
            self.pick_new_mode()
        
        self.pos = new_pos
    
    def run(self):
        dt = 1.0 / 10.0  # 对应10 Hz
        while not rospy.is_shutdown():
            self.update_position(dt)
            
            # 创建并发布PoseStamped消息
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"  # 假设ENU坐标系为map
            msg.pose.position.x = self.pos[0]
            msg.pose.position.y = self.pos[1]
            msg.pose.position.z = 0.0    # 水面目标，Z=0
            msg.pose.orientation.z = math.sin(0.5 * self.heading)
            msg.pose.orientation.w = math.cos(0.5 * self.heading)
            
            self.pose_pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = DynamicTargetGenerator()
        node.run()
    except rospy.ROSInterruptException:
        pass
