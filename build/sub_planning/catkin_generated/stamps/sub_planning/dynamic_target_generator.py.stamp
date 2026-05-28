#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from shapely.geometry import Point, Polygon

class DynamicTargetGenerator:
    def __init__(self):
        rospy.init_node('dynamic_target_generator')
        
        # 读取边界参数
        boundary_points = rospy.get_param('/boundary_loader/enu_points', [])
        if len(boundary_points) < 3:
            rospy.logerr("边界点不足，至少需要3个点构成多边形")
            rospy.signal_shutdown("边界参数错误")
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
        self.velocity = np.random.uniform(-1.0, 1.0, 2)  # 初始随机速度 (m/s)
        self.speed_limit = 2.0                           # 最大速率 (m/s)
        self.accel_std = 0.5                             # 加速度噪声标准差
        
        # ROS发布者
        self.pose_pub = rospy.Publisher('/dynamic_target/pose', PoseStamped, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        
        rospy.loginfo("动态目标生成器已启动，边界内初始位置: (%.2f, %.2f)", self.pos[0], self.pos[1])
    
    def is_inside(self, point):
        """检查点是否在边界内部"""
        return self.boundary_polygon.contains(Point(point))
    
    def update_position(self, dt):
        """更新位置，若出界则调整速度"""
        # 随机改变速度方向/大小（平滑随机游走）
        self.velocity += np.random.normal(0, self.accel_std, 2) * dt
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed_limit:
            self.velocity = self.velocity / speed * self.speed_limit
        
        new_pos = self.pos + self.velocity * dt
        
        # 边界检查与反弹逻辑
        if not self.is_inside(new_pos):
            # 简单反弹：反转速度分量并回退一步
            self.velocity = -self.velocity * 0.8  # 能量损失
            # 尝试微调位置使其回到内部
            new_pos = self.pos + self.velocity * dt
            if not self.is_inside(new_pos):
                # 如果反弹后仍在外，将位置重置到内部附近
                new_pos = self.pos - self.velocity * dt * 0.5
        
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
            msg.pose.orientation.w = 1.0 # 无旋转
            
            self.pose_pub.publish(msg)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = DynamicTargetGenerator()
        node.run()
    except rospy.ROSInterruptException:
        pass