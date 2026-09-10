#!/usr/bin/env python3
"""
功能：在边界约束内生成动态目标和障碍物，为规划与感知链路提供仿真场景。
输入：通过 ROS 参数以及目标、里程计、边界和感知话题接收规划输入。
输出：发布目标、速度、偏航、障碍物或规划结果，并维护优化器内部状态。
"""
import math
import random

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from shapely.geometry import Point, Polygon
from visualization_msgs.msg import Marker, MarkerArray

class DynamicTargetGenerator:
    """
    功能：在边界内生成动态目标轨迹及障碍物场景。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('dynamic_target_generator')

        # --- 随机种子（用于场景复现）---
        # 使用两条独立随机流：
        #   rng_init   – 初始位置采样与障碍物创建（拒绝采样次数可变）
        #   rng_motion – 目标/障碍物运动演化（每循环固定消耗，保证确定性）
        # 两者互不影响：即使初始采样的拒绝次数因机器人位置微差而变化，
        # 运动随机序列仍然完全一致。
        random_seed = rospy.get_param('~random_seed', -1)
        if random_seed >= 0:
            seed = int(random_seed)
            np.random.seed(seed)
            random.seed(seed)
            self.rng_init = np.random.RandomState(seed)
            self.rng_motion = np.random.RandomState(seed)
            rospy.loginfo('[DynamicTarget] random seed set to %d', seed)
        else:
            self.rng_init = np.random
            self.rng_motion = np.random

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/dynamic_target/pose')
        self.target_marker_topic = rospy.get_param('~target_marker_topic', '/dynamic_target/marker')
        self.target_mesh_resource = rospy.get_param(
            '~target_mesh_resource',
            'package://sub_planning/meshes/AUV.STL',
        )
        self.target_mesh_scale = float(rospy.get_param('~target_mesh_scale', 1.0))
        self.target_mesh_z = float(rospy.get_param('~target_mesh_z', 0.0))
        self.target_mesh_color = rospy.get_param('~target_mesh_color', [1.0, 0.84, 0.0, 0.92])
        self.obstacle_marker_topic = rospy.get_param('~obstacle_marker_topic', '/dynamic_obstacles/markers')
        self.obstacle_count = int(rospy.get_param('~obstacle_count', 6))
        self.obstacle_max_marker_id = int(rospy.get_param('~obstacle_max_marker_id', 64))
        self.obstacle_size_min = float(rospy.get_param('~obstacle_size_min', 1.0))
        self.obstacle_size_max = float(rospy.get_param('~obstacle_size_max', 2.5))
        self.obstacle_speed_min = float(rospy.get_param('~obstacle_speed_min', 0.03))
        self.obstacle_speed_max = float(rospy.get_param('~obstacle_speed_max', 0.15))
        self.obstacle_shapes = rospy.get_param('~obstacle_shapes', ['sphere', 'box', 'cylinder'])
        self.obstacle_boundary_margin = float(rospy.get_param('~obstacle_boundary_margin', 2.0))
        self.init_near_robot = rospy.get_param('~init_near_robot', True)
        self.init_min_radius = float(rospy.get_param('~init_min_radius', 6.0))
        self.init_max_radius = float(rospy.get_param('~init_max_radius', 12.0))
        self.init_sample_attempts = int(rospy.get_param('~init_sample_attempts', 80))
        self.wait_robot_odom_timeout = float(rospy.get_param('~wait_robot_odom_timeout', 10.0))
        self.have_robot_odom = False
        self.robot_xy = np.array([0.0, 0.0], dtype=float)
        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)

        # --- 运动启动控制：等待飞控解锁后再开始目标/障碍物运动 ---
        self.mavros_state_topic = rospy.get_param('~mavros_state_topic', '/mavros/state')
        self.motion_start_on_arm = bool(rospy.get_param('~motion_start_on_arm', True))
        self.arm_wait_timeout = float(rospy.get_param('~arm_wait_timeout', 120.0))
        self.armed = False
        self.motion_started = False
        self.arm_wait_start = None
        self.arm_timeout_fallback_logged = False
        
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
        
        self.wait_for_robot_odom()
        self.pos = self.sample_initial_position()
        
        # 运动参数
        # 使用平滑的“速度 + 航向角 + 转率”模型，避免布朗式抖动轨迹。
        # 机动切换采用迭代计数而非墙钟时间，保证运动序列完全确定性。
        # 速度范围可在 launch 中通过 target_speed_min / target_speed_max 配置
        self.speed_min = max(0.05, float(rospy.get_param('~target_speed_min', 0.25)))
        self.speed_max = max(self.speed_min + 0.05, float(rospy.get_param('~target_speed_max', 0.75)))
        init_lo = min(self.speed_min + 0.10, self.speed_max)
        init_hi = max(self.speed_max - 0.20, self.speed_min)
        self.speed = self.rng_motion.uniform(min(init_lo, init_hi), max(init_lo, init_hi))
        self.heading = self.rng_motion.uniform(-math.pi, math.pi)
        self.turn_rate = 0.0
        self.speed_tau = 2.5
        self.turn_tau = 4.0
        self.speed_noise_std = 0.06
        self.turn_noise_std = 0.08
        self.max_turn_rate = 0.32
        self.boundary_margin = 2.0
        self.maneuver_mode = 'straight'
        self.mode_duration = self.rng_motion.uniform(4.0, 8.0)
        self.mode_steps_remaining = int(round(self.mode_duration * 10.0))
        self.mode_target_speed = self.speed
        self.mode_target_turn_rate = 0.0
        self.velocity = np.array(
            [self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)],
            dtype=float,
        )
        
        # ROS发布者
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher(self.target_marker_topic, Marker, queue_size=10)
        self.obstacle_pub = rospy.Publisher(self.obstacle_marker_topic, MarkerArray, queue_size=10)
        self.obstacles = self.create_obstacles()
        self.rate = rospy.Rate(10)  # 10 Hz

        # 飞控状态订阅（用于在 ARM 上升沿启动运动）
        self.state_sub = rospy.Subscriber(self.mavros_state_topic, State, self.state_cb, queue_size=1)
        
        rospy.loginfo("动态目标生成器已启动，边界内初始位置: (%.2f, %.2f)", self.pos[0], self.pos[1])

    def odom_cb(self, msg):
        """
        功能：执行 `odom_cb` 对应的核心处理逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        self.robot_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float)
        self.have_robot_odom = True

    def state_cb(self, msg):
        """
        功能：监听飞控状态，在 ARM 上升沿启动目标与障碍物运动。
        输入：`msg` 为 mavros_msgs/State 类型的 ROS 消息。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        was_armed = self.armed
        self.armed = msg.armed
        if not was_armed and msg.armed:
            # ARM 上升沿：运动时钟起点（与 metrics_logger 的 t_sec=0 对齐）
            self.motion_started = True
            rospy.loginfo('[DynamicTarget] vehicle ARMED — target/obstacle motion started')

    def motion_enabled(self):
        """
        功能：判断当前是否允许目标与障碍物运动。
        输入：无显式输入参数；依赖对象缓存状态与参数配置。
        输出：返回布尔值，True 表示运动已启动。
        """
        if not self.motion_start_on_arm:
            return True
        if self.armed:
            return True
        # 回退：超过等待时间仍未 ARM 时自动开始运动（非 MAVROS 场景）
        if self.arm_wait_start is not None and self.arm_wait_timeout >= 0:
            elapsed = (rospy.Time.now() - self.arm_wait_start).to_sec()
            if elapsed > self.arm_wait_timeout:
                if not self.arm_timeout_fallback_logged:
                    rospy.logwarn(
                        '[DynamicTarget] no ARM after %.1fs — starting motion anyway (fallback)',
                        self.arm_wait_timeout,
                    )
                    self.arm_timeout_fallback_logged = True
                return True
        return False

    def wait_for_robot_odom(self):
        """
        功能：等待机器人里程计就绪后再继续初始化流程。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        if not self.init_near_robot:
            return
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and not self.have_robot_odom:
            if (rospy.Time.now() - start).to_sec() > self.wait_robot_odom_timeout:
                rospy.logwarn(
                    "[DynamicTarget] waiting for robot odom timed out after %.1fs, falling back to global boundary sampling.",
                    self.wait_robot_odom_timeout,
                )
                return
            rospy.loginfo_throttle(2.0, "[DynamicTarget] waiting for robot odom: %s", self.robot_odom_topic)
            rate.sleep()

    def sample_global_position(self, rng=None):
        """
        功能：在允许区域内采样一个全局位置。
        输入：`rng` 可选随机数发生器，缺省使用初始化随机流 rng_init。
        输出：返回与该功能对应的计算结果。
        """
        if rng is None:
            rng = self.rng_init
        min_x, min_y, max_x, max_y = self.boundary_polygon.bounds
        while not rospy.is_shutdown():
            pos = np.array([rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)], dtype=float)
            if self.boundary_polygon.contains(Point(pos)):
                return pos
        return np.array([0.0, 0.0], dtype=float)

    def sample_initial_position(self):
        """
        功能：采样动态目标的初始位置。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        if not self.init_near_robot or not self.have_robot_odom:
            pos = self.sample_global_position()
            rospy.loginfo("[DynamicTarget] global initial target position: (%.2f, %.2f)", pos[0], pos[1])
            return pos

        r_min = max(0.0, min(self.init_min_radius, self.init_max_radius))
        r_max = max(r_min, self.init_max_radius)
        for _ in range(max(self.init_sample_attempts, 1)):
            radius = math.sqrt(self.rng_init.uniform(r_min * r_min, r_max * r_max))
            angle = self.rng_init.uniform(-math.pi, math.pi)
            pos = self.robot_xy + radius * np.array([math.cos(angle), math.sin(angle)], dtype=float)
            if self.boundary_polygon.contains(Point(pos)):
                rospy.loginfo(
                    "[DynamicTarget] initial target sampled near robot: robot=(%.2f, %.2f), target=(%.2f, %.2f), radius=%.2f",
                    self.robot_xy[0],
                    self.robot_xy[1],
                    pos[0],
                    pos[1],
                    radius,
                )
                return pos

        pos = self.sample_global_position()
        rospy.logwarn(
            "[DynamicTarget] failed to sample near robot after %d attempts, global fallback target=(%.2f, %.2f)",
            self.init_sample_attempts,
            pos[0],
            pos[1],
        )
        return pos
    
    def is_inside(self, point):
        """
        功能：判断给定点是否位于当前边界内部。
        输入：`point` 为该函数执行所需的输入参数。
        输出：返回布尔值或状态判定结果。
        """
        return self.boundary_polygon.contains(Point(point))

    @staticmethod
    def wrap_pi(angle):
        """
        功能：将输入角度归一化到 [-pi, pi] 区间。
        输入：`angle` 为该函数执行所需的输入参数。
        输出：返回归一化后的弧度角。
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def distance_to_boundary(self, point):
        """
        功能：计算给定点到边界的最近距离。
        输入：`point` 为该函数执行所需的输入参数。
        输出：返回与该功能对应的计算结果。
        """
        return self.boundary_polygon.exterior.distance(Point(point))

    def pick_new_mode(self):
        """
        功能：为动态目标选择新的运动模式或机动状态。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        modes = ['straight', 'turn_left', 'turn_right', 'accelerate', 'decelerate']
        weights = np.array([0.52, 0.12, 0.12, 0.14, 0.10], dtype=float)
        self.maneuver_mode = self.rng_motion.choice(modes, p=weights / np.sum(weights))
        self.mode_duration = self.rng_motion.uniform(4.0, 8.0)
        self.mode_steps_remaining = int(round(self.mode_duration * 10.0))

        if self.maneuver_mode == 'straight':
            self.mode_target_turn_rate = 0.0
            self.mode_target_speed = np.clip(self.speed + self.rng_motion.uniform(-0.03, 0.06), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'turn_left':
            self.mode_target_turn_rate = self.rng_motion.uniform(0.10, 0.22)
            self.mode_target_speed = np.clip(self.speed + self.rng_motion.uniform(-0.04, 0.02), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'turn_right':
            self.mode_target_turn_rate = -self.rng_motion.uniform(0.10, 0.22)
            self.mode_target_speed = np.clip(self.speed + self.rng_motion.uniform(-0.04, 0.02), self.speed_min, self.speed_max)
        elif self.maneuver_mode == 'accelerate':
            self.mode_target_turn_rate = self.rng_motion.uniform(-0.05, 0.05)
            # 加速目标速度下限随速度范围缩放（默认 0.25+0.25=0.50，与原行为一致）
            accel_lo = min(self.speed_min + 0.25, self.speed_max)
            self.mode_target_speed = self.rng_motion.uniform(max(self.speed, accel_lo), self.speed_max)
        else:
            self.mode_target_turn_rate = self.rng_motion.uniform(-0.05, 0.05)
            # 减速目标速度上限随速度范围缩放（默认 0.75-0.35=0.40，与原行为一致）
            decel_hi = max(self.speed_max - 0.35, self.speed_min)
            self.mode_target_speed = self.rng_motion.uniform(self.speed_min, min(self.speed, decel_hi))
    
    def update_position(self, dt):
        """
        功能：根据当前模式推进目标位置与运动状态。
        输入：`dt` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        center = np.array(self.boundary_polygon.centroid.coords[0], dtype=float)
        to_center = center - self.pos
        center_heading = math.atan2(to_center[1], to_center[0])

        dist_to_boundary = self.distance_to_boundary(self.pos)
        boundary_gain = np.clip(
            (self.boundary_margin - dist_to_boundary) / max(self.boundary_margin, 1e-6),
            0.0,
            1.0,
        )
        # 机动切换使用迭代计数（10Hz 固定步长），不受墙钟抖动影响
        if self.mode_steps_remaining <= 0:
            self.pick_new_mode()
        self.mode_steps_remaining -= 1

        if boundary_gain > 0.05:
            heading_error = self.wrap_pi(center_heading - self.heading)
            desired_turn_rate = np.clip(1.4 * heading_error / max(self.turn_tau, 1e-3), -self.max_turn_rate, self.max_turn_rate)
            desired_speed = max(self.speed_min, 0.42 - 0.12 * boundary_gain)
        else:
            desired_turn_rate = self.mode_target_turn_rate
            desired_speed = self.mode_target_speed
            heading_error = 0.0

        self.turn_rate += (-(self.turn_rate - desired_turn_rate) / max(self.turn_tau, 1e-3)) * dt
        self.turn_rate += self.rng_motion.normal(0.0, self.turn_noise_std) * math.sqrt(dt)
        self.turn_rate = np.clip(self.turn_rate, -self.max_turn_rate, self.max_turn_rate)

        self.speed += ((desired_speed - self.speed) / max(self.speed_tau, 1e-3)) * dt
        self.speed += self.rng_motion.normal(0.0, self.speed_noise_std) * math.sqrt(dt)
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
            self.heading = self.wrap_pi(center_heading + self.rng_motion.uniform(-0.12, 0.12))
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

    def create_obstacles(self):
        """
        功能：创建用于仿真的障碍物集合。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：返回与该功能对应的计算结果。
        """
        obstacles = []
        shapes = [str(item).lower() for item in self.obstacle_shapes]
        shapes = [item for item in shapes if item in ('sphere', 'box', 'cylinder')]
        if not shapes:
            shapes = ['sphere', 'box', 'cylinder']

        size_min = max(0.05, min(self.obstacle_size_min, self.obstacle_size_max))
        size_max = max(size_min, self.obstacle_size_max)
        speed_min = max(0.0, min(self.obstacle_speed_min, self.obstacle_speed_max))
        speed_max = max(speed_min, self.obstacle_speed_max)

        for idx in range(max(0, self.obstacle_count)):
            shape = str(self.rng_init.choice(shapes))
            pos = self.sample_global_position()
            heading = self.rng_init.uniform(-math.pi, math.pi)
            speed = self.rng_init.uniform(speed_min, speed_max)
            obstacle = {
                'id': idx,
                'shape': shape,
                'pos': pos,
                'heading': heading,
                'speed': speed,
                'scale': self.sample_obstacle_scale(shape, size_min, size_max),
                'yaw': self.rng_init.uniform(-math.pi, math.pi),
                'yaw_rate': self.rng_init.uniform(-0.08, 0.08),
                'color': self.sample_obstacle_color(),
            }
            obstacles.append(obstacle)
        rospy.loginfo("[DynamicTarget] generated %d sonar obstacle markers on %s", len(obstacles), self.obstacle_marker_topic)
        return obstacles

    def sample_obstacle_scale(self, shape, size_min, size_max):
        """
        功能：为障碍物采样合适的尺度参数。
        输入：`shape`、`size_min`、`size_max` 为该函数执行所需的输入参数。
        输出：返回与该功能对应的计算结果。
        """
        if shape == 'sphere':
            size = self.rng_init.uniform(size_min, size_max)
            return np.array([size, size, size], dtype=float)
        if shape == 'cylinder':
            diameter = self.rng_init.uniform(size_min, size_max)
            height = self.rng_init.uniform(size_min, size_max)
            return np.array([diameter, diameter, height], dtype=float)
        return np.array(
            [
                self.rng_init.uniform(size_min, size_max),
                self.rng_init.uniform(size_min, size_max),
                self.rng_init.uniform(size_min, size_max),
            ],
            dtype=float,
        )

    def sample_obstacle_color(self):
        """
        功能：生成障碍物可视化颜色。
        输入：无显式输入参数；使用初始化随机流 rng_init。
        输出：返回采样得到的颜色配置。
        """
        palette = [
            [0.25, 0.75, 1.0, 0.72],
            [1.0, 0.45, 0.22, 0.72],
            [0.55, 0.95, 0.35, 0.72],
            [0.95, 0.85, 0.25, 0.72],
        ]
        return list(palette[self.rng_init.randint(0, len(palette))])

    def update_obstacles(self, dt):
        """
        功能：更新障碍物的运行状态或可视化内容。
        输入：`dt` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        if not self.obstacles:
            return
        center = np.array(self.boundary_polygon.centroid.coords[0], dtype=float)
        for obstacle in self.obstacles:
            pos = obstacle['pos']
            to_center = center - pos
            center_heading = math.atan2(to_center[1], to_center[0])
            dist_to_boundary = self.distance_to_boundary(pos)
            boundary_gain = np.clip(
                (self.obstacle_boundary_margin - dist_to_boundary) / max(self.obstacle_boundary_margin, 1e-6),
                0.0,
                1.0,
            )
            if boundary_gain > 0.0:
                heading_error = self.wrap_pi(center_heading - obstacle['heading'])
                obstacle['heading'] = self.wrap_pi(obstacle['heading'] + np.clip(heading_error, -0.5, 0.5) * boundary_gain)
            else:
                obstacle['heading'] = self.wrap_pi(obstacle['heading'] + self.rng_motion.normal(0.0, 0.035) * math.sqrt(dt))

            velocity = obstacle['speed'] * np.array([math.cos(obstacle['heading']), math.sin(obstacle['heading'])], dtype=float)
            new_pos = pos + velocity * dt
            if not self.is_inside(new_pos):
                obstacle['heading'] = self.wrap_pi(center_heading + self.rng_motion.uniform(-0.18, 0.18))
                norm = max(np.linalg.norm(to_center), 1e-6)
                new_pos = pos + min(0.4, obstacle['speed'] * dt + 0.05) * to_center / norm
                if not self.is_inside(new_pos):
                    new_pos = self.sample_global_position(self.rng_motion)
            obstacle['pos'] = new_pos
            obstacle['yaw'] = self.wrap_pi(obstacle['yaw'] + obstacle['yaw_rate'] * dt)
    
    def run(self):
        """
        功能：执行 `run` 对应的核心处理逻辑。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；持续进入事件循环并保持节点运行。
        """
        dt = 1.0 / 10.0  # 对应10 Hz
        self.arm_wait_start = rospy.Time.now()
        while not rospy.is_shutdown():
            # 仅在运动已启动（ARM 或超时回退）时推进目标与障碍物
            # 未启动时保持初始位姿不变，但仍发布位姿/标记（场景就绪信号）
            if self.motion_enabled():
                self.update_position(dt)
                self.update_obstacles(dt)

            # 创建并发布PoseStamped消息
            msg = PoseStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"  # 假设ENU坐标系为map
            msg.pose.position.x = self.pos[0]
            msg.pose.position.y = self.pos[1]
            msg.pose.position.z = self.target_mesh_z
            msg.pose.orientation.z = math.sin(0.5 * self.heading)
            msg.pose.orientation.w = math.cos(0.5 * self.heading)

            self.pose_pub.publish(msg)
            self.publish_mesh_marker(msg)
            self.publish_obstacle_markers(msg.header.stamp)
            self.rate.sleep()

    def publish_mesh_marker(self, pose_msg):
        """
        功能：发布目标网格可视化标记。
        输入：`pose_msg` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = "dynamic_target"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.scale.x = self.target_mesh_scale
        marker.scale.y = self.target_mesh_scale
        marker.scale.z = self.target_mesh_scale
        marker.mesh_resource = self.target_mesh_resource
        marker.mesh_use_embedded_materials = False
        color = list(self.target_mesh_color)
        while len(color) < 4:
            color.append(1.0)
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        self.marker_pub.publish(marker)

    def publish_obstacle_markers(self, stamp):
        """
        功能：发布障碍物集合的可视化标记。
        输入：`stamp` 为该函数执行所需的输入参数。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        marker_array = MarkerArray()
        active_ids = set()
        for obstacle in self.obstacles:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.ns = "dynamic_obstacles"
            marker.id = int(obstacle['id'])
            active_ids.add(marker.id)
            marker.action = Marker.ADD
            if obstacle['shape'] == 'sphere':
                marker.type = Marker.SPHERE
            elif obstacle['shape'] == 'cylinder':
                marker.type = Marker.CYLINDER
            else:
                marker.type = Marker.CUBE
            marker.pose.position.x = float(obstacle['pos'][0])
            marker.pose.position.y = float(obstacle['pos'][1])
            marker.pose.position.z = float(obstacle['scale'][2] * 0.5)
            marker.pose.orientation.z = math.sin(0.5 * obstacle['yaw'])
            marker.pose.orientation.w = math.cos(0.5 * obstacle['yaw'])
            marker.scale.x = float(obstacle['scale'][0])
            marker.scale.y = float(obstacle['scale'][1])
            marker.scale.z = float(obstacle['scale'][2])
            color = obstacle['color']
            marker.color.r = float(color[0])
            marker.color.g = float(color[1])
            marker.color.b = float(color[2])
            marker.color.a = float(color[3])
            marker_array.markers.append(marker)
        for marker_id in range(max(self.obstacle_max_marker_id, self.obstacle_count)):
            if marker_id in active_ids:
                continue
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.ns = "dynamic_obstacles"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.obstacle_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        node = DynamicTargetGenerator()
        node.run()
    except rospy.ROSInterruptException:
        pass
