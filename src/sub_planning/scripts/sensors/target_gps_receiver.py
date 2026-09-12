#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：实物实验中的目标 GPS 接收与声学方位解算节点。

背景：实物实验中没有仿真传感器（acoustic_tdoa_sensor_sim）与动态目标场景
（dynamic_target_generator）。目标（如另一条艇）通过局域网 UDP 周期性向机器人
发送其真实 GPS 位置（lat,lon,alt）。本节点完成四件事：
  1. 接收目标 GPS，逐条以 NavSatFix 转发到 /target_gps/fix（按接收频率，
     保留原始 GPS 数据供 rosbag 记录与离线分析）；
  2. 转换为以飞控 home 为原点的 ENU 坐标，以 PoseStamped 发布到
     /dynamic_target/pose（作为目标位置真值，frame_id 为 map，与仿真
     dynamic_target_generator 保持同一话题约定，可直接被 metrics_logger
     与 auto_arm_guided 复用）；
  3. 结合机器人当前位姿（/mavros/local_position/odom，ENU）解算目标相对
     机器人机体系的方位角（机头方向为零、逆时针为正，单位弧度），以 Float32
     发布到 /target_acoustic_bearing —— 与 tdoa_bearing_estimator 的输出约定
     完全一致，可直接被 target_pdaf_ekf 与 mpc_belief_target_tracker 消费；
  4. 方位解算频率保持与仿真 TDOA 量测频率一致（~bearing_publish_rate，
     默认 1 Hz），不随目标 GPS 高频发送而提高。

UDP 报文格式（UTF-8 文本，两种均可）：
  JSON: {"lat": 30.123456, "lon": 120.123456, "alt": 0.0}   (alt 可省略)
  CSV : "30.123456,120.123456,0.0"                          (alt 可省略)

话题参数：
  ~udp_port            – UDP 监听端口（默认 4002）
  ~target_pose_topic   – 目标真值位姿话题（默认 /dynamic_target/pose）
  ~bearing_topic       – 声学方位话题（默认 /target_acoustic_bearing）
  ~fix_topic           – 原始目标 GPS 转发话题（默认 /target_gps/fix，NavSatFix）
  ~odom_topic          – 机器人 ENU 里程计话题（默认 /mavros/local_position/odom）
  ~home_topic          – 飞控 home 位置话题（默认 /mavros/home_position/home）
"""
import json
import math
import socket
import threading

import pymap3d as pm
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import HomePosition
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Float32, Float32MultiArray


class TargetGpsReceiver:
    """
    功能：接收目标 GPS，发布目标 ENU 真值位姿与相对机体系的方位角。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('target_gps_receiver')

        # 1. UDP 接收参数
        self.bind_ip = rospy.get_param('~bind_ip', '0.0.0.0')
        self.udp_port = int(rospy.get_param('~udp_port', 4002))

        # 2. 话题接口参数
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/dynamic_target/pose')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.fix_topic = rospy.get_param('~fix_topic', '/target_gps/fix')
        self.debug_topic = rospy.get_param('~debug_topic', '/target_gps/debug')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.home_topic = rospy.get_param('~home_topic', '/mavros/home_position/home')
        self.target_frame = rospy.get_param('~target_frame', 'map')

        # 3. 发布频率与数据新鲜度超时参数
        # GPS 位置短时间不变，真值位姿可以稍宽松；方位解算对新鲜度要求更高。
        self.pose_publish_rate = float(rospy.get_param('~pose_publish_rate', 5.0))
        # 与仿真 TDOA 量测频率保持一致（1 Hz），保证实验口径对齐。
        self.bearing_publish_rate = float(rospy.get_param('~bearing_publish_rate', 1.0))
        self.pose_timeout = float(rospy.get_param('~pose_timeout', 30.0))
        self.bearing_timeout = float(rospy.get_param('~bearing_timeout', 5.0))

        # 4. 状态缓存
        self.home = None            # (lat, lon, alt)，飞控 home 位置
        self.robot_x = 0.0          # 机器人 ENU 位置
        self.robot_y = 0.0
        self.robot_yaw = 0.0        # 机器人 ENU 航向角
        self.have_odom = False
        self.target_enu = None      # 目标 ENU 位置 (x, y, z)
        self.last_rx_time = None    # 最近一次收到目标 GPS 的时间
        self.msg_count = 0
        self.parse_error_count = 0

        # 5. 订阅与发布
        rospy.Subscriber(self.home_topic, HomePosition, self.home_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.pose_pub = rospy.Publisher(self.target_pose_topic, PoseStamped, queue_size=1)
        self.bearing_pub = rospy.Publisher(self.bearing_topic, Float32, queue_size=1)
        self.fix_pub = rospy.Publisher(self.fix_topic, NavSatFix, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, Float32MultiArray, queue_size=1)

        # 6. 等待 home 位置（与 boundary_loader 相同的约定）
        rospy.loginfo('Waiting for home position from FCU...')
        rate = rospy.Rate(1)
        while self.home is None and not rospy.is_shutdown():
            rate.sleep()
        if self.home is None:
            rospy.logerr('[TargetGpsReceiver] home position not received (FCU down or shutting down)')
            return
        rospy.loginfo('Home position received: lat=%.7f, lon=%.7f, alt=%.2f',
                      self.home[0], self.home[1], self.home[2])

        # 7. 启动 UDP 接收线程与发布定时器
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.udp_port))
        self.sock.settimeout(0.1)
        self.rx_thread = threading.Thread(target=self.rx_loop, daemon=True)
        self.rx_thread.start()
        rospy.Timer(rospy.Duration(1.0 / max(self.pose_publish_rate, 0.1)), self.pose_timer_cb)
        rospy.Timer(rospy.Duration(1.0 / max(self.bearing_publish_rate, 0.1)), self.bearing_timer_cb)
        rospy.loginfo('[TargetGpsReceiver] listening on UDP %s:%d, pose=%s bearing=%s',
                      self.bind_ip, self.udp_port, self.target_pose_topic, self.bearing_topic)

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------
    def home_cb(self, msg):
        """功能：缓存飞控 home 位置（GPS 转 ENU 的原点）。"""
        self.home = (msg.geo.latitude, msg.geo.longitude, msg.geo.altitude)

    def odom_cb(self, msg):
        """功能：缓存机器人 ENU 位置与航向角。"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_odom = True

    # ------------------------------------------------------------------
    # UDP 接收
    # ------------------------------------------------------------------
    def rx_loop(self):
        """功能：UDP 接收线程，解析目标 GPS 并缓存最新 ENU 位置。"""
        while not rospy.is_shutdown():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            gps = self.parse_gps(data)
            if gps is None:
                self.parse_error_count += 1
                rospy.logwarn_throttle(5.0, '[TargetGpsReceiver] unparsable datagram from %s (total %d): %r',
                                       addr, self.parse_error_count, data[:64])
                continue

            self.target_enu = pm.geodetic2enu(gps[0], gps[1], gps[2], *self.home)
            self.last_rx_time = rospy.Time.now()
            self.msg_count += 1

            # 逐条转发原始目标 GPS（NavSatFix），供 rosbag 记录与离线分析
            fix = NavSatFix()
            fix.header.stamp = self.last_rx_time
            fix.header.frame_id = 'gps'
            fix.status.status = NavSatStatus.STATUS_FIX
            fix.status.service = NavSatStatus.SERVICE_GPS
            fix.latitude = gps[0]
            fix.longitude = gps[1]
            fix.altitude = gps[2]
            self.fix_pub.publish(fix)

            range_m, bearing = self._rel_range_bearing()
            rospy.loginfo_throttle(5.0,
                                   '[TargetGpsReceiver] GPS from %s (msg %d): lat=%.7f lon=%.7f alt=%.2f'
                                   ' -> enu=(%.2f, %.2f) range=%.1f m bearing=%.2f deg',
                                   addr, self.msg_count, gps[0], gps[1], gps[2],
                                   self.target_enu[0], self.target_enu[1],
                                   range_m, math.degrees(bearing))

    def parse_gps(self, raw):
        """
        功能：解析 UDP 报文中的目标 GPS 坐标。
        输入：`raw` 为接收到的原始字节串；其余输入依赖当前对象缓存状态。
        输出：返回 (lat, lon, alt) 三元组；解析失败返回 None。
        """
        text = raw.decode('utf-8', errors='replace').strip()
        if not text:
            return None

        # JSON 格式：{"lat": .., "lon": .., "alt": ..}
        if text.startswith('{'):
            try:
                data = json.loads(text)
                return float(data['lat']), float(data['lon']), float(data.get('alt', 0.0))
            except (ValueError, KeyError, TypeError):
                return None

        # CSV 格式：lat,lon[,alt]，支持逗号/空格/分号分隔
        for sep in (',', ' ', ';'):
            parts = text.split(sep)
            if len(parts) >= 2:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    alt = float(parts[2]) if len(parts) >= 3 and parts[2].strip() else 0.0
                    return lat, lon, alt
                except ValueError:
                    continue
        return None

    # ------------------------------------------------------------------
    # 定时发布
    # ------------------------------------------------------------------
    def pose_timer_cb(self, _event):
        """功能：按频率发布目标 ENU 真值位姿（PoseStamped，map 系）。"""
        if self.target_enu is None or not self._fresh(self.pose_timeout):
            return
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.target_frame
        msg.pose.position.x = self.target_enu[0]
        msg.pose.position.y = self.target_enu[1]
        msg.pose.position.z = self.target_enu[2]
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

    def bearing_timer_cb(self, _event):
        """功能：按频率解算并发布目标相对机体系的方位角（Float32，弧度）。"""
        if self.target_enu is None or not self.have_odom:
            return
        if not self._fresh(self.bearing_timeout):
            rospy.logwarn_throttle(5.0, '[TargetGpsReceiver] target GPS stale (%.1fs), bearing not published',
                                   (rospy.Time.now() - self.last_rx_time).to_sec())
            return

        range_m, bearing = self._rel_range_bearing()
        msg = Float32()
        msg.data = bearing
        self.bearing_pub.publish(msg)

        debug = Float32MultiArray()
        debug.data = [range_m, self._wrap_pi(self.robot_yaw + bearing), bearing]
        self.debug_pub.publish(debug)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _fresh(self, timeout):
        """功能：判断最近一次目标 GPS 是否在 timeout 秒内收到。"""
        return self.last_rx_time is not None and \
            (rospy.Time.now() - self.last_rx_time).to_sec() <= timeout

    def _rel_range_bearing(self):
        """
        功能：计算目标相对机器人的距离与机体系方位角。
        输出：返回 (range_m, bearing_rad) 二元组，bearing 为机体系方位
              （机头为零、逆时针为正，与 tdoa_bearing_estimator 约定一致）。
        """
        dx = self.target_enu[0] - self.robot_x
        dy = self.target_enu[1] - self.robot_y
        range_m = math.hypot(dx, dy)
        world_angle = math.atan2(dy, dx)
        bearing = self._wrap_pi(world_angle - self.robot_yaw)
        return range_m, bearing

    @staticmethod
    def _wrap_pi(angle):
        """功能：将角度归一化到 [-pi, pi)。"""
        return math.atan2(math.sin(angle), math.cos(angle))


if __name__ == '__main__':
    try:
        TargetGpsReceiver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
