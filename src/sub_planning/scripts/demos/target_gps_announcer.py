#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：模拟目标端的 GPS 广播器（实物实验联调与测试用）。

在目标计算机（与机器人同一局域网 192.168.0.x 网段）上运行，周期性向机器人
发送目标 GPS 位置，报文格式与 target_gps_receiver 约定一致（CSV 文本
"lat,lon,alt"）。正式实验时由目标端真实的定位回传程序替代本节点。

两种模式：
  fixed  – 固定点（默认）：持续发送 (lat0, lon0, alt0)
  circle – 圆周运动：围绕 (lat0, lon0) 做半径 radius 的匀速圆周运动，
           用于验证方位解算与跟踪闭环
注意 lat0/lon0 应设为机器人 home 附近的坐标，使目标出现在声呐量程内。

话题参数：
  ~udp_ip   – 机器人 IP（默认 192.168.0.100）
  ~udp_port – 目标端口（默认 4002，与 target_gps_receiver 一致）
  ~rate     – 发送频率 Hz（默认 1.0）
  ~mode     – fixed | circle（默认 fixed）
  ~lat0 ~lon0 ~alt0 – 目标基准 GPS 坐标（默认 30.0, 120.0, 0.0）
  ~radius   – circle 模式半径 m（默认 20.0）
  ~speed    – circle 模式线速度 m/s（默认 0.4）
"""
import math
import socket
import time

import rospy


class TargetGpsAnnouncer:
    """
    功能：周期性地向机器人广播目标 GPS 位置。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建套接字并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('target_gps_announcer', anonymous=True)

        self.udp_ip = rospy.get_param('~udp_ip', '192.168.0.100')
        self.udp_port = int(rospy.get_param('~udp_port', 4002))
        self.rate = float(rospy.get_param('~rate', 1.0))
        self.mode = rospy.get_param('~mode', 'fixed').lower()
        self.lat0 = float(rospy.get_param('~lat0', 30.0))
        self.lon0 = float(rospy.get_param('~lon0', 120.0))
        self.alt0 = float(rospy.get_param('~alt0', 0.0))
        self.radius = float(rospy.get_param('~radius', 20.0))
        self.speed = float(rospy.get_param('~speed', 0.4))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.omega = self.speed / max(self.radius, 1e-3)
        self.t0 = time.time()

        rospy.Timer(rospy.Duration(1.0 / max(self.rate, 0.1)), self.timer_cb)
        rospy.loginfo('[TargetGpsAnnouncer] broadcasting to %s:%d (mode=%s, rate=%.1f Hz)',
                      self.udp_ip, self.udp_port, self.mode, self.rate)

    def timer_cb(self, _event):
        """
        功能：计算当前时刻的目标 GPS 并发送 UDP 报文。
        输入：`_event` 为定时器事件；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过 UDP 发送目标 GPS 报文。
        """
        lat, lon = self.lat0, self.lon0
        if self.mode == 'circle':
            t = time.time() - self.t0
            dn = self.radius * math.cos(self.omega * t)
            de = self.radius * math.sin(self.omega * t)
            lat = self.lat0 + dn / 111320.0
            lon = self.lon0 + de / (111320.0 * math.cos(math.radians(self.lat0)))

        payload = '%.7f,%.7f,%.2f' % (lat, lon, self.alt0)
        self.sock.sendto(payload.encode('utf-8'), (self.udp_ip, self.udp_port))
        rospy.loginfo_throttle(10.0, '[TargetGpsAnnouncer] sent lat=%.7f lon=%.7f alt=%.2f',
                               lat, lon, self.alt0)


if __name__ == '__main__':
    try:
        TargetGpsAnnouncer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
