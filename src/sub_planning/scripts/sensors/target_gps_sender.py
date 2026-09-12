#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：目标端 GPS 转发节点（运行在目标机器人的机载电脑上）。

背景：实物实验中的目标是一台与跟踪机器人同架构的水下机器人，其机载电脑同样
运行 ROS + MAVROS。本节点订阅目标自身的 GPS 位置（/mavros/global_position/global），
事件驱动转发：每次收到有效 GPS 更新立即通过局域网 UDP 发送到跟踪机器人的
target_gps_receiver 节点，报文格式为 CSV 文本 "lat,lon,alt"（与接收端解析
约定一致）。发送频率即目标机 GPS 更新率，无定时器引入的额外延迟。

话题参数：
  ~udp_ip        – 跟踪机器人 IP（默认 192.168.0.101）
  ~udp_port      – 目标端口（默认 4002，与 target_gps_receiver 一致）
  ~gps_topic     – 目标自身 GPS 话题（默认 /mavros/global_position/global）
  ~min_interval  – 两次发送的最小间隔 s，仅作异常高频限流保护（默认 0.05）
  ~stale_timeout – GPS 新鲜度超时 s，超过则告警提示定位失效（默认 5.0）
"""
import socket

import rospy
from sensor_msgs.msg import NavSatFix, NavSatStatus


class TargetGpsSender:
    """
    功能：订阅目标自身 GPS 并通过 UDP 转发给跟踪机器人。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建套接字并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('target_gps_sender')

        self.udp_ip = rospy.get_param('~udp_ip', '192.168.0.101')
        self.udp_port = int(rospy.get_param('~udp_port', 4002))
        self.gps_topic = rospy.get_param('~gps_topic', '/mavros/global_position/global')
        self.min_interval = float(rospy.get_param('~min_interval', 0.05))
        self.stale_timeout = float(rospy.get_param('~stale_timeout', 5.0))

        self.latest_gps = None     # (lat, lon, alt)
        self.last_gps_time = None
        self.last_send_time = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rospy.Subscriber(self.gps_topic, NavSatFix, self.gps_cb, queue_size=1)
        # 低频定时器仅用于 GPS 失效告警，不参与数据发送
        rospy.Timer(rospy.Duration(1.0), self.stale_warn_timer_cb)
        rospy.loginfo('[TargetGpsSender] event-driven forwarding %s -> UDP %s:%d',
                      self.gps_topic, self.udp_ip, self.udp_port)

    def gps_cb(self, msg):
        """
        功能：缓存目标自身最新 GPS 位置，并在定位有效时立即转发。
        输入：`msg` 为 /mavros/global_position/global 的 NavSatFix 消息。
        输出：无显式返回；更新内部缓存状态并通过 UDP 发送 "lat,lon,alt" 报文。
        """
        if msg.status.status < NavSatStatus.STATUS_FIX:
            rospy.logwarn_throttle(10.0, '[TargetGpsSender] no GPS fix (status=%d), not caching',
                                   msg.status.status)
            return

        self.latest_gps = (msg.latitude, msg.longitude, msg.altitude)
        self.last_gps_time = rospy.Time.now()
        self._send()

    def _send(self):
        """功能：将最新 GPS 立即发送到跟踪机器人（带最小间隔限流保护）。"""
        now = rospy.Time.now()
        if self.last_send_time is not None and \
                (now - self.last_send_time).to_sec() < self.min_interval:
            return

        payload = '%.7f,%.7f,%.2f' % self.latest_gps
        self.sock.sendto(payload.encode('utf-8'), (self.udp_ip, self.udp_port))
        self.last_send_time = now
        rospy.loginfo_throttle(10.0, '[TargetGpsSender] sent lat=%.7f lon=%.7f alt=%.2f',
                               self.latest_gps[0], self.latest_gps[1], self.latest_gps[2])

    def stale_warn_timer_cb(self, _event):
        """
        功能：低频检查 GPS 新鲜度，定位长时间未更新时输出告警。
        输入：`_event` 为定时器事件；其余输入依赖当前对象缓存状态。
        输出：无显式返回；仅输出告警日志。
        """
        if self.latest_gps is None:
            rospy.logwarn_throttle(10.0, '[TargetGpsSender] no valid GPS received yet')
            return
        stale = (rospy.Time.now() - self.last_gps_time).to_sec()
        if stale > self.stale_timeout:
            rospy.logwarn_throttle(5.0, '[TargetGpsSender] GPS stale (%.1fs > %.1fs), no data forwarded',
                                   stale, self.stale_timeout)


if __name__ == '__main__':
    try:
        TargetGpsSender()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
