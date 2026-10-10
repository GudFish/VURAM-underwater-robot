#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：Water Linked DVL A50 驱动与定位导航节点（TCP JSON API，json_v3.3）。

背景：DVL A50 通过以太网接入机载电脑（TCP server 端口 16171，换行分隔 JSON）。
本节点完成三件事：
  1. 驱动 DVL：TCP 连接管理（指数退避重连）、JSON 报文解析、命令收发
     （reset_dead_reckoning / calibrate_gyro / set_config 等）；
  2. 定位导航：以 DVL 内置 dead-reckoning（DR，5 Hz，IMU/AHRS + 声学速度
     卡尔曼滤波）为锚点，两次 DR 报告之间用最新速度报告（2-15 Hz）按锚点
     姿态前向推算，以 odom_rate（默认 20 Hz）发布 nav_msgs/Odometry
     （frame_id=odom，child=base_link）——注意 20 Hz 中仅 5 Hz 为真实量测；
  3. 发布原始 DVL 数据：/dvl/velocity（DVL 机体系，z 向下，未转换）、
     /dvl/altitude、/dvl/fom、/dvl/connected。

Reset 语义：/dvl_odometry/reset 服务发送 reset_dead_reckoning 命令，DVL 将
位置清零、roll/pitch/yaw 清零，并按当前机体系重新建立坐标系（约 50 ms 生效），
即"重新从原点开始建立坐标系并开始定位导航"。reset 后 odom 位姿会跳变到原点，
这是 REP-105 odom 漂移帧的合法语义，下游滤波需容忍跳变或自行重启。
启动时若 reset_on_start=true，首次连上 DVL 后自动 reset 一次（每次节点运行
仅一次，重连不重发）。

坐标系：DVL 机体系 x 向前（A50 LED 标 +x）、y 向右、z 向下；ROS REP-103 为
x 向前、y 向左、z 向上。转换等效绕 x 轴被动旋转 180°（R=diag(1,-1,-1)）：
  位置/速度 (x,-y,-z)；姿态 roll 不变、pitch→-pitch、yaw→-yaw。

时间戳：一律使用主机接收时刻（DVL 时钟未必 NTP 同步）；time_of_validity 仅
作 debug 日志。完整参数表见 README.md。
"""
import json
import math
import socket
import threading
from collections import deque

import rospy
import tf2_ros
from tf.transformations import (quaternion_conjugate, quaternion_from_euler,
                                quaternion_matrix, quaternion_multiply)
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger, TriggerResponse

# 行缓冲上限：设备异常持续输出无换行字节时防止内存无限增长
MAX_LINE_BUFFER = 1 << 20

# DVL→ROS 协方差符号翻转模式（对应 R=diag(1,-1,-1) 的被动旋转）
COV_FLIP_SIGN = (1.0, -1.0, -1.0)


class DvlOdometryNode:
    """
    功能：驱动 Water Linked DVL A50，发布原始数据话题与 20 Hz odom，
          并提供 reset / calibrate_gyro 服务。
    输入：构造阶段无显式位置参数，依赖 ROS 参数、DVL TCP 数据流与内部成员状态。
    输出：初始化并维护该对象的运行状态，供连接线程、定时回调与服务回调使用。
    """
    def __init__(self):
        """
        功能：读取参数、创建发布器/服务/TF 广播器并启动连接线程。
        输入：无显式构造参数；通过 ROS 参数与话题配置完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
        rospy.init_node('dvl_odometry_node')

        # 1. 连接参数
        self.ip = rospy.get_param('~ip', '192.168.1.95')
        self.port = int(rospy.get_param('~port', 16171))
        self.reconnect_min_delay = float(rospy.get_param('~reconnect_min_delay', 1.0))
        self.reconnect_max_delay = float(rospy.get_param('~reconnect_max_delay', 30.0))

        # 2. 行为参数
        self.reset_on_start = bool(rospy.get_param('~reset_on_start', True))
        self.apply_config = bool(rospy.get_param('~apply_config', False))
        self.speed_of_sound = float(rospy.get_param('~speed_of_sound', 0.0))
        self.mounting_rotation_offset = float(rospy.get_param('~mounting_rotation_offset', 0.0))

        # 3. 输出参数
        self.odom_rate = float(rospy.get_param('~odom_rate', 20.0))
        self.publish_tf = bool(rospy.get_param('~publish_tf', False))
        self.publish_z = bool(rospy.get_param('~publish_z', True))
        self.odom_topic = rospy.get_param('~odom_topic', '/dvl/odom')
        self.velocity_topic = rospy.get_param('~velocity_topic', '/dvl/velocity')
        self.altitude_topic = rospy.get_param('~altitude_topic', '/dvl/altitude')
        self.fom_topic = rospy.get_param('~fom_topic', '/dvl/fom')
        self.connected_topic = rospy.get_param('~connected_topic', '/dvl/connected')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.dvl_frame = rospy.get_param('~dvl_frame', 'dvl_link')

        # 4. 新鲜度与命令超时参数
        self.vel_timeout = float(rospy.get_param('~vel_timeout', 1.0))
        self.dr_timeout = float(rospy.get_param('~dr_timeout', 3.0))
        self.reset_timeout = float(rospy.get_param('~reset_timeout', 2.0))
        self.gyro_calib_timeout = float(rospy.get_param('~gyro_calib_timeout', 20.0))

        # 5. 协方差参数
        self.dr_z_std = float(rospy.get_param('~dr_z_std', 1.0))
        self.dr_rp_std = float(rospy.get_param('~dr_rp_std', 3.0))     # deg
        self.dr_yaw_std = float(rospy.get_param('~dr_yaw_std', 5.0))   # deg
        self.cov_growth_pos = float(rospy.get_param('~cov_growth_pos', 0.01))  # m^2/s

        # 6. 状态缓存（_state_lock 保护）
        self._state_lock = threading.Lock()
        self.anchor_pos = [0.0, 0.0, 0.0]          # ROS 系传播锚点位置
        self.anchor_quat = [0.0, 0.0, 0.0, 1.0]    # (x,y,z,w) 单位四元数
        self.anchor_time = None
        self.anchor_set = False
        self.dr_pos_dvl = None
        self.dr_rpy_deg = None
        self.dr_std = 0.0
        self.dr_stamp = None
        self.have_dr = False
        self.vel_dvl = (0.0, 0.0, 0.0)
        self.vel_cov_dvl = [[0.0] * 3 for _ in range(3)]
        self.vel_valid = False
        self.fom = -1.0
        self.altitude = -1.0
        self.vel_stamp = None
        self.have_vel = False
        self.parse_error_count = 0

        # 7. 命令子系统（_cmd_cond 保护 _cmd_queue/_inflight/connected）
        self._cmd_cond = threading.Condition()
        self._cmd_queue = deque()
        self._inflight = None
        self.connected = False
        self.reset_sent = False

        # 8. 发布器
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)
        self.vel_pub = rospy.Publisher(self.velocity_topic,
                                       TwistWithCovarianceStamped, queue_size=1)
        self.alt_pub = rospy.Publisher(self.altitude_topic, Range, queue_size=1)
        self.fom_pub = rospy.Publisher(self.fom_topic, Float32, queue_size=1)
        # latch=True：晚到的订阅者也能立即拿到当前连接状态
        self.connected_pub = rospy.Publisher(self.connected_topic, Bool, queue_size=1, latch=True)

        # 9. TF 广播器（默认关闭，避免与 mavros 的 map→base_link 冲突）
        self.tf_broadcaster = tf2_ros.TransformBroadcaster() if self.publish_tf else None

        # 10. 服务（绝对名，避免与其他节点的 /reset 冲突）
        rospy.Service('/dvl_odometry/reset', Trigger, self._srv_reset)
        rospy.Service('/dvl_odometry/calibrate_gyro', Trigger, self._srv_calibrate_gyro)

        # 11. 启动 20 Hz odom 定时器（主线程）与连接线程
        rospy.Timer(rospy.Duration(1.0 / max(self.odom_rate, 0.1)), self._odom_timer_cb)
        self.conn_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.conn_thread.start()
        rospy.loginfo('[DvlOdometry] started: ip=%s port=%d odom_rate=%.1f Hz '
                      'publish_tf=%s reset_on_start=%s',
                      self.ip, self.port, self.odom_rate,
                      self.publish_tf, self.reset_on_start)

    # ------------------------------------------------------------------
    # 连接与接收（连接线程独占 socket）
    # ------------------------------------------------------------------
    def _connection_loop(self):
        """功能：连接线程：指数退避重连，连接期间运行读循环并驱动命令泵。"""
        backoff = self.reconnect_min_delay
        while not rospy.is_shutdown():
            try:
                sock = socket.create_connection((self.ip, self.port), timeout=5.0)
            except OSError as exc:
                # 逐次记录（退避上限 30 s，最多 ~2 条/分钟，不刷屏）
                rospy.logwarn('[DvlOdometry] connect to %s:%d failed: %s (retry in %.0fs)',
                              self.ip, self.port, exc, backoff)
                self._sleep_interruptible(backoff)
                backoff = min(backoff * 2.0, self.reconnect_max_delay)
                continue

            backoff = self.reconnect_min_delay
            rospy.loginfo('[DvlOdometry] connected to %s:%d', self.ip, self.port)
            with self._cmd_cond:
                self.connected = True
            self.connected_pub.publish(Bool(True))
            self._enqueue_startup_commands()
            try:
                self._read_loop(sock)
            except OSError as exc:
                rospy.logdebug('[DvlOdometry] socket error: %s', exc)
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
                with self._cmd_cond:
                    self.connected = False
                self.connected_pub.publish(Bool(False))
                self._fail_all_pending('DVL disconnected')
                rospy.logwarn('[DvlOdometry] disconnected from %s:%d', self.ip, self.port)

    @staticmethod
    def _sleep_interruptible(duration):
        """功能：分段睡眠（0.2 s 粒度），节点关闭时可及时退出退避等待。"""
        deadline = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.2)

    def _enqueue_startup_commands(self):
        """功能：连接后入队启动命令（set_config 可选、reset_on_start 每节点一次）。"""
        if self.apply_config:
            params = {}
            if self.speed_of_sound > 0.0:
                params['speed_of_sound'] = self.speed_of_sound
            if self.mounting_rotation_offset != 0.0:
                params['mounting_rotation_offset'] = self.mounting_rotation_offset
            if params:
                self._enqueue({'command': 'set_config', 'parameters': params},
                              self.reset_timeout, wait=False)
        if self.reset_on_start and not self.reset_sent:
            # 只发一次：无论成功与否，本次节点运行不再自动重发（重连亦不重发）
            self.reset_sent = True
            self._enqueue({'command': 'reset_dead_reckoning'}, self.reset_timeout, wait=False)
            rospy.loginfo('[DvlOdometry] reset_dead_reckoning sent on startup')

    def _read_loop(self, sock):
        """功能：读循环：命令泵 + 按行解析 JSON 报文并分派。"""
        sock.settimeout(0.1)
        buf = b''
        while not rospy.is_shutdown():
            self._pump_commands(sock)
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                break  # 对端关闭
            buf += data
            if len(buf) > MAX_LINE_BUFFER:
                rospy.logwarn('[DvlOdometry] line buffer overflow (%d bytes), dropping',
                              len(buf))
                buf = buf[-MAX_LINE_BUFFER:]
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                line = line.strip()
                if not line:
                    continue
                # 单条报文异常不得终止接收线程（接收线程必须对坏数据免疫）
                try:
                    self._dispatch(json.loads(line.decode('utf-8', errors='replace')))
                except Exception as exc:
                    self.parse_error_count += 1
                    rospy.logwarn_throttle(5.0,
                                           '[DvlOdometry] error processing line (total %d): %s: %s',
                                           self.parse_error_count, type(exc).__name__, exc)

    def _dispatch(self, msg):
        """功能：按 type 分派报文：velocity / position_local / response。"""
        mtype = msg.get('type', '')
        if mtype in ('velocity', 'velocity_water'):
            self._on_velocity(msg)
        elif mtype == 'position_local':
            self._on_dr(msg)
        elif mtype == 'response':
            self._on_response(msg)
        else:
            rospy.logwarn_throttle(10.0, '[DvlOdometry] unknown message type %r ignored', mtype)

    # ------------------------------------------------------------------
    # 报文处理（RX 线程内：更新状态并发布原始数据话题）
    # ------------------------------------------------------------------
    def _on_velocity(self, msg):
        """功能：处理速度报告：更新状态缓存并发布 /dvl/velocity、/dvl/fom、/dvl/altitude。"""
        now = rospy.Time.now()
        vx = float(msg.get('vx', 0.0))
        vy = float(msg.get('vy', 0.0))
        vz = float(msg.get('vz', 0.0))
        cov = self._sanitize_covariance(msg.get('covariance'))
        vel_valid = bool(msg.get('velocity_valid', False))
        fom = float(msg.get('fom', -1.0))
        altitude = float(msg.get('altitude', -1.0))

        with self._state_lock:
            self.vel_dvl = (vx, vy, vz)
            self.vel_cov_dvl = cov
            self.vel_valid = vel_valid
            self.fom = fom
            self.altitude = altitude
            self.vel_stamp = now
            self.have_vel = True

        # 原始速度：DVL 机体系（z 向下），不做任何转换
        tw = TwistWithCovarianceStamped()
        tw.header.stamp = now
        tw.header.frame_id = self.dvl_frame
        tw.twist.twist.linear.x = vx
        tw.twist.twist.linear.y = vy
        tw.twist.twist.linear.z = vz
        for i in range(3):
            for j in range(3):
                tw.twist.covariance[i * 6 + j] = cov[i][j]
        self.vel_pub.publish(tw)

        self.fom_pub.publish(Float32(fom))

        if altitude >= 0.0:
            rng = Range()
            rng.header.stamp = now
            rng.header.frame_id = self.dvl_frame
            rng.radiation_type = Range.ULTRASOUND
            rng.field_of_view = 0.0
            rng.min_range = 0.05
            rng.max_range = 50.0
            rng.range = altitude
            self.alt_pub.publish(rng)

    def _on_dr(self, msg):
        """功能：处理 DR 报告：帧转换后更新传播锚点（最新 DR 即锚点）。"""
        now = rospy.Time.now()
        x = float(msg.get('x', 0.0))
        y = float(msg.get('y', 0.0))
        z = float(msg.get('z', 0.0))
        std = float(msg.get('std', 0.0))
        roll = float(msg.get('roll', 0.0))
        pitch = float(msg.get('pitch', 0.0))
        yaw = float(msg.get('yaw', 0.0))

        with self._state_lock:
            self.dr_pos_dvl = (x, y, z)
            self.dr_rpy_deg = (roll, pitch, yaw)
            self.dr_std = std
            self.dr_stamp = now
            self.have_dr = True
            # DVL→ROS 转换：位置 (x,-y,-z)；姿态 (roll, -pitch, -yaw)
            self.anchor_pos = [x, -y, -z]
            self.anchor_quat = list(quaternion_from_euler(math.radians(roll),
                                                          math.radians(-pitch),
                                                          math.radians(-yaw)))
            self.anchor_time = now
            self.anchor_set = True

    def _on_response(self, msg):
        """功能：匹配命令响应（同一时刻仅一条在途命令，按 response_to 匹配）。"""
        name = msg.get('response_to', '')
        with self._cmd_cond:
            pending = self._inflight
            if pending is None or pending['cmd'].get('command') != name:
                rospy.logwarn_throttle(10.0, '[DvlOdometry] unexpected response %r ignored', name)
                return
            with pending['cond']:
                pending['ok'] = bool(msg.get('success', False))
                pending['error'] = str(msg.get('error_message') or '')
                pending['result'] = msg.get('result') or {}
                pending['done'] = True
                pending['cond'].notify_all()
            self._inflight = None

    @staticmethod
    def _sanitize_covariance(raw):
        """功能：清洗 3×3 协方差矩阵，非法输入返回全零矩阵。"""
        try:
            rows = [[float(v) for v in row[:3]] for row in raw[:3]]
            if len(rows) < 3 or any(len(r) < 3 for r in rows):
                return [[0.0] * 3 for _ in range(3)]
            return rows
        except (TypeError, ValueError):
            return [[0.0] * 3 for _ in range(3)]

    # ------------------------------------------------------------------
    # 命令子系统
    # ------------------------------------------------------------------
    def _enqueue(self, cmd, timeout, wait):
        """
        功能：命令入队。wait=True 时阻塞等待 ACK/NAK/超时。
        输出：返回 (ok, error, result)；未连接或超时 ok=False。
        """
        pending = {'cond': threading.Condition(), 'cmd': cmd, 'timeout': timeout,
                   'sent_time': None, 'ok': False, 'error': '', 'result': {}, 'done': False}
        with self._cmd_cond:
            if not self.connected:
                return False, 'DVL not connected', {}
            self._cmd_queue.append((cmd, pending))
            self._cmd_cond.notify_all()
        if not wait:
            return True, '', {}
        with pending['cond']:
            if not pending['done']:
                pending['cond'].wait(timeout)
            if not pending['done']:
                # 调用方超时：命令可能仍在队/在途，由连接线程兜底完成或超时失败
                return False, 'response timeout', {}
            return pending['ok'], pending['error'], pending['result']

    def _pump_commands(self, sock):
        """功能：命令泵：扫描在途命令超时，空闲时发送下一条（同一时刻仅一条在途）。"""
        to_send = None
        now = rospy.Time.now()
        with self._cmd_cond:
            if self._inflight is not None:
                pending = self._inflight
                with pending['cond']:
                    if not pending['done'] and pending['sent_time'] is not None and \
                            (now - pending['sent_time']).to_sec() > pending['timeout']:
                        pending['error'] = 'response timeout'
                        pending['done'] = True
                        pending['cond'].notify_all()
                        rospy.logwarn('[DvlOdometry] command %s response timeout',
                                      pending['cmd'].get('command'))
                    if pending['done']:
                        self._inflight = None
            if self._inflight is None and self._cmd_queue:
                cmd, pending = self._cmd_queue.popleft()
                pending['sent_time'] = now
                self._inflight = pending
                to_send = (cmd, pending)
        if to_send is not None:
            cmd, _ = to_send
            try:
                sock.sendall((json.dumps(cmd) + '\n').encode('utf-8'))
            except OSError:
                pass  # 断连由读循环退出路径统一处理（_fail_all_pending）

    def _fail_all_pending(self, reason):
        """功能：断连时使所有等待中的命令立即失败返回，服务调用不会挂死。"""
        with self._cmd_cond:
            pendings = [p for _, p in self._cmd_queue]
            if self._inflight is not None:
                pendings.append(self._inflight)
            self._cmd_queue.clear()
            self._inflight = None
        for pending in pendings:
            with pending['cond']:
                if not pending['done']:
                    pending['error'] = reason
                    pending['done'] = True
                    pending['cond'].notify_all()

    # ------------------------------------------------------------------
    # 服务回调（rospy 服务线程内执行）
    # ------------------------------------------------------------------
    def _srv_reset(self, _req):
        """功能：/dvl_odometry/reset 服务：重置 DVL dead-reckoning 并本地归零锚点。"""
        ok, err, _res = self._enqueue({'command': 'reset_dead_reckoning'},
                                      self.reset_timeout, wait=True)
        if ok:
            with self._state_lock:
                self.anchor_pos = [0.0, 0.0, 0.0]
                self.anchor_quat = [0.0, 0.0, 0.0, 1.0]
                self.anchor_time = rospy.Time.now()
                self.anchor_set = True
            rospy.loginfo('[DvlOdometry] dead reckoning reset OK (new frame at origin)')
            return TriggerResponse(True, 'dead reckoning reset OK')
        rospy.logwarn('[DvlOdometry] reset_dead_reckoning failed: %s', err)
        return TriggerResponse(False, err)

    def _srv_calibrate_gyro(self, _req):
        """功能：/dvl_odometry/calibrate_gyro 服务：陀螺标定（DVL 必须静止，最长 15 s）。"""
        rospy.logwarn('[DvlOdometry] calibrating gyro: keep the DVL stationary (up to 15 s)...')
        ok, err, _res = self._enqueue({'command': 'calibrate_gyro'},
                                      self.gyro_calib_timeout, wait=True)
        if ok:
            rospy.loginfo('[DvlOdometry] gyro calibration OK')
            return TriggerResponse(True, 'gyro calibration OK')
        rospy.logwarn('[DvlOdometry] calibrate_gyro failed: %s', err)
        return TriggerResponse(False, err)

    # ------------------------------------------------------------------
    # 20 Hz odom 组合发布（主线程定时器）
    # ------------------------------------------------------------------
    def _odom_timer_cb(self, _event):
        """功能：按 odom_rate 组合发布 Odometry：DR 锚点 + 最新速度前向推算。"""
        now = rospy.Time.now()
        with self._state_lock:
            if not self.anchor_set:
                rospy.loginfo_throttle(5.0, '[DvlOdometry] waiting for dead-reckoning report...')
                return
            anchor_pos = list(self.anchor_pos)
            anchor_quat = list(self.anchor_quat)
            anchor_time = self.anchor_time
            dr_age = (now - self.dr_stamp).to_sec() if self.dr_stamp is not None \
                else float('inf')
            vel_age = (now - self.vel_stamp).to_sec() if self.vel_stamp is not None \
                else float('inf')
            vel_ok = self.have_vel and self.vel_valid and vel_age <= self.vel_timeout
            if vel_ok:
                vel = self.vel_dvl
                vel_cov = self.vel_cov_dvl
            else:
                vel = (0.0, 0.0, 0.0)
                vel_cov = None
            dr_std = self.dr_std

        # DR 过期：持位不再外推；速度过期/无效：仅靠 DR 锚点
        if dr_age > self.dr_timeout:
            dt = 0.0
            v_ros = (0.0, 0.0, 0.0)
            t_stale = dr_age - self.dr_timeout
            rospy.logwarn_throttle(5.0, '[DvlOdometry] DR report stale (%.1fs), holding position',
                                   dr_age)
        elif not vel_ok:
            dt = 0.0
            v_ros = (0.0, 0.0, 0.0)
            t_stale = max(0.0, vel_age - self.vel_timeout)
        else:
            dt = max(0.0, (now - anchor_time).to_sec())
            v_ros = (vel[0], -vel[1], -vel[2])
            t_stale = 0.0

        rotated = self._rotate_vec(anchor_quat, v_ros)
        pos = [anchor_pos[i] + rotated[i] * dt for i in range(3)]
        if not self.publish_z:
            pos[2] = 0.0

        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.robot_frame
        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        msg.pose.pose.orientation.x = anchor_quat[0]
        msg.pose.pose.orientation.y = anchor_quat[1]
        msg.pose.pose.orientation.z = anchor_quat[2]
        msg.pose.pose.orientation.w = anchor_quat[3]
        msg.twist.twist.linear.x = v_ros[0]
        msg.twist.twist.linear.y = v_ros[1]
        msg.twist.twist.linear.z = v_ros[2]
        self._fill_covariances(msg, anchor_quat, vel_cov, dt, t_stale, dr_std)
        self.odom_pub.publish(msg)

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = now
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.robot_frame
            tf_msg.transform.translation.x = pos[0]
            tf_msg.transform.translation.y = pos[1]
            tf_msg.transform.translation.z = pos[2]
            tf_msg.transform.rotation.x = anchor_quat[0]
            tf_msg.transform.rotation.y = anchor_quat[1]
            tf_msg.transform.rotation.z = anchor_quat[2]
            tf_msg.transform.rotation.w = anchor_quat[3]
            self.tf_broadcaster.sendTransform(tf_msg)

    def _fill_covariances(self, msg, anchor_quat, vel_cov_dvl, dt, t_stale, dr_std):
        """功能：填充 pose/twist 6×6 协方差（位置=DR std²+传播+增长，姿态=固定先验）。"""
        # 位置块：DR std 为 X-Y 平面位置标准差；z 分量取 max(std, dr_z_std)
        std_xy = max(dr_std, 1e-3)
        pos_cov = [[0.0] * 3 for _ in range(3)]
        pos_cov[0][0] = std_xy ** 2
        pos_cov[1][1] = std_xy ** 2
        pos_cov[2][2] = max(std_xy ** 2, self.dr_z_std ** 2)

        # 传播不确定度：R_q · C_ros · R_qᵀ · dt²
        if vel_cov_dvl is not None and dt > 0.0:
            c_ros = self._flip_covariance(vel_cov_dvl)
            r_q = quaternion_matrix(anchor_quat)[:3, :3]
            prop = r_q.dot(c_ros).dot(r_q.T) * (dt * dt)
            pos_cov = [[pos_cov[i][j] + prop[i][j] for j in range(3)] for i in range(3)]

        # 数据过期增长项：协方差随时间放大，表征"不可信"
        if t_stale > 0.0:
            growth = self.cov_growth_pos * t_stale
            for i in range(3):
                pos_cov[i][i] += growth

        for i in range(3):
            for j in range(3):
                msg.pose.covariance[i * 6 + j] = pos_cov[i][j]

        # 姿态块：固定先验（DR 姿态来自 IMU/AHRS，短期可信）
        rp = math.radians(self.dr_rp_std)
        yw = math.radians(self.dr_yaw_std)
        msg.pose.covariance[21] = rp * rp
        msg.pose.covariance[28] = rp * rp
        msg.pose.covariance[35] = yw * yw

        # twist 块：C_ros（child 系速度），无有效速度时保持全零
        if vel_cov_dvl is not None:
            c_ros = self._flip_covariance(vel_cov_dvl)
            for i in range(3):
                for j in range(3):
                    msg.twist.covariance[i * 6 + j] = c_ros[i][j]

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _flip_covariance(cov):
        """功能：DVL→ROS 协方差转换：C_ros[i][j] = s_i·s_j·C_dvl[i][j]，s=(1,-1,-1)。"""
        s = COV_FLIP_SIGN
        return [[s[i] * s[j] * cov[i][j] for j in range(3)] for i in range(3)]

    @staticmethod
    def _rotate_vec(quat, vec):
        """功能：用四元数旋转向量（q ⊗ v ⊗ q⁻¹），返回旋转后的 [x,y,z]。"""
        qv = quaternion_multiply(quat, [vec[0], vec[1], vec[2], 0.0])
        qv = quaternion_multiply(qv, quaternion_conjugate(quat))
        return [qv[0], qv[1], qv[2]]


if __name__ == '__main__':
    try:
        DvlOdometryNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
