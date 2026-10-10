#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：Water Linked DVL A50 TCP JSON API 模拟器，用于无硬件联调 dvl_odometry_node。

背景：DVL A50 通过以太网提供 TCP server（端口 16171，换行分隔 JSON）。本脚本
模拟其行为，供驱动节点开发与回归验证：
  - 周期输出 velocity 报告（默认 10 Hz）与 position_local dead-reckoning 报告
    （默认 5 Hz），字段与官方 json_v3.3 一致；
  - 应答命令：reset_dead_reckoning（约 50 ms 后清零位置与姿态）、calibrate_gyro
    （模拟耗时 2 s，期间报告暂停）、get_version_info、get_config/set_config；
  - 测试钩子 sim_dropout：主动断开当前连接，验证驱动节点的重连逻辑。

运动模型：随机游走航向（±1.5°/s 高斯噪声）+ 恒定前进速度（0.3 m/s），
z（向下）缓慢起伏；std 随时间缓慢增长，模拟无底锁时的发散。

用法：
  python3 dvl_simulator.py [--host 127.0.0.1] [--port 16171]
                           [--velocity-rate 10.0] [--dr-rate 5.0]
  或：roslaunch dvl_odometry dvl_simulator.launch

无 rospy 依赖，可独立运行。
"""
import argparse
import json
import math
import random
import socket
import threading
import time


class DvlSimulator:
    """
    功能：模拟 Water Linked DVL A50 的 TCP JSON API 服务器。
    输入：构造参数为监听地址、端口与报告频率；运行时依赖 TCP 连接与内部运动状态。
    输出：向每个连接周期性发送 velocity / position_local 报告并应答命令。
    """
    def __init__(self, host='127.0.0.1', port=16171, velocity_rate=10.0, dr_rate=5.0):
        self.host = host
        self.port = port
        self.velocity_rate = max(velocity_rate, 0.5)
        self.dr_rate = max(dr_rate, 0.5)
        self.lock = threading.Lock()
        # DVL 系运动状态：x 向前、y 向右、z 向下
        self.pos = [0.0, 0.0, 2.0]
        self.rpy = [0.0, 0.0, 0.0]  # roll, pitch, yaw (deg)
        self.speed = 0.3             # 前进速度 m/s
        self.std = 0.001             # 位置 std，随时间缓慢增长
        self.altitude = 2.0
        self.config = {
            'speed_of_sound': 1475.0,
            'mounting_rotation_offset': 0.0,
            'acoustic_enabled': True,
            'dark_mode_enabled': False,
            'range_mode': 'auto',
            'periodic_cycling_enabled': True,
        }

    def run(self):
        """功能：监听端口并接受连接，每个连接一个服务线程。"""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(4)
        print('[DvlSimulator] listening on %s:%d (velocity %.1f Hz, DR %.1f Hz)'
              % (self.host, self.port, self.velocity_rate, self.dr_rate))
        while True:
            conn, addr = srv.accept()
            print('[DvlSimulator] client connected: %s' % (addr,))
            threading.Thread(target=self._serve, args=(conn, addr), daemon=True).start()

    # ------------------------------------------------------------------
    # 单连接服务
    # ------------------------------------------------------------------
    def _serve(self, conn, addr):
        """功能：单连接服务线程：周期输出报告，逐行解析并应答命令。"""
        conn.settimeout(0.05)
        buf = b''
        last_vel = 0.0
        last_dr = 0.0
        try:
            while True:
                now = time.time()
                if now - last_vel >= 1.0 / self.velocity_rate:
                    last_vel = now
                    self._step_motion(1.0 / self.velocity_rate)
                    self._send(conn, self._velocity_report())
                if now - last_dr >= 1.0 / self.dr_rate:
                    last_dr = now
                    self._send(conn, self._dr_report())
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line.decode('utf-8'))
                    except ValueError:
                        continue
                    if not self._handle_command(conn, cmd):
                        return  # sim_dropout：主动断连，验证驱动重连
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
            print('[DvlSimulator] client disconnected: %s' % (addr,))

    # ------------------------------------------------------------------
    # 运动模型与报告
    # ------------------------------------------------------------------
    def _step_motion(self, dt):
        """功能：推进运动模型：随机游走航向 + 前进速度，z 缓慢起伏，std 缓慢增长。"""
        with self.lock:
            self.rpy[2] += random.gauss(0.0, 1.5) * dt
            yaw = math.radians(self.rpy[2])
            self.pos[0] += self.speed * math.cos(yaw) * dt
            self.pos[1] += self.speed * math.sin(yaw) * dt
            self.pos[2] += random.gauss(0.0, 0.02) * dt
            self.pos[2] = max(0.5, min(5.0, self.pos[2]))
            self.std = min(2.0, self.std + random.uniform(0.0, 0.0005) * dt)

    def _velocity_report(self):
        """功能：构造 velocity 报告（DVL 机体系，z 向下，字段对齐官方 json_v3.3）。"""
        with self.lock:
            yaw = math.radians(self.rpy[2])
            vx = self.speed * math.cos(yaw)
            vy = self.speed * math.sin(yaw)
            vz = random.gauss(0.0, 0.005)
            altitude = self.altitude
        now_us = int(time.time() * 1e6)
        return {
            'time': 1000.0 / self.velocity_rate,  # 距上次报告毫秒数
            'vx': vx,
            'vy': vy,
            'vz': vz,
            'fom': 0.005,
            'covariance': [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]],
            'altitude': altitude,
            'transducers': [
                {'id': i, 'velocity': 0.0, 'distance': altitude, 'rssi': -30.0,
                 'nsd': -90.0, 'beam_valid': True} for i in range(4)
            ],
            'velocity_valid': True,
            'status': 0,
            'format': 'json_v3.3',
            'type': 'velocity',
            'time_of_validity': now_us,
            'time_of_transmission': now_us,
        }

    def _dr_report(self):
        """功能：构造 position_local dead-reckoning 报告。"""
        with self.lock:
            payload = {
                'ts': time.time(),
                'x': self.pos[0],
                'y': self.pos[1],
                'z': self.pos[2],
                'std': self.std,
                'roll': self.rpy[0],
                'pitch': self.rpy[1],
                'yaw': self.rpy[2],
                'type': 'position_local',
                'status': 0,
                'format': 'json_v3.3',
            }
        return payload

    # ------------------------------------------------------------------
    # 命令处理
    # ------------------------------------------------------------------
    def _handle_command(self, conn, cmd):
        """
        功能：处理单条命令并应答。
        输出：返回 False 表示主动断连（sim_dropout），其余返回 True。
        """
        name = cmd.get('command', '')
        if name == 'sim_dropout':
            print('[DvlSimulator] sim_dropout: closing connection to test reconnection')
            return False
        if name == 'reset_dead_reckoning':
            time.sleep(0.05)  # 官方语义：约 50 ms 后清零
            with self.lock:
                self.pos = [0.0, 0.0, 0.0]
                self.rpy = [0.0, 0.0, 0.0]
                self.std = 0.001
            self._respond(conn, name, True, '', None)
        elif name == 'calibrate_gyro':
            print('[DvlSimulator] calibrating gyro (2 s, reports paused)...')
            time.sleep(2.0)
            self._respond(conn, name, True, '', None)
        elif name == 'get_version_info':
            self._respond(conn, name, True, '', {
                'chipid': '0xsim0000000001',
                'hardware_revision': 4,
                'product_id': 21035,
                'product_name': 'DVL A50 (simulator)',
                'variant': 'standard',
                'version_short': '2.7.2',
                'is_ready': True,
            })
        elif name == 'get_config':
            with self.lock:
                result = dict(self.config)
            self._respond(conn, name, True, '', result)
        elif name == 'set_config':
            params = cmd.get('parameters') or {}
            with self.lock:
                self.config.update({k: v for k, v in params.items() if k in self.config})
            self._respond(conn, name, True, '', None)
        else:
            self._respond(conn, name, False, 'unknown command %r' % name, None)
        return True

    # ------------------------------------------------------------------
    # 发送
    # ------------------------------------------------------------------
    @staticmethod
    def _respond(conn, name, success, error, result):
        """功能：发送命令响应（response_to/success/error_message/result）。"""
        DvlSimulator._send(conn, {
            'response_to': name,
            'success': success,
            'error_message': error,
            'result': result,
            'format': 'json_v3.3',
            'type': 'response',
        })

    @staticmethod
    def _send(conn, payload):
        """功能：以换行分隔 JSON 发送一条报文，断连静默忽略。"""
        try:
            conn.sendall((json.dumps(payload) + '\n').encode('utf-8'))
        except OSError:
            pass


def main():
    """功能：解析命令行参数并启动模拟器。"""
    parser = argparse.ArgumentParser(description='Water Linked DVL A50 TCP JSON simulator')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=16171, help='监听端口（默认 16171）')
    parser.add_argument('--velocity-rate', type=float, default=10.0, help='速度报告频率 Hz（默认 10）')
    parser.add_argument('--dr-rate', type=float, default=5.0, help='DR 报告频率 Hz（默认 5）')
    # parse_known_args：忽略 roslaunch 注入的 __name/__log 等 remapping 参数
    args, _unknown = parser.parse_known_args()
    DvlSimulator(args.host, args.port, args.velocity_rate, args.dr_rate).run()


if __name__ == '__main__':
    main()
