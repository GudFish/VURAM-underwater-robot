#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：等待 SITL 飞控的 EKF 开始使用 GPS（"EKF3 IMU0 is using GPS"）。
      通过 pymavlink 监听 MAVProxy 输出的 UDP 端口（默认 127.0.0.1:14550），
      等待 STATUSTEXT 消息中出现 "EKF3 IMU0 is using GPS"（MAVProxy 控制台
      中显示的 AP: 提示即为该消息的渲染结果）。

      仅 GPS fix 不足以判定就绪：飞控需在 EKF 建立原点并采用 GPS 后
      才能正常切换 GUIDED 模式，因此必须等待 EKF 级信号。

      辅助判定（次选信号）：
        - EKF_STATUS_REPORT.flags 置位 EKF_POS_HORIZ_ABS (0x10)，
          表示 EKF 已输出 GPS 绝对位置。
      主信号（STATUSTEXT）优先，出现即成功。

用法：
  python3 wait_sitl_gps.py --timeout 120 --port 14550
"""
import argparse
import sys
import time

from pymavlink import mavutil

# EKF_STATUS_REPORT.flags 中"EKF 使用绝对位置（GPS）"位
EKF_POS_HORIZ_ABS = 0x10


def decode_text(raw):
    """pymavlink 不同版本返回 str 或 bytes，统一转 str。"""
    if isinstance(raw, bytes):
        return raw.decode('utf-8', 'ignore')
    return str(raw)


def main():
    parser = argparse.ArgumentParser(description='等待 SITL EKF 开始使用 GPS')
    parser.add_argument('--timeout', type=float, default=120.0, help='超时（秒），默认 120')
    parser.add_argument('--port', type=int, default=14550, help='MAVProxy 输出 UDP 端口，默认 14550')
    args = parser.parse_args()

    conn = mavutil.mavlink_connection('udpin:127.0.0.1:%d' % args.port)
    deadline = time.time() + args.timeout
    start = time.time()
    last_fix = -1
    last_ekf_flags = -1

    print('[WaitGPS] listening on udp:127.0.0.1:%d (timeout %.0fs)' % (args.port, args.timeout))
    print('[WaitGPS] waiting for EKF signal: "EKF3 IMU0 is using GPS"')

    while time.time() < deadline:
        msg = conn.recv_match(
            type=['STATUSTEXT', 'GPS_RAW_INT', 'EKF_STATUS_REPORT'],
            blocking=True,
            timeout=2.0,
        )
        if msg is None:
            continue

        mtype = msg.get_type()
        elapsed = time.time() - start

        if mtype == 'GPS_RAW_INT':
            # 进度提示：GPS 锁定状态
            if msg.fix_type != last_fix:
                print('[WaitGPS] GPS fix_type=%d sats=%d (%.1fs)'
                      % (msg.fix_type, msg.satellites_visible, elapsed))
                last_fix = msg.fix_type

        elif mtype == 'EKF_STATUS_REPORT':
            # 次选信号：EKF 已输出绝对位置
            if msg.flags != last_ekf_flags:
                print('[WaitGPS] EKF flags=0x%02x (%.1fs)' % (msg.flags, elapsed))
                last_ekf_flags = msg.flags
            if msg.flags & EKF_POS_HORIZ_ABS:
                print('[WaitGPS] EKF reports absolute position (GPS) — ready (%.1fs)' % elapsed)
                return 0

        elif mtype == 'STATUSTEXT':
            # 主信号：控制台 "AP: EKF3 IMU0 is using GPS" 的原始消息
            text = decode_text(msg.text)
            if 'EKF3 IMU0 is using GPS' in text:
                print('[WaitGPS] EKF READY: "%s" (%.1fs)' % (text.strip(), elapsed))
                return 0

    print('[WaitGPS] TIMEOUT after %.0fs (last GPS fix_type=%d, last EKF flags=0x%02x)'
          % (args.timeout, last_fix, last_ekf_flags), file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
