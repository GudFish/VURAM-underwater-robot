#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从实验 rosbag 直接导出前视声呐 debug 图像为视频。

与 rosbag_play.launch 的 record_sonar_video（实时回放采集）不同，
本脚本直接顺序读取 bag 中的图像话题写入视频文件：
  - 帧率由消息时间戳反推：fps = (N-1)/(t_last - t_first)，
    因此视频总时长与 bag 实际录制时长严格一致，与机器回放速度无关，
    也不会因回放掉帧而缩短；
  - 默认不修改画面；需要核对时刻时可加 --time-overlay 在每帧左上角
    叠加 bag 相对时间与消息时间戳（HH:MM:SS）。

用法:
  python3 bag_sonar_debug_to_video.py -i <bag路径>
      [-o 输出视频路径]                # 默认与 bag 同目录、同名 _sonar_debug.mp4
      [--topic /forward_sonar/debug_image]
      [--time-overlay]                # 叠加时间文字（默认不叠加）
      [--fps 10]                      # 强制指定帧率（默认按时间戳反推）
      [--max-frames N]                # 只写前 N 帧（快速试跑用）

示例:
  python3 bag_sonar_debug_to_video.py \
      -i ~/ardupilot_ws/bagfiles/experiment/xiawu/proposed_real_ts_nvidia_desktop_20260914_145352.bag
"""
import argparse
import datetime
import os
import sys

import rosbag

import numpy as np

try:
    import cv2
except ImportError:
    print('[ERROR] 需要 python3-opencv: sudo apt install python3-opencv', file=sys.stderr)
    sys.exit(1)

DEFAULT_TOPIC = '/forward_sonar/debug_image'


def image_to_bgr(msg):
    """sensor_msgs/Image -> OpenCV BGR ndarray（支持常见编码）。"""
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    if msg.encoding in ('bgr8',):
        return img
    if msg.encoding in ('rgb8',):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if msg.encoding in ('bgra8',):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if msg.encoding in ('mono8', '8UC1'):
        gray = img[:, :, 0] if img.ndim == 3 else img
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError('unsupported encoding: %s' % msg.encoding)


def draw_time_overlay(frame, stamp, t_first):
    """左上角叠加 bag 相对时间与消息时间戳。"""
    rel = stamp.to_sec() - t_first
    wall = datetime.datetime.fromtimestamp(stamp.to_sec()).strftime('%H:%M:%S')
    text = 't = %.1f s   %s' % (rel, wall)
    h, w = frame.shape[:2]
    scale = max(0.5, w / 640.0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(1.2 * scale)))
    (tw, th), base = cv2.getTextSize(text, font, 0.5 * scale, thickness)
    cv2.rectangle(frame, (4, 4), (10 + tw, 12 + th + base), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 10 + th), font, 0.5 * scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', required=True, help='输入 bag 路径')
    parser.add_argument('-o', '--output', default=None, help='输出视频路径')
    parser.add_argument('--topic', default=DEFAULT_TOPIC, help='图像话题（默认 %s）' % DEFAULT_TOPIC)
    parser.add_argument('--time-overlay', action='store_true', help='叠加时间文字（默认不叠加）')
    parser.add_argument('--fps', type=float, default=None, help='强制帧率（默认按时间戳反推）')
    parser.add_argument('--max-frames', type=int, default=None, help='只写前 N 帧（试跑用）')
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.input)
    if not os.path.isfile(bag_path):
        print('[ERROR] bag 不存在: %s' % bag_path, file=sys.stderr)
        sys.exit(1)

    # ---- 第一遍：收集时间戳，反推帧率 ----
    stamps = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for _topic, msg, t in bag.read_messages(topics=[args.topic]):
            stamps.append(t)
            if args.max_frames and len(stamps) >= args.max_frames:
                break
    n = len(stamps)
    if n == 0:
        print('[ERROR] 话题 %s 无消息。可用: rosbag info %s' % (args.topic, bag_path), file=sys.stderr)
        sys.exit(1)

    duration = stamps[-1].to_sec() - stamps[0].to_sec()
    if args.fps is not None:
        fps = args.fps
    else:
        fps = (n - 1) / max(duration, 1e-3)
        fps = min(max(fps, 1.0), 60.0)   # 夹到合理范围，避免异常时间戳产生极端帧率
    print('帧数: %d  首帧 %s  末帧 %s' % (n, stamps[0], stamps[-1]))
    print('bag 时长: %.1f s  反推帧率: %.3f fps' % (duration, fps))

    # ---- 第二遍：写视频 ----
    if args.output is None:
        out_path = os.path.splitext(bag_path)[0] + '_sonar_debug.mp4'
    else:
        out_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    writer = None
    written = 0
    with rosbag.Bag(bag_path, 'r') as bag:
        for _topic, msg, t in bag.read_messages(topics=[args.topic]):
            if args.max_frames and written >= args.max_frames:
                break
            frame = image_to_bgr(msg)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (w, h))
                if not writer.isOpened():
                    print('[ERROR] 无法创建视频文件: %s' % out_path, file=sys.stderr)
                    sys.exit(1)
            if args.time_overlay:
                frame = draw_time_overlay(frame, t, stamps[0].to_sec())
            writer.write(frame)
            written += 1
            if written % 500 == 0:
                print('  已写 %d/%d 帧' % (written, n))
    if writer is not None:
        writer.release()
    print('完成: %d 帧 -> %s (%.1f s @ %.3f fps)' % (written, out_path, written / max(fps, 1e-3), fps))


if __name__ == '__main__':
    main()
