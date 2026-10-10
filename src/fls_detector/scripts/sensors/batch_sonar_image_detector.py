#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：离线批量对 AUV_sonar_images/<frame>/fan_output/ 下的扇形声呐图像
     复用 forward_sonar_target_detector.py 的检测流程进行目标检测，
     将潜在目标画框标注后保存到 <frame>/detect_output/ 中。

用法：
    python3 batch_sonar_image_detector.py <frame路径> [--region] [--range-norm]
示例：
    python3 batch_sonar_image_detector.py ~/AUV_sonar_images/frame3 --region --range-norm
    python3 batch_sonar_image_detector.py ~/AUV_sonar_images/frame_test --region --range-norm
    python3 batch_sonar_image_detector.py <frame路径> --region --threshold-percentile 95
输出：
    <frame>/detect_output/sonar_<序号>_det.png       （检测标注图）
    <frame>/denoised_output/sonar_<序号>_den.png    （滤波后未标注图）

说明：
    1. fan_output 中的扇形图原点位于顶部中心（近距在上），而检测器约定
       第 0 行为最远距离，因此处理前将图像上下翻转，标注后翻回原方向保存。
    2. 检测参数与 launch/sonar_target_detector.launch 中保持一致（DETECTOR_PARAMS）。
    3. 标注画法复刻检测器 publish_debug_image：绿框为最终候选（白色十字为
       加权质心）、青框为仅场景候选、红框为被抑制的边界/近场噪声区域。
    4. --region 区域模式：检测器参数在高百分位阈值下只会框出目标上的小亮斑，
       区域模式改用全局阈值 + 连通域提取整片亮区画大框；质心量程超过
       max_region_range（默认 25 m）的远处大亮带不画框；面积只用于剔除
       过小/过大的极端区域（min_blob_area=300 / max_blob_area=3000），
       不参与排序；区域按权重排序，权重 = 区域平均亮度 *
       (1 + near_gain * (1 - 量程/最大量程))，亮斑越强、距离原点越近权重越高，
       配合 --max-regions 可只保留最靠前的中近距离目标。
    5. 检测前默认做中值去噪（--denoise 可选 none/median/gaussian/bilateral），
       保存的标注图基于去噪后的图像；--range-norm 可进一步去除等距弧形
       噪声条带（按距离环估计背景亮度并相减）。
    6. 无需 ROS 运行环境；检测器模块中的 rospy/消息导入在无 ROS 时由桩模块代替。
"""

import argparse
import importlib.util
import math
import os
import re
import sys
import types

import cv2
import numpy as np

# 与 launch/sonar_target_detector.launch 保持一致的检测参数（离线复现）
DETECTOR_PARAMS = {
    'fov_deg': 130.0,
    'max_range': 30.0,
    'min_range': 0.2,
    'threshold_percentile': 98.5,
    'min_threshold': 50.0,
    'min_blob_area': 16,
    'publish_min_score': 0.15,
    'candidate_min_score': 0.025,
    'max_candidates': 20,
    'publish_candidates_without_bearing': True,
    'max_scene_candidates': 16,
    'bearing_candidate_gate_deg': 12.0,
    'min_mean_contrast': 7.0,
    'min_peak_delta': 6.0,
    'min_excess_score': 0.025,
    'use_local_threshold': True,
    'local_threshold_window_px': 31,
    'local_threshold_std_scale': 0.9,
    'local_threshold_min_delta': 3.0,
    'local_threshold_global_floor_ratio': 0.25,
    'suppress_boundary_like_blobs': True,
    'boundary_min_area': 120,
    'boundary_min_long_axis_px': 45.0,
    'boundary_min_physical_length_m': 4.0,
    'boundary_min_aspect_ratio': 4.0,
    'boundary_min_perimeter_area_ratio': 1.2,
    'boundary_mask_dilate_px': 2,
    'suppress_near_noise_blobs': True,
    'near_noise_range_m': 2.5,
    'near_noise_min_area': 32,
    'near_noise_min_physical_length_m': 0.45,
    # 区域模式：质心量程超过该值的区域视为远处大亮带，不画框（<=0 表示不限制）
    'max_region_range': 25.0,
    # 区域模式：面积只用于剔除过小/过大的极端区域（<=0 表示不限制）
    'max_blob_area': 3000,
    # 区域模式：近处优先增益，权重 = 区域平均亮度 * (1 + near_gain * (1 - 量程/最大量程))
    'near_gain': 1.0,
}


def _install_ros_stubs():
    """
    功能：在无 ROS 环境下安装 rospy 与消息包的桩模块，使检测器模块可独立加载。
    输入：无显式输入参数。
    输出：无显式返回；向 sys.modules 注册桩模块。
    """
    rospy_stub = types.ModuleType('rospy')
    rospy_stub.get_param = lambda name, default=None: default
    rospy_stub.init_node = lambda *args, **kwargs: None
    rospy_stub.loginfo = lambda *args, **kwargs: None
    rospy_stub.logwarn = lambda *args, **kwargs: None
    rospy_stub.logwarn_throttle = lambda *args, **kwargs: None
    rospy_stub.logerr = lambda *args, **kwargs: None
    rospy_stub.signal_shutdown = lambda *args, **kwargs: None
    rospy_stub.spin = lambda: None
    rospy_stub.Subscriber = object
    rospy_stub.Publisher = object
    rospy_stub.Time = object

    class _DummyMsg(object):
        """占位消息类型：离线处理流程不会构造或使用 ROS 消息。"""
        def __init__(self, *args, **kwargs):
            self.header = None
            self.data = None

    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.Image = _DummyMsg
    sensor_msgs.msg = sensor_msgs_msg
    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Float32 = _DummyMsg
    std_msgs_msg.Float32MultiArray = _DummyMsg
    std_msgs.msg = std_msgs_msg

    sys.modules['rospy'] = rospy_stub
    sys.modules['sensor_msgs'] = sensor_msgs
    sys.modules['sensor_msgs.msg'] = sensor_msgs_msg
    sys.modules['std_msgs'] = std_msgs
    sys.modules['std_msgs.msg'] = std_msgs_msg


def _load_detector_module():
    """
    功能：加载 forward_sonar_target_detector 模块；优先真实 ROS 环境，失败则用桩模块。
    输入：无显式输入参数。
    输出：返回加载完成的检测器模块。
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, 'forward_sonar_target_detector.py')
    spec = importlib.util.spec_from_file_location('forward_sonar_target_detector', module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        _install_ros_stubs()
        spec = importlib.util.spec_from_file_location('forward_sonar_target_detector', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def make_offline_detector(params=None):
    """
    功能：绕过 rospy.init_node 直接构造检测器实例，参数与 launch 文件保持一致。
    输入：`params` 为可选参数覆盖字典。
    输出：返回已配置好检测参数的 ForwardSonarTargetDetector 实例。
    """
    merged = dict(DETECTOR_PARAMS)
    merged.update(params or {})
    detector = object.__new__(DETECTOR_MODULE.ForwardSonarTargetDetector)
    for key, value in merged.items():
        setattr(detector, key, value)
    detector.kernel = np.ones((3, 3), dtype=np.uint8)
    detector.latest_bearing = None
    detector.latest_bearing_stamp = None
    return detector


def detect_image(detector, image, extra_mask=None):
    """
    功能：复现 ForwardSonarTargetDetector.image_cb 的检测流程（无方位引导分支）。
    输入：`detector` 为检测器实例，`image` 为单通道声呐图像，
          `extra_mask` 为可选附加有效区域掩膜（如扇形 alpha 掩膜）。
    输出：返回 (pdaf 候选, 场景候选, 被抑制区域) 三元组。
    """
    height, width = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    footprint_mask = detector.valid_mask(height, width)
    if extra_mask is not None:
        footprint_mask = footprint_mask & extra_mask
    valid_pixels = blurred[footprint_mask]
    if valid_pixels.size == 0:
        return [], [], []

    global_threshold = max(
        detector.min_threshold,
        float(np.percentile(valid_pixels, np.clip(detector.threshold_percentile, 0.0, 100.0))),
    )

    raw_binary, threshold_map = detector.build_detection_binary(blurred, footprint_mask, global_threshold)
    boundary_binary = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, detector.kernel)
    boundary_mask, suppressed = detector.extract_boundary_mask(boundary_binary, height, width)
    binary = np.array(raw_binary, copy=True)
    if np.any(boundary_mask):
        binary[boundary_mask] = 0
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, detector.kernel)
    if np.any(boundary_mask):
        binary[boundary_mask] = 0
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, detector.kernel)
    else:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, detector.kernel)

    background_level = float(np.median(valid_pixels))
    scene_candidates, candidate_suppressed = detector.extract_candidates(
        image, binary, threshold_map, background_level, None, detector.max_scene_candidates
    )
    suppressed.extend(candidate_suppressed)
    candidates = detector.filter_candidates_for_pdaf(scene_candidates, None)
    return candidates, scene_candidates, suppressed


def annotate_image(image, candidates, scene_candidates, suppressed):
    """
    功能：按 publish_debug_image 的画法在图像上标注检测结果。
    输入：`image` 为单通道图像，其余为检测流程输出的候选与抑制区域列表。
    输出：返回标注后的 BGR 三通道图像。
    """
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for item in suppressed or []:
        x, y, w, h = item['bbox']
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
    pdaf_labels = set(item.get('label') for item in candidates or [])
    for detection in scene_candidates or []:
        if detection.get('label') in pdaf_labels:
            continue
        x, y, w, h = detection['bbox']
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 220, 220), 1)
    for detection in candidates or []:
        x, y, w, h = detection['bbox']
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 220, 0), 1)
        center = (int(round(detection['col'])), int(round(detection['row'])))
        cv2.circle(debug, center, 4, (255, 255, 255), 1)
        cv2.line(debug, (center[0] - 5, center[1]), (center[0] + 5, center[1]), (255, 255, 255), 1)
        cv2.line(debug, (center[0], center[1] - 5), (center[0], center[1] + 5), (255, 255, 255), 1)
    return debug


def load_fan_image(path):
    """
    功能：读取扇形图像为单通道灰度图；若带 alpha 通道则同时返回扇形有效掩膜。
    输入：`path` 为图像文件路径。
    输出：返回 (灰度图, 扇形掩膜) 二元组，掩膜可为 None。
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    if raw.ndim == 3 and raw.shape[2] == 4:
        gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
        mask = raw[:, :, 3] > 0
    elif raw.ndim == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        mask = None
    else:
        gray = raw
        mask = None
    return gray, mask


def natural_key(name):
    """
    功能：按文件名中的数字序号自然排序。
    输入：`name` 为文件名。
    输出：返回用于排序的键。
    """
    match = re.search(r'\d+', name)
    return (int(match.group()) if match else 0, name)


def output_name(name, suffix='_det'):
    """
    功能：由输入文件名生成输出文件名，如 sonar_0_fan.png + '_det' -> sonar_0_det.png。
    输入：`name` 为输入文件名，`suffix` 为输出名后缀。
    输出：返回输出文件名。
    """
    stem, _ = os.path.splitext(name)
    if stem.endswith('_fan'):
        stem = stem[:-len('_fan')]
    return '{0}{1}.png'.format(stem, suffix)


def preprocess_image(image, denoise):
    """
    功能：检测前的图像去噪预处理（抑制声呐斑点噪声）。
    输入：`image` 为单通道声呐图像，`denoise` 为去噪方法名。
    输出：返回去噪后的图像（`none` 时原样返回）。
    """
    if denoise == 'median':
        return cv2.medianBlur(image, 3)
    if denoise == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    if denoise == 'bilateral':
        return cv2.bilateralFilter(image, 5, 25, 25)
    return image


def remove_range_arcs(image, mask=None, smooth=15):
    """
    功能：去除等距弧形噪声条带——按距离环估计背景亮度（环上像素的中值，
          目标只占少数方位角，中值近似纯背景），平滑后从图像中相减，
          抑制沿方位均匀分布、随量程起伏的增益条带。
    输入：`image` 为单通道声呐图像（扇形原点位于底部中心的翻转坐标系），
          `mask` 为有效区域掩膜，`smooth` 为背景剖面的平滑窗口（px）。
    输出：返回去除弧形条带后的图像。
    """
    height, width = image.shape[:2]
    cx = width // 2
    apex_y = height - 1
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.hypot(xx - cx, yy - apex_y).astype(np.int32)
    valid = (radius >= 0) & (radius <= height)
    if mask is not None:
        valid &= mask
    flat_r = radius[valid]
    flat_v = image[valid].astype(np.float32)
    if flat_r.size == 0:
        return image

    num_bins = height + 1
    counts = np.bincount(flat_r, minlength=num_bins)
    # 按 (半径, 亮度) 二级排序: 同一环内亮度升序, 便于取中值
    order = np.lexsort((flat_v, flat_r))
    v_sorted = flat_v[order]
    cum = np.cumsum(counts)
    nonzero = counts > 0
    # 每个距离环的精确中值: 奇数个取中间元素, 偶数个取中间两元素的平均
    upper = cum - counts + counts // 2
    lower = np.where((counts % 2 == 0) & nonzero, upper - 1, upper)
    ring_median = np.zeros(num_bins, dtype=np.float32)
    ring_median[nonzero] = (v_sorted[upper[nonzero]] + v_sorted[lower[nonzero]]) * 0.5
    # 填充无像素的距离环（线性插值）
    valid_idx = np.nonzero(nonzero)[0]
    ring_median = np.interp(np.arange(num_bins), valid_idx, ring_median[valid_idx])
    # 平滑背景剖面，避免过拟合真实目标
    window = max(int(smooth), 1)
    kernel = np.ones(window, np.float32) / window
    ring_bg = np.convolve(ring_median, kernel, mode='same')

    bg_map = ring_bg[np.clip(radius, 0, height)]
    result = image.astype(np.float32) - bg_map
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def detect_regions(detector, image, extra_mask=None):
    """
    功能：区域级检测——用全局阈值提取整片连通亮区（供“目标是一片区域”场景画大框）。
    输入：`detector` 为检测器实例，`image` 为单通道声呐图像，
          `extra_mask` 为可选附加有效区域掩膜（如扇形 alpha 掩膜）。
    输出：返回区域列表，每项含 bbox、质心、物理量测与形状度量；
          质心量程超过 detector.max_region_range 的远处大亮带被剔除；
          面积只用于剔除过小/过大的极端区域（min_blob_area/max_blob_area）；
          按权重降序排列，权重 = 区域平均亮度 * (1 + near_gain * (1 - 量程/最大量程))，
          即亮斑越强、距离原点越近，权重越高。
    """
    height, width = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    footprint_mask = detector.valid_mask(height, width)
    if extra_mask is not None:
        footprint_mask = footprint_mask & extra_mask
    valid_pixels = blurred[footprint_mask]
    if valid_pixels.size == 0:
        return []

    background_level = float(np.median(valid_pixels))
    threshold = max(
        detector.min_threshold,
        float(np.percentile(valid_pixels, np.clip(detector.threshold_percentile, 0.0, 100.0))),
    )
    binary = np.zeros_like(blurred, dtype=np.uint8)
    binary[(blurred >= threshold) & footprint_mask] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = []
    max_blob_area = getattr(detector, 'max_blob_area', 0)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < detector.min_blob_area:
            continue
        if max_blob_area > 0 and area > max_blob_area:
            continue
        component_mask = labels == label_idx
        bbox = (
            int(stats[label_idx, cv2.CC_STAT_LEFT]),
            int(stats[label_idx, cv2.CC_STAT_TOP]),
            int(stats[label_idx, cv2.CC_STAT_WIDTH]),
            int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
        )
        shape = detector.component_shape_metrics(component_mask, bbox, area, height, width)
        pixels = image[component_mask].astype(np.float32)
        rows, cols = np.nonzero(component_mask)
        weights = pixels + 1.0e-3
        weight_sum = float(np.sum(weights))
        row = float(np.dot(rows.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
        col = float(np.dot(cols.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
        measurement = detector.row_col_to_measurement(row, col, height, width)
        max_region_range = getattr(detector, 'max_region_range', 0.0)
        if max_region_range > 0.0 and measurement['range'] > max_region_range:
            continue
        mean_intensity = float(np.mean(pixels))
        near_gain = max(getattr(detector, 'near_gain', 1.0), 0.0)
        near_weight = 1.0 + near_gain * max(
            0.0, 1.0 - measurement['range'] / max(detector.max_range, 1.0e-6))
        weight = mean_intensity * float(near_weight)
        score = float(np.clip(weight / max(255.0 * (1.0 + near_gain), 1.0e-6), 0.0, 1.0))
        mean_contrast = mean_intensity - background_level
        uncertainty = detector.estimate_candidate_uncertainty(
            score, max(mean_contrast, 0.0), -1.0)
        regions.append({
            'label': int(label_idx),
            'area': area,
            'bbox': bbox,
            'row': row,
            'col': col,
            'range': measurement['range'],
            'bearing': measurement['bearing'],
            'forward': measurement['forward'],
            'lateral': measurement['lateral'],
            'mean_intensity': mean_intensity,
            'peak_intensity': float(np.max(pixels)),
            'shape': shape,
            'near_weight': float(near_weight),
            'weight': weight,
            'score': score,
            'mean_contrast': float(mean_contrast),
            'uncertainty': uncertainty,
        })
    regions.sort(key=lambda item: item['weight'], reverse=True)
    return regions


def annotate_regions(image, regions):
    """
    功能：在图像上为每个区域绘制外接框与质心十字。
    输入：`image` 为单通道图像，`regions` 为 detect_regions 返回的区域列表。
    输出：返回标注后的 BGR 三通道图像。
    """
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for region in regions:
        x, y, w, h = region['bbox']
        cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 220, 0), 2)
        center = (int(round(region['col'])), int(round(region['row'])))
        cv2.circle(debug, center, 4, (255, 255, 255), 1)
        cv2.line(debug, (center[0] - 5, center[1]), (center[0] + 5, center[1]), (255, 255, 255), 1)
        cv2.line(debug, (center[0], center[1] - 5), (center[0], center[1] + 5), (255, 255, 255), 1)
    return debug


def main():
    """
    功能：解析命令行参数并执行批量检测标注。
    输入：依赖命令行参数。
    输出：返回进程退出码。
    """
    parser = argparse.ArgumentParser(description='离线批量声呐图像目标检测与画框标注')
    parser.add_argument('path', help='frame 目录（包含 fan_output/）或直接指向图像目录')
    parser.add_argument('--fan-dir', default='fan_output', help='扇形图像子目录名（默认 fan_output）')
    parser.add_argument('--output-dir', default=None, help='输出目录（默认 <frame>/detect_output）')
    parser.add_argument('--ext', nargs='+', default=('png', 'jpg', 'jpeg'), help='处理的图像扩展名')
    parser.add_argument('--max-images', type=int, default=0, help='最多处理张数（0 表示全部）')
    parser.add_argument('--show', action='store_true', help='逐帧弹出标注结果窗口')
    parser.add_argument('--region', action='store_true',
                        help='区域模式：按整片连通亮区画大框（默认只框检测器候选小斑块）')
    parser.add_argument('--threshold-percentile', type=float, default=None,
                        help='区域模式的全局阈值百分位（默认 93.0）')
    parser.add_argument('--min-threshold', type=float, default=None,
                        help='区域模式的绝对阈值下限（默认 40.0）')
    parser.add_argument('--min-blob-area', type=int, default=None,
                        help='区域模式的最小连通域面积，剔除小斑点（默认 300）')
    parser.add_argument('--max-blob-area', type=int, default=None,
                        help='区域模式的最大连通域面积，剔除过大区域（默认 3000，<=0 不限制）')
    parser.add_argument('--max-region-range', type=float, default=None,
                        help='区域模式的质心最大量程，超过视为远处大亮带不画框（默认 25.0 m，<=0 不限制）')
    parser.add_argument('--near-gain', type=float, default=None,
                        help='区域模式近处优先增益，权重=亮度*(1+增益*(1-量程/最大量程))（默认 1.0）')
    parser.add_argument('--max-regions', type=int, default=0,
                        help='区域模式最多保留的区域数，按近处优先权重取前 N（0 表示全部）')
    parser.add_argument('--denoise', choices=('none', 'median', 'gaussian', 'bilateral'),
                        default='median', help='检测前的去噪预处理方法（默认 median）')
    parser.add_argument('--range-norm', action='store_true',
                        help='去除等距弧形噪声条带：按距离环估计背景亮度并相减')
    parser.add_argument('--range-norm-smooth', type=int, default=15,
                        help='距离环背景剖面的平滑窗口（px，默认 15）')
    args = parser.parse_args()

    path = os.path.abspath(os.path.expanduser(args.path))
    fan_sub = os.path.join(path, args.fan_dir)
    input_dir = fan_sub if os.path.isdir(fan_sub) else path
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir), 'detect_output')
    os.makedirs(output_dir, exist_ok=True)
    denoised_dir = os.path.join(os.path.dirname(input_dir), 'denoised_output')
    os.makedirs(denoised_dir, exist_ok=True)

    extensions = tuple('.' + ext.lower() for ext in args.ext)
    files = [name for name in os.listdir(input_dir) if name.lower().endswith(extensions)]
    files.sort(key=natural_key)
    if not files:
        print('[batch_sonar] 未在 {0} 中找到图像文件'.format(input_dir))
        return 1
    if args.max_images > 0:
        files = files[:args.max_images]

    params = {}
    if args.region:
        params.setdefault('threshold_percentile', 93.0)
        params.setdefault('min_threshold', 40.0)
        params.setdefault('min_blob_area', 300)
    if args.threshold_percentile is not None:
        params['threshold_percentile'] = args.threshold_percentile
    if args.min_threshold is not None:
        params['min_threshold'] = args.min_threshold
    if args.min_blob_area is not None:
        params['min_blob_area'] = args.min_blob_area
    if args.max_blob_area is not None:
        params['max_blob_area'] = args.max_blob_area
    if args.max_region_range is not None:
        params['max_region_range'] = args.max_region_range
    if args.near_gain is not None:
        params['near_gain'] = args.near_gain
    detector = make_offline_detector(params)

    mode = '区域模式' if args.region else '检测器模式'
    print('[batch_sonar] 输入目录: {0}（{1} 张）'.format(input_dir, len(files)))
    print('[batch_sonar] 输出目录: {0} | 滤波图: {1} | {2} | 去噪 {3}'.format(
        output_dir, denoised_dir, mode, args.denoise))

    total_count = 0
    images_with_target = 0
    for index, name in enumerate(files):
        source = os.path.join(input_dir, name)
        image, fan_mask = load_fan_image(source)
        if image is None:
            print('[{0}/{1}] [skip] 无法读取: {2}'.format(index + 1, len(files), source))
            continue

        # 扇形图原点在上方（近距在上），检测器约定第 0 行为最远距离，翻转后处理
        flipped = np.flipud(image)
        flipped_mask = np.flipud(fan_mask) if fan_mask is not None else None
        processed = preprocess_image(flipped, args.denoise)
        if args.range_norm:
            footprint = detector.valid_mask(*flipped.shape)
            if flipped_mask is not None:
                footprint = footprint & flipped_mask
            processed = remove_range_arcs(processed, footprint, args.range_norm_smooth)
        # 滤波后（未加检测框）的图像另存到 denoised_output/
        cv2.imwrite(os.path.join(denoised_dir, output_name(name, '_den')),
                    np.flipud(processed))

        if args.region:
            regions = detect_regions(detector, processed, flipped_mask)
            if args.max_regions > 0:
                regions = regions[:args.max_regions]
            annotated = annotate_regions(processed, regions)
            annotated = np.flipud(annotated)  # 翻回原方向保存
            if regions:
                images_with_target += 1
                total_count += len(regions)
            detail = ' | '.join(
                'a={0} i={1:.0f} w={2:.0f} r={3:.1f}m b={4:+.0f}deg len={5:.1f}m'.format(
                    item['area'], item['mean_intensity'], item['weight'], item['range'],
                    math.degrees(item['bearing']), item['shape']['physical_length_m'])
                for item in regions
            )
            print('[{0}/{1}] {2} -> {3} | 区域 {4} | {5}'.format(
                index + 1, len(files), name, output_name(name), len(regions), detail))
        else:
            candidates, scene_candidates, suppressed = detect_image(detector, processed, flipped_mask)
            annotated = annotate_image(processed, candidates, scene_candidates, suppressed)
            annotated = np.flipud(annotated)  # 翻回原方向保存
            if candidates:
                images_with_target += 1
                total_count += len(candidates)
            detail = ' | '.join(
                'r={0:.1f}m b={1:+.0f}deg s={2:.2f}'.format(
                    item['range'], math.degrees(item['bearing']), item['score'])
                for item in candidates
            )
            print('[{0}/{1}] {2} -> {3} | 候选 {4} 抑制 {5} | {6}'.format(
                index + 1, len(files), name, output_name(name),
                len(candidates), len(suppressed), detail))

        out_path = os.path.join(output_dir, output_name(name))
        cv2.imwrite(out_path, annotated)

        if args.show:
            cv2.imshow(name, annotated)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()
    print('[batch_sonar] 完成: 处理 {0} 张, 检出 {1} 个, 有目标的图像 {2} 张'.format(
        len(files), total_count, images_with_target))
    return 0


DETECTOR_MODULE = _load_detector_module()


if __name__ == '__main__':
    sys.exit(main())
