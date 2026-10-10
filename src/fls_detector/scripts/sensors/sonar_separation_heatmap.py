#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：生成主动场景分离代价热力图。
     对 <frame>/fan_output/ 下的扇形声呐图像，采用 batch_sonar_image_detector.py
     的降噪与区域检测流程（中值去噪 + range_norm + region），按 proposed 方法
     （mpc_belief_target_tracker.py）中的主动场景分离代价，在 FOV 内逐点计算
     代价并绘制热力图，用点标注候选区域中心。

代价流程（与论文 Active scene disambiguation 及 mpc_belief_target_tracker.py 一致）：
    1) 候选权重归一化：w^i ∝ c^i (1-ε^i) / sqrt(max[ρ^i, 1])，归一化使 Σw=1
       （c^i 为置信度，ε^i 为不确定性，量程归一化抑制远距杂波）；
    2) 相对参考方位角做 Gaussian focus boost：
       w^i ← w^i (1 + κ·g_f·exp(-½(e^i_θ/σ_f)²))，κ=1−α
       （参考为 prior 估计方位，1−α 为对应信念置信度），
       boost 后不再归一化，截断到 [0,1]；
    3) 将横纵坐标离散为网格，对 FOV 内每个网格点，计算各候选相对该点的
       方位角，两两作差代入高斯重叠代价并按权重求和（从该点看出去所有
       候选方向的重叠/歧义程度）：
       theta_i(x, y) = atan2(lat_i - lat(x, y), fwd_i - fwd(x, y))
       z(x, y) = s_alpha * w_s
                 * sum_{i<j} w^i w^j exp(-0.5 * ((theta_i - theta_j) / sigma_s)^2)
       s_alpha = base_fraction + (1 - base_fraction) * alpha
    FOV 外的点设为透明；色阶按每帧 FOV 内代价的参考分位数归一
    （--norm-percentile，默认 95）并做伽马校正（--gamma，默认 0.5）
    以增强低值区对比。
常数默认取自 launch/mpc_sonar_bearing_tracking.launch（proposed 执行值）：
    w_s = 2.0, sigma_s = 6.0deg, sigma_f = 12.0deg, g_f = 3.0,
    active_scene_base_fraction = 0.3, N_s = 8, min_score = 0.08

用法：
    python3 sonar_separation_heatmap.py <frame路径> --alpha <0..1> [--bearing <方位角deg>]
示例：
    python3 sonar_separation_heatmap.py ~/AUV_sonar_images/frame_test --alpha 0.7 --bearing 16
输出：
    <frame>/separation_heatmap/sonar_<序号>_heat.png
        （独立热力图，仅 FOV 内着色；FOV 外默认透明，--bg white 时为纯白）

说明：
    方位角约定与检测器一致（正值方位在图像左侧，即 col=0 一侧）。
    热力图配色为 cool 渐变（低代价青色 -> 高代价品红），候选中心为
    橙色实心黑边圆点；bearing 参考射线默认绘制（--no-bearing-line 关闭）。
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

import batch_sonar_image_detector as batch

# proposed 方法执行时的常数（launch/mpc_sonar_bearing_tracking.launch）
W_ACTIVE_SCENE_SEPARATION = 2.0
ACTIVE_SCENE_BASE_FRACTION = 0.3
ACTIVE_SCENE_SEP_SIGMA_DEG = 6.0

# 热力图色阶: cool 渐变（低代价青色 -> 高代价品红，MATLAB cool 同款线性插值）
_HEAT_STOPS = (
    (0.00, (0, 255, 255)),     # cyan    低代价
    (1.00, (255, 0, 255)),     # magenta 高代价
)
# 候选中心标记: 橙色 #fab219 实心 + 黑色边线
MARKER_COLOR = (25, 178, 250)   # BGR
MARKER_RING = (0, 0, 0)
MARKER_RADIUS = 20


def build_heat_lut():
    """
    功能：按蓝序贯色阶构建 256 级查找表。
    输入：无显式输入参数。
    输出：返回 (256, 1, 3) 的 uint8 BGR 查找表。
    """
    t_stops = np.array([stop[0] for stop in _HEAT_STOPS], dtype=np.float32)
    rgb_stops = np.array([stop[1] for stop in _HEAT_STOPS], dtype=np.float32)
    levels = np.arange(256, dtype=np.float32) / 255.0
    rgb = np.stack(
        [np.interp(levels, t_stops, rgb_stops[:, c]) for c in range(3)], axis=1
    ).astype(np.uint8)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:, 0, :] = rgb[:, ::-1]  # RGB -> BGR
    return lut


def physical_grid(detector, height, width):
    """
    功能：计算图像每个像素的物理坐标（前向与侧向，米）。
    输入：`detector` 为检测器实例，`height`、`width` 为图像尺寸。
    输出：返回 (forward, lateral) 两个 (height, width) 矩阵。
    """
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    forward = detector.min_range + (1.0 - rows / max(height - 1, 1)) * (
        detector.max_range - detector.min_range)
    max_lateral = detector.max_lateral_span()
    lateral = max_lateral - cols * (2.0 * max_lateral / max(width - 1, 1))
    return (np.broadcast_to(forward, (height, width)),
            np.broadcast_to(lateral, (height, width)))


def scene_weights(regions, bearing0, focus_sigma_deg, focus_gain, kappa,
                  max_candidates, min_score):
    """
    功能：按论文 Active scene disambiguation 流程生成候选权重。
          1) 基础权重 w^i ∝ c^i (1-ε^i) / sqrt(max[ρ^i, 1])，归一化使 Σw=1；
          2) 相对参考方位角做 Gaussian focus boost：
             w^i ← w^i (1 + κ·g_f·exp(-½(e^i_θ/σ_f)²))，
             κ=1−α（参考为 prior 估计方位，α 为不确定性混合指数，
             对应信念置信度 1−α），无参考方位时不做 boost；
          3) 截断到 [0,1]（与 mpc_belief_target_tracker 一致，boost 后不再归一化）。
    输入：`regions` 为区域候选列表（含 score/uncertainty/range/bearing），
          `bearing0` 为 prior 参考方位角（rad，可为 None），`kappa` 为
          boost 增益系数（=1−α），其余为 focus 常数与数量上限。
    输出：返回 (选中的候选列表, 最终权重列表)。
    """
    candidates = []
    for item in regions:
        score = float(item.get('score', 0.0))
        if score < min_score:
            continue
        base = score * (1.0 - float(item.get('uncertainty', 0.0))) / math.sqrt(
            max(item['range'], 1.0))
        candidates.append((base, item))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    candidates = candidates[:max(int(max_candidates), 0)]

    base_norm = max(sum(pair[0] for pair in candidates), 1.0e-6)
    sigma_f = math.radians(focus_sigma_deg)
    selected, weights = [], []
    for base, item in candidates:
        weight = base / base_norm
        if bearing0 is not None and kappa > 0.0 and focus_gain > 0.0:
            error = math.atan2(math.sin(item['bearing'] - bearing0),
                               math.cos(item['bearing'] - bearing0))
            focus = math.exp(-0.5 * (error / sigma_f) * (error / sigma_f))
            weight *= 1.0 + kappa * focus_gain * focus
        selected.append(item)
        weights.append(float(np.clip(weight, 0.0, 1.0)))
    return selected, weights


def separation_cost(forward_map, lateral_map, cand_forward, cand_lateral,
                    weights, alpha, sep_sigma_deg, w_sep, base_fraction,
                    grid_step=1):
    """
    功能：按论文场景分离代价（Eq. scene）计算 FOV 内逐点代价。
          将横纵坐标离散为网格，对每个网格点，计算所有候选相对于该点的
          方位角，两两作差代入高斯重叠代价并按权重求和（从该点看出去
          所有候选方向的重叠/歧义程度）：
              theta_i(x, y) = atan2(lat_i - lat(x, y), fwd_i - fwd(x, y))
              z(x, y) = s_alpha * w_s
                        * sum_{i<j} w^i w^j exp(-0.5 * ((theta_i - theta_j) / sigma_s)^2)
    输入：`forward_map`/`lateral_map` 为网格点物理坐标，`cand_forward`/
          `cand_lateral` 为候选形心物理坐标，`weights` 为 focus boost 后的
          候选权重，其余为分离代价常数，`grid_step` 为网格离散步长。
    输出：返回与原图同尺寸的代价矩阵。
    """
    height, width = forward_map.shape
    if grid_step > 1:
        fwd = forward_map[::grid_step, ::grid_step]
        lat = lateral_map[::grid_step, ::grid_step]
    else:
        fwd, lat = forward_map, lateral_map
    z = np.zeros(fwd.shape, dtype=np.float32)
    sigma = math.radians(sep_sigma_deg)
    n = len(cand_forward)
    for i in range(n):
        theta_i = np.arctan2(cand_lateral[i] - lat, cand_forward[i] - fwd)
        for j in range(i + 1, n):
            theta_j = np.arctan2(cand_lateral[j] - lat, cand_forward[j] - fwd)
            dtheta = np.arctan2(np.sin(theta_i - theta_j), np.cos(theta_i - theta_j))
            z += weights[i] * weights[j] * np.exp(
                -0.5 * (dtheta / sigma) * (dtheta / sigma))
    alpha_sep = base_fraction + (1.0 - base_fraction) * np.clip(alpha, 0.0, 1.0)
    z *= alpha_sep * w_sep
    if grid_step > 1:
        z = cv2.resize(z, (width, height), interpolation=cv2.INTER_LINEAR)
    return z


def render_heatmap(mask, cost_norm, lut, white_bg=False):
    """
    功能：渲染独立热力图——仅在 FOV 内绘制代价色阶，
          FOV 外为透明（BGRA，alpha=0）或纯白。
    输入：`mask` 为 FOV 掩膜，`cost_norm` 为 [0,1] 归一化代价，
          `lut` 为色阶查找表，`white_bg` 为 True 时 FOV 外用纯白填充。
    输出：返回 BGRA 四通道（透明背景）或 BGR 三通道（白色背景）图像。
    """
    idx = (np.clip(cost_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = lut[idx, 0]  # (H, W, 3) BGR
    height, width = mask.shape
    if white_bg:
        out = np.full((height, width, 3), 255, dtype=np.uint8)
        out[mask] = colored[mask]
    else:
        out = np.zeros((height, width, 4), dtype=np.uint8)
        out[mask, :3] = colored[mask]
        out[mask, 3] = 255
    return out


def draw_candidates(image, regions):
    """
    功能：用橙色实心 + 黑色边线的圆点标注候选区域中心。
    输入：`image` 为 BGR/BGRA 图像，`regions` 为区域候选列表。
    输出：无显式返回；原地修改图像。
    """
    has_alpha = image.shape[2] == 4
    for region in regions:
        center = (int(round(region['col'])), int(round(region['row'])))
        if has_alpha:
            cv2.circle(image, center, MARKER_RADIUS, MARKER_RING + (255,), -1)
            cv2.circle(image, center, max(MARKER_RADIUS - 2, 1), MARKER_COLOR + (255,), -1)
        else:
            cv2.circle(image, center, MARKER_RADIUS, MARKER_RING, -1)
            cv2.circle(image, center, max(MARKER_RADIUS - 2, 1), MARKER_COLOR, -1)


def draw_bearing_ray(image, detector, bearing):
    """
    功能：沿给定声学方位角绘制参考射线（扇形顶点到远边界）。
    输入：`image` 为 BGR/BGRA 图像（翻转坐标系，顶点在底部中心），
          `detector` 为检测器实例，`bearing` 为方位角（rad）。
    输出：无显式返回；原地修改图像。
    """
    height, width = image.shape[:2]
    has_alpha = image.shape[2] == 4
    color = (255, 255, 255, 255) if has_alpha else (255, 255, 255)
    pts = []
    for row in range(0, height, max(1, height // 40)):
        forward = detector.min_range + (1.0 - row / max(height - 1, 1)) * (
            detector.max_range - detector.min_range)
        col = detector.bearing_to_col(bearing, forward, width)
        pts.append((col, row))
    if len(pts) >= 2:
        cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, color, 1)


def add_colorbar(image, z_ref, lut):
    """
    功能：在图像右侧添加 cool 色阶 colorbar，并标注绝对代价值（0 到 z_ref）。
    输入：`image` 为 BGR/BGRA 图像，`z_ref` 为色阶参考值（顶=品红=z_ref），
          `lut` 为色阶查找表。
    输出：返回加宽后的图像。
    """
    height, width = image.shape[:2]
    has_alpha = image.shape[2] == 4
    bar_w = 56
    if has_alpha:
        canvas = np.zeros((height, width + bar_w, 4), dtype=np.uint8)
        canvas[:, :width] = image
        canvas[:, width:, 3] = 255
        canvas[:, width:, :3] = 255
    else:
        canvas = np.full((height, width + bar_w, 3), 255, dtype=np.uint8)
        canvas[:, :width] = image

    grad = np.linspace(255, 0, height, dtype=np.uint8)  # 顶=品红(高值) 底=青(0)
    bar = lut[grad, 0][:, None, :]  # (height, 1, 3)
    x0, x1 = width + 12, width + 28
    if has_alpha:
        canvas[:, x0:x1, :3] = bar
    else:
        canvas[:, x0:x1] = bar

    ink = (0, 0, 0, 255) if has_alpha else (0, 0, 0)
    for frac, label in ((0.0, '0'), (0.5, '{0:.2f}'.format(z_ref * 0.5)),
                        (1.0, '{0:.2f}'.format(z_ref))):
        y = int(height - 1 - frac * (height - 1))
        cv2.line(canvas, (x1, y), (x1 + 5, y), ink, 1)
        cv2.putText(canvas, label, (x1 + 7, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ink, 1, cv2.LINE_AA)
    return canvas


def main():
    """
    功能：解析命令行参数并生成分离代价热力图。
    输入：依赖命令行参数。
    输出：返回进程退出码。
    """
    parser = argparse.ArgumentParser(description='主动场景分离代价热力图生成')
    parser.add_argument('path', help='frame 目录（包含 fan_output/）或直接指向图像目录')
    parser.add_argument('--bearing', type=float, default=None,
                        help='prior 估计方位角（deg，检测器约定：正值在图像左侧），'
                             '作为 Gaussian focus boost 的参考方位，κ=1−alpha（信念置信度越低'
                             ' boost 越弱），并绘制参考射线；缺省时不做 boost，'
                             '权重退化为归一化基础权重')
    parser.add_argument('--no-bearing-line', action='store_true',
                        help='不绘制 bearing 参考射线（默认绘制）')
    parser.add_argument('--alpha', type=float, required=True,
                        help='不确定性加权系数 alpha（0..1）')
    parser.add_argument('--grid-step', type=int, default=2,
                        help='代价场网格离散步长 px（默认 2；1 为全分辨率）')
    parser.add_argument('--norm-percentile', type=float, default=95.0,
                        help='色阶参考分位数：按 FOV 内代价的该百分位归一'
                             '（100=最大值，默认 95，越小对比越强）')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='色阶伽马校正（<1 提升低值区对比，默认 0.5；1.0 为线性）')
    parser.add_argument('--fixed-ref', type=float, default=None,
                        help='固定参考标度：按该常数线性归一（真值模式，跨帧/跨参数可比），'
                             '通常配 --gamma 1.0 --colorbar 使用')
    parser.add_argument('--colorbar', action='store_true',
                        help='在图像右侧绘制 colorbar 并标注绝对代价值')
    parser.add_argument('--truth', action='store_true',
                        help='真值模式预设：单图内线性归一化（按本图最大代价，'
                             '不跨图共用标度）并绘制 colorbar，等价于 '
                             '--norm-percentile 100 --gamma 1.0 --colorbar')
    parser.add_argument('--fan-dir', default='fan_output', help='扇形图像子目录名（默认 fan_output）')
    parser.add_argument('--output-dir', default=None,
                        help='输出目录（默认 <frame>/separation_heatmap）')
    parser.add_argument('--max-images', type=int, default=0, help='最多处理张数（0 表示全部）')
    parser.add_argument('--bg', choices=('transparent', 'white'), default='transparent',
                        help='FOV 外背景：transparent 透明（默认）或 white 纯白')
    parser.add_argument('--show', action='store_true', help='逐帧弹出结果窗口')
    # 分离代价常数（默认与 launch/mpc_sonar_bearing_tracking.launch 一致）
    parser.add_argument('--sep-sigma-deg', type=float, default=ACTIVE_SCENE_SEP_SIGMA_DEG,
                        help='分离代价角度带宽 sigma_s（deg，默认 6.0）')
    parser.add_argument('--w-sep', type=float, default=W_ACTIVE_SCENE_SEPARATION,
                        help='分离代价静态权重 w_s（默认 2.0）')
    parser.add_argument('--base-fraction', type=float, default=ACTIVE_SCENE_BASE_FRACTION,
                        help='分离代价基础占比（默认 0.3）')
    # Gaussian focus boost 常数（launch: active_scene_bearing_focus_deg / gain）
    parser.add_argument('--focus-sigma-deg', type=float, default=12.0,
                        help='focus boost 角度带宽 sigma_f（deg，默认 12.0）')
    parser.add_argument('--focus-gain', type=float, default=3.0,
                        help='focus boost 增益 g_f（默认 3.0）')
    parser.add_argument('--min-score', type=float, default=0.08,
                        help='候选最小置信度（对应 active_scene_min_score，默认 0.08）')
    parser.add_argument('--max-candidates', type=int, default=8,
                        help='参与代价的候选数量上限 N_s（对应 active_scene_max_points，默认 8）')
    # 区域检测参数（默认与 batch 工具一致）
    parser.add_argument('--threshold-percentile', type=float, default=93.0)
    parser.add_argument('--min-threshold', type=float, default=40.0)
    parser.add_argument('--min-blob-area', type=int, default=300)
    parser.add_argument('--max-blob-area', type=int, default=3000)
    parser.add_argument('--max-region-range', type=float, default=25.0)
    parser.add_argument('--near-gain', type=float, default=1.0)
    args = parser.parse_args()

    if args.truth:
        args.norm_percentile = 100.0
        args.gamma = 1.0
        args.colorbar = True

    path = os.path.abspath(os.path.expanduser(args.path))
    fan_sub = os.path.join(path, args.fan_dir)
    input_dir = fan_sub if os.path.isdir(fan_sub) else path
    output_dir = args.output_dir or os.path.join(os.path.dirname(input_dir), 'separation_heatmap')
    os.makedirs(output_dir, exist_ok=True)

    extensions = ('.png', '.jpg', '.jpeg')
    files = [name for name in os.listdir(input_dir) if name.lower().endswith(extensions)]
    files.sort(key=batch.natural_key)
    if not files:
        print('[sep_heatmap] 未在 {0} 中找到图像文件'.format(input_dir))
        return 1
    if args.max_images > 0:
        files = files[:args.max_images]

    detector = batch.make_offline_detector({
        'threshold_percentile': args.threshold_percentile,
        'min_threshold': args.min_threshold,
        'min_blob_area': args.min_blob_area,
        'max_blob_area': args.max_blob_area,
        'max_region_range': args.max_region_range,
        'near_gain': args.near_gain,
    })
    lut = build_heat_lut()
    bearing0 = math.radians(args.bearing) if args.bearing is not None else None
    alpha_sep = args.base_fraction + (1.0 - args.base_fraction) * np.clip(args.alpha, 0.0, 1.0)
    max_cost = alpha_sep * args.w_sep

    print('[sep_heatmap] 输入目录: {0}（{1} 张）'.format(input_dir, len(files)))
    print('[sep_heatmap] 输出目录: {0}'.format(output_dir))
    print('[sep_heatmap] alpha={0:.2f} alpha_sep={1:.2f} 单对代价={2:.2f} '
          'sigma_s={3:.1f}deg 网格步长={4}px 色阶参考分位={5:.1f} gamma={6:.2f}'.format(
              args.alpha, alpha_sep, max_cost, args.sep_sigma_deg, args.grid_step,
              args.norm_percentile, args.gamma))
    print('[sep_heatmap] focus: sigma_f={0:.1f}deg g_f={1:.1f} min_score={2:.2f} '
          'N_s={3}'.format(args.focus_sigma_deg, args.focus_gain,
                           args.min_score, args.max_candidates))
    if bearing0 is not None:
        kappa = 1.0 - np.clip(args.alpha, 0.0, 1.0)
        print('[sep_heatmap] focus prior 参考方位 {0:+.1f}deg (kappa=1-alpha={1:.2f})'.format(
            args.bearing, kappa))

    for index, name in enumerate(files):
        source = os.path.join(input_dir, name)
        image, fan_mask = batch.load_fan_image(source)
        if image is None:
            print('[{0}/{1}] [skip] 无法读取: {2}'.format(index + 1, len(files), source))
            continue

        # 与 batch 工具一致: 翻转 -> 中值去噪 -> 去弧形条带 -> 区域检测
        flipped = np.flipud(image)
        flipped_mask = np.flipud(fan_mask) if fan_mask is not None else None
        processed = batch.preprocess_image(flipped, 'median')
        footprint = detector.valid_mask(*flipped.shape)
        if flipped_mask is not None:
            footprint = footprint & flipped_mask
        processed = batch.remove_range_arcs(processed, footprint, 15)
        regions = batch.detect_regions(detector, processed, footprint)

        forward_map, lateral_map = physical_grid(detector, *processed.shape)
        kappa = 1.0 - np.clip(args.alpha, 0.0, 1.0)  # prior 参考: κ=1−α
        selected, weights = scene_weights(
            regions, bearing0, args.focus_sigma_deg, args.focus_gain, kappa,
            args.max_candidates, args.min_score)
        cost = separation_cost(
            forward_map, lateral_map,
            [item['forward'] for item in selected],
            [item['lateral'] for item in selected],
            weights,
            args.alpha, args.sep_sigma_deg, args.w_sep, args.base_fraction,
            args.grid_step)
        if args.fixed_ref is not None:
            z_scale = max(args.fixed_ref, 1.0e-9)  # 固定标度: 颜色线性对应绝对代价
        else:
            z_scale = max(float(np.percentile(
                cost[footprint], np.clip(args.norm_percentile, 0.0, 100.0))), 1.0e-9)
        cost_norm = np.clip(cost / z_scale, 0.0, 1.0)
        if args.gamma != 1.0:
            cost_norm = np.power(cost_norm, args.gamma)
        result = render_heatmap(footprint, cost_norm, lut, args.bg == 'white')
        draw_candidates(result, selected)
        if bearing0 is not None and not args.no_bearing_line:
            draw_bearing_ray(result, detector, bearing0)
        result = np.flipud(result)  # 翻回原方向保存
        if args.colorbar:
            result = add_colorbar(result, z_scale, lut)

        out_path = os.path.join(output_dir, batch.output_name(name, '_heat'))
        cv2.imwrite(out_path, result)

        detail = ' | '.join(
            'r={0:.1f}m b={1:+.0f}deg w={2:.3f}'.format(
                item['range'], math.degrees(item['bearing']), weight)
            for item, weight in zip(selected, weights)
        )
        z_min = float(cost[footprint].min())
        z_max = float(cost[footprint].max())
        print('[{0}/{1}] {2} -> {3} | 候选 {4}/{5} | z[{6:.2f},{7:.2f}] 色阶参考={8:.2f} | {9}'.format(
            index + 1, len(files), name, batch.output_name(name, '_heat'),
            len(selected), len(regions), z_min, z_max, z_scale, detail))

        if args.show:
            cv2.imshow(name, result)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()
    print('[sep_heatmap] 完成: 处理 {0} 张'.format(len(files)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
