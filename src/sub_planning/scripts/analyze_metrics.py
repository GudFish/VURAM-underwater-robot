#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper-facing figures from compact metrics CSVs.

图1-3  各方法的 XY 轨迹图（proposed / bearing-only / CE-NMPC）：
       - 轨迹透明度随时间渐显（起始 0.25 → 结束 1.0），目标轨迹为虚线
       - 每隔 20s：机器人处绘制朝向 yaw 的声呐 FOV 扇形（颜色/透明度与
         轨迹该时刻一致），目标处绘制正方形标记（透明度同样随时刻变化）
       - 起点标注为圆、终点标注为叉；图例仅标注黑色目标轨迹的起点与终点
       - 三张图共用同一坐标区间
图4    估计误差随时间变化（3 方法叠加）
图5    置信度（cov_trace）随时间变化（3 方法叠加）
图6    机器人-目标距离随时间变化（3 方法叠加 + 期望站位/容差带）
图7-10 所有种子测试的柱状统计（RMSE / standoff success / control effort / FOV），
       并在终端打印均值 ± 标准差表格，便于论文表格填数。

用法:
  python3 analyze_metrics.py                      # 默认 ~/sub_planning_metrics
  python3 analyze_metrics.py -i <csv目录> -o <输出目录>
  python3 analyze_metrics.py --run 2              # 图1-6 使用每个方法的第 3 次运行
  # 手动指定坐标范围（负数需用等号形式传参）：
  python3 analyze_metrics.py --traj-xlim=-30,5 --traj-ylim=-10,20 \
                             --fig4-ylim=0,15 --fig5-ylim=0,10 --fig6-ylim=0,12

  当前论文用：python3 ./src/sub_planning/scripts/analyze_metrics.py --run 5 --traj-xlim=-29,2 --traj-ylim=-6,14.1 --fig4-ylim=-0.5,46 --fig5-ylim=-10,750 --fig6-ylim=-0.2,15.3

说明:
  CSV 文件名格式为 {method}_seed{seed}_{timestamp}.csv，脚本会打印
  图1-6 数据来源的随机种子；用 --run N 选择每个方法按文件名排序后的
  第 N 次运行（默认 0）。所有图片不绘制标题。
  图片输出为 PDF 矢量格式（fig1~fig10）；每张图的尺寸与边距通过
  文件顶部的 FIGURE_CONFIG 配置独立调整。
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# 全局字体设置：Times New Roman 15pt，数学公式用 LaTeX 风格（stix 数学字体）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'mathtext.fontset': 'stix',
    # 坐标轴标签与坐标轴之间的距离（磅），数值越小标签越靠近坐标轴
    'axes.labelpad': 0,
})


METHODS = ['proposed', 'b1_bearing_only', 'b2_ce_nmpc']
METHOD_LABELS = {
    'proposed': 'Proposed',
    'b1_bearing_only': 'BO-AT',
    'b2_ce_nmpc': 'CE-NMPC',
}
METHOD_COLORS = {
    'proposed': '#C5272D',
    'b1_bearing_only': '#037F77',
    'b2_ce_nmpc': '#0001A1',
}
# METHOD_COLORS = {
#     'proposed': '#d62728',
#     'b1_bearing_only': '#1f77b4',
#     'b2_ce_nmpc': '#ff7f0e',
# }
SONAR_FOV_DEG = 130.0          # 声呐视场角（扇形标注用）
DESIRED_STANDOFF = 3.0         # 期望站位距离
STANDOFF_TOLERANCE = 0.8       # 站位容差
ALPHA_START = 0.25             # 轨迹起始透明度（不为 0，保证可见）
MARK_INTERVAL_S = 20.0         # 轨迹标记间隔
FOV_FAN_RADIUS = 1.4           # FOV 扇形半径（m）
SMOOTH_WINDOW = 20             # 时间曲线平滑窗口（2s @ 10Hz）

REQUIRED_COLUMNS = {
    'estimation_pos_error_m', 'cov_trace_m2', 'gt_standoff_m', 'standoff_error_m',
    'within_standoff_tolerance', 'control_effort_sq', 'target_in_sonar_fov',
}

# ====================================================================
#  每张图片的尺寸与边距配置
#   使用方法：修改对应图片的数值后重新运行脚本即可。
#     figsize        – 图片尺寸（宽, 高）英寸
#     left/right     – 画布左右留白比例（左小右大时可容纳图例）
#     top/bottom     – 画布上下留白比例
# ====================================================================
FIGURE_CONFIG = {
    # 图1-3 轨迹图
    'trajectory': dict(figsize=(6.4, 4.8),
                       left=0.10, right=1, top=1, bottom=0.12),
    # 图4 估计误差
    'fig4_estimation_error': dict(figsize=(6.4, 3.5),
                                  left=0.09, right=0.99, top=0.99, bottom=0.14),
    # 图5 置信度
    'fig5_confidence': dict(figsize=(6.4, 3.5),
                            left=0.11, right=0.99, top=0.99, bottom=0.14),
    # 图6 跟踪时间线（含图例）
    'fig6_tracking_timeline': dict(figsize=(6.4, 3.5),
                                   left=0.09, right=0.99, top=0.99, bottom=0.14),
    # 图7-10 柱状统计图
    'bars': dict(figsize=(5.5, 4.2),
                 left=0.12, right=0.97, top=0.95, bottom=0.12),
}


# ====================================================================
#  数据加载
# ====================================================================

def load_csvs(input_dir):
    """加载目录下所有 CSV，返回 (data, file_info)（按文件名排序）。

    data:      {method: [DataFrame, ...]}
    file_info: {method: [(文件名, seed), ...]}，与 data 列表顺序一致
    """
    data = {}
    file_info = {}
    for path in sorted(glob.glob(os.path.join(input_dir, '*.csv'))):
        name = os.path.basename(path)
        method = next((m for m in METHODS if name.startswith(m)), None)
        if method is None:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print('[WARN] cannot read {}: {}'.format(name, exc))
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            print('[WARN] skip legacy/incomplete {}: missing {}'.format(name, sorted(missing)))
            continue
        if len(frame) < 10:
            print('[WARN] skip {}: fewer than 10 samples'.format(name))
            continue
        seed_match = re.search(r'_seed(\d+)_', name)
        seed = int(seed_match.group(1)) if seed_match else None
        data.setdefault(method, []).append(frame)
        file_info.setdefault(method, []).append((name, seed))
        print('  loaded {} ({})'.format(name, method))
    return data, file_info


def time_axis(frame):
    if 't_sec' in frame and frame['t_sec'].notna().any():
        return frame['t_sec'].to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float) / 10.0


def pick_run(data, method, run_idx):
    """选择某方法的第 run_idx 次运行；越界时退回最后一次并告警。"""
    frames = data.get(method)
    if not frames:
        return None
    if run_idx >= len(frames):
        print('[WARN] {} 只有 {} 次运行，--run {} 越界，使用最后一次'.format(
            METHOD_LABELS[method], len(frames), run_idx))
        return frames[-1]
    return frames[run_idx]


def smooth(series, window=SMOOTH_WINDOW):
    return series.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def alpha_at(idx, n):
    """第 idx 个采样点（共 n 个）对应的轨迹透明度。"""
    return ALPHA_START + (1.0 - ALPHA_START) * (float(idx) / max(n - 1, 1))


def style_axes(ax):
    """统一坐标轴样式：无网格，仅保留左侧与下侧边线。"""
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



# ====================================================================
#  图1-3：轨迹图
# ====================================================================

def trajectory_bounds(frames):
    """计算多帧数据（机器人+目标）的全局 x/y 范围。"""
    xs, ys = [], []
    for frame in frames:
        if frame is None:
            continue
        xs.append(frame['robot_x'].to_numpy(dtype=float))
        xs.append(frame['gt_x'].to_numpy(dtype=float))
        ys.append(frame['robot_y'].to_numpy(dtype=float))
        ys.append(frame['gt_y'].to_numpy(dtype=float))
    if not xs:
        return None
    xs = np.concatenate([v[np.isfinite(v)] for v in xs])
    ys = np.concatenate([v[np.isfinite(v)] for v in ys])
    pad = 0.06 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
    return (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)


def plot_trajectory(frame, method, bounds, output_dir):
    """单方法轨迹图：渐显轨迹 + 每 20s 的 FOV 扇形 / 目标方形标记 + 起终点。"""
    has_yaw = 'robot_psi' in frame.columns
    if not has_yaw:
        print('[WARN] {} 的 CSV 缺少 robot_psi 列 — 不绘制 FOV 扇形（旧数据可用 bag_to_csv.sh 重新生成）'.format(method))

    t = time_axis(frame)
    rx = frame['robot_x'].to_numpy(dtype=float)
    ry = frame['robot_y'].to_numpy(dtype=float)
    psi = frame['robot_psi'].to_numpy(dtype=float) if has_yaw else None
    gx = frame['gt_x'].to_numpy(dtype=float)
    gy = frame['gt_y'].to_numpy(dtype=float)
    n = len(t)

    cfg = FIGURE_CONFIG['trajectory']
    # 按共享坐标区间的横纵比动态设置 figure 高度：
    # 等比例模式下绘图区为正方形，figure 高度随数据比例调整，避免顶部空白
    width = cfg['figsize'][0]
    if bounds is not None:
        x_span = max(bounds[1] - bounds[0], 1e-6)
        y_span = max(bounds[3] - bounds[2], 1e-6)
        height = width * (y_span / x_span)
    else:
        height = cfg['figsize'][1]
    fig, ax = plt.subplots(figsize=(width, height))
    color = METHOD_COLORS[method]

    # --- 渐显轨迹：分段绘制，透明度随时间递增（目标为虚线） ---
    n_chunks = max(2, min(120, n // 10))
    bounds_idx = np.linspace(0, n, n_chunks + 1).astype(int)
    for idx in range(n_chunks):
        sl = slice(bounds_idx[idx], bounds_idx[idx + 1] + 1)
        mid = (bounds_idx[idx] + bounds_idx[idx + 1]) // 2
        alpha = alpha_at(mid, n)
        ax.plot(rx[sl], ry[sl], color=color, linewidth=1.1, alpha=alpha,
                solid_capstyle='round')
        ax.plot(gx[sl], gy[sl], color='#333333', linewidth=0.9, linestyle='--',
                alpha=alpha, dash_capstyle='round')

    # --- 每隔 20s 的标记（颜色/透明度与轨迹该时刻一致） ---
    half_fov = 0.5 * np.radians(SONAR_FOV_DEG)
    for mark_t in np.arange(0.0, t[-1] + 1e-6, MARK_INTERVAL_S):
        idx = int(np.argmin(np.abs(t - mark_t)))
        alpha = alpha_at(idx, n)
        if has_yaw:
            wedge = Wedge((rx[idx], ry[idx]), FOV_FAN_RADIUS,
                          np.degrees(psi[idx] - half_fov), np.degrees(psi[idx] + half_fov),
                          alpha=alpha, facecolor=color, edgecolor='none')
            ax.add_patch(wedge)
        # 目标：正方形标记，透明度随时刻变化
        ax.plot(gx[idx], gy[idx], marker='s', markersize=4.5, color='black',
                alpha=alpha, zorder=5)

    # --- 起点（圆）/ 终点（叉）：机器人与目标均标注 ---
    ax.plot(rx[0], ry[0], 'o', markersize=6, color=color, zorder=6)
    ax.plot(rx[-1], ry[-1], 'x', markersize=7, color=color, mew=2, zorder=6)
    ax.plot(gx[0], gy[0], 'o', markersize=6, color='black', zorder=6, label='Start point')
    ax.plot(gx[-1], gy[-1], 'x', markersize=7, color='black', mew=2, zorder=6,
            label='End point')

    # --- 图例：仅标注黑色目标轨迹的起点/终点，其余为常规图例项 ---
    ax.plot([], [], color=color, linewidth=1.5, label='AUV trajectory')
    ax.plot([], [], color='#333333', linewidth=1.2, linestyle='--', label='Target trajectory')
    if has_yaw:
        ax.add_patch(Wedge((0, 0), FOV_FAN_RADIUS, 0, SONAR_FOV_DEG, alpha=0.6,
                           facecolor=color, edgecolor='none', label='Sonar FOV'))
    ax.plot([], [], marker='s', markersize=4.5, color='black', linestyle='none')

    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    if bounds is not None:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    # 等比例坐标轴：x/y 相同数值差对应相同物理长度
    ax.set_aspect('equal', adjustable='box')
    style_axes(ax)
    ax.legend(loc='best')
    # 尺寸/边距调整：见文件顶部 FIGURE_CONFIG 配置
    fig.subplots_adjust(left=cfg['left'], right=cfg['right'],
                        top=cfg['top'], bottom=cfg['bottom'])

    fname = 'fig{}_trajectory_{}.pdf'.format(METHODS.index(method) + 1, method)
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print('  -> {}'.format(fname))


# ====================================================================
#  图4-6：时间曲线（单指标单图）
# ====================================================================

def plot_timeline(data, run_idx, column, ylabel, fname, output_dir,
                  reference=None, band=None, band_label=None,
                  legend_loc='best', cfg=None, ylim=None):
    if cfg is None:
        cfg = FIGURE_CONFIG.get(fname.rsplit('.', 1)[0], FIGURE_CONFIG['fig4_estimation_error'])
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    for method in METHODS:
        frame = pick_run(data, method, run_idx)
        if frame is None or column not in frame:
            continue
        ax.plot(time_axis(frame), smooth(frame[column]), color=METHOD_COLORS[method],
                linewidth=1.2, label=METHOD_LABELS[method])
    if reference is not None:
        ax.axhline(reference, color='#555555', linestyle='--', linewidth=0.9,
                   label='Desired standoff')
    if band is not None:
        ax.axhspan(band[0], band[1], color='gray', alpha=0.12, label=band_label)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.legend(loc=legend_loc)  # 纵向排布（ncol=1）
    # 尺寸/边距调整：见文件顶部 FIGURE_CONFIG 配置
    fig.subplots_adjust(left=cfg['left'], right=cfg['right'],
                        top=cfg['top'], bottom=cfg['bottom'])
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print('  -> {}'.format(fname))


# ====================================================================
#  图7-10：跨种子柱状统计
# ====================================================================

def per_run_scalar(frame, metric):
    if metric == 'rmse':
        err = frame['estimation_pos_error_m'].to_numpy(dtype=float)
        return float(np.sqrt(np.nanmean(err * err)))
    if metric == 'standoff_success':
        return float(100.0 * np.nanmean(frame['within_standoff_tolerance']))
    if metric == 'control_effort':
        return float(np.nanmean(frame['control_effort_sq']))
    if metric == 'fov':
        return float(100.0 * np.nanmean(frame['target_in_sonar_fov']))
    raise ValueError(metric)


def collect_stats(data, metric):
    """返回 {method: {'values': [...], 'mean': m, 'std': s, 'n': n}}。"""
    stats = {}
    for method in METHODS:
        frames = data.get(method, [])
        values = [per_run_scalar(frame, metric) for frame in frames]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        stats[method] = {
            'values': values,
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'n': len(values),
        }
    return stats


BAR_SPECS = [
    ('rmse', 'Estimation RMSE (m)', 'fig7_rmse_bars.pdf'),
    ('standoff_success', 'Standoff success (%)', 'fig8_standoff_success_bars.pdf'),
    ('control_effort', 'Mean control effort ($u^2+v^2$)', 'fig9_control_effort_bars.pdf'),
    ('fov', 'Target in sonar FOV (%)', 'fig10_fov_bars.pdf'),
]


def plot_bars(stats, ylabel, fname, output_dir):
    cfg = FIGURE_CONFIG['bars']
    fig, ax = plt.subplots(figsize=cfg['figsize'])
    labels, means, stds, colors = [], [], [], []
    for method in METHODS:
        stat = stats.get(method)
        if stat is None:
            continue
        labels.append(METHOD_LABELS[method])
        means.append(stat['mean'])
        stds.append(stat['std'])
        colors.append(METHOD_COLORS[method])
    if not means:
        plt.close(fig)
        print('[WARN] no data for {}'.format(fname))
        return
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                '{:.2f}'.format(mean), ha='center', va='bottom')
    # 尺寸/边距调整：见文件顶部 FIGURE_CONFIG 配置
    fig.subplots_adjust(left=cfg['left'], right=cfg['right'],
                        top=cfg['top'], bottom=cfg['bottom'])
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print('  -> {}'.format(fname))


def print_stat_table(all_stats):
    """终端打印均值 ± 标准差表格，便于论文表格填数。"""
    methods = [m for m in METHODS if any(m in s for s in all_stats.values())]
    print('')
    print('=' * 100)
    print('  汇总统计（均值 ± 标准差，跨所有种子运行）')
    print('=' * 100)
    header = '{:<26}'.format('指标')
    for m in methods:
        n = next(s[m]['n'] for s in all_stats.values() if m in s)
        header += ' {:>22}'.format('{} (N={})'.format(METHOD_LABELS[m], n))
    print(header)
    print('-' * 100)
    for metric, label, _fname in BAR_SPECS:
        stats = all_stats[metric]
        row = '{:<26}'.format(label)
        for m in methods:
            stat = stats.get(m)
            if stat is None:
                row += ' {:>22}'.format('N/A')
            else:
                row += ' {:>22}'.format('{:.3f} ± {:.3f}'.format(stat['mean'], stat['std']))
        print(row)
    print('=' * 100)


# ====================================================================
#  主入口
# ====================================================================

def parse_range(text):
    """解析 "min,max" 字符串为 (min, max)；空/非法时返回 None。"""
    if not text:
        return None
    try:
        lo, hi = (float(v.strip()) for v in text.split(','))
    except (ValueError, TypeError):
        print('[WARN] 范围参数格式错误: {}（应为 "min,max"），忽略'.format(text))
        return None
    if lo >= hi:
        print('[WARN] 范围参数无效: min >= max ({})，忽略'.format(text))
        return None
    return (lo, hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', default=os.path.expanduser('~/sub_planning_metrics'),
                        help='CSV 输入目录（默认 ~/sub_planning_metrics）')
    parser.add_argument('-o', '--output', default=os.path.expanduser('~/sub_planning_metrics/figures'),
                        help='图表输出目录')
    parser.add_argument('--run', type=int, default=1,
                        help='图1-6 使用每个方法按文件名排序后的第 N 次运行（默认 0）')
    parser.add_argument('--traj-xlim', default=None,
                        help='图1-3 手动指定 x 轴范围，格式 "xmin,xmax"（默认自动计算）')
    parser.add_argument('--traj-ylim', default=None,
                        help='图1-3 手动指定 y 轴范围，格式 "ymin,ymax"（默认自动计算）')
    parser.add_argument('--fig4-ylim', default=None,
                        help='图4 手动指定 y 轴范围，格式 "ymin,ymax"（默认自动计算）')
    parser.add_argument('--fig5-ylim', default=None,
                        help='图5 手动指定 y 轴范围，格式 "ymin,ymax"（默认自动计算）')
    parser.add_argument('--fig6-ylim', default=None,
                        help='图6 手动指定 y 轴范围，格式 "ymin,ymax"（默认自动计算）')
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input)
    output_dir = os.path.expanduser(args.output)
    if not os.path.isdir(input_dir):
        print('[ERROR] 输入目录不存在: {}'.format(input_dir), file=sys.stderr)
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    data, file_info = load_csvs(input_dir)
    if not data:
        print('[ERROR] 未找到有效 CSV 文件', file=sys.stderr)
        sys.exit(1)
    print('方法数据: {}'.format({m: len(data.get(m, [])) for m in METHODS}))

    # 打印图1-6 使用的数据文件及其随机种子
    print('')
    print('=' * 100)
    print('  图1-6 数据来源（--run {}，按文件名排序的第 {} 次运行）'.format(args.run, args.run + 1))
    print('=' * 100)
    for method in METHODS:
        info = file_info.get(method)
        if not info:
            print('  {:<22} 无数据'.format(METHOD_LABELS[method]))
            continue
        idx = min(args.run, len(info) - 1)
        name, seed = info[idx]
        seed_text = 'seed {}'.format(seed) if seed is not None else 'seed 未知（旧命名）'
        print('  {:<22} {}  <- {}'.format(METHOD_LABELS[method], seed_text, name))
    print('=' * 100)

    # 图1-3：轨迹（三图共用同一坐标区间；支持手动指定范围）
    frames = {method: pick_run(data, method, args.run) for method in METHODS}
    bounds = trajectory_bounds([frames[m] for m in METHODS])
    if args.traj_xlim is not None or args.traj_ylim is not None:
        xlim = parse_range(args.traj_xlim)
        ylim = parse_range(args.traj_ylim)
        if xlim is not None or ylim is not None:
            if xlim is None:
                xlim = (bounds[0], bounds[1]) if bounds else (0.0, 1.0)
            if ylim is None:
                ylim = (bounds[2], bounds[3]) if bounds else (0.0, 1.0)
            bounds = (xlim[0], xlim[1], ylim[0], ylim[1])
            print('[INFO] 图1-3 使用手动坐标范围: x={} y={}'.format(xlim, ylim))
    for method in METHODS:
        if frames[method] is not None:
            plot_trajectory(frames[method], method, bounds, output_dir)

    # 图4-6：单指标时间曲线（与轨迹同一次运行；支持手动指定 y 轴范围）
    plot_timeline(data, args.run, 'estimation_pos_error_m', 'Estimation error (m)',
                  'fig4_estimation_error.pdf', output_dir, legend_loc='upper left',
                  ylim=parse_range(args.fig4_ylim))
    plot_timeline(data, args.run, 'cov_trace_m2', 'tr($\mathbf{P}^{p}$) (m$^2$)',
                  'fig5_confidence.pdf', output_dir,
                  ylim=parse_range(args.fig5_ylim))
    plot_timeline(data, args.run, 'gt_standoff_m', 'Robot-target distance (m)',
                  'fig6_tracking_timeline.pdf', output_dir,
                  reference=DESIRED_STANDOFF,
                  band=(DESIRED_STANDOFF - STANDOFF_TOLERANCE, DESIRED_STANDOFF + STANDOFF_TOLERANCE),
                  band_label='Tolerance band',
                  ylim=parse_range(args.fig6_ylim))

    # 图7-10：跨种子柱状统计 + 终端打印数据表
    all_stats = {}
    for metric, _label, fname in BAR_SPECS:
        stats = collect_stats(data, metric)
        all_stats[metric] = stats
    print_stat_table(all_stats)
    for metric, label, fname in BAR_SPECS:
        plot_bars(all_stats[metric], label, fname, output_dir)

    print('Figures written to {}'.format(output_dir))


if __name__ == '__main__':
    main()
