#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实物实验 metrics 分析与论文图（参考 analyze_metrics.py 的图1-6）。

输入：~/ardupilot_ws/results/sub_planning_metrics_exp/eval/ 下三种方法的 CSV
      （每种方法一次实验，CSV 与 bag 同名，如
       proposed_real_ts_nvidia_desktop_20260914_104634.csv）。

图1-3  各方法的 XY 轨迹图（proposed / bearing-only / CE-NMPC）：
       - 轨迹透明度随时间渐显（起始 0.25 → 结束 1.0），目标轨迹为虚线
       - 每隔 20s：机器人处绘制朝向 yaw 的声呐 FOV 扇形，目标处绘制方形标记
       - 起点圆 / 终点叉；三张图共用同一坐标区间
图4    估计误差随时间变化（3 方法叠加）
图5    置信度（cov_trace）随时间变化（3 方法叠加）
图6    机器人-目标距离随时间变化（3 方法叠加 + 期望站位/容差带）

终端输出：每种方法一次实验的数值统计：
       估计误差 RMSE、control effort、平均协方差、平均机器人与目标距离
       按时间采样求均值 ± 标准差；目标在 FOV 内占比为单值
       （FOV 内时刻数 / 总时刻数）。

用法:
  python3 analyze_real_experiment.py                      # 默认 eval/ 目录
  python3 analyze_real_experiment.py -i <csv目录> -o <输出目录>
  # 手动指定坐标范围（负数需用等号形式传参）：
  python3 analyze_real_experiment.py --traj-xlim=-30,5 --traj-ylim=-10,20 \
                             --fig4-ylim=0,15 --fig5-ylim=0,10 --fig6-ylim=0,12
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


# 全局字体设置：与 analyze_metrics.py 一致（Times New Roman 15pt，stix 数学字体）
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
SONAR_FOV_DEG = 130.0          # 声呐视场角（扇形标注用）
DESIRED_STANDOFF = 3.0         # 期望站位距离
STANDOFF_TOLERANCE = 0.8       # 站位容差
ALPHA_START = 0.25             # 轨迹起始透明度
MARK_INTERVAL_S = 20.0         # 轨迹标记间隔
FOV_FAN_RADIUS = 1.4           # FOV 扇形半径（m）
SMOOTH_WINDOW = 20             # 时间曲线平滑窗口（2s @ 10Hz）

REQUIRED_COLUMNS = {
    'estimation_pos_error_m', 'cov_trace_m2', 'gt_standoff_m', 'standoff_error_m',
    'within_standoff_tolerance', 'control_effort_sq', 'target_in_sonar_fov',
}

# ====================================================================
#  每张图片的尺寸与边距配置（与 analyze_metrics.py 一致）
# ====================================================================
FIGURE_CONFIG = {
    'trajectory': dict(figsize=(6.4, 4.8),
                       left=0.10, right=1, top=1, bottom=0.12),
    'fig4_estimation_error': dict(figsize=(6.4, 3.5),
                                  left=0.09, right=0.99, top=0.99, bottom=0.14),
    'fig5_confidence': dict(figsize=(6.4, 3.5),
                            left=0.11, right=0.99, top=0.99, bottom=0.14),
    'fig6_tracking_timeline': dict(figsize=(6.4, 3.5),
                                   left=0.09, right=0.99, top=0.99, bottom=0.14),
}


# ====================================================================
#  数据加载
# ====================================================================

def load_csvs(input_dir):
    """加载目录下所有 CSV，返回 (data, file_info)（按文件名排序）。

    data:      {method: [DataFrame, ...]}
    file_info: {method: [(文件名, ...), ...]}，与 data 列表顺序一致
    方法名按文件名前缀匹配（proposed_real_... / b1_bearing_only_real_... 等）。
    """
    data = {}
    file_info = {}
    for path in sorted(glob.glob(os.path.join(input_dir, '*.csv'))):
        name = os.path.basename(path)
        method = next((m for m in METHODS if name.startswith(m)), None)
        if method is None:
            print('[WARN] skip {}: 无法识别方法名'.format(name))
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print('[WARN] cannot read {}: {}'.format(name, exc))
            continue
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            print('[WARN] skip incomplete {}: missing {}'.format(name, sorted(missing)))
            continue
        if len(frame) < 10:
            print('[WARN] skip {}: fewer than 10 samples'.format(name))
            continue
        data.setdefault(method, []).append(frame)
        file_info.setdefault(method, []).append((name,))
        print('  loaded {} ({})'.format(name, method))
    return data, file_info


def time_axis(frame):
    """时间轴（相对秒）。

    兼容两类数据：正常情况 t_sec 为 ARM 后的相对秒；个别真实实验因
    /use_sim_time 状态异常导致 t_sec 为 Unix 墙钟秒（> 1e6），
    此时重基准为相对秒（减去首帧时间）。
    """
    if 't_sec' in frame and frame['t_sec'].notna().any():
        t = frame['t_sec'].to_numpy(dtype=float)
        finite = t[np.isfinite(t)]
        if len(finite) and finite.min() > 1e6:
            t = t - finite.min()
        return t
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
#  图1-3：轨迹图（与 analyze_metrics.py 相同画法）
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
        print('[WARN] {} 的 CSV 缺少 robot_psi 列 — 不绘制 FOV 扇形'.format(method))

    t = time_axis(frame)
    rx = frame['robot_x'].to_numpy(dtype=float)
    ry = frame['robot_y'].to_numpy(dtype=float)
    psi = frame['robot_psi'].to_numpy(dtype=float) if has_yaw else None
    gx = frame['gt_x'].to_numpy(dtype=float)
    gy = frame['gt_y'].to_numpy(dtype=float)
    n = len(t)

    cfg = FIGURE_CONFIG['trajectory']
    # 按共享坐标区间的横纵比动态设置 figure 高度（等比例绘图区）
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
    # 实物数据开头存在 NaN（GPS 未固定/EKF 未初始化），NaN 圆心会使
    # matplotlib 的 Wedge 陷入死循环，绘制前必须跳过非有限值。
    # 标记时间点带防御性上限，避免异常时间轴产生海量标记
    half_fov = 0.5 * np.radians(SONAR_FOV_DEG)
    t_finite = t[np.isfinite(t)]
    t_end = float(np.max(t_finite)) if len(t_finite) else 0.0
    mark_ts = np.arange(0.0, t_end + 1e-6, MARK_INTERVAL_S)
    if len(mark_ts) > 200:
        mark_ts = mark_ts[:200]
    for mark_t in mark_ts:
        idx = int(np.argmin(np.abs(t - mark_t)))
        alpha = alpha_at(idx, n)
        if np.isfinite(rx[idx]) and np.isfinite(ry[idx]):
            if has_yaw and np.isfinite(psi[idx]):
                wedge = Wedge((rx[idx], ry[idx]), FOV_FAN_RADIUS,
                              np.degrees(psi[idx] - half_fov), np.degrees(psi[idx] + half_fov),
                              alpha=alpha, facecolor=color, edgecolor='none')
                ax.add_patch(wedge)
        if np.isfinite(gx[idx]) and np.isfinite(gy[idx]):
            ax.plot(gx[idx], gy[idx], marker='s', markersize=4.5, color='black',
                    alpha=alpha, zorder=5)

    # --- 起点（圆）/ 终点（叉）：机器人与目标均标注（跳过非有限值） ---
    if np.isfinite(rx[0]) and np.isfinite(ry[0]):
        ax.plot(rx[0], ry[0], 'o', markersize=6, color=color, zorder=6)
    if np.isfinite(rx[-1]) and np.isfinite(ry[-1]):
        ax.plot(rx[-1], ry[-1], 'x', markersize=7, color=color, mew=2, zorder=6)
    if np.isfinite(gx[0]) and np.isfinite(gy[0]):
        ax.plot(gx[0], gy[0], 'o', markersize=6, color='black', zorder=6, label='Start point')
    if np.isfinite(gx[-1]) and np.isfinite(gy[-1]):
        ax.plot(gx[-1], gy[-1], 'x', markersize=7, color='black', mew=2, zorder=6,
                label='End point')

    # --- 图例 ---
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
    ax.set_aspect('equal', adjustable='box')
    style_axes(ax)
    ax.legend(loc='best')
    fig.subplots_adjust(left=cfg['left'], right=cfg['right'],
                        top=cfg['top'], bottom=cfg['bottom'])

    fname = 'fig{}_trajectory_{}.pdf'.format(METHODS.index(method) + 1, method)
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print('  -> {}'.format(fname))


# ====================================================================
#  图4-6：时间曲线（与 analyze_metrics.py 相同画法）
# ====================================================================

def plot_timeline(data, run_idx, column, ylabel, fname, output_dir,
                  reference=None, band=None, band_label=None,
                  legend_loc='best', ylim=None):
    cfg = FIGURE_CONFIG[fname.rsplit('.', 1)[0]]
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
    ax.legend(loc=legend_loc)
    fig.subplots_adjust(left=cfg['left'], right=cfg['right'],
                        top=cfg['top'], bottom=cfg['bottom'])
    fig.savefig(os.path.join(output_dir, fname))
    plt.close(fig)
    print('  -> {}'.format(fname))


# ====================================================================
#  数值统计：单次实验内对时间采样做统计（均值 ± 标准差）
# ====================================================================

def run_stats(frame):
    """一次实验的时间序列统计。

    返回 dict: est_rmse/est_std/control_mean/control_std/fov_mean/fov_std/
              cov_mean/cov_std/dist_mean/dist_std
    """
    err = frame['estimation_pos_error_m'].to_numpy(dtype=float)
    ctrl = frame['control_effort_sq'].to_numpy(dtype=float)
    fov = frame['target_in_sonar_fov'].to_numpy(dtype=float)
    cov = frame['cov_trace_m2'].to_numpy(dtype=float)
    dist = frame['gt_standoff_m'].to_numpy(dtype=float)
    return {
        'est_rmse': float(np.sqrt(np.nanmean(err * err))),
        'est_std': float(np.nanstd(err)),
        'control_mean': float(np.nanmean(ctrl)),
        'control_std': float(np.nanstd(ctrl)),
        'fov_mean': float(100.0 * np.nanmean(fov)),
        'fov_std': float(100.0 * np.nanstd(fov)),
        'cov_mean': float(np.nanmean(cov)),
        'cov_std': float(np.nanstd(cov)),
        'dist_mean': float(np.nanmean(dist)),
        'dist_std': float(np.nanstd(dist)),
    }


# (key, 指标名, 均值字段, 标准差字段, 是否显示 ±)：
# FOV 是 0/1 指示量的时间占比，一次实验只有一个百分比，无 ± 波动项；
# 其余指标对时间采样求均值 ± 标准差
STAT_ROWS = [
    ('est', 'Estimation RMSE (m)', 'est_rmse', 'est_std', True),
    ('control', 'Control effort (mean $u^2+v^2$)', 'control_mean', 'control_std', True),
    ('fov', 'Target in sonar FOV (%)', 'fov_mean', None, False),
    ('cov', 'Mean covariance tr($\\mathbf{P}^{p}$) (m$^2$)', 'cov_mean', 'cov_std', True),
    ('dist', 'Mean robot-target distance (m)', 'dist_mean', 'dist_std', True),
]


def print_stat_table(data):
    """终端打印三种方法的数值统计（单次实验，均值 ± 标准差）。"""
    stats = {}
    for method in METHODS:
        frames = data.get(method, [])
        if frames:
            stats[method] = run_stats(frames[0])
    print('')
    print('=' * 110)
    print('  实物实验数值统计（每种方法 1 次实验；对单次实验的时间采样做均值 ± 标准差）')
    print('=' * 110)
    header = '{:<42}'.format('指标')
    for m in METHODS:
        header += ' {:>21}'.format(METHOD_LABELS[m])
    print(header)
    print('-' * 110)
    for _key, label, mean_key, std_key, with_std in STAT_ROWS:
        row = '{:<42}'.format(label)
        for m in METHODS:
            stat = stats.get(m)
            if stat is None:
                row += ' {:>21}'.format('N/A')
            elif with_std:
                row += ' {:>21}'.format('{:.3f} ± {:.3f}'.format(stat[mean_key], stat[std_key]))
            else:
                row += ' {:>21}'.format('{:.3f}'.format(stat[mean_key]))
        print(row)
    print('=' * 110)


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
    parser.add_argument('-i', '--input',
                        default=os.path.expanduser('~/ardupilot_ws/results/sub_planning_metrics_exp/eval'),
                        help='CSV 输入目录（默认 ~/ardupilot_ws/results/sub_planning_metrics_exp/eval）')
    parser.add_argument('-o', '--output',
                        default=os.path.expanduser('~/ardupilot_ws/results/sub_planning_metrics_exp/eval/figures'),
                        help='图表输出目录')
    parser.add_argument('--run', type=int, default=0,
                        help='图1-6 使用每个方法按文件名排序后的第 N 次运行（默认 0）')
    parser.add_argument('--traj-xlim', default=None, help='图1-3 x 轴范围 "xmin,xmax"')
    parser.add_argument('--traj-ylim', default=None, help='图1-3 y 轴范围 "ymin,ymax"')
    parser.add_argument('--fig4-ylim', default=None, help='图4 y 轴范围 "ymin,ymax"')
    parser.add_argument('--fig5-ylim', default=None, help='图5 y 轴范围 "ymin,ymax"')
    parser.add_argument('--fig6-ylim', default=None, help='图6 y 轴范围 "ymin,ymax"')
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

    # 图4-6：单指标时间曲线（支持手动指定 y 轴范围）
    plot_timeline(data, args.run, 'estimation_pos_error_m', 'Estimation error (m)',
                  'fig4_estimation_error.pdf', output_dir, legend_loc='upper left',
                  ylim=parse_range(args.fig4_ylim))
    plot_timeline(data, args.run, 'cov_trace_m2', 'tr($\\mathbf{P}^{p}$) (m$^2$)',
                  'fig5_confidence.pdf', output_dir,
                  ylim=parse_range(args.fig5_ylim))
    plot_timeline(data, args.run, 'gt_standoff_m', 'Robot-target distance (m)',
                  'fig6_tracking_timeline.pdf', output_dir,
                  reference=DESIRED_STANDOFF,
                  band=(DESIRED_STANDOFF - STANDOFF_TOLERANCE, DESIRED_STANDOFF + STANDOFF_TOLERANCE),
                  band_label='Tolerance band',
                  ylim=parse_range(args.fig6_ylim))

    # 终端打印数值统计
    print_stat_table(data)

    print('Figures written to {}'.format(output_dir))


if __name__ == '__main__':
    main()
