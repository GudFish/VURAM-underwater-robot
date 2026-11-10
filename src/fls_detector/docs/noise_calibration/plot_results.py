#!/usr/bin/env python3
"""Plot held-out background statistics; scenes are deliberately not image-paired."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', required=True, type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    package = output.parents[1]
    spec = importlib.util.spec_from_file_location('calibration', package / 'scripts/calibrate_sonar_noise.py')
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    report = json.loads((output / 'statistics.json').read_text())
    samples = np.load(output / 'review_samples.npz')
    synthetic = {}
    for name in ['motion', 'stationary', 'default']:
        data = np.fromfile(args.render_dir / (name + '_background.bin'), np.uint8).reshape(64, 494, 256)
        background = calibration._gray_statistics(data[:, :, 200:232], np.arange(200, 232),
                                                  [(0, 20), (128, 160), (205, 230)])
        excess = data[:, 40:450, 126:131].mean() - np.concatenate(
            [data[:, 40:450, 118:123], data[:, 40:450, 134:139]], axis=2).mean()
        synthetic[name] = dict(background=background, center_excess_gray=float(excess))
    report['actual_cpp_renderer'] = synthetic
    (output / 'statistics.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(10, 6), layout='constrained')
    for ax, title, img in zip(axes,
            ['Real moving sonar (held out)', 'Previous default simulation', 'Fitted background, no stripe'],
            [samples['motion_validation'],
             cv2.imread(str(args.render_dir / 'default_target_raw.pgm'), 0),
             cv2.imread(str(args.render_dir / 'motion_target_raw.pgm'), 0)]):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto', origin='upper')
        ax.set(title=title, xlabel='Native beam column', ylabel='Range row (top = near)')
    fig.suptitle('Same mono8 [0,255] scale; different scenes, no pixel-wise similarity claim', fontsize=10)
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for label, stats in [('Real held out', report['motion_validation']),
                         ('Fitted C++ simulation', synthetic['motion']['background']),
                         ('Previous C++ simulation', synthetic['default']['background'])]:
        axes[0, 0].plot(stats['per_range_mean'], label=label)
        axes[0, 1].plot(stats['per_range_std'], label=label)
    for ax in axes[0]:
        ax.set_xlabel('Native range row (increasing distance)');ax.legend(fontsize=8)
    axes[0, 0].set_ylabel('Background gray mean')
    axes[0, 1].set_ylabel('Background gray standard deviation')
    import yaml
    profile = yaml.safe_load((package / 'config/bhq750d_noise_experiment_10m.yaml').read_text())
    bias = profile['noise_beam_bias_db'] or np.zeros(256)
    axes[1, 0].plot(np.arange(118, 139), bias[118:139])
    axes[1, 0].set(xlabel='Native beam column', ylabel='Empirical bias (relative dB)', title='Center stripe removed from simulation')
    values = [report['motion_validation']['mean'], synthetic['motion']['background']['mean'],
              report['center_validation']['mean_excess_gray'], synthetic['motion']['center_excess_gray']]
    axes[1, 1].bar(['Real bg', 'Sim bg', 'Real stripe', 'Sim stripe'], values,
                   color=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'])
    axes[1, 1].set(ylabel='Gray level / stripe excess', title='Held-out real vs actual C++ renderer')
    fig.savefig(output / 'statistics.png', dpi=160)
    plt.close(fig)
    # Optional cross-range checks use identical native angular sampling and
    # distance spacing to distinguish range mapping from sampling differences.
    if (args.render_dir / 'range30_background.bin').exists():
        step = float((output / 'motion_renderer_input.txt').read_text().splitlines()[0].split()[2])
        reference = np.fromfile(args.render_dir / 'range30_background.bin', np.uint8).reshape(64, -1, 256)
        checks = []
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        image_figure, image_axes = plt.subplots(1, 4, figsize=(12, 5), layout='constrained')
        for maximum, image_ax in zip([5, 10, 15, 30], image_axes):
            data = np.fromfile(args.render_dir / f'range{maximum}_background.bin', np.uint8).reshape(64, -1, 256)
            count = int(np.floor(maximum / step + 1e-9)) + 1
            shared = data[:, :count]
            error = np.abs(shared.astype(float) - reference[:, :count].astype(float))
            checks.append(dict(max_range_m=maximum, rows=data.shape[1], beams=data.shape[2],
                               common_distance_max_gray_error=float(error.max())))
            if error.max() != 0:
                raise AssertionError('same-distance noise changed with selected range')
            distance = np.arange(count) * step
            axes[0].plot(distance, shared.mean(axis=(0, 2)), label=f'{maximum} m selected')
            axes[1].plot(distance, shared.std(axis=(0, 2)), label=f'{maximum} m selected')
            image_ax.imshow(shared[0], cmap='gray', vmin=0, vmax=255, aspect='auto', origin='upper',
                            extent=[0, 255, (count-1)*step, 0])
            image_ax.set(title=f'{maximum} m selected', xlabel='Native beam column', ylabel='Distance (m; top = near)')
        for ax, label in zip(axes, ['Background gray mean', 'Background gray standard deviation']):
            ax.axvline(10, color='black', linestyle='--', alpha=.6, label='Measured profile ends at 10 m')
            ax.set(xlabel='Physical distance (m)', ylabel=label);ax.legend(fontsize=8)
        fig.suptitle('Same-distance noise is independent of the selected maximum range')
        fig.savefig(output / 'range_consistency.png', dpi=160)
        image_figure.suptitle('Measured background only, no center stripe; endpoint statistics held beyond 10 m', fontsize=10)
        image_figure.savefig(output / 'range_backgrounds.png', dpi=160)
        plt.close(fig);plt.close(image_figure)
        report['range_consistency'] = dict(profile_range_m=10., extrapolation='hold endpoint statistics',
                                           seed=7, native_range_spacing_m=step, checks=checks)
        (output / 'statistics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
