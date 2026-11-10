#!/usr/bin/env python3
"""Reproduce the manually reviewed 2026 BHQ background calibration.

This is a dataset-specific analysis recipe, not automatic background detection.
Images keep native orientation: row zero is nearest to the sonar. Historical
display settings and physical range sampling are assumptions, not SDK metadata.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rosbag
import yaml

import calibrate_sonar_noise as calibration


TOPIC = '/forward_sonar/image_raw'
BAG = 'bagfiles/experiment/xiawu/proposed_real_ts_nvidia_desktop_20260914_144517.bag'
EXCLUDED = [(0, 20), (128, 160), (205, 230)]
BEAMS = np.arange(200, 232)


def sample_windows(path):
    """Disjoint fit/validation times; uniform sampling within each window."""
    with rosbag.Bag(str(path)) as bag:
        times = np.array([t.to_sec() for _, _, t in bag.read_messages(topics=[TOPIC], raw=True)])
        if not len(times):
            raise ValueError('bag contains no native sonar images')
        selections = {}
        # Anchors were visually reviewed at ~50% and ~80% of bag duration.
        for name, start, duration in [('fit', 56.35, 6.), ('validation', 98.23, 6.)]:
            candidates = np.flatnonzero((times >= times[0] + start) & (times < times[0] + start + duration))
            if len(candidates) < 2:
                raise ValueError('reviewed time window has insufficient images')
            selected = candidates[calibration.uniform_frame_indices(times[candidates], 60)]
            selections[name] = dict(indices=selected.tolist(), times_sec=times[selected].tolist(),
                                    start_offset_sec=start, duration_sec=duration)
        wanted = {index for group in selections.values() for index in group['indices']}
        images = {}
        for index, (_, msg, _) in enumerate(bag.read_messages(topics=[TOPIC])):
            if index in wanted:
                if msg.encoding != 'mono8' or (msg.height, msg.width) != (494, 256):
                    raise ValueError('reviewed geometry changed')
                images[index] = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width].copy()
        arrays = {name: np.stack([images[i] for i in group['indices']]) for name, group in selections.items()}
        return arrays, selections


def gray_report(frames):
    return calibration._gray_statistics(frames[:, :, BEAMS], BEAMS, EXCLUDED)


def center_report(frames):
    values = frames[:, 40:450].astype(float)
    excess = values[:, :, 126:131].mean(axis=2) - np.concatenate(
        [values[:, :, 118:123], values[:, :, 134:139]], axis=2).mean(axis=2)
    return dict(mean_excess_gray=float(excess.mean()),
                per_frame_excess_gray=excess.mean(axis=1).tolist(),
                rows=[40, 450], center_columns=[126, 131], flank_columns=[[118, 123], [134, 139]])


def stationary_statistics(root):
    """Keep native groups separate; there is no known meters-per-row mapping."""
    summaries = {}
    frame3_fit = None
    for group, count, roi in [('frame1', 37, (110, 35, 235, 70)),
                              ('frame2', 64, (100, 35, 235, 120)),
                              ('frame3', 894, (100, 35, 190, 110))]:
        paths = [root / group / ('sonar_%d.jpg' % i) for i in range(count)]
        images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
        if any(image is None for image in images):
            raise ValueError('stationary source sequence is incomplete: ' + group)
        images = np.stack(images)
        x0, y0, x1, y1 = roi
        cut = images[:, y0:y1, x0:x1]
        residual = cut.astype(float) - cut.mean(axis=0, keepdims=True)
        c = images[:, y0:y1, 126:130].mean()
        flank = np.concatenate([images[:, y0:y1, 118:124], images[:, y0:y1, 132:138]], axis=2).mean()
        summaries[group] = dict(frames=count, shape=list(images.shape[1:]), roi_xyxy=list(roi),
            gray_mean=float(cut.mean()), gray_std=float(cut.std()),
            temporal_pixel_sd_median=float(np.median(cut.std(axis=0))),
            temporal_residual_adjacent_correlation=calibration._adjacent_correlation(residual),
            center_to_flank_ratio=float(c / flank),
            source_format='grayscale JPEG; processed display values, not raw acoustic samples')
        if group == 'frame3':
            # Separate time blocks, not alternating neighboring frames.
            frame3_fit = (cut[:600:5], cut[600::5], paths[:600:5])
    return summaries, frame3_fit


def dump_profile(path, values, comment):
    path.write_text('# ' + comment + '\n' + yaml.safe_dump(values, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--stationary-dir', type=Path, default=Path.home() / 'AUV_sonar_images')
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    package = args.workspace / 'src/fls_detector'
    report_dir = args.report_dir or package / 'docs/noise_calibration'
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir = package / 'config'
    arrays, selection = sample_windows(args.workspace / BAG)
    fit, validation = arrays['fit'], arrays['validation']
    profile = calibration.estimate(fit[:, :, BEAMS], beam_indices=BEAMS, exclude_range_rows=EXCLUDED)
    # Keep a stationary reference separate from the moving bag's range profile.
    stationary, static_data = stationary_statistics(args.stationary_dir)
    static_fit, static_validation, static_paths = static_data
    static_stats = calibration.estimate(static_fit)
    fixed_display = dict(gamma=150, display_min_db=0., display_max_db=75., gain_db=0., gain_assist=False)
    source = dict(bag=BAG, topic=TOPIC, frames=len(fit), source_shape=[494, 256],
                  origin='top center; row zero is nearest; no vertical flip',
                  beam_rois=[[200, 232]], excluded_range_rows=[list(x) for x in EXCLUDED],
                  excluded_row_fill='linear interpolation; nearest endpoint outside valid rows',
                  sampling=selection['fit'], validation_sampling=selection['validation'],
                  trim_percentile=5., assumed_range_m=10., gamma_byte=150, display_db=[0., 75.],
                  calibrated=False, sdk_metadata_present=False,
                  interpretation='relative display-domain background including unresolved diffuse clutter; not absolute receiver power',
                  range_mapping='profile samples span 0 to 10 m; independent of selected sonar range',
                  range_extrapolation='hold last measured mean/std beyond 10 m; not measured calibration',
                  directional_interference='excluded; no center-stripe template')
    profile.update(noise_floor_power=1., noise_profile_range_m=10., noise_beam_bias_db=[], **fixed_display)
    profile['noise_profile_source'] = source
    dump_profile(config_dir / 'bhq750d_noise_experiment_10m.yaml', profile,
                 'Measured background only, no center stripe. Samples span 0..10 m; hold endpoint beyond 10 m.')
    static_profile = dict(noise_floor_power=1., noise_rayleigh_db=0.,
                          noise_mean_db=[float(np.mean(static_stats['noise_mean_db']))],
                          noise_std_db=[float(np.mean(static_stats['noise_std_db']))],
                          noise_correlation=static_stats['noise_correlation'], noise_beam_bias_db=[], **fixed_display)
    static_profile['noise_profile_source'] = dict(directory=str(args.stationary_dir / 'frame3'),
        fit_files=[p.name for p in static_paths], validation_file_indices=list(range(600, 894, 5)),
        roi_xyxy=[100, 35, 190, 110], source_shape=[625, 256],
        origin='top center; row zero is nearest; no vertical flip', calibrated=False,
        sdk_metadata_present=False, trim_percentile=5., range_mapping_known=False,
        interpretation='near-water JPEG background reference, constant with range; not a full-range physical calibration')
    dump_profile(config_dir / 'bhq750d_noise_stationary_reference.yaml', static_profile,
                 'Stationary frame3 near-water reference; JPEG, unknown range/gain metadata; no center stripe.')
    report = dict(motion_source=source, motion_fit=gray_report(fit), motion_validation=gray_report(validation),
                  center_fit=center_report(fit), center_validation=center_report(validation),
                  stationary_groups=stationary,
                  stationary_fit_gray=calibration._gray_statistics(static_fit, np.arange(static_fit.shape[2])),
                  stationary_validation_gray=calibration._gray_statistics(static_validation, np.arange(static_validation.shape[2])),
                  assumptions=['Top is near in every native image; fan/UI/processed derivatives excluded.',
                               'Frame1/255 duplicate, frame_test duplicates, and anomalous-number files are excluded.',
                               '5/95% Winsorized SD is conservative; Gaussian display noise cannot reproduce all positive skew or temporal texture.',
                               'Stationary 468/625 rows and moving 494 rows are never treated as the same range sampling.'])
    (report_dir / 'statistics.json').write_text(json.dumps(report, indent=2) + '\n')
    # Inputs to the actual C++ renderer verification (native shape, fixed seed).
    for name, values in [('motion', profile), ('stationary', static_profile),
                         ('default', dict(noise_floor_power=1., noise_rayleigh_db=1.5, noise_correlation=.35,
                                          noise_mean_db=[0.], noise_std_db=[0.], noise_beam_bias_db=[]))]:
        lines = ['494 256 %.17g 10' % (10. / 493.),
                 '%.17g %.17g %.17g %.17g' % (values['noise_floor_power'], values['noise_rayleigh_db'],
                                             values['noise_correlation'], values.get('noise_profile_range_m', 0.))]
        for key, n in [('noise_mean_db', 494), ('noise_std_db', 494), ('noise_beam_bias_db', 256)]:
            v = values[key] or [0.]
            lines.append(' '.join('%.17g' % x for x in np.interp(np.linspace(0, len(v)-1, n), np.arange(len(v)), v)))
        (report_dir / (name + '_renderer_input.txt')).write_text('\n'.join(lines) + '\n')
    np.savez_compressed(report_dir / 'review_samples.npz', motion_fit=fit[30], motion_validation=validation[30],
                        stationary=cv2.imread(str(args.stationary_dir / 'frame3/sonar_300.jpg'), cv2.IMREAD_GRAYSCALE))
    print(json.dumps(dict(motion_fit_gray=report['motion_fit']['mean'], motion_validation_gray=report['motion_validation']['mean'],
                         center_stripe_enabled=False, profile_range_m=profile['noise_profile_range_m'],
                         noise_correlation=profile['noise_correlation'],
                         stationary_mean_db=static_profile['noise_mean_db'], stationary_std_db=static_profile['noise_std_db']), indent=2))


if __name__ == '__main__':
    main()
