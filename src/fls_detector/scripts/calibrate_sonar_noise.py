#!/usr/bin/env python3
"""Estimate a relative dB noise profile from an explicitly selected empty-water ROI.

The SDK mono8 data already includes gain/gamma. Its inverse display transfer is
an approximation, never an absolute acoustic power calibration. Use fixed device
gain, an empty scene, and the actual display limits for meaningful comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _adjacent_correlation(values, beam_indices=None):
    """Correlate actual adjacent beams, excluding joins between disjoint ROIs."""
    indices = np.arange(values.shape[2]) if beam_indices is None else np.asarray(beam_indices)
    if indices.ndim != 1 or len(indices) != values.shape[2] or np.any(np.diff(indices) <= 0):
        raise ValueError('beam indices must be strictly increasing and match selected beams')
    adjacent = np.diff(indices) == 1
    if not np.any(adjacent):
        return 0.0
    left, right = values[:, :, :-1][:, :, adjacent].ravel(), values[:, :, 1:][:, :, adjacent].ravel()
    return 0.0 if np.std(left) < 1e-9 or np.std(right) < 1e-9 else float(np.corrcoef(left, right)[0, 1])


def _valid_range_rows(count, exclude_range_rows=None):
    """Apply manually identified exclusions; this does not identify background."""
    valid = np.ones(count, dtype=bool)
    for begin, end in exclude_range_rows or []:
        if not 0 <= begin < end <= count:
            raise ValueError('excluded range rows must lie inside the native image')
        valid[begin:end] = False
    if not np.any(valid):
        raise ValueError('at least one unmasked range row is required')
    return valid


def estimate(frames, gamma=150.0 / 255.0, db_min=0.0, db_max=75.0,
             trim_percentile=5.0, beam_indices=None, exclude_range_rows=None):
    """Fit manually selected background; frames shape: time x range x beams.

    Excluded rows never contribute to statistics. Their output profile values
    are interpolated from remaining rows, extending the nearest endpoint.
    """
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError('need at least two frames, range rows, and background beams')
    if not np.isfinite(gamma + db_min + db_max) or gamma <= 0 or db_max <= db_min:
        raise ValueError('invalid display transfer')
    if not np.isfinite(trim_percentile) or not 0 <= trim_percentile < 50:
        raise ValueError('trim percentile must be in [0, 50)')
    valid = _valid_range_rows(values.shape[1], exclude_range_rows)
    db = np.power(np.clip(values[:, valid, :] / 255.0, 0, 1), 1.0 / gamma) * (db_max - db_min) + db_min
    # Robust trimming reduces isolated contamination; it cannot identify targets.
    lo, hi = np.percentile(db, [trim_percentile, 100 - trim_percentile], axis=(0, 2))
    trimmed = np.clip(db, lo[None, :, None], hi[None, :, None])
    mean = trimmed.mean(axis=(0, 2))
    std = trimmed.std(axis=(0, 2))
    residual = trimmed - mean[None, :, None]
    correlation = _adjacent_correlation(residual, beam_indices)
    rows = np.arange(values.shape[1])
    mean = np.interp(rows, rows[valid], mean)
    std = np.interp(rows, rows[valid], std)
    return dict(noise_mean_db=mean.tolist(), noise_std_db=std.tolist(),
                noise_correlation=float(np.clip(correlation, 0, .95)), noise_rayleigh_db=0.0)


def uniform_frame_indices(timestamps, count):
    """Select distinct frames nearest evenly spaced acquisition times."""
    stamps = np.asarray(timestamps, dtype=np.float64)
    if stamps.ndim != 1 or not len(stamps) or count < 1 or np.any(np.diff(stamps) < 0):
        raise ValueError('need ordered timestamps and a positive frame count')
    count = min(count, len(stamps))
    selected = []
    for j, target in enumerate(np.linspace(stamps[0], stamps[-1], count)):
        lower = selected[-1] + 1 if selected else 0
        upper = len(stamps) - (count - j)
        at = int(np.searchsorted(stamps, target))
        candidates = {int(np.clip(at, lower, upper)), int(np.clip(at - 1, lower, upper))}
        selected.append(min(candidates, key=lambda i: (abs(stamps[i] - target), i)))
    return selected


def _gray_statistics(frames, beam_indices, exclude_range_rows=None):
    original = np.asarray(frames, dtype=np.float64)
    valid = _valid_range_rows(original.shape[1], exclude_range_rows)
    values = original[:, valid, :]
    mean = values.mean(axis=(0, 2))
    def native_rows(values):
        # Null denotes an excluded measurement, rather than an observed zero.
        result = [None] * original.shape[1]
        for row, value in zip(np.flatnonzero(valid), values):
            result[row] = float(value)
        return result
    return dict(mean=float(values.mean()), std=float(values.std()),
                percentiles=dict(zip(['p10', 'p50', 'p90', 'p95', 'p99'],
                                     np.percentile(values, [10, 50, 90, 95, 99]).tolist())),
                zero_fraction=float(np.mean(values == 0)), saturated_fraction=float(np.mean(values == 255)),
                per_range_mean=native_rows(mean), per_range_std=native_rows(values.std(axis=(0, 2))),
                per_range_percentiles=[native_rows(p) for p in np.percentile(values, [10, 50, 90], axis=(0, 2))],
                valid_range_rows=np.flatnonzero(valid).tolist(),
                spatial_correlation=_adjacent_correlation(values - mean[None, :, None], beam_indices),
                temporal_residual_spatial_correlation=_adjacent_correlation(
                    values - values.mean(axis=0, keepdims=True), beam_indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag')
    parser.add_argument('output')
    parser.add_argument('--additional-bag', action='append', default=[], help='another bag with matching acquisition geometry; repeatable')
    parser.add_argument('--topic', default='/forward_sonar/image_raw')
    parser.add_argument('--beam-roi', required=True, action='append', help='empty-water beam slice START:END; repeat for disjoint ROIs; end exclusive')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--sampling', choices=['first', 'uniform'], default='first', help='frame selection per bag; uniform spans sonar message times')
    parser.add_argument('--start-offset', type=float, default=0., help='window start seconds after the first sonar message in each bag')
    parser.add_argument('--duration', type=float, help='window length in seconds; end exclusive; default continues to bag end')
    parser.add_argument('--exclude-range-rows', action='append', default=[], help='manually excluded native range rows START:END; repeatable; output profile interpolates these rows')
    parser.add_argument('--trim-percentile', type=float, default=5., help='Winsorize each tail by this percentile; 0 disables clipping')
    parser.add_argument('--report', help='write JSON statistics and sampling provenance')
    parser.add_argument('--gamma-byte', type=int, default=150)
    parser.add_argument('--db-min', type=float, default=0.)
    parser.add_argument('--db-max', type=float, default=75.)
    parser.add_argument('--profile-range-m', type=float,
                        help='physical distance of last native range row; default uses SDK metadata if available')
    parser.add_argument('--metadata-output', help='save the latest recorded SDK/simulator metadata JSON')
    parser.add_argument('--device-config-output', help='export SDK sampling/bearing geometry as a simulator YAML overlay')
    args = parser.parse_args()
    import rosbag
    try:
        rois = sorted(tuple(map(int, roi.split(':'))) for roi in args.beam_roi)
        excluded_rows = sorted(tuple(map(int, rows.split(':'))) for rows in args.exclude_range_rows)
        if any(len(roi) != 2 for roi in rois + excluded_rows):
            raise ValueError()
    except ValueError:
        parser.error('beam ROI and excluded range rows must have the form START:END')
    if (any(begin < 0 or end <= begin for begin, end in rois) or
            any(rois[i][1] > rois[i + 1][0] for i in range(len(rois) - 1)) or
            sum(end - begin for begin, end in rois) < 2 or args.frames < 2 or
            not 0 <= args.gamma_byte <= 255 or not np.isfinite(args.trim_percentile) or
            not 0 <= args.trim_percentile < 50):
        parser.error('invalid ROI, frame count or gamma byte')
    if (not np.isfinite(args.start_offset) or args.start_offset < 0 or
            (args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0))):
        parser.error('start offset must be nonnegative and duration must be positive')
    if args.profile_range_m is not None and (not np.isfinite(args.profile_range_m) or args.profile_range_m <= 0):
        parser.error('profile range must be finite and positive')
    beam_indices = np.concatenate([np.arange(begin, end) for begin, end in rois])
    frames, shape, metadata, bag_sources = [], None, None, []
    for bag_name in [args.bag] + args.additional_bag:
        local_frames, selected_times, local_metadata, local_indices = [], [], None, []
        first_sonar_time, window_end = None, None
        with rosbag.Bag(bag_name, 'r') as bag:
            topics = bag.get_type_and_topic_info().topics
            if args.topic not in topics:
                parser.error(f'{bag_name}: no {args.topic} topic')
            has_metadata = '/forward_sonar/metadata' in topics
            selected = None
            if args.sampling == 'uniform' or args.start_offset != 0 or args.duration is not None:
                timestamps = [t.to_sec() for _, _, t in bag.read_messages(topics=[args.topic], raw=True)]
                if not timestamps:
                    parser.error(f'{bag_name}: no sonar frames')
                first_sonar_time = timestamps[0]
                window_start = first_sonar_time + args.start_offset
                window_end = None if args.duration is None else window_start + args.duration
                eligible = [i for i, t in enumerate(timestamps)
                            if t >= window_start and (window_end is None or t < window_end)]
                if not eligible:
                    parser.error(f'{bag_name}: no sonar frames inside the selected time window')
                if args.sampling == 'uniform':
                    local_selected = uniform_frame_indices([timestamps[i] for i in eligible], args.frames)
                    selected = {eligible[i] for i in local_selected}
                else:
                    selected = set(eligible[:args.frames])
            sonar_index = -1
            for topic, msg, stamp in bag.read_messages(topics=[args.topic, '/forward_sonar/metadata']):
                if topic == '/forward_sonar/metadata':
                    local_metadata = json.loads(msg.data)
                    continue
                sonar_index += 1
                if args.sampling == 'first' and len(local_frames) >= args.frames:
                    if local_metadata is not None or not has_metadata:
                        break
                    continue
                if selected is not None and sonar_index not in selected:
                    continue
                if msg.encoding != 'mono8' or beam_indices[-1] >= msg.width:
                    parser.error(f'{bag_name}: ROI exceeds image width or topic is not mono8')
                if shape is not None and shape != (msg.height, msg.width):
                    parser.error('sonar configuration changed; split bags by configuration first')
                shape = (msg.height, msg.width)
                try:
                    _valid_range_rows(msg.height, excluded_rows)
                except ValueError as exc:
                    parser.error(str(exc))
                raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
                local_frames.append(raw[:, beam_indices].copy())
                selected_times.append(stamp.to_sec())
                local_indices.append(sonar_index)
        if len(local_frames) < 2:
            parser.error(f'{bag_name}: not enough image frames')
        frames.extend(local_frames)
        if first_sonar_time is None:
            first_sonar_time = selected_times[0]
        if local_metadata is not None:
            geometry_keys = ['n_beams', 'n_ranges', 'range_resolution_m', 'beam_angles_deg', 'gamma_byte', 'frequency_hz']
            if metadata is not None and any(metadata.get(key) != local_metadata.get(key) for key in geometry_keys):
                parser.error('recorded SDK geometry/display settings differ; fit separate profiles')
            metadata = local_metadata
        resolved = Path(bag_name).resolve()
        try:
            relative = str(resolved.relative_to(Path.cwd()))
        except ValueError:
            import os
            relative = os.path.relpath(str(resolved), str(Path.cwd()))
        bag_sources.append(dict(input_path=bag_name, absolute_path=str(resolved), relative_path=relative,
                                frames=len(local_frames), shape=list(shape), beam_rois=[list(roi) for roi in rois],
                                frame_indices=local_indices, frame_times_sec=selected_times,
                                first_sonar_time_sec=first_sonar_time,
                                window_start_time_sec=first_sonar_time + args.start_offset,
                                window_end_time_sec=window_end,
                                metadata_present=local_metadata is not None))
    if len(frames) < 2:
        parser.error('not enough image frames')
    gamma = 1. if args.gamma_byte in (0, 255) else args.gamma_byte / 255.
    result = estimate(frames, gamma, args.db_min, args.db_max, args.trim_percentile, beam_indices, excluded_rows)
    # Mean describes receiver background, therefore the synthetic base floor is 0 dB.
    result['noise_floor_power'] = 1.0
    profile_range = args.profile_range_m
    if profile_range is None and metadata is not None:
        profile_range = (shape[0] - 1) * float(metadata['range_resolution_m'])
        if not np.isfinite(profile_range) or profile_range <= 0:
            parser.error('recorded SDK range resolution must be finite and positive')
    result['noise_profile_range_m'] = float(profile_range or 0.)
    result['noise_beam_bias_db'] = []
    result['noise_profile_source'] = dict(bag=str(Path(args.bag).resolve()), frames=len(frames),
                                        beam_roi=list(rois[0]), beam_rois=[list(roi) for roi in rois], bags=bag_sources,
                                        sampling=args.sampling, frames_requested_per_bag=args.frames,
                                        start_offset_sec=args.start_offset, duration_sec=args.duration,
                                        excluded_range_rows=[list(rows) for rows in excluded_rows],
                                        excluded_row_fill='linear interpolation; nearest endpoint outside valid rows',
                                        trim_percentile=args.trim_percentile, calibrated=False,
                                        gamma_byte=args.gamma_byte, display_db=[args.db_min, args.db_max],
                                        profile_range_m=float(profile_range or 0.),
                                        range_mapping='meters' if profile_range is not None else 'legacy normalized rows',
                                        range_extrapolation='hold endpoint statistics')
    Path(args.output).write_text('# Relative background statistics; verify ROI is free of targets.\n'
                                 + yaml.safe_dump(result, sort_keys=False))
    if args.report:
        report = dict(source=result['noise_profile_source'],
                      raw_gray=_gray_statistics(frames, beam_indices, excluded_rows),
                      relative_db=dict(per_range_mean=result['noise_mean_db'], per_range_std=result['noise_std_db'],
                                       spatial_correlation=result['noise_correlation']),
                      notes=['ROIs must be selected explicitly; bright surfaces are not automatically rejected.',
                             'Relative display-domain statistics, not absolute acoustic power calibration.',
                             'Nonzero trim_percentile reports Winsorized standard deviation.',
                             'Excluded range rows do not enter statistics; noise profiles interpolate them; gray reports use null.'])
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    if args.metadata_output and metadata is not None:
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + '\n')
    if args.device_config_output:
        if metadata is None:
            parser.error('bag has no SDK metadata; historical native calibration cannot be recovered from Image alone')
        overlay = dict(num_beams=int(metadata['n_beams']), num_range_bins=int(metadata['n_ranges']),
                       range_resolution=float(metadata['range_resolution_m']),
                       beam_angles_deg=metadata['beam_angles_deg'], gamma=int(metadata['gamma_byte']),
                       mode=1 if float(metadata['frequency_hz']) < 1e6 else 2)
        Path(args.device_config_output).write_text(yaml.safe_dump(overlay, sort_keys=False))
    print(f'Saved {shape[0]} range rows from {len(frames)} frames to {args.output}; approximate display-domain calibration')


if __name__ == '__main__':
    main()
