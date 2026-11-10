import importlib.util
from pathlib import Path
import sys
import json
import subprocess

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from sensors.sonar_geometry import cartesian_mask

spec = importlib.util.spec_from_file_location('calibrate_sonar_noise', SCRIPTS / 'calibrate_sonar_noise.py')
calibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibrate)


def load_launch_parameters(path, arguments):
    # Parse the real include graph without starting MAVROS or a controller.
    import roslaunch
    config = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(str(path), config, argv=arguments, verbose=False)
    return {name: param.value for name, param in config.params.items()}, config


@pytest.mark.parametrize('method', ['proposed', 'b1_bearing_only', 'b2_ce_nmpc'])
@pytest.mark.parametrize('distance', [5., 10., 15., 30.])
def test_tracking_launch_uses_one_sonar_geometry(method, distance):
    launch = SCRIPTS.parents[1] / 'sub_planning/launch/target_tracking_simulation.launch'
    params, config = load_launch_parameters(launch, [f'method:={method}', f'sonar_range:={distance}'])
    planner = 'mpc_dynamic_target_tracker' if method == 'b2_ce_nmpc' else 'mpc_belief_target_tracker'
    for node in [planner, 'metrics_logger']:
        assert params[f'/{node}/sonar_max_range'] == distance
        assert params[f'/{node}/sonar_min_range'] == .2
        assert params[f'/{node}/sonar_fov_deg'] == 130.
    if method == 'b1_bearing_only':
        assert '/forward_sonar_sim/max_range' not in params
        return
    assert params['/forward_sonar_sim/max_range'] == distance
    assert params['/forward_sonar_target_detector/max_range'] == distance
    assert params['/forward_sonar_range_viz/max_range'] == distance
    assert params['/forward_sonar/geometry/max_range'] == distance
    assert '/forward_sonar_sim' in [name.rstrip('/') for name in config.clear_params]
    if distance == 10.:
        assert params['/forward_sonar_sim/num_range_bins'] == 494
        assert params['/forward_sonar_sim/range_resolution'] * 493 == pytest.approx(distance)
    else:
        assert params['/forward_sonar_sim/num_range_bins'] == 0
    assert len(params['/forward_sonar_sim/noise_mean_db']) == 494
    assert len(params['/forward_sonar_sim/noise_std_db']) == 494
    assert params['/forward_sonar_sim/noise_beam_bias_db'] == []
    assert params['/forward_sonar_sim/noise_profile_range_m'] == 10.


def test_tracking_default_range_and_custom_aperture():
    launch = SCRIPTS.parents[1] / 'sub_planning/launch/target_tracking_simulation.launch'
    params, _ = load_launch_parameters(launch, ['sonar_fov_deg:=120', 'sonar_min_range:=0.3'])
    for node in ['forward_sonar_sim', 'forward_sonar_target_detector', 'forward_sonar_range_viz']:
        assert params[f'/{node}/max_range'] == 15.
        assert params[f'/{node}/min_range'] == .3
        assert params[f'/{node}/fov_deg'] == 120.
    for node in ['mpc_belief_target_tracker', 'metrics_logger']:
        assert params[f'/{node}/sonar_max_range'] == 15.
        assert params[f'/{node}/sonar_min_range'] == .3
        assert params[f'/{node}/sonar_fov_deg'] == 120.


@pytest.mark.parametrize('distance', [5., 10., 15., 30.])
def test_compare_launch_does_not_force_a_ten_meter_grid(distance):
    launch = SCRIPTS.parent / 'launch/bhq750d_compare.launch'
    params, _ = load_launch_parameters(launch, [f'max_range:={distance}'])
    assert params['/forward_sonar_sim/max_range'] == distance
    assert params['/forward_sonar_target_detector/max_range'] == distance
    assert params['/forward_sonar_sim/num_range_bins'] == (494 if distance == 10. else 0)
    assert params['/forward_sonar_sim/noise_profile_range_m'] == 10.
    assert len(params['/forward_sonar_sim/noise_mean_db']) == 494
    assert params['/forward_sonar_sim/noise_beam_bias_db'] == []


def test_noise_profile_with_sparse_bright_contamination():
    rng = np.random.default_rng(5)
    expected = np.linspace(12., 30., 100)
    db = expected[None, :, None] + rng.normal(0., 2., (30, 100, 80))
    gray = 255 * np.power(np.clip(db / 75., 0, 1), 150. / 255)
    gray[:, :, 0] = 255  # a small contamination is trimmed, not fitted as receiver noise
    profile = calibrate.estimate(gray)
    assert np.max(np.abs(np.array(profile['noise_mean_db']) - expected)) < .4
    assert np.allclose(profile['noise_std_db'], 2., atol=.35)
    assert 0 <= profile['noise_correlation'] < .1


def test_noise_profile_can_disable_winsorization():
    frames = np.arange(80., dtype=float).reshape(4, 5, 4)
    profile = calibrate.estimate(frames, gamma=1., db_min=0., db_max=255., trim_percentile=0.)
    assert np.allclose(profile['noise_mean_db'], frames.mean(axis=(0, 2)))
    assert np.allclose(profile['noise_std_db'], frames.std(axis=(0, 2)))
    clipped = calibrate.estimate(frames, gamma=1., db_min=0., db_max=255., trim_percentile=10.)
    assert np.all(np.array(clipped['noise_std_db']) < np.array(profile['noise_std_db']))
    with pytest.raises(ValueError, match='trim percentile'):
        calibrate.estimate(frames, trim_percentile=50.)


def test_noise_correlation_excludes_disjoint_roi_joins():
    rng = np.random.default_rng(17)
    db = rng.normal(30., 2., (100, 60, 4))
    db[:, :, 2] = db[:, :, 1]  # far-apart beams happen to share a contaminating signal
    gray = 255 * np.power(db / 75., 150. / 255)
    joined = calibrate.estimate(gray, trim_percentile=0.)
    separated = calibrate.estimate(gray, trim_percentile=0., beam_indices=[0, 1, 50, 51])
    assert joined['noise_correlation'] > .25
    assert separated['noise_correlation'] < .05


def test_excluded_range_rows_never_affect_statistics_and_are_interpolated():
    rng = np.random.default_rng(41)
    rows = np.arange(7.)
    values = 30 + rows[None, :, None] * 4 + rng.normal(size=(60, 7, 20)) * (1 + rows[None, :, None])
    excluded = [(0, 2), (3, 4), (6, 7)]
    values[:, [0, 1, 3, 6], :] = 255.
    valid = np.array([2, 4, 5])
    profile = calibrate.estimate(values, gamma=1., db_max=255., trim_percentile=0., exclude_range_rows=excluded)
    reference = calibrate.estimate(values[:, valid, :], gamma=1., db_max=255., trim_percentile=0.)
    assert np.allclose(profile['noise_mean_db'], np.interp(rows, valid, reference['noise_mean_db']))
    assert np.allclose(profile['noise_std_db'], np.interp(rows, valid, reference['noise_std_db']))
    assert profile['noise_correlation'] == reference['noise_correlation']
    report = calibrate._gray_statistics(values, np.arange(20), excluded)
    assert np.isclose(report['mean'], values[:, valid, :].mean())
    assert report['saturated_fraction'] == 0.
    assert report['valid_range_rows'] == [2, 4, 5]
    assert len(report['per_range_mean']) == 7
    assert [report['per_range_mean'][i] for i in [0, 1, 3, 6]] == [None] * 4
    with pytest.raises(ValueError, match='unmasked range row'):
        calibrate.estimate(values, exclude_range_rows=[(0, 7)])
    with pytest.raises(ValueError, match='inside the native image'):
        calibrate.estimate(values, exclude_range_rows=[(5, 8)])


def test_uniform_sampling_is_time_based_and_never_repeats_frames():
    assert calibrate.uniform_frame_indices([0., 1., 2., 5., 9., 10., 20.], 3) == [0, 5, 6]
    timestamps = [0., .001, .002, .003, 20.]
    selected = calibrate.uniform_frame_indices(timestamps, 4)
    assert len(selected) == len(set(selected)) == 4
    assert selected[0] == 0 and selected[-1] == 4
    assert calibrate.uniform_frame_indices(timestamps, 20) == list(range(5))


def test_circular_range_and_blind_zone_masks():
    mask = cartesian_mask(257, 513, .2, 30., 130.)
    assert mask[0, 256]  # exactly maximum range on the forward axis
    assert not mask[0, 0]  # formerly accepted 40.49 m corner
    assert not mask[-1, 0]  # outside horizontal aperture
    assert mask[-1, 256]  # exactly minimum range on the forward axis
    assert np.array_equal(mask, mask[:, ::-1])


def test_thin_physical_echo_survives_region_morphology():
    sys.path.insert(0, str(SCRIPTS / 'sensors'))
    import batch_sonar_image_detector as batch
    detector = batch.make_offline_detector(dict(
        max_range=10., region_threshold_percentile=93., region_min_threshold=40.,
        region_min_blob_area=40, region_max_blob_area=600, region_max_range=25., region_near_gain=1.))
    image = np.full((256, 512), 10, np.uint8)
    image[52:55, 240:274] = 200
    regions = detector.detect_regions(image, detector.valid_mask(256, 512))
    assert len(regions) == 1
    assert 7. < regions[0]['range'] < 9.


def test_noise_cli_exports_recorded_sdk_geometry(tmp_path):
    import rosbag
    import rospy
    import yaml
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    metadata = dict(n_beams=8, n_ranges=20, range_resolution_m=.02,
                    beam_angles_deg=np.linspace(-2, 2, 8).tolist(), gamma_byte=150, frequency_hz=1.2e6)
    bag_path = tmp_path / 'noise.bag'
    with rosbag.Bag(str(bag_path), 'w') as bag:
        bag.write('/forward_sonar/metadata', String(data=json.dumps(metadata)), rospy.Time.from_sec(1.))
        for i in range(4):
            msg = Image(height=20, width=8, encoding='mono8', step=8,
                        data=np.full((20, 8), 40 + i, np.uint8).tobytes())
            bag.write('/forward_sonar/image_raw', msg, rospy.Time.from_sec(2. + i))
    output, config = tmp_path / 'noise.yaml', tmp_path / 'device.yaml'
    subprocess.run([sys.executable, str(SCRIPTS / 'calibrate_sonar_noise.py'), str(bag_path), str(output),
                    '--beam-roi', '1:6', '--frames', '3', '--device-config-output', str(config)], check=True)
    noise, device = yaml.safe_load(output.read_text()), yaml.safe_load(config.read_text())
    assert len(noise['noise_mean_db']) == 20
    assert noise['noise_profile_range_m'] == pytest.approx(19 * .02)
    assert noise['noise_beam_bias_db'] == []
    assert noise['noise_profile_source']['frames'] == 3
    assert device['beam_angles_deg'] == metadata['beam_angles_deg']
    assert device['num_range_bins'] == 20 and device['mode'] == 2


def test_noise_cli_combines_bags_and_reports_actual_uniform_samples(tmp_path):
    import rosbag
    import rospy
    import yaml
    from sensor_msgs.msg import Image
    bags = [tmp_path / 'first.bag', tmp_path / 'second.bag']
    offsets = [0., 1., 2., 5., 9., 10., 20.]
    for bag_index, bag_path in enumerate(bags):
        with rosbag.Bag(str(bag_path), 'w') as bag:
            for frame_index, offset in enumerate(offsets):
                values = np.full((12, 8), 30 + frame_index + bag_index * 10, np.uint8)
                values[:, 3:5] = 255  # explicit ROI exclusion, not automatic target removal
                msg = Image(height=12, width=8, encoding='mono8', step=8, data=values.tobytes())
                bag.write('/forward_sonar/image_raw', msg, rospy.Time.from_sec(100. + bag_index * 100. + offset))
    output, report_path = tmp_path / 'combined.yaml', tmp_path / 'report.json'
    subprocess.run([sys.executable, str(SCRIPTS / 'calibrate_sonar_noise.py'), str(bags[0]), str(output),
                    '--additional-bag', str(bags[1]), '--beam-roi', '1:3', '--beam-roi', '5:7',
                    '--frames', '3', '--sampling', 'uniform', '--trim-percentile', '0',
                    '--report', str(report_path)], check=True)
    profile, report = yaml.safe_load(output.read_text()), json.loads(report_path.read_text())
    source = profile['noise_profile_source']
    assert source['frames'] == 6 and source['sampling'] == 'uniform'
    assert source['beam_rois'] == [[1, 3], [5, 7]]
    assert source['trim_percentile'] == 0.
    assert source['bags'][0]['frame_indices'] == source['bags'][1]['frame_indices'] == [0, 5, 6]
    assert source['bags'][0]['frame_times_sec'] == [100., 110., 120.]
    assert source['bags'][1]['frame_times_sec'] == [200., 210., 220.]
    assert source['bags'][0]['shape'] == [12, 8]
    assert Path(source['bags'][0]['absolute_path']) == bags[0]
    expected = np.array([30, 35, 36, 40, 45, 46], dtype=float)
    assert np.isclose(report['raw_gray']['mean'], expected.mean())
    assert np.isclose(report['raw_gray']['std'], expected.std())
    assert report['raw_gray']['saturated_fraction'] == 0.
    assert len(report['raw_gray']['per_range_mean']) == 12


@pytest.mark.parametrize('sampling,expected_indices', [('first', [1, 2]), ('uniform', [1, 3])])
def test_noise_cli_selects_time_window_and_masks_contaminated_rows(tmp_path, sampling, expected_indices):
    import rosbag
    import rospy
    import yaml
    from sensor_msgs.msg import Image
    bag_path = tmp_path / 'window.bag'
    with rosbag.Bag(str(bag_path), 'w') as bag:
        for frame_index in range(6):
            values = np.full((8, 6), 40 + frame_index, np.uint8)
            values[:2, :] = values[6:, :] = 255
            msg = Image(height=8, width=6, encoding='mono8', step=6, data=values.tobytes())
            bag.write('/forward_sonar/image_raw', msg, rospy.Time.from_sec(100. + 2 * frame_index))
    output, report_path = tmp_path / 'window.yaml', tmp_path / 'window.json'
    subprocess.run([sys.executable, str(SCRIPTS / 'calibrate_sonar_noise.py'), str(bag_path), str(output),
                    '--beam-roi', '1:5', '--frames', '2', '--sampling', sampling,
                    '--start-offset', '2', '--duration', '6', '--profile-range-m', '10',
                    '--exclude-range-rows', '0:2', '--exclude-range-rows', '6:8',
                    '--trim-percentile', '0', '--report', str(report_path)], check=True)
    profile, report = yaml.safe_load(output.read_text()), json.loads(report_path.read_text())
    source = profile['noise_profile_source']
    assert source['bags'][0]['frame_indices'] == expected_indices
    assert source['bags'][0]['first_sonar_time_sec'] == 100.
    assert source['bags'][0]['window_start_time_sec'] == 102.
    assert source['bags'][0]['window_end_time_sec'] == 108.
    assert source['start_offset_sec'] == 2. and source['duration_sec'] == 6.
    assert source['excluded_range_rows'] == [[0, 2], [6, 8]]
    assert len(profile['noise_mean_db']) == len(profile['noise_std_db']) == 8
    assert profile['noise_profile_range_m'] == 10.
    assert source['range_mapping'] == 'meters'
    assert np.allclose(profile['noise_mean_db'], profile['noise_mean_db'][2])
    assert np.isclose(report['raw_gray']['mean'], 40 + np.mean(expected_indices))
    assert report['raw_gray']['saturated_fraction'] == 0.
    assert report['raw_gray']['per_range_mean'][:2] == [None, None]
    assert report['raw_gray']['per_range_mean'][6:] == [None, None]
