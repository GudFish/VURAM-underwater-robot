"""PD wrench signs, coordinate transforms and normalized action contract."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from control.observation import build_obs, build_ref11, load_spec
from control.pd_controller import PDController
from utils.math3d import quat_mul, quat_to_R, yaw_quat


@pytest.fixture
def allocator():
    # Deliberately asymmetric, so the wrong reverse scale is observable.
    return SimpleNamespace(scales=np.array([100., 100., 100., 10., 10.]),
                           negative_scales=np.array([50., 50., 50., 5., 5.]))


@pytest.fixture
def spec():
    return load_spec(str(PACKAGE / 'config/obs_act.yaml'))


@pytest.mark.parametrize('obs_index, action_index, gain', [
    (0, 0, 80.), (1, 1, 80.), (2, 2, 60.),
    (3, 3, 2.49), (5, 4, 2.60)])
def test_five_controlled_axes_use_signed_physical_limits(
        allocator, obs_index, action_index, gain):
    controller = PDController(allocator)
    observation = np.zeros(21)
    observation[obs_index] = .1
    positive = controller.step(observation)
    assert positive[action_index] == pytest.approx(
        .1 * gain / allocator.scales[action_index])
    assert np.count_nonzero(positive) == 1
    observation[obs_index] = -.1
    negative = controller.step(observation)
    assert negative[action_index] == pytest.approx(
        -.1 * gain / allocator.negative_scales[action_index])


def test_velocity_feedback_yaw_feedforward_and_no_pitch_command(allocator):
    controller = PDController(allocator)
    observation = np.zeros(21)
    observation[6:12] = [.1, -.1, .2, .2, 50., -.1]
    observation[12:16] = [.2, -.2, .1, .3]
    # Pitch attitude/rate and previous actions have no PD output channel.
    observation[4] = 50.
    observation[16:] = .8
    actual = controller.step(observation, np.ones((20, 18)), 20)
    wrench = np.array([4., -4., -3.2, -1.245 * .2, 1.30 * .4])
    expected = wrench / np.where(wrench >= 0, allocator.scales,
                                 allocator.negative_scales)
    np.testing.assert_allclose(actual, expected)


def test_body_heading_rotates_world_position_and_velocity_error(allocator, spec):
    controller = PDController(allocator)
    state = np.r_[np.zeros(6), yaw_quat(np.pi / 2), [3., -1., -2.]]
    reference = build_ref11([3.1, -1., -2.], state[6:10], [.2, 0., 0.], 0.)
    observation = build_obs(state, reference, np.zeros(5), spec['obs_layout'])
    actual = controller.step(observation)
    np.testing.assert_allclose(actual, [0., -.32, 0., 0., 0.], atol=1e-12)


def test_yaw_error_uses_short_rotation_across_pi(allocator, spec):
    controller = PDController(allocator)
    state = np.r_[np.zeros(6), yaw_quat(np.pi - .05), np.zeros(3)]
    reference = build_ref11(np.zeros(3), yaw_quat(-np.pi + .05), np.zeros(3), 0.)
    observation = build_obs(state, reference, np.zeros(5), spec['obs_layout'])
    assert observation[5] == pytest.approx(.1)
    assert controller.step(observation)[4] == pytest.approx(.026)
    reference[3:7] *= -1  # The same orientation's other quaternion sign.
    np.testing.assert_allclose(controller.step(build_obs(
        state, reference, np.zeros(5), spec['obs_layout'])),
        controller.step(observation))


def test_reference_world_translation_and_heading_do_not_change_body_command(
        allocator, spec):
    controller = PDController(allocator)
    state = np.r_[[.03, -.02, .01, .01, .02, -.03], yaw_quat(.2), [2., 1., -3.]]
    reference = build_ref11([2.1, .9, -2.9], yaw_quat(.3), [.1, .2, .05], .04)
    expected = controller.step(build_obs(state, reference, np.zeros(5), spec['obs_layout']))
    rotation_quat = yaw_quat(1.3)
    rotation = quat_to_R(rotation_quat)
    state[6:10] = quat_mul(rotation_quat, state[6:10])
    state[10:13] = rotation @ state[10:13] + [30., -40., -2.]
    reference[:3] = rotation @ reference[:3] + [30., -40., -2.]
    reference[3:7] = quat_mul(rotation_quat, reference[3:7])
    reference[7:10] = rotation @ reference[7:10]
    actual = controller.step(build_obs(state, reference, np.zeros(5), spec['obs_layout']))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_saturation_and_invalid_inputs(allocator):
    controller = PDController(allocator)
    observation = np.zeros(21)
    observation[[0, 1, 2, 3, 5]] = [100., -100., 100., -100., 100.]
    np.testing.assert_array_equal(controller.step(observation), [1., -1., 1., -1., 1.])
    for invalid in (np.zeros(20), np.full(21, np.nan), np.full(21, np.inf)):
        with pytest.raises(ValueError, match='finite observation'):
            controller.step(invalid)


@pytest.mark.parametrize('kwargs', [
    dict(kp_xyz=[1., 2.]), dict(kd_xyz=[1., -2., 3.]),
    dict(kp_att=[1., 2., 3.]), dict(kd_att=[np.inf, 1.])])
def test_bad_gain_configuration_rejected(allocator, kwargs):
    with pytest.raises(ValueError):
        PDController(allocator, **kwargs)


def test_ros_gain_overrides_and_controller_metadata(allocator, spec):
    params = {'~kp_xyz': [1., 2., 3.], '~kd_att': [0., 0.]}
    controller = PDController.from_ros_params(allocator,
                                             lambda name, default: params.get(name, default))
    controller.validate_spec(spec)
    assert controller.version == -1 and controller.controller_type == 'pd'
    assert controller.parameters['kp_xyz'] == [1., 2., 3.]
    assert controller.parameters['attitude_axes'] == ['roll', 'yaw']
    np.testing.assert_array_equal(controller.step(np.zeros(21)), np.zeros(5))


@pytest.mark.parametrize('name', ['e_pos', 'e_att', 'v_body', 'v_ref_lin',
                                  'w_ref_z', 'a_prev'])
@pytest.mark.parametrize('field', ['start', 'len'])
def test_changed_observation_field_layout_is_rejected(allocator, spec, name, field):
    invalid = copy.deepcopy(spec)
    invalid['obs_layout'][name][field] += 1
    with pytest.raises(ValueError, match='obs_layout|obs\\[21\\]'):
        PDController(allocator).validate_spec(invalid)


@pytest.mark.parametrize('change', ['swapped', 'renamed', 'float', 'boolean'])
def test_same_dimension_incompatible_layout_is_rejected(allocator, spec, change):
    invalid = copy.deepcopy(spec)
    layout = invalid['obs_layout']
    if change == 'swapped':
        layout['e_pos']['start'], layout['e_att']['start'] = 3, 0
    elif change == 'renamed':
        layout['position_error'] = layout.pop('e_pos')
    elif change == 'float':
        layout['e_pos']['start'] = 0.
    else:
        layout['w_ref_z']['len'] = True
    with pytest.raises(ValueError, match='obs_layout'):
        PDController(allocator).validate_spec(invalid)


@pytest.mark.parametrize('name, value', [
    ('dt', 0.), ('dt', -.05), ('dt', np.inf), ('dt', np.nan),
    ('control_rate', 10.), ('control_rate', -20.), ('control_rate', 0.),
    ('control_rate', np.inf), ('control_rate', np.nan)])
def test_inconsistent_or_invalid_control_timing_is_rejected(allocator, spec, name, value):
    invalid = copy.deepcopy(spec)
    invalid[name] = value
    with pytest.raises(ValueError, match='control_rate must equal 1 / dt'):
        PDController(allocator).validate_spec(invalid)


@pytest.mark.parametrize('encoding', ['world_tracking_v1', '', None])
def test_unsupported_observation_encoding_is_rejected(allocator, spec, encoding):
    invalid = copy.deepcopy(spec)
    invalid['observation_encoding'] = encoding
    with pytest.raises(ValueError, match='observation encoding'):
        PDController(allocator).validate_spec(invalid)


def test_defaults_match_training_warm_start_for_horizontal_pose(spec):
    learning_root = Path('~/mujoco_auv_env').expanduser()
    if not (learning_root / 'auv_learning').is_dir():
        pytest.skip('requires the learner checkout for the shared wrench contract')
    sys.path.insert(0, str(learning_root))
    from auv_learning.control.warm_start import WarmStartController
    from auv_learning.task.math_utils import load_yaml
    environment = load_yaml(str(learning_root / 'config/env_config.yaml'))
    gains = load_yaml(str(learning_root / 'config/controller_config.yaml'))['warm_start']
    baseline = WarmStartController(gains, com=environment['vehicle']['com'],
                                   thruster_config=environment['thruster'])
    controller = PDController(baseline.allocator)
    for yaw in (-2., 0., 1.4):
        state = np.r_[[.01, .02, -.01, .02, 0., -.02], yaw_quat(yaw), [1., 2., -3.]]
        reference = build_ref11([1.1, 1.9, -2.9], yaw_quat(yaw + .1), [.1, -.1, .05], .1)
        observation = build_obs(state, reference, np.zeros(5), spec['obs_layout'])
        np.testing.assert_allclose(controller.step(observation), baseline.step(state, reference),
                                   atol=1e-12)

