"""ROS adapter contract regressions; no ROS master or vehicle required."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from nav_msgs.msg import Odometry

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from vuram_policy.policy_inference import PolicyInference, PoseState, _history_tensor
from vuram_policy.spec import (OBSERVATION_ENCODING,
                               build_obs, build_ref11, load_spec)
from vuram_policy.utils import quat_mul, quat_to_R, yaw_quat


@pytest.fixture
def spec():
    return load_spec(str(PACKAGE / 'config' / 'obs_act.yaml'))


def inputs(yaw=0.0):
    state = np.r_[np.arange(6) * .03, yaw_quat(yaw), [1., 2., -3.]]
    reference = build_ref11([2., 2., -3.], yaw_quat(yaw), [.4, 0., 0.], .2)
    return state, reference, np.arange(5) * .1


@pytest.mark.parametrize('yaw, direction', [
    (0., [1., 0., 0.]), (np.pi / 2, [0., -1., 0.]),
    (np.pi, [-1., 0., 0.]), (-np.pi / 2, [0., 1., 0.])])
def test_heading_changes_body_error_and_reference_velocity(spec, yaw, direction):
    state, ref, previous = inputs(yaw)
    obs = build_obs(state, ref, previous, spec['obs_layout'])
    np.testing.assert_allclose(obs[:3], direction, atol=1e-12)
    np.testing.assert_allclose(obs[12:15], np.array(direction) * .4, atol=1e-12)
    np.testing.assert_allclose(obs[3:6], 0., atol=1e-12)
    np.testing.assert_array_equal(obs[6:12], state[:6])
    np.testing.assert_array_equal(obs[16:], previous)
    assert obs[15] == .2


def test_full_attitude_rotation_and_world_origin_invariance(spec):
    state, ref, previous = inputs()
    state[6:10] = np.array([.7, -.2, .3, .4])
    state[6:10] /= np.linalg.norm(state[6:10])
    ref[:3] += [1., -2., .5]
    obs = build_obs(state, ref, previous, spec['obs_layout'])
    rotation = quat_to_R(state[6:10]).T
    np.testing.assert_allclose(obs[:3], rotation @ (ref[:3] - state[10:13]))
    np.testing.assert_allclose(obs[12:15], rotation @ ref[7:10])
    turn = yaw_quat(1.1)
    world_rotation = quat_to_R(turn)
    transformed, target = state.copy(), ref.copy()
    transformed[6:10] = quat_mul(turn, state[6:10])
    transformed[10:13] = world_rotation @ state[10:13] + [10., -4., 5.]
    target[:3] = world_rotation @ ref[:3] + [10., -4., 5.]
    target[3:7] = quat_mul(turn, ref[3:7])
    target[7:10] = world_rotation @ ref[7:10]
    np.testing.assert_allclose(build_obs(transformed, target, previous,
                                       spec['obs_layout']), obs, atol=1e-12)


class Actor(torch.nn.Module):
    def __init__(self, encoding):
        super().__init__()
        self.observation_encoding = encoding

    def forward(self, obs):
        return torch.tanh(obs[:, :5])


class Deploy(torch.nn.Module):
    def __init__(self, encoding):
        super().__init__()
        self.policy = Actor(encoding)

    def forward(self, obs, history, valid):
        return self.policy(obs)


def snapshot(tmp_path, encoding=OBSERVATION_ENCODING):
    path = tmp_path / 'deploy.pt'
    torch.jit.script(Deploy(encoding)).save(str(path))
    manifest = dict(schema_version=4, version=12, obs_dim=21, action_dim=5,
                    history_shape=[20, 18], dt=.05,
                    observation_encoding=encoding,
                    history_encoding='window_relative_position_v1',
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_manifest(tmp_path, manifest)
    return manifest


def write_manifest(path, manifest):
    (path / 'spec.json').write_text(json.dumps(manifest))


def test_snapshot_uses_body_coordinates(tmp_path, spec):
    snapshot(tmp_path)
    inference = PolicyInference(str(tmp_path))
    inference.validate_spec(spec)
    state, ref, previous = inputs(np.pi / 2)
    obs = inference.observe(state, ref, previous, spec['obs_layout'])
    np.testing.assert_allclose(obs[:3], [0., -1., 0.], atol=1e-12)
    action = inference.step(obs, np.zeros((20, 18)), 20)
    np.testing.assert_allclose(action, np.tanh(obs[:5]), atol=1e-7)


def test_missing_manifest_is_rejected(tmp_path):
    snapshot(tmp_path)
    (tmp_path / 'spec.json').unlink()
    with pytest.raises(FileNotFoundError):
        PolicyInference(str(tmp_path))


def test_missing_encoding_is_rejected(tmp_path):
    manifest = snapshot(tmp_path)
    del manifest['observation_encoding']
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match='expected observation encoding'):
        PolicyInference(str(tmp_path))


def test_scripted_actor_encoding_must_match_manifest(tmp_path):
    manifest = snapshot(tmp_path, 'unsupported')
    manifest['observation_encoding'] = OBSERVATION_ENCODING
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match='differs'):
        PolicyInference(str(tmp_path))


@pytest.mark.parametrize('field,value,error', [
    ('observation_encoding', 'unknown', 'expected observation encoding'),
    ('model_sha256', 'bad', 'digest'),
    ('history_shape', [20, 19], 'expected obs'),
    ('history_encoding', 'unknown', 'unsupported history'),
    ('actor_future_spec', {'steps': 2}, 'future-reference')])
def test_incompatible_snapshot_is_rejected(tmp_path, field, value, error):
    manifest = snapshot(tmp_path)
    manifest[field] = value
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match=error):
        PolicyInference(str(tmp_path))


def test_control_period_checked_before_deployment(tmp_path, spec):
    snapshot(tmp_path)
    inference = PolicyInference(str(tmp_path))
    wrong = copy.deepcopy(spec)
    wrong['control_rate'] = 10
    with pytest.raises(ValueError, match='control_rate'):
        inference.validate_spec(wrong)
    wrong['dt'] = .1
    with pytest.raises(ValueError, match='control period'):
        inference.validate_spec(wrong)


def test_history_stays_raw_state_action_with_left_padding():
    state, _, previous = inputs(np.pi / 2)
    row = np.r_[state, previous].astype(np.float32)
    history = _history_tensor([row], 20, 18)
    np.testing.assert_array_equal(history[:-1], 0.)
    np.testing.assert_array_equal(history[-1], row)
    # The encoder inside deploy.pt performs relative-position conversion.
    np.testing.assert_array_equal(history[-1, 10:13], state[10:13])


@pytest.mark.parametrize('world', [False, True])
def test_odometry_twist_is_rotated_only_when_world_flag_set(world):
    msg = Odometry()
    q = yaw_quat(np.pi / 2)
    msg.pose.pose.orientation.w, msg.pose.pose.orientation.z = q[0], q[3]
    msg.twist.twist.linear.x = .4
    msg.twist.twist.angular.x = .2
    with patch('vuram_policy.policy_inference.rospy.Subscriber'):
        pose = PoseState('/test/odom', twist_in_world=world)
    pose._cb(msg)
    state = pose.step()['state']
    np.testing.assert_allclose(state[:3], [0., -.4, 0.] if world else [.4, 0., 0.],
                               atol=1e-12)
    np.testing.assert_allclose(state[3:6], [0., -.2, 0.] if world else [.2, 0., 0.],
                               atol=1e-12)
