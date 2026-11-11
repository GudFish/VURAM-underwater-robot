"""Collection contracts and fault handling; no ROS master or vehicle required."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from data_collection.contract import TrainingContract
from control import mavros_startup
from data_collection.episodes import EpisodeWriter, LapRecorder, next_iteration, write_episode
from trajectory.figure8 import FixedFigure8
from vuram_policy.model_reload import PolicyReloader
from control import tracking_node as node
from data_collection import episodes
from vuram_policy import policy_inference as policy_node
from vuram_policy.policy_inference import PolicyInference
from control.observation import load_spec
from utils.math3d import quat_to_R, yaw_quat

LEARNING_ROOT = Path(os.environ.get('AUV_LEARNING_ROOT', '~/mujoco_auv_env')).expanduser()


@pytest.fixture
def contract():
    if not (LEARNING_ROOT / 'auv_learning').is_dir():
        pytest.skip('set AUV_LEARNING_ROOT to the training checkout')
    return TrainingContract(LEARNING_ROOT)


@pytest.fixture
def spec():
    return load_spec(str(PACKAGE / 'config/obs_act.yaml'))


def state_at(pos=(1., 2., -3.), yaw=.4):
    return np.r_[np.zeros(6), yaw_quat(yaw), pos]


class ConstantPolicy:
    history_shape = (20, 18)

    def __init__(self, version=0, action=.1):
        self.version = version
        self.action = action
        self.valids = []

    def step(self, obs, history, valid):
        self.valids.append(valid)
        return np.full(5, self.action)


def record_lap(contract, spec, policy=None, duration=30., iteration=0):
    trajectory = FixedFigure8(state_at(), duration=duration)
    policy = policy or ConstantPolicy()
    lap = LapRecorder(contract, spec, trajectory, policy, iteration)
    issued = []
    for index in range(trajectory.intervals + 1):
        pos, quat, _, _ = trajectory.get_ref(trajectory.params, index * .05)
        state = np.r_[np.zeros(6), quat, pos]
        terminal = lap.record(state, 100. + index * .05, 100. + index * .05,
                              lambda action: issued.append(action.copy()))
        assert terminal == (index == trajectory.intervals)
    return lap, issued


@pytest.mark.parametrize('yaw', [-2., 0., 1.3])
def test_figure8_anchor_plane_closure_speed_and_heading(yaw):
    state = state_at(pos=(-4., 7., -.3), yaw=yaw)
    curve = FixedFigure8(state)
    start = curve.get_ref(curve.params, 0.)
    np.testing.assert_allclose(start[0], state[10:13])
    np.testing.assert_allclose(quat_to_R(start[1]), quat_to_R(state[6:10]), atol=1e-14)
    for duration in (30., 60.):
        for actual, expected in zip(curve.get_ref(curve.params, duration), start):
            np.testing.assert_allclose(actual, expected, atol=1e-14)
    samples = [curve.get_ref(curve.params, t) for t in np.linspace(0., 30., 1001)]
    assert np.ptp(np.asarray([s[0] for s in samples])[:, 2]) == 0.
    assert max(np.linalg.norm(s[2]) for s in samples) == pytest.approx(.5)
    np.testing.assert_allclose(curve.params['amplitudes'], [1.52832767545, .91699660527])
    eps = 1e-5
    left, right = curve.get_ref(curve.params, 30.-eps), curve.get_ref(curve.params, eps)
    np.testing.assert_allclose(left[2], right[2], atol=1e-9)
    np.testing.assert_allclose(left[1], right[1], atol=1e-9)


@pytest.mark.parametrize('kwargs', [dict(duration=0), dict(speed=-1), dict(ratio=0),
                                  dict(speed=float('nan')), dict(duration=.001)])
def test_invalid_trajectory_rejected(kwargs):
    with pytest.raises(ValueError):
        FixedFigure8(state_at(), **kwargs)


def test_figure8_parameters_reconstruct_in_training_generator(contract):
    from auv_learning.task.references import ReferenceGenerator
    curve = FixedFigure8(state_at(), duration=30.02, speed=.4, ratio=.7)
    assert curve.duration == 30.
    generator = ReferenceGenerator()
    for t in np.linspace(0, curve.duration, 80):
        for actual, expected in zip(curve.get_ref(curve.params, t), generator.get_ref(curve.params, t)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_complete_lap_has_600_transitions_and_persistent_motor_state(contract, spec):
    lap, issued = record_lap(contract, spec)
    bundle, extras = lap.finish()
    assert bundle.n_steps == 601
    assert len(issued) == 600
    np.testing.assert_allclose(bundle.arrays['time'].ravel(), np.arange(601) * .05)
    assert bundle.arrays['done'].sum() == 1 and bundle.arrays['done'][-1, 0] == 1
    assert extras['action_applied'].sum() == 600
    np.testing.assert_array_equal(extras['dof_pwm'][-1], extras['dof_pwm'][-2])
    np.testing.assert_array_equal(bundle.arrays['ref'][0], bundle.arrays['ref'][-1])
    assert lap.policy.valids[:3] == [0, 1, 2]
    assert lap.policy.valids[-1] == 20
    assert bundle.attrs['motor_force_source'] == 'nominal_allocator_estimate'
    motor = contract.allocator.motor_force.copy()
    np.testing.assert_array_equal(bundle.arrays['motor_force_pre'][-1], motor)
    next_policy = ConstantPolicy(version=1, action=-.2)
    next_lap = LapRecorder(contract, spec, lap.trajectory, next_policy, 1)
    next_lap.record(bundle.arrays['state'][-1], 130., 130., lambda action: None)
    np.testing.assert_array_equal(next_lap.rows[0]['motor_force_pre'], motor)
    assert next_policy.valids == [0]
    np.testing.assert_array_equal(next_lap.previous, np.full(5, -.2))
    with pytest.raises(ValueError, match='incomplete'):
        next_lap.finish()


def test_disabled_saving_keeps_control_and_bounded_history_without_episode_arrays(contract, spec, monkeypatch):
    monkeypatch.setattr(contract, 'reward', Mock(side_effect=AssertionError('reward in test mode')))
    policy = ConstantPolicy(action=.1)
    trajectory = FixedFigure8(state_at())
    lap = LapRecorder(contract, spec, trajectory, policy, 0, save_data=False)
    issued = []
    for index in range(trajectory.intervals + 1):
        assert lap.record(state_at(), index * .05, index * .05,
                          lambda action: issued.append(action.copy())) == (index == 600)
        assert lap.rows == lap.extra_rows == []
        assert len(lap.history) <= 20
    assert len(issued) == 600
    assert lap.step_index == 601
    contract.reward.assert_not_called()
    with pytest.raises(ValueError, match='disabled'):
        lap.finish()
    with pytest.raises(ValueError, match='already complete'):
        lap.record(state_at(), 31., 31., lambda action: None)
    motor = contract.allocator.motor_force.copy()
    next_lap = LapRecorder(contract, spec, trajectory, policy, 1, save_data=False)
    np.testing.assert_array_equal(next_lap.previous, np.zeros(5))
    assert not next_lap.history
    np.testing.assert_array_equal(contract.allocator.motor_force, motor)
    next_lap.record(state_at(), 30., 30., lambda action: None)
    assert policy.valids[-1] == 0


@pytest.mark.parametrize('action', [np.full(5, np.nan), np.zeros(6)])
def test_invalid_controller_output_never_reaches_publisher(contract, spec, action):
    policy = ConstantPolicy()
    policy.step = Mock(return_value=action)
    lap = LapRecorder(contract, spec, FixedFigure8(state_at()), policy, 0,
                              save_data=False)
    send = Mock()
    with pytest.raises(ValueError, match='five finite'):
        lap.record(state_at(), 0., 0., send)
    send.assert_not_called()
    np.testing.assert_array_equal(contract.allocator.motor_force, np.zeros(6))


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
def test_h5_roundtrip_ingestion_and_one_training_batch(contract, spec, tmp_path, monkeypatch, controller_kind):
    import torch
    from auv_learning.deployment.episode import EpisodeBundle
    from auv_learning.dynamics.fossen import FossenPrior
    from auv_learning.learning.dataset import GrowingBuffer
    from auv_learning.learning.models.encoder import HistoryEncoder
    from auv_learning.learning.models.policy import LatentPolicy
    from auv_learning.learning.models.vnd import VNDModel
    from auv_learning.learning.sampling import sample_vnd, sample_rollouts
    from auv_learning.learning.vnd import workflow
    from auv_learning.learning.vnd.optimization import train_vnd, train_policy_stage

    torch.set_num_threads(2)
    from control.pd_controller import PDController
    controller = PDController(contract.allocator) if controller_kind == 'pd' else None
    lap, _ = record_lap(contract, spec, policy=controller)
    bundle, extras = lap.finish()
    assert bundle.attrs['controller_type'] == controller_kind
    if controller_kind == 'pd':
        assert bundle.attrs['controller_parameters'] == controller.parameters
        assert bundle.attrs['policy_version'] == -1
    inbox = tmp_path / 'inbox'
    inbox.mkdir()
    path = write_episode(bundle, extras, str(inbox))
    decoded = EpisodeBundle.from_hdf5(path)
    assert decoded.n_steps == 601
    with h5py.File(path) as stream:
        assert stream['episode/dof_pwm'].shape == (601, 6)
        assert stream['episode/odom_stamp'].shape == (601, 1)
        assert stream['episode/action_applied'][-1, 0] == 0
    buffer = GrowingBuffer(str(tmp_path / 'rollouts'))
    monkeypatch.setattr(workflow, 'VND_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(workflow, 'VND_INBOX_DIR', str(inbox))
    monkeypatch.setattr(workflow, 'VND_ROLLOUT_DIR', str(tmp_path / 'rollouts'))
    assert workflow._ingest(buffer) == [bundle.episode_id]
    assert sample_vnd(buffer.episodes, 2)[0].shape == (2, 19)
    assert sample_rollouts(buffer.episodes, 2, 4)[1].shape == (2, 5, 11)
    model = VNDModel(FossenPrior(contract.env_config, .05), HistoryEncoder())
    policy = LatentPolicy()
    dyn_opt = torch.optim.Adam(model.parameters(), lr=.0001)
    pol_opt = torch.optim.Adam(policy.parameters(), lr=.0001)
    rng = np.random.default_rng(5)
    dynamics = train_vnd(model, buffer.episodes, dyn_opt,
                         dict(epochs=1, batches_per_epoch=1, batch=2, future=2), 'cpu', rng)
    assert dynamics and all(np.isfinite(v) for v in dynamics.values())
    result = train_policy_stage(policy, model, buffer.episodes, pol_opt,
                                dict(steps=1, batch=2, horizon=2), spec['obs_layout'],
                                contract.reward_weights, 'cpu', rng)
    assert result['updates'] == 1 and np.isfinite(result['policy_loss'])
    assert next_iteration(str(inbox)) == 1


def test_atomic_spooling_is_invisible_until_complete_and_never_overwrites(contract, spec, tmp_path):
    lap, _ = record_lap(contract, spec, duration=.2)
    bundle, extras = lap.finish()
    writing, release = threading.Event(), threading.Event()
    original = bundle.to_hdf5

    def delayed(path):
        original(path)
        writing.set()
        assert release.wait(5.)

    with patch.object(bundle, 'to_hdf5', delayed):
        writer = EpisodeWriter(str(tmp_path))
        try:
            writer.submit(bundle, extras)
            assert writing.wait(5.)
            assert not list(tmp_path.glob('*.h5'))
            assert list(tmp_path.glob('*.tmp-*'))
        finally:
            release.set()
            writer.close()
    path = next(tmp_path.glob('*.h5'))
    previous = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_episode(bundle, extras, str(tmp_path))
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp-*'))


def test_writer_queue_is_bounded_and_write_errors_propagate(contract, spec, tmp_path):
    lap, _ = record_lap(contract, spec, duration=.2)
    bundle, extras = lap.finish()
    blocked, release = threading.Event(), threading.Event()

    def slow_write(*args):
        blocked.set()
        assert release.wait(5.)
        return 'saved'

    with patch.object(episodes, 'write_episode', slow_write):
        writer = EpisodeWriter(str(tmp_path))
        try:
            writer.submit(bundle, extras)
            assert blocked.wait(5.)
            writer.submit(bundle, extras)
            writer.submit(bundle, extras)
            with pytest.raises(RuntimeError, match='queue is full'):
                writer.submit(bundle, extras)
        finally:
            release.set()
            writer.close()
    with patch.object(episodes, 'write_episode', side_effect=OSError('disk full')):
        writer = EpisodeWriter(str(tmp_path))
        writer.submit(bundle, extras)
        with pytest.raises(RuntimeError, match='disk full'):
            writer.close()


def fake_snapshot(root, version):
    snapshot = root / ('policy_v{:06d}'.format(version))
    snapshot.mkdir()
    (snapshot / 'deploy.pt').write_bytes(b'placeholder')
    (snapshot / 'spec.json').write_text('{}')
    (root / 'latest.json').write_text(json.dumps(dict(version=version, path=snapshot.name)))
    return snapshot


def test_reloader_prepares_without_changing_active_and_handles_bad_publication(tmp_path):
    fake_snapshot(tmp_path, 0)
    loader = lambda path: ConstantPolicy(int(Path(path).name[8:]))
    reloader = PolicyReloader(str(tmp_path), loader)
    reloader.poll_once()
    active = reloader.take_ready()
    assert active.version == 0
    second = fake_snapshot(tmp_path, 1)
    reloader.poll_once()
    assert active.version == 0  # no mid-lap swap
    active = reloader.take_ready()
    assert active.version == 1
    (tmp_path / 'latest.json').unlink()
    with pytest.raises(FileNotFoundError):
        reloader.poll_once()
    assert reloader.take_ready() is None and active.version == 1
    fake_snapshot(tmp_path, 2)
    reloader.load_policy = Mock(side_effect=ValueError('bad checksum'))
    with pytest.raises(ValueError, match='checksum'):
        reloader.poll_once()
    assert reloader.take_ready() is None and active.version == 1
    # A pinned directory ignores the pointer, even if a newer version exists.
    pinned = PolicyReloader(str(tmp_path), loader, str(second))
    pinned.poll_once()
    assert pinned.take_ready().version == 1
    pinned.load_policy = Mock(return_value=ConstantPolicy(10))
    (second / 'deploy.pt').write_bytes(b'changed in place')
    pinned.poll_once()
    assert pinned.take_ready() is None
    pinned.load_policy.assert_not_called()


def test_slow_background_loading_does_not_block_active_policy(tmp_path):
    fake_snapshot(tmp_path, 3)
    started, release = threading.Event(), threading.Event()

    def slow_load(path):
        started.set()
        assert release.wait(5.)
        return ConstantPolicy(3)

    reloader = PolicyReloader(str(tmp_path), slow_load)
    reloader.start()
    try:
        assert started.wait(5.)
        assert reloader.take_ready() is None
        active = ConstantPolicy(2)
        for _ in range(10):
            active.step(None, None, 0)
        assert len(active.valids) == 10
    finally:
        release.set()
        reloader.close()
    assert reloader.take_ready().version == 3


def test_real_snapshot_checksums_and_motor_contract_are_checked(contract, spec, tmp_path):
    from auv_learning.deployment.snapshot import export_deploy
    from auv_learning.learning.models.encoder import HistoryEncoder
    from auv_learning.learning.models.policy import LatentPolicy
    snapshot = export_deploy(LatentPolicy(), HistoryEncoder(), True, str(tmp_path),
                             0, spec, contract.allocator)
    policy = PolicyInference(snapshot)
    policy.validate_spec(spec)
    contract.validate_policy(policy)
    for valid in (0, 20):
        assert np.isfinite(policy.step(np.zeros(21), np.zeros((20, 18)), valid)).all()
    policy.manifest['action_scales'][0] *= 2
    with pytest.raises(ValueError, match='action_scales'):
        contract.validate_policy(policy)
    with open(Path(snapshot) / 'deploy.pt', 'ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='digest'):
        PolicyInference(snapshot)


def healthy_gate():
    gate = node.ControlGate()
    gate.ready = gate.enabled = True
    gate.update_state(SimpleNamespace(connected=True, armed=True, mode='MANUAL'))
    now = time.monotonic()
    snapshot = dict(state=state_at(), time=100., received_monotonic=now)
    assert gate.reason(snapshot, now, 100.) is None
    return gate, snapshot, now


@pytest.mark.parametrize('field,value', [('armed', False), ('connected', False), ('mode', 'ALT_HOLD')])
def test_fcu_conditions_gate_collection(field, value):
    gate, snapshot, now = healthy_gate()
    setattr(gate.state, field, value)
    assert 'vehicle must' in gate.reason(snapshot, now, 100.)


def test_bridge_disable_stale_replayed_and_backwards_odometry():
    gate, snapshot, now = healthy_gate()
    gate.enabled = False
    assert 'disabled' in gate.reason(snapshot, now, 100.)
    gate.enabled = True
    assert 'receive timeout' in gate.reason(snapshot, now + .21, 100.21)
    # Replayed messages with fresh receipt times cannot hide a frozen stamp.
    snapshot['received_monotonic'] = now + .21
    assert 'timestamp stalled' in gate.reason(snapshot, now + .21, 100.21)
    snapshot.update(time=99., received_monotonic=now + .22)
    assert 'backwards' in gate.reason(snapshot, now + .22, 100.22)


def test_gate_does_not_mistake_concurrent_older_snapshot_for_rollback():
    gate, snapshot, now = healthy_gate()
    newer = dict(state=state_at(), time=100.05, received_monotonic=now + .05)
    assert gate.reason(newer, now + .05, 100.05) is None
    assert gate.reason(snapshot, now + .06, 100.06) is None


def test_gate_reports_header_delay_separately_from_callback_age():
    gate, snapshot, now = healthy_gate()
    reason = gate.reason(snapshot, now, 100.3)
    assert 'header is stale' in reason and 'header_age=0.300s callback_age=0.000s' in reason


def test_gate_can_reestablish_clock_baseline_only_when_startup_allows_it():
    gate, snapshot, now = healthy_gate()
    snapshot.update(time=10., received_monotonic=now + .05)
    assert 'backwards' in gate.reason(snapshot, now + .05, 10., allow_time_reset=True)
    snapshot.update(time=10.05, received_monotonic=now + .1)
    assert gate.reason(snapshot, now + .1, 10.05) is None
    snapshot.update(time=9., received_monotonic=now + .15)
    assert 'backwards' in gate.reason(snapshot, now + .15, 9.)
    snapshot.update(time=9.05, received_monotonic=now + .2)
    assert 'backwards' in gate.reason(snapshot, now + .2, 9.05)


@pytest.mark.parametrize('failure', ['not_ready', 'heartbeat', 'zero_quaternion', 'nan_state'])
def test_gate_rejects_missing_readiness_heartbeat_and_invalid_state(failure):
    gate, snapshot, now = healthy_gate()
    if failure == 'not_ready':
        gate.ready = False
    elif failure == 'heartbeat':
        gate.state_received = now - 4.
    elif failure == 'zero_quaternion':
        snapshot['state'][6:10] = 0.
    else:
        snapshot['state'][0] = np.nan
    assert gate.reason(snapshot, now, 100.) is not None


def test_watchdog_neutralizes_stalled_control_and_rejects_late_action():
    neutral = threading.Event()
    watchdog = node.CollectionWatchdog(lambda: None, neutral.set, timeout=.04)
    try:
        assert neutral.wait(2.)
        send = Mock()
        with pytest.raises(RuntimeError, match='interval exceeded'):
            watchdog.publish(np.ones(5), send)
        send.assert_not_called()
    finally:
        watchdog.close()


def test_watchdog_neutralizes_when_gate_or_writer_fails():
    neutral = threading.Event()
    watchdog = node.CollectionWatchdog(lambda: 'bridge disabled', neutral.set)
    try:
        assert neutral.wait(2.)
        assert watchdog.fault == 'bridge disabled'
    finally:
        watchdog.close()


@pytest.mark.parametrize('broken_detail', [False, True])
def test_watchdog_cannot_hide_a_slow_publish_and_reports_actual_gap(monkeypatch, broken_detail):
    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr(node, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    neutral, check, send = Mock(), Mock(return_value=None), Mock()

    def detail():
        if broken_detail:
            raise RuntimeError('diagnostic failure')
        return 'phase=command_publish'

    watchdog = node.CollectionWatchdog(check, neutral, detail=detail)
    try:
        def slow_send(action):
            send(action)
            clock.now += .101

        with pytest.raises(RuntimeError, match=r'actual=0.101s'):
            watchdog.publish(np.zeros(5), slow_send)
        assert ('phase=command_publish' in watchdog.fault) is not broken_detail
        neutral.assert_called_once()
        with pytest.raises(RuntimeError, match=r'actual=0.101s'):
            watchdog.publish(np.zeros(5), send)
        send.assert_called_once()
    finally:
        watchdog.close()



def test_bridge_enable_callback_publishes_status_and_neutral():
    module_spec = importlib.util.spec_from_file_location(
        'six_dof_test', PACKAGE / 'scripts/control/set_6dof_cmd.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    bridge = object.__new__(module.SixDofCmdNode)
    bridge.enabled_pub, bridge._publish_neutral = Mock(), Mock()
    response = bridge.enable_cb(SimpleNamespace(data=False))
    assert response.success and not bridge.enabled
    assert bridge.enabled_pub.publish.call_args[0][0].data is False
    bridge._publish_neutral.assert_called_once()


def test_collect_dispatch_preserves_external_and_internal_entrypoints(spec):
    from vuram_policy import policy_inference
    with patch.object(policy_inference.rospy, 'get_param',
                      side_effect=lambda key, default=None: 'collect' if key == '~ref_source' else default), \
         patch.object(policy_node, 'run_collection_node') as run:
        policy_inference.run_policy_node('', str(PACKAGE / 'config/obs_act.yaml'), episode_len_s=30.)
        assert run.call_args[0][0] == '' and run.call_args[0][2] == 30.
    with patch.object(policy_inference.rospy, 'get_param', side_effect=lambda key, default=None: default), \
         patch.object(policy_inference, 'PolicyInference') as policy:
        policy.return_value.loaded = False
        policy_inference.run_policy_node('', str(PACKAGE / 'config/obs_act.yaml'))
        assert policy.call_args[0][0].endswith('models/policy_v000012')


@pytest.mark.parametrize('save_data', [True, False])
@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
@pytest.mark.parametrize('disconnect,check_imu,odom_hz,imu_hz,rollback', [
    (False, False, 20., 20., False), (False, True, 20., 20., False),
    (True, True, 20., 20., False), (False, True, 4., 30., False),
    (False, True, 30., 4., False), (False, True, 20., 20., True),
])
def test_runtime_waits_for_arming_switches_only_between_laps_and_drains_h5(contract, spec,
                                                                         tmp_path, monkeypatch,
                                                                         disconnect, check_imu,
                                                                         odom_hz, imu_hz, rollback,
                                                                         save_data, controller_kind):
    from vuram_policy import policy_inference
    commands, zeros, errors = [], [], []
    state = SimpleNamespace(connected=True, armed=False, mode='MANUAL')
    policies = [ConstantPolicy(7, .1), ConstantPolicy(8, -.1)]
    callbacks = {}
    configured = []
    monkeypatch.setattr(policy_inference, 'configure_inference_threads',
                        lambda value: configured.append((threading.get_ident(), value)))

    class Clock:
        now = 0.

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(node, 'time', clock)
    monkeypatch.setattr(mavros_startup, 'time', clock)

    class Pose:
        sub = Mock()
        samples = 0

        def __init__(self, *args, on_sample=None, **kwargs):
            self.on_sample = on_sample
            self.snapshot = None
            self.last_odom = self.last_imu = -float('inf')

        def step(self):
            self.samples += 1
            if self.samples >= 3:
                state.armed = True
            if disconnect and len(commands) >= 2:
                state.connected = False
            callbacks['state'](state)
            if clock.now - self.last_odom + 1e-9 >= 1. / odom_hz:
                self.last_odom = clock.now
                if rollback and len(commands) == 4:
                    # A backwards callback immediately followed by recovery
                    # between control ticks must still prevent terminal export.
                    self.on_sample(99. + clock.now, clock.now)
                self.on_sample(100. + clock.now, clock.now)
                self.snapshot = dict(state=state_at((9. + self.samples * .01, 2., -1.)),
                                     time=100. + clock.now, received_monotonic=clock.now)
            if 'imu' in callbacks and clock.now - self.last_imu + 1e-9 >= 1. / imu_hz:
                self.last_imu = clock.now
                callbacks['imu'](SimpleNamespace(header=SimpleNamespace(
                    stamp=SimpleNamespace(to_sec=lambda: 100. + clock.now))))
            return self.snapshot

    class Thrusters:
        wrench_gain = 1.

        def set_zero(self):
            zeros.append(clock.now)

        def set_wrench_action(self, action):
            assert state.armed
            commands.append((clock.now, action.copy()))

        def shutdown(self):
            self.set_zero()

    class Loader:
        initial = True
        switched = False

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            assert configured == [(threading.get_ident(), 1)]

        def close(self):
            pass

        def take_ready(self):
            if self.initial:
                self.initial = False
                return policies[0]
            if len(commands) >= 4 and not self.switched:
                self.switched = True
                return policies[1]
            return None

    class Watchdog:
        fault = None

        def __init__(self, check, neutral, **kwargs):
            self.check = check

        def publish(self, action, send):
            assert self.check() is None
            send(action)
            return 100. + clock.now

        def close(self):
            pass

    def subscribe(topic, kind, callback, **kwargs):
        if kind is node.Imu:
            callbacks['imu'] = callback
        else:
            if topic == '/mavros/state':
                callbacks['state'] = callback
                callback(state)
            else:
                callback(SimpleNamespace(data=True))
        return Mock()

    params = {'~learning_root': str(LEARNING_ROOT), '~data_dir': str(tmp_path / 'inbox'),
              '~model_dir': '', '~wait_param_list': True, '~viz': False,
              '~check_mavros_imu': check_imu, '~save_data': save_data}
    monkeypatch.setattr(node.rospy, 'get_param', lambda key, default=None: params.get(key, default))
    monkeypatch.setattr(node.rospy, 'is_shutdown', lambda: len(commands) >= 9 or clock.now > 5.)
    monkeypatch.setattr(node.rospy, 'on_shutdown', Mock())
    monkeypatch.setattr(node.rospy, 'Subscriber', subscribe)
    monkeypatch.setattr(node.rospy.Time, 'now', lambda: SimpleNamespace(to_sec=lambda: 100. + clock.now))
    monkeypatch.setattr(node.rospy, 'logerr', lambda *args: errors.append(args))
    monkeypatch.setattr(node.rospy, 'loginfo', Mock())
    monkeypatch.setattr(node.rospy, 'loginfo_throttle', Mock())
    monkeypatch.setattr(node.rospy, 'logwarn', Mock())
    monkeypatch.setattr(node.rospy, 'wait_for_service', Mock())
    pull = Mock(side_effect=[SimpleNamespace(success=False, param_received=400),
                            SimpleNamespace(success=True, param_received=1600)])
    services = Mock(return_value=pull)
    monkeypatch.setattr(node.rospy, 'ServiceProxy', services)
    monkeypatch.setattr(node, 'ThrusterInterface', lambda topic: Thrusters())
    monkeypatch.setattr(node, 'PoseState', Pose)
    monkeypatch.setattr(policy_node, 'PolicyReloader', Loader)
    monkeypatch.setattr(node, 'CollectionWatchdog', Watchdog)
    if not save_data:
        monkeypatch.setattr(node, 'EpisodeWriter', Mock(side_effect=AssertionError('writer in test mode')))
        monkeypatch.setattr(node, 'next_iteration', Mock(side_effect=AssertionError('scan in test mode')))
    if controller_kind == 'pd':
        from control.pd_controller_node import run_pd_node
        run_pd_node(spec, .2)
        assert not configured  # PD never configures Torch or loads a snapshot.
    else:
        policy_node.run_collection_node('', spec, .2)
    assert pull.call_count == 2
    assert all(call.args[1] is mavros_startup.ParamPull for call in services.call_args_list)
    if odom_hz < 19. or imu_hz < 19.:
        assert not commands and not errors
        assert not list((tmp_path / 'inbox').glob('*.h5'))
        return
    if disconnect:
        assert len(commands) == 2
        assert errors
        assert 'vehicle must' in str(errors[0][-1])
        assert not list((tmp_path / 'inbox').glob('*.h5'))
        assert zeros[-1] >= commands[-1][0]
        return
    if rollback:
        assert len(commands) == 4
        assert errors and 'timestamp moved backwards' in str(errors[0][-1])
        assert not list((tmp_path / 'inbox').glob('*.h5'))
        assert zeros[-1] >= commands[-1][0]
        return
    assert not errors
    assert len(commands) == 9
    assert commands[0][0] >= 2.  # param retry + a full second of actual odometry readiness
    np.testing.assert_allclose(np.diff([t for t, _ in commands]), .05)
    if controller_kind == 'policy':
        for _, action in commands[:4]:
            np.testing.assert_array_equal(action, np.full(5, .1))
        for _, action in commands[4:]:
            np.testing.assert_array_equal(action, np.full(5, -.1))
        assert policies[0].valids[0] == policies[1].valids[0] == 0
    else:
        assert all(np.isfinite(action).all() and np.max(np.abs(action)) <= 1. for _, action in commands)
    if not save_data:
        assert not (tmp_path / 'inbox').exists()
        node.EpisodeWriter.assert_not_called()
        node.next_iteration.assert_not_called()
        return
    files = sorted((tmp_path / 'inbox').glob('*.h5'))
    assert len(files) == 2
    bundles = [contract.bundle_type.from_hdf5(str(path)) for path in files]
    assert [b.attrs['policy_version'] for b in bundles] == ([7, 8] if controller_kind == 'policy' else [-1, -1])
    assert [b.attrs['controller_type'] for b in bundles] == [controller_kind, controller_kind]
    assert [b.n_steps for b in bundles] == [5, 5]
    np.testing.assert_array_equal(bundles[0].arrays['state'][0, 10:13],
                                  bundles[0].arrays['ref'][0, :3])
    np.testing.assert_array_equal(bundles[0].arrays['state'][-1], bundles[1].arrays['state'][0])
    np.testing.assert_array_equal(bundles[0].arrays['motor_force_pre'][-1],
                                  bundles[1].arrays['motor_force_pre'][0])
    np.testing.assert_array_equal(bundles[0].arrays['ref'], bundles[1].arrays['ref'])
    assert zeros[-1] >= commands[-1][0]


def test_launch_expansion_custom_topics_empty_model_dir_and_external_compatibility(monkeypatch):
    import rospkg
    import roslaunch
    from roslaunch import substitution_args
    paths = [str(PACKAGE.parent)] + rospkg.get_ros_paths()
    monkeypatch.setattr(substitution_args, '_rospack', rospkg.RosPack(ros_paths=paths))
    config = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(str(PACKAGE / 'launch/policy_deploy.launch'), config,
                                       argv=['enable_rviz:=false', 'dof_cmd_topic:=/custom_control',
                                             'ref_speed:=0.5', 'figure8_ratio:=0.6',
                                             'episode_len_s:=30.0', 'inference_threads:=2'],
                                       verbose=False)
    values = {name: param.value for name, param in config.params.items()}
    assert values['/policy_inference/ref_source'] == 'collect'
    assert values['/policy_inference/model_dir'] == ''
    assert values['/policy_inference/cmd_topic'] == values['/set_6dof_cmd/cmd_topic'] == '/custom_control'
    assert values['/policy_inference/enabled_topic'] == values['/set_6dof_cmd/enabled_topic']
    assert values['/policy_inference/ref_speed'] == .5
    assert values['/policy_inference/figure8_ratio'] == .6
    assert values['/policy_inference/inference_threads'] == 2
    assert values['/policy_inference/param_wait_timeout'] == 180.
    assert values['/policy_inference/check_mavros_imu'] is True
    assert values['/mavros_telemetry/mavros_telemetry_rate'] == 20.
    assert values['/policy_inference/odom_ready_window'] == 1.
    infer = next(n for n in config.nodes if n.name == 'policy_inference')
    assert infer.required and '--episode_len_s 30.0' in infer.args
    external = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(
        str(PACKAGE.parent / 'vuram_planning/launch/policy_goal_deploy.launch'), external,
        argv=['enable_rviz:=false'], verbose=False)
    assert external.params['/policy_inference/ref_source'].value == 'external'
    custom_odom = roslaunch.config.ROSLaunchConfig()
    roslaunch.xmlloader.XmlLoader().load(str(PACKAGE / 'launch/policy_deploy.launch'), custom_odom,
                                       argv=['enable_rviz:=false', 'odom_topic:=/dvl/odom'], verbose=False)
    assert custom_odom.params['/policy_inference/check_mavros_imu'].value is False


def test_bad_transport_mode_fails_before_publisher_or_training_import(monkeypatch, spec):
    monkeypatch.setattr(node.rospy, 'get_param',
                        lambda name, default=None: 'bad' if name == '~transport_mode' else default)
    with patch.object(node, 'TrainingContract') as contract, \
            patch.object(node, 'ThrusterInterface') as thrusters:
        with pytest.raises(ValueError, match='transport_mode'):
            policy_node.run_collection_node('', spec, 30.)
        contract.assert_not_called()
        thrusters.assert_not_called()


def test_local_numbering_honors_lan_receipts_after_archive_deleted(contract, spec, tmp_path):
    from auv_learning.deployment.lan_store import EpisodeStore, sha256_file
    lap, _ = record_lap(contract, spec, duration=.2)
    (tmp_path / 'source').mkdir()
    source = write_episode(*lap.finish(), str(tmp_path / 'source'))
    store = EpisodeStore(tmp_path / 'host')
    try:
        ack = store.receive(source, 'auv01', Path(source).stem, sha256_file(source))
    finally:
        store.close()
    for path in (tmp_path / 'host/inbox').glob('*.h5'):
        path.unlink()
    assert next_iteration(str(tmp_path / 'host/inbox')) == ack['iteration'] + 1
