"""DVL deployment gates exercised through the shared ROS control lifecycle.

Callbacks, clocks and command transport are simulated; the trajectory, stream
checks, DVL health checks, PD controller and optional H5 writer stay real.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from geometry_msgs.msg import (PoseWithCovarianceStamped,
                               TwistWithCovarianceStamped)

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from control import tracking_node as node
from control.observation import load_spec
from control.pd_controller import PDController
from data_collection.contract import TrainingContract


class ConstantPolicy:
    history_shape = (20, 18)
    version = 7

    def __init__(self):
        self.valids = []

    def step(self, observation, history, valid):
        self.valids.append(valid)
        return np.full(5, .1)


@pytest.fixture
def tracking(monkeypatch, tmp_path):
    """Install a finite deterministic runtime; no rospy timer or socket runs."""
    def run(controller_kind='policy', source='dvl', mode='mavros_fused',
            startup_hold='velocity_valid', valid_at=.4,
            fault=None, fault_after=2, save_data=False):
        callbacks, subscriptions = {}, {}
        commands, neutral, errors, waits, dvl_events = [], [], [], [], []
        clock = SimpleNamespace(now=0.)
        state = SimpleNamespace(connected=True, armed=True, mode='MANUAL')
        injected = False
        last_sample = None
        data_dir = tmp_path / 'inbox'
        spec = load_spec(str(PACKAGE / 'config/obs_act.yaml'))
        params = {
            '~odom_source': source, '~dvl_mode': mode, '~dvl_timeout': .15,
            '~odom_topic': '/dvl/odom' if source == 'dvl' else '/mavros/local_position/odom',
            '~odom_ready_window': .2, '~check_mavros_imu': False,
            '~wait_param_list': False, '~save_data': save_data,
            '~data_dir': str(data_dir), '~viz': False,
        }
        if save_data:
            learning_root = Path(os.environ.get('AUV_LEARNING_ROOT', '~/mujoco_auv_env')).expanduser()
            if not (learning_root / 'auv_learning').is_dir():
                pytest.skip('set AUV_LEARNING_ROOT to the training checkout')
            contract = TrainingContract(learning_root)
        else:
            # Test mode still advances nominal motor state. It needs no reward
            # function, model snapshot, training checkout or episode bundle.
            contract = SimpleNamespace(allocator=SimpleNamespace(
                scales=np.full(5, 100.), negative_scales=np.full(5, 110.), step=Mock()))
        monkeypatch.setattr(node, 'TrainingContract', lambda root: contract)
        controller = PDController(contract.allocator) if controller_kind == 'pd' else ConstantPolicy()
        if controller_kind == 'pd':
            controller.validate_spec(spec)
        loader = node.StaticControllerSource(controller)

        def deliver(topic, message):
            if topic in callbacks:
                callbacks[topic](message)
                if topic.startswith('/dvl/'):
                    dvl_events.append((clock.now, topic, getattr(message, 'data', None)))

        def boolean(topic, value):
            deliver(topic, SimpleNamespace(data=value))

        def healthy_samples():
            nonlocal last_sample, injected
            if last_sample == clock.now:
                return
            last_sample = clock.now
            valid = clock.now + 1e-9 >= valid_at
            boolean('/dvl/connected', valid or startup_hold != 'connected')
            boolean('/dvl/resetting', not valid and startup_hold == 'resetting')
            boolean('/dvl/velocity_valid', valid or startup_hold != 'velocity_valid')
            boolean('/dvl/dr_valid', valid or startup_hold != 'dr_valid')
            boolean('/dvl/angular_velocity_valid',
                    valid or startup_hold != 'angular_velocity_valid')
            velocity_stopped = fault == 'velocity_timeout' and len(commands) >= fault_after
            if (valid or startup_hold != 'velocity_missing') and not velocity_stopped:
                velocity = TwistWithCovarianceStamped()
                velocity.header.stamp = node.rospy.Time.from_sec(100. + clock.now)
                velocity.twist.covariance = (np.eye(6) * .01).ravel().tolist()
                deliver('/dvl/velocity', velocity)
            if valid or startup_hold != 'dr_missing':
                pose = PoseWithCovarianceStamped()
                pose.header.stamp = node.rospy.Time.from_sec(100. + clock.now)
                pose.pose.pose.orientation.w = 1.
                pose.pose.covariance = (np.eye(6) * .01).ravel().tolist()
                deliver('/dvl/dead_reckoning', pose)
            if fault and fault != 'velocity_timeout' and not injected and len(commands) >= fault_after:
                injected = True
                # The next healthy callback has already replaced the false
                # status before either the main loop or watchdog reads it.
                # A health check that only retains the last Bool is unsafe.
                topic = '/dvl/' + fault
                bad, good = (True, False) if fault == 'resetting' else (False, True)
                boolean(topic, bad)
                boolean(topic, good)

        def sleep(seconds):
            assert seconds >= 0.
            clock.now += seconds
            healthy_samples()

        def subscribe(topic, kind, callback, **kwargs):
            callbacks[topic] = callback
            subscriptions[topic] = Mock()
            if topic == '/mavros/state':
                callback(state)
            elif not topic.startswith('/dvl/'):
                callback(SimpleNamespace(data=True))
            return subscriptions[topic]

        class Pose:
            def __init__(self, *args, on_sample=None, **kwargs):
                self.on_sample, self.sub = on_sample, Mock()
                self.last_tick, self.snapshot = None, None

            def step(self):
                deliver('/mavros/state', state)
                healthy_samples()
                if self.last_tick != clock.now:
                    self.last_tick = clock.now
                    self.on_sample(100. + clock.now, clock.now)
                    self.snapshot = dict(
                        state=np.r_[np.zeros(6), [1., 0., 0., 0.], [2., -1., -.4]],
                        time=100. + clock.now, received_monotonic=clock.now)
                return self.snapshot

        class Thrusters:
            wrench_gain = 1.

            def set_zero(self):
                neutral.append(clock.now)

            def set_wrench_action(self, action):
                commands.append((clock.now, action.copy()))

            def shutdown(self):
                self.set_zero()

        class Watchdog:
            fault = None

            def __init__(self, check, neutral, **kwargs):
                self.check, self.neutral = check, neutral

            def publish(self, action, send):
                self.fault = self.fault or self.check()
                if self.fault:
                    self.neutral()
                    raise RuntimeError(self.fault)
                send(action)
                return 100. + clock.now

            def close(self):
                pass

        # Keep real rospy.Time serialization for ROS message headers, but
        # replace every runtime operation that could contact ROS or a device.
        import rospy
        ros_time = SimpleNamespace(
            from_sec=rospy.Time.from_sec,
            now=lambda: SimpleNamespace(to_sec=lambda: 100. + clock.now))
        ros = SimpleNamespace(
            get_param=lambda key, default=None: params.get(key, default),
            is_shutdown=lambda: len(commands) >= 9 or clock.now > 2.,
            on_shutdown=Mock(), Subscriber=subscribe, Time=ros_time,
            loginfo=Mock(), logwarn=Mock(),
            loginfo_throttle=lambda *args: waits.append(args),
            logerr=lambda *args: errors.append(args))
        monkeypatch.setattr(node, 'rospy', ros)
        monkeypatch.setattr(node, 'time', SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
        monkeypatch.setattr(node, 'PoseState', Pose)
        monkeypatch.setattr(node, 'ThrusterInterface', lambda topic: Thrusters())
        monkeypatch.setattr(node, 'CollectionWatchdog', Watchdog)
        visualizer = Mock()
        monkeypatch.setattr(node, 'TrajVisualizer', lambda *args, **kwargs: visualizer)
        if not save_data:
            monkeypatch.setattr(node, 'EpisodeWriter', Mock(side_effect=AssertionError('writer in test mode')))
            monkeypatch.setattr(node, 'next_iteration', Mock(side_effect=AssertionError('scan in test mode')))
        node.run_tracking_node(spec, .2, lambda contract, model_root: loader)
        assert all(subscription.unregister.call_count == 1 for subscription in subscriptions.values())
        return SimpleNamespace(commands=commands, neutral=neutral, errors=errors,
                               waits=waits, dvl_events=dvl_events, subscriptions=subscriptions,
                               data_dir=data_dir, controller=controller, viz=visualizer,
                               contract=contract)
    return run


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
@pytest.mark.parametrize('mode', ['mavros_fused', 'dvl_raw', 'dvl_fused'])
@pytest.mark.parametrize('startup_hold', [
    'connected', 'resetting', 'velocity_valid', 'velocity_missing',
])
def test_dvl_readiness_holds_neutral_then_tracks_across_laps(tracking, controller_kind,
                                                            mode, startup_hold):
    result = tracking(controller_kind=controller_kind, mode=mode, startup_hold=startup_hold)
    assert not result.errors
    assert len(result.commands) == 9
    assert result.commands[0][0] >= .4 - 1e-9
    assert any(t < result.commands[0][0] for t in result.neutral)
    np.testing.assert_allclose(np.diff([t for t, _ in result.commands]), .05, atol=1e-12)
    assert all(action.shape == (5,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.
               for _, action in result.commands)
    assert result.viz.start_episode.call_count == 3
    assert not result.data_dir.exists()
    if controller_kind == 'policy':
        # Terminal state uses a controller inference without issuing an action.
        assert result.controller.valids == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]
    dr_topics = {'/dvl/dr_valid', '/dvl/dead_reckoning'}
    if mode == 'mavros_fused':
        assert not dr_topics.intersection(result.subscriptions)
    else:
        assert dr_topics.issubset(result.subscriptions)
    assert ('/dvl/angular_velocity_valid' in result.subscriptions) == (mode == 'dvl_fused')


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
@pytest.mark.parametrize('mode', ['dvl_raw', 'dvl_fused'])
@pytest.mark.parametrize('startup_hold', ['dr_valid', 'dr_missing'])
def test_dead_reckoning_modes_require_valid_pose_before_first_command(
        tracking, controller_kind, mode, startup_hold):
    result = tracking(controller_kind=controller_kind, mode=mode, startup_hold=startup_hold)
    assert not result.errors and len(result.commands) == 9
    assert result.commands[0][0] >= .4 - 1e-9


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
def test_device_fusion_waits_for_valid_angular_velocity_before_tracking(
        tracking, controller_kind):
    result = tracking(controller_kind=controller_kind, mode='dvl_fused',
                      startup_hold='angular_velocity_valid')
    assert not result.errors and len(result.commands) == 9
    assert result.commands[0][0] >= .4 - 1e-9
    assert any('angular velocity estimate is invalid' in str(wait) for wait in result.waits)


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
@pytest.mark.parametrize('mode,fault', [
    ('mavros_fused', 'velocity_valid'), ('dvl_raw', 'velocity_valid'),
    ('dvl_fused', 'velocity_valid'), ('mavros_fused', 'connected'),
    ('mavros_fused', 'resetting'), ('dvl_raw', 'dr_valid'),
    ('dvl_fused', 'dr_valid'), ('dvl_fused', 'angular_velocity_valid'),
])
def test_transient_dvl_fault_latches_despite_healthy_high_rate_odometry(
        tracking, controller_kind, mode, fault):
    result = tracking(controller_kind=controller_kind, mode=mode, fault=fault)
    assert len(result.commands) == 2
    assert result.errors and 'dvl' in str(result.errors[0]).lower()
    assert result.neutral[-1] > result.commands[-1][0]
    assert not result.data_dir.exists()
    values = [value for _, topic, value in result.dvl_events if topic == '/dvl/' + fault]
    assert values[-2:] == ([True, False] if fault == 'resetting' else [False, True])


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
def test_mavros_source_does_not_subscribe_or_wait_for_dvl(tracking, controller_kind):
    result = tracking(controller_kind=controller_kind, source='mavros', valid_at=50.)
    assert not result.errors and len(result.commands) == 9
    assert result.commands[0][0] < .4
    assert not any(topic.startswith('/dvl/') for topic in result.subscriptions)


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
@pytest.mark.parametrize('mode', ['mavros_fused', 'dvl_raw', 'dvl_fused'])
def test_dvl_velocity_timeout_stops_even_when_fused_odometry_keeps_updating(
        tracking, controller_kind, mode):
    result = tracking(controller_kind=controller_kind, mode=mode, fault='velocity_timeout')
    assert 4 <= len(result.commands) <= 6
    assert result.errors and 'dvl' in str(result.errors[0]).lower()
    last_velocity = max(t for t, topic, _ in result.dvl_events if topic == '/dvl/velocity')
    assert result.neutral[-1] - last_velocity > .15
    assert result.commands[-1][0] - last_velocity <= .15 + 1e-9
    assert not result.data_dir.exists()


@pytest.mark.parametrize('controller_kind', ['policy', 'pd'])
def test_dvl_fault_preserves_completed_h5_and_discards_current_lap(tracking, controller_kind):
    result = tracking(controller_kind=controller_kind, mode='dvl_fused',
                      fault='velocity_valid', fault_after=6, save_data=True)
    assert len(result.commands) == 6
    assert result.errors and 'dvl' in str(result.errors[0]).lower()
    files = list(result.data_dir.glob('*.h5'))
    assert len(files) == 1
    assert not list(result.data_dir.glob('*.tmp-*'))
    bundle = result.contract.bundle_type.from_hdf5(str(files[0]))
    assert bundle.attrs['schema_version'] == 4
    assert bundle.attrs['controller_type'] == controller_kind
    assert bundle.attrs['odom_source'] == 'dvl'
    assert bundle.attrs['dvl_mode'] == 'dvl_fused'
    assert bundle.attrs['angular_velocity_source'] == 'dvl_dr_quaternion_difference_lowpass'
    assert bundle.n_steps == 5
    assert bundle.arrays['done'].ravel().tolist() == [0, 0, 0, 0, 1]
    with h5py.File(files[0]) as handle:
        assert handle['episode/action_applied'][:, 0].tolist() == [1, 1, 1, 1, 0]
        assert handle['episode/dof_pwm'].shape == (5, 6)
        assert handle['episode/odom_stamp'].shape == (5, 1)
