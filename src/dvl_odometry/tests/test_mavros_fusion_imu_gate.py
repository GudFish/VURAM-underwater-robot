"""Exercise the public-output IMU gate without a ROS master or EKF process."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import Mock

import numpy as np
import pytest
import rospy
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


NODE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'mavros_dvl_fusion_node.py'
NODE_SPEC = importlib.util.spec_from_file_location('fusion_imu_gate_under_test', NODE_PATH)
NODE_MODULE = importlib.util.module_from_spec(NODE_SPEC)
NODE_SPEC.loader.exec_module(NODE_MODULE)


def imu(stamp):
    message = Imu()
    message.header.stamp = rospy.Time.from_sec(stamp)
    message.header.frame_id = 'base_link'
    message.orientation.w = 1.
    message.linear_acceleration.z = 9.80665
    message.orientation_covariance = np.diag([.03 ** 2] * 3).ravel().tolist()
    message.angular_velocity_covariance = np.diag([.01 ** 2] * 3).ravel().tolist()
    message.linear_acceleration_covariance = np.diag([.1 ** 2] * 3).ravel().tolist()
    return message


@pytest.fixture
def rig(monkeypatch):
    clock = SimpleNamespace(now=10.)
    monkeypatch.setattr(rospy.Time, 'now', staticmethod(lambda: rospy.Time.from_sec(clock.now)))
    monkeypatch.setattr(rospy, 'is_shutdown', lambda: False)
    monkeypatch.setattr(rospy.core, 'is_shutdown_requested', lambda: False)
    warnings = []
    monkeypatch.setattr(rospy, 'logwarn_throttle',
                        lambda _period, message, *args: warnings.append(message % args))
    node = object.__new__(NODE_MODULE.MavrosDvlFusion)
    node.lock = threading.RLock()
    node.output = SimpleNamespace(robot_frame='base_link', publish=Mock())
    node.imu_timeout = .5
    node.imu_future_tolerance = .02
    node.vel_timeout = 1.
    node.imu = imu(9.95)
    node.velocity = None
    node.vel_valid = False
    node.vel_invalid_at = 0.
    node.last_output = None
    node.started = True
    node.suspended = False
    node.minimum_output_stamp = 0.
    node.last_imu_stamp = node.imu.header.stamp.to_sec()
    node.imu_pub = Mock()
    node.velocity_pub = Mock()
    node.restore_pose = Mock(return_value=True)
    return SimpleNamespace(node=node, clock=clock, warnings=warnings)


def public_odom(stamp=10.):
    message = Odometry()
    message.header.stamp = rospy.Time.from_sec(stamp)
    return message


def dvl_velocity(stamp):
    message = TwistWithCovarianceStamped()
    message.header.stamp = rospy.Time.from_sec(stamp)
    return message


@pytest.mark.parametrize('stamp, reason', [(10.03, 'future'), (9.4, 'stale')])
def test_bad_timestamp_keeps_recent_good_imu_and_public_output(rig, stamp, reason):
    node, previous = rig.node, rig.node.imu
    rejected = imu(stamp)

    assert not node.fresh(rejected, node.imu_timeout)
    node.on_imu(rejected)

    assert node.imu is previous
    assert node.last_imu_stamp == previous.header.stamp.to_sec()
    assert not node.suspended
    node.imu_pub.publish.assert_not_called()
    node.restore_pose.assert_not_called()
    node.on_output(public_odom())
    node.output.publish.assert_called_once()
    warning = rig.warnings[-1].lower()
    assert reason in warning
    for field in ('header_age=', 'last_good_age=', 'limit='):
        assert field in warning


@pytest.mark.parametrize('field, value', [('orientation_covariance', -1.),
                                        ('angular_velocity_covariance', float('nan')),
                                        ('linear_acceleration_covariance', -1.)])
def test_bad_covariance_does_not_erase_good_imu(rig, field, value):
    node, previous = rig.node, rig.node.imu
    rejected = imu(9.99)
    covariance = list(getattr(rejected, field))
    covariance[0] = value
    setattr(rejected, field, covariance)

    node.on_imu(rejected)

    assert node.imu is previous
    assert not node.suspended
    node.imu_pub.publish.assert_not_called()
    node.restore_pose.assert_not_called()


@pytest.mark.parametrize('bad_stamp', [10.03, 9.4])
def test_good_packet_after_single_bad_packet_does_not_reset_pose(rig, bad_stamp):
    node = rig.node
    node.on_imu(imu(bad_stamp))
    accepted = imu(9.99)

    node.on_imu(accepted)

    assert not node.suspended
    assert node.imu.header.stamp == accepted.header.stamp
    node.restore_pose.assert_not_called()
    node.imu_pub.publish.assert_called_once_with(node.imu)


def test_repeated_bad_packets_cannot_keep_expired_good_imu_alive(rig):
    node, previous = rig.node, rig.node.imu
    for now in (10.1, 10.3, 10.44):
        rig.clock.now = now
        node.on_imu(imu(now + .03))
        assert not node.suspended
        assert node.imu is previous
        node.on_output(public_odom(now))
    assert node.output.publish.call_count == 3

    rig.clock.now = 10.451
    node.on_imu(imu(10.481))

    assert node.suspended
    assert node.imu is previous
    assert node.last_imu_stamp == previous.header.stamp.to_sec()
    node.on_output(public_odom(rig.clock.now))
    assert node.output.publish.call_count == 3
    node.imu_pub.publish.assert_not_called()
    node.restore_pose.assert_not_called()


def test_watchdog_pauses_output_after_true_imu_outage(rig):
    node = rig.node
    rig.clock.now = 10.45  # The existing 0.5-second bound remains inclusive.
    node.watchdog(None)
    assert not node.suspended
    node.on_output(public_odom(rig.clock.now))
    node.output.publish.assert_called_once()

    rig.clock.now = 10.451
    node.watchdog(None)

    assert node.suspended
    node.on_output(public_odom(rig.clock.now))
    node.output.publish.assert_called_once()
    node.restore_pose.assert_not_called()


def test_bad_packet_without_any_good_sample_suspends_initialized_gate(rig):
    node = rig.node
    node.imu = None
    node.last_imu_stamp = None

    node.on_imu(imu(10.03))

    assert node.suspended
    assert node.imu is None
    node.imu_pub.publish.assert_not_called()


@pytest.mark.parametrize('trigger', ['bad_packet', 'watchdog', 'next_good_packet'])
def test_first_good_packet_after_true_outage_restores_pose_before_forwarding(rig, trigger):
    node = rig.node
    rig.clock.now = 10.6
    if trigger == 'bad_packet':
        node.on_imu(imu(10.63))
    elif trigger == 'watchdog':
        node.watchdog(None)
    accepted = imu(10.59)
    events = []
    node.restore_pose.side_effect = lambda **_kwargs: events.append('restore') or True
    node.imu_pub.publish.side_effect = lambda _message: events.append('publish')

    node.on_imu(accepted)

    assert events == ['restore', 'publish']
    node.restore_pose.assert_called_once_with(zero_position=False, stamp=accepted.header.stamp)
    assert not node.suspended
    assert node.imu.header.stamp == accepted.header.stamp
    node.on_output(public_odom(rig.clock.now))
    node.output.publish.assert_called_once()


def test_failed_outage_recovery_holds_output_until_pose_restored(rig):
    node = rig.node
    rig.clock.now = 10.6
    node.watchdog(None)
    node.restore_pose.return_value = False

    node.on_imu(imu(10.59))

    assert node.suspended
    node.imu_pub.publish.assert_not_called()
    node.on_output(public_odom(rig.clock.now))
    node.output.publish.assert_not_called()
    node.restore_pose.return_value = True
    node.on_imu(imu(10.6))
    assert not node.suspended
    assert node.restore_pose.call_count == 2
    node.imu_pub.publish.assert_called_once()


@pytest.mark.parametrize('stamp', [10.03, 9.4])
def test_rejected_first_sample_does_not_initialize_ekf(rig, stamp):
    node = rig.node
    node.started = False
    node.imu = None
    node.last_imu_stamp = None

    node.on_imu(imu(stamp))
    node.watchdog(None)

    assert not node.started
    assert not node.suspended
    assert node.imu is None
    node.imu_pub.publish.assert_not_called()
    node.velocity_pub.publish.assert_not_called()
    node.restore_pose.assert_not_called()


@pytest.mark.parametrize('stamp', [9.95, 9.94])
def test_duplicate_or_older_valid_sample_does_not_replace_recent_good_imu(rig, stamp):
    node, previous = rig.node, rig.node.imu

    node.on_imu(imu(stamp))

    assert node.imu is previous
    assert not node.suspended
    node.imu_pub.publish.assert_not_called()
    node.restore_pose.assert_not_called()


@pytest.mark.parametrize('lead', [.003, .007, .02])
def test_small_future_imu_keeps_all_public_gates_open_without_retimestamping(rig, lead):
    node = rig.node
    accepted = imu(rig.clock.now+lead)
    stamp = accepted.header.stamp
    node.diagnostics = Mock()

    node.on_imu(accepted)
    node.watchdog(None)
    node.on_output(public_odom())
    node.on_velocity(dvl_velocity(9.99))
    node.diagnose(None)
    assert node.reset(None).success

    assert node.imu_fresh()
    assert node.imu.header.stamp == stamp
    assert accepted.header.stamp == stamp
    assert not node.suspended
    assert not rig.warnings
    node.imu_pub.publish.assert_called_once_with(node.imu)
    node.output.publish.assert_called_once()
    node.velocity_pub.publish.assert_called_once()
    node.diagnostics.send.assert_called_once()
    assert node.diagnostics.send.call_args.kwargs['imu_fresh'] is True
    # Only the explicit reset service, not receipt of a future-dated IMU,
    # is allowed to call SetPose in this healthy sequence.
    node.restore_pose.assert_called_once_with(True, rospy.Time.now())


def test_small_future_imu_initializes_with_valid_dvl(rig):
    node = rig.node
    node.started = False
    node.imu = None
    node.last_imu_stamp = None
    node.velocity = dvl_velocity(9.99)
    node.vel_valid = True

    node.on_imu(imu(10.005))

    assert node.started
    node.imu_pub.publish.assert_called_once_with(node.imu)
    node.velocity_pub.publish.assert_called_once_with(node.velocity)
    node.restore_pose.assert_not_called()


def test_sustained_small_future_imus_do_not_freeze_or_reset_ekf(rig):
    node = rig.node
    for index in range(200):
        rig.clock.now = 10.+index*.01
        node.on_imu(imu(rig.clock.now+.005))
        node.watchdog(None)
        node.on_output(public_odom(rig.clock.now))
        assert not node.suspended
    assert node.imu_pub.publish.call_count == 200
    assert node.output.publish.call_count == 200
    node.restore_pose.assert_not_called()
    assert not rig.warnings


def test_small_future_imu_still_expires_and_restores_pose_after_real_outage(rig):
    node = rig.node
    node.on_imu(imu(10.005))
    rig.clock.now = 10.506
    node.watchdog(None)
    node.on_output(public_odom(rig.clock.now))
    assert node.suspended
    node.output.publish.assert_not_called()

    recovered = imu(10.511)
    node.on_imu(recovered)
    node.on_output(public_odom(rig.clock.now))
    assert not node.suspended
    node.restore_pose.assert_called_once_with(zero_position=False, stamp=recovered.header.stamp)
    node.output.publish.assert_called_once()


@pytest.mark.parametrize('lead', [.020001, .1, 2.])
def test_future_imu_outside_tolerance_is_not_forwarded(rig, lead):
    node, previous = rig.node, rig.node.imu
    node.on_imu(imu(rig.clock.now+lead))
    assert node.imu is previous
    assert not node.suspended
    node.imu_pub.publish.assert_not_called()
    assert 'future IMU timestamp' in rig.warnings[-1]


def test_zero_tolerance_restores_strict_imu_clock_check(rig):
    node, previous = rig.node, rig.node.imu
    node.imu_future_tolerance = 0.
    node.on_imu(imu(10.001))
    assert node.imu is previous
    node.imu_pub.publish.assert_not_called()


def test_imu_clock_tolerance_does_not_accept_future_dvl_velocity(rig):
    node = rig.node
    node.on_velocity(dvl_velocity(10.005))
    assert node.velocity is None
    assert not node.vel_valid
    node.velocity_pub.publish.assert_not_called()
