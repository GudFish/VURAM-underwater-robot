"""Actual DVL input health is independent of fused odometry output rate."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from control.dvl_health import DvlHealth


def healthy(mode='mavros_fused', timeout=1.):
    gate = DvlHealth(mode=mode, timeout=timeout, now=lambda: 20.)
    gate.update_connected(True)
    gate.update_resetting(False)
    gate.update_velocity_valid(True)
    gate.update_dr_valid(True)
    gate.update_angular_velocity_valid(True)
    gate.update_velocity(100., 20.)
    gate.update_dr(100., 20.)
    assert gate.reason(20., 100.) is None
    return gate


@pytest.mark.parametrize('timeout', [0., -1., float('nan'), float('inf')])
def test_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match='positive and finite'):
        DvlHealth(timeout=timeout)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='unknown DVL mode'):
        DvlHealth(mode='odometry')


def test_waits_for_explicit_status_and_actual_sample():
    gate = DvlHealth(mode='mavros_fused', now=lambda: 20.)
    assert 'connected' in gate.reason(20., 100.)
    gate.update_connected(True)
    assert 'resetting' in gate.reason(20., 100.)
    gate.update_resetting(False)
    assert 'velocity_valid' in gate.reason(20., 100.)
    gate.update_velocity_valid(True)
    for _ in range(10):
        gate.update_velocity_valid(True)  # Cached Bool does not update sample age.
    assert 'actual DVL velocity sample' in gate.reason(20., 100.)
    gate.update_velocity(100.)  # Uses the injected monotonic clock.
    assert gate.reason(20., 100.) is None


def test_startup_can_recover_without_arming():
    gate = healthy()
    gate.update_connected(False)
    assert 'disconnected' in gate.reason(20., 100.)
    gate.update_connected(True)
    assert gate.reason(20., 100.) is None
    assert 'timeout' in gate.reason(22., 102.)
    gate.update_velocity(102., 22.)
    assert gate.reason(22., 102.) is None


@pytest.mark.parametrize('method, bad, good, text', [
    ('update_connected', False, True, 'disconnected'),
    ('update_resetting', True, False, 'reset'),
    ('update_velocity_valid', False, True, 'velocity is invalid'),
])
def test_short_status_fault_is_latched_between_control_checks(method, bad, good, text):
    gate = healthy()
    gate.arm()
    getattr(gate, method)(bad)
    getattr(gate, method)(good)
    gate.update_velocity(100.1, 20.1)
    assert text in gate.reason(20.1, 100.1)
    gate.arm()  # A second arm is not a reset.
    assert text in gate.reason(20.1, 100.1)


@pytest.mark.parametrize('now, ros_now, text', [
    (21.01, 100., 'receive timeout'),
    (20., 101.01, 'source timestamp is stale'),
    (20., 98.99, 'ahead of ROS clock'),
    (18.99, 100., 'ahead of monotonic clock'),
])
def test_source_and_receipt_freshness_are_checked_and_latched(now, ros_now, text):
    gate = healthy()
    gate.enable()
    assert text in gate.reason(now, ros_now)
    gate.update_velocity(200., 50.)
    assert text in gate.reason(50., 200.)


def test_timeout_boundary_is_inclusive():
    gate = healthy(timeout=.2)
    assert gate.reason(20.2, 100.2) is None
    assert gate.reason(20., 99.8) is None
    assert 'timeout' in gate.reason(20.20001, 100.2)


@pytest.mark.parametrize('mode', ['dvl_raw', 'dvl_fused'])
def test_device_modes_require_actual_fresh_valid_dead_reckoning(mode):
    gate = DvlHealth(mode=mode, now=lambda: 20.)
    gate.update_connected(True)
    gate.update_resetting(False)
    gate.update_velocity_valid(True)
    gate.update_velocity(100., 20.)
    gate.update_angular_velocity_valid(True)
    assert 'dead-reckoning validity' in gate.reason(20., 100.)
    gate.update_dr_valid(True)
    assert 'actual DVL dr sample' in gate.reason(20., 100.)
    gate.update_dr(98., 20.)
    assert 'dr source timestamp is stale' in gate.reason(20., 100.)
    gate.update_dr(100., 18.)
    assert 'dr receive timeout' in gate.reason(20., 100.)
    gate.update_dr(100., 20.)
    assert gate.reason(20., 100.) is None
    gate.arm()
    gate.update_dr_valid(False)
    gate.update_dr_valid(True)
    assert 'dead-reckoning is invalid' in gate.reason(20., 100.)


def test_mavros_mode_is_independent_of_device_dr():
    gate = DvlHealth(mode='mavros_fused', now=lambda: 20.)
    gate.update_connected(True)
    gate.update_resetting(False)
    gate.update_velocity_valid(True)
    gate.update_velocity(100.)
    assert gate.reason(20., 100.) is None
    gate.arm()
    gate.update_dr_valid(False)
    gate.update_dr(float('nan'))
    assert gate.reason(20., 100.) is None


def test_default_device_fusion_waits_for_angular_velocity_validity():
    gate = DvlHealth(now=lambda: 20.)
    assert gate.mode == 'dvl_fused'
    gate.update_connected(True)
    gate.update_resetting(False)
    gate.update_velocity_valid(True)
    gate.update_dr_valid(True)
    gate.update_velocity(100.)
    gate.update_dr(100.)
    assert 'angular velocity validity status' in gate.reason(20., 100.)
    gate.update_angular_velocity_valid(False)
    assert 'angular velocity estimate is invalid' in gate.reason(20., 100.)
    gate.update_angular_velocity_valid(True)
    assert gate.reason(20., 100.) is None


def test_transient_angular_velocity_fault_is_latched_in_device_fusion():
    gate = healthy(mode='dvl_fused')
    gate.arm()
    gate.update_angular_velocity_valid(False)
    gate.update_angular_velocity_valid(True)
    assert 'angular velocity estimate is invalid' in gate.reason(20., 100.)


@pytest.mark.parametrize('mode', ['mavros_fused', 'dvl_raw'])
def test_other_modes_ignore_device_angular_velocity_validity(mode):
    gate = healthy(mode=mode)
    gate.arm()
    gate.update_angular_velocity_valid(False)
    assert gate.reason(20., 100.) is None
    gate.update_angular_velocity_valid(True)
    assert gate.reason(20., 100.) is None


@pytest.mark.parametrize('stamp, received', [
    (float('nan'), 20.), (float('inf'), 20.),
    (100., float('nan')), (100., float('inf')), ('invalid', 20.),
])
def test_invalid_timestamps_fail_without_callback_exception(stamp, received):
    gate = healthy()
    gate.update_velocity(stamp, received)
    assert 'invalid timestamp' in gate.reason(20., 100.)
    gate.update_velocity(100., 20.)
    assert gate.reason(20., 100.) is None
    gate.arm()
    gate.update_velocity(stamp, received)
    gate.update_velocity(100., 20.)
    assert 'invalid timestamp' in gate.reason(20., 100.)


def test_arming_during_unknown_status_fails_closed():
    gate = DvlHealth()
    gate.arm()
    gate.update_connected(True)
    gate.update_resetting(False)
    gate.update_velocity_valid(True)
    gate.update_velocity(100., 20.)
    assert 'connected' in gate.reason(20., 100.)


def test_nonfinite_clock_does_not_pass_health_check():
    gate = healthy()
    assert 'invalid' in gate.reason(float('nan'), 100.)
    assert gate.reason(20., 100.) is None
