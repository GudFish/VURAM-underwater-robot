"""Shared MAVROS startup helpers, without a master or vehicle connection."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from control import mavros_startup as node


@pytest.fixture
def parameter_wait_mocks(monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(node, 'time', SimpleNamespace(
        monotonic=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, 'now', clock.now + seconds)))
    monkeypatch.setattr(node.rospy, 'is_shutdown', lambda: False)
    monkeypatch.setattr(node.rospy, 'logwarn', Mock())
    monkeypatch.setattr(node.rospy, 'loginfo', Mock())
    monkeypatch.setattr(node.rospy, 'loginfo_throttle', Mock())
    available = Mock()
    monkeypatch.setattr(node.rospy, 'wait_for_service', available)
    return clock, available


def test_parameter_pull_timeout_responses_retry_same_download(parameter_wait_mocks, monkeypatch):
    clock, available = parameter_wait_mocks
    request = Mock(side_effect=[SimpleNamespace(success=False, param_received=400),
                                SimpleNamespace(success=False, param_received=900),
                                SimpleNamespace(success=True, param_received=1600)])
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=request))
    neutral = Mock()
    assert node.wait_for_parameter_list('/mavros/param/pull', neutral)
    assert request.call_count == available.call_count == 3
    assert all(call.kwargs == {'force_pull': False} for call in request.call_args_list)
    assert neutral.call_count >= 20 and clock.now >= 2.


def test_parameter_service_becoming_available_is_retried(parameter_wait_mocks, monkeypatch):
    _, available = parameter_wait_mocks
    available.side_effect = [node.rospy.ROSException('not available yet'), None]
    request = Mock(return_value=SimpleNamespace(success=True, param_received=1600))
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=request))
    assert node.wait_for_parameter_list('/mavros/param/pull', Mock())
    assert available.call_count == 2 and request.call_count == 1


def test_parameter_wait_overall_timeout_reports_progress(parameter_wait_mocks, monkeypatch):
    request = Mock(return_value=SimpleNamespace(success=False, param_received=123))
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=request))
    with pytest.raises(RuntimeError, match='within 0.5s .*123 parameters'):
        node.wait_for_parameter_list('/mavros/param/pull', Mock(), timeout=.5)
    assert request.call_count == 1


@pytest.mark.parametrize('shutdown', [False, True])
def test_parameter_wait_remains_neutral_and_cancellable_during_blocked_rpc(monkeypatch, shutdown):
    release, finished, stop = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(node.rospy, 'wait_for_service', Mock())
    monkeypatch.setattr(node.rospy, 'is_shutdown', stop.is_set)
    monkeypatch.setattr(node.rospy, 'loginfo_throttle', Mock())

    def request(**kwargs):
        try:
            if shutdown:
                stop.set()
            assert release.wait(2.)
            return SimpleNamespace(success=True, param_received=1600)
        finally:
            finished.set()

    rpc = Mock(side_effect=request)
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=rpc))
    neutral = Mock()
    started = time.monotonic()
    try:
        if shutdown:
            assert node.wait_for_parameter_list('/mavros/param/pull', neutral) is False
        else:
            with pytest.raises(RuntimeError, match='within 0.1s'):
                node.wait_for_parameter_list('/mavros/param/pull', neutral, timeout=.1)
        assert time.monotonic() - started < 1.
        assert rpc.call_count == 1
        assert neutral.call_count >= 1
    finally:
        release.set()
        assert finished.wait(1.)


@pytest.mark.parametrize('success', [True, False])
def test_telemetry_setup_requests_raw_imu_before_attitude_and_position(monkeypatch, success):
    request = Mock(return_value=SimpleNamespace(success=success))
    monkeypatch.setattr(node.rospy, 'wait_for_service', Mock())
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=request))
    monkeypatch.setattr(node.rospy, 'is_shutdown', lambda: False)
    monkeypatch.setattr(node.rospy, 'loginfo', Mock())
    monkeypatch.setattr(node.rospy, 'logwarn', Mock())
    neutral = Mock()
    assert node.configure_mavros_telemetry('/mavros/set_message_interval', 30., neutral) is success
    assert [call.kwargs for call in request.call_args_list] == [
        dict(message_id=27, message_rate=30.), dict(message_id=31, message_rate=30.),
        dict(message_id=32, message_rate=30.)]
    assert neutral.called
    assert node.rospy.logwarn.called is not success


def test_rejected_acceleration_stream_is_reported_even_if_other_requests_succeed(monkeypatch):
    request = Mock(side_effect=[SimpleNamespace(success=False),
                                SimpleNamespace(success=True), SimpleNamespace(success=True)])
    monkeypatch.setattr(node.rospy, 'wait_for_service', Mock())
    monkeypatch.setattr(node.rospy, 'ServiceProxy', Mock(return_value=request))
    monkeypatch.setattr(node.rospy, 'is_shutdown', lambda: False)
    warning = Mock()
    monkeypatch.setattr(node.rospy, 'logwarn', warning)
    assert not node.configure_mavros_telemetry('/mavros/set_message_interval', 20., Mock())
    assert [call.kwargs['message_id'] for call in request.call_args_list] == [27, 31, 32]
    assert 'message 27 request rejected' in warning.call_args.args[-1]
