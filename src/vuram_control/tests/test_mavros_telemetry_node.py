"""Shared telemetry setup ordering and cancellation, without FC/RC traffic."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def runtime(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/control/mavros_telemetry_node.py'
    spec = importlib.util.spec_from_file_location('telemetry_node_test', path)
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    callbacks = {}
    subscribed, stop = threading.Event(), threading.Event()
    subscription = Mock()
    params = {'~ready_topic': '/robot/pwm/ready', '~mavros_telemetry_rate': 30.,
              '~param_pull_service': '/robot/param/pull', '~param_wait_timeout': 45.,
              '~message_interval_service': '/robot/set_message_interval'}

    def subscribe(topic, kind, callback, **kwargs):
        callbacks[topic] = callback
        callback(SimpleNamespace(data=False))
        subscribed.set()
        return subscription

    ros = SimpleNamespace(
        get_param=lambda key, default=None: params.get(key, default),
        Subscriber=subscribe, is_shutdown=stop.is_set,
        loginfo_throttle=Mock(), spin=Mock())
    monkeypatch.setattr(node, 'rospy', ros)
    events = []
    wait = Mock(side_effect=lambda *args: events.append('params') or True)
    configure = Mock(side_effect=lambda *args: events.append('rates') or True)
    monkeypatch.setattr(node, 'wait_for_parameter_list', wait)
    monkeypatch.setattr(node, 'configure_mavros_telemetry', configure)
    return SimpleNamespace(node=node, ros=ros, params=params, callbacks=callbacks,
                           subscribed=subscribed, stop=stop, subscription=subscription,
                           wait=wait, configure=configure, events=events)


def test_setup_waits_for_bridge_and_parameters_before_requesting_rates(runtime):
    errors = []

    def run():
        try:
            runtime.node.run()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert runtime.subscribed.wait(1.)
        runtime.wait.assert_not_called()
        runtime.configure.assert_not_called()
        runtime.callbacks['/robot/pwm/ready'](SimpleNamespace(data=True))
        worker.join(1.)
        assert not worker.is_alive() and not errors
        assert runtime.events == ['params', 'rates']
        assert runtime.wait.call_args.args[0] == '/robot/param/pull'
        assert runtime.wait.call_args.args[2] == 45.
        assert runtime.configure.call_args.args[:2] == ('/robot/set_message_interval', 30.)
        runtime.subscription.unregister.assert_called_once()
        runtime.ros.spin.assert_called_once()
    finally:
        runtime.stop.set()
        worker.join(1.)


def test_shutdown_while_bridge_not_ready_does_not_request_services(runtime):
    worker = threading.Thread(target=runtime.node.run)
    worker.start()
    try:
        assert runtime.subscribed.wait(1.)
        runtime.stop.set()
        worker.join(1.)
        assert not worker.is_alive()
        runtime.wait.assert_not_called()
        runtime.configure.assert_not_called()
        runtime.subscription.unregister.assert_called_once()
        runtime.ros.spin.assert_not_called()
    finally:
        runtime.stop.set()
        worker.join(1.)


@pytest.mark.parametrize('rate', [0., -1., float('nan'), float('inf')])
def test_invalid_rate_is_rejected_before_waiting_or_requesting_services(runtime, rate):
    runtime.params['~mavros_telemetry_rate'] = rate
    with pytest.raises(ValueError, match='positive and finite'):
        runtime.node.run()
    assert not runtime.subscribed.is_set()
    runtime.wait.assert_not_called()
    runtime.configure.assert_not_called()


def test_optional_waits_and_request_rejection_still_leave_node_running(runtime):
    runtime.params.update({'~wait_ready': False, '~wait_param_list': False})
    runtime.configure.side_effect = None
    runtime.configure.return_value = False
    runtime.node.run()
    assert not runtime.subscribed.is_set()
    runtime.wait.assert_not_called()
    runtime.configure.assert_called_once()
    runtime.ros.spin.assert_called_once()
