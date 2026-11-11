"""Construction faults and shutdown still release control and completed data."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))

from control import tracking_node as node
from control.observation import load_spec
from data_collection.episodes import EpisodeWriter


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    params = {'~learning_root': str(tmp_path / 'learning'),
              '~data_dir': str(tmp_path / 'inbox')}
    ros = SimpleNamespace(
        get_param=lambda key, default: params.get(key, default),
        on_shutdown=Mock(), Subscriber=Mock(), is_shutdown=lambda: True,
        loginfo=Mock(), logwarn=Mock(), logerr=Mock())
    monkeypatch.setattr(node, 'rospy', ros)
    monkeypatch.setattr(node, 'TrainingContract', Mock(return_value=object()))
    writer = Mock(data_dir=str(tmp_path / 'inbox'))
    monkeypatch.setattr(node, 'EpisodeWriter', Mock(return_value=writer))
    thrusters = Mock(wrench_gain=1.)
    monkeypatch.setattr(node, 'ThrusterInterface', Mock(return_value=thrusters))
    pose = Mock()
    monkeypatch.setattr(node, 'PoseState', Mock(return_value=pose))
    source = Mock()
    factory = Mock(return_value=source)
    spec = load_spec(str(PACKAGE / 'config/obs_act.yaml'))
    return SimpleNamespace(ros=ros, writer=writer, thrusters=thrusters,
                           pose=pose, source=source, factory=factory, spec=spec,
                           data_dir=tmp_path / 'inbox')


def test_odometry_construction_failure_stops_writer_source_and_commands(runtime):
    node.PoseState.side_effect = RuntimeError('odometry construction failed')
    node.run_tracking_node(runtime.spec, .2, runtime.factory)
    runtime.thrusters.set_zero.assert_called_once_with()
    runtime.thrusters.shutdown.assert_called_once_with()
    runtime.writer.close.assert_called_once_with()
    runtime.source.close.assert_called_once_with()
    runtime.source.start.assert_not_called()
    runtime.ros.Subscriber.assert_not_called()
    assert 'odometry construction failed' in str(runtime.ros.logerr.call_args)


@pytest.mark.parametrize('failure_index', [2, 4])
def test_partial_subscriber_construction_unregisters_every_created_subscription(
        runtime, failure_index):
    subscriptions = [Mock() for _ in range(failure_index - 1)]
    runtime.ros.Subscriber.side_effect = subscriptions + [RuntimeError('subscription failed')]
    node.run_tracking_node(runtime.spec, .2, runtime.factory)
    for subscription in subscriptions:
        subscription.unregister.assert_called_once_with()
    runtime.pose.sub.unregister.assert_called_once_with()
    runtime.thrusters.set_zero.assert_called_once_with()
    runtime.thrusters.shutdown.assert_called_once_with()
    runtime.writer.close.assert_called_once_with()
    runtime.source.close.assert_called_once_with()
    assert 'subscription failed' in str(runtime.ros.logerr.call_args)


def test_reloader_cleanup_failure_still_drains_completed_episode(runtime, monkeypatch):
    release = threading.Event()
    writers = []

    class CompletedEpisode:
        episode_id = 'ep_000007_completed_before_shutdown'

        def to_hdf5(self, path):
            assert release.wait(3.), 'source cleanup must release the pending write'
            with h5py.File(path, 'w') as handle:
                handle.create_group('episode').create_dataset('state', data=np.zeros((3, 13)))

    def create_writer(data_dir, saved):
        writer = EpisodeWriter(data_dir, saved=saved)
        writers.append(writer)
        writer.submit(CompletedEpisode(), {})
        return writer

    def close_source():
        release.set()
        raise RuntimeError('reloader close failed')

    monkeypatch.setattr(node, 'EpisodeWriter', create_writer)
    runtime.source.close.side_effect = close_source
    node.PoseState.side_effect = RuntimeError('odometry construction failed')
    try:
        with pytest.raises(RuntimeError, match='reloader close failed'):
            node.run_tracking_node(runtime.spec, .2, runtime.factory)
    finally:
        drained_by_runtime = bool(writers) and all(not writer._thread.is_alive() for writer in writers)
        release.set()
        for writer in writers:
            if writer._thread.is_alive():
                writer.close()
    assert len(writers) == 1 and drained_by_runtime
    files = list(runtime.data_dir.glob('*.h5'))
    assert [path.name for path in files] == [CompletedEpisode.episode_id + '.h5']
    assert not list(runtime.data_dir.glob('*.tmp-*'))
    with h5py.File(files[0], 'r') as handle:
        assert handle['episode/state'].shape == (3, 13)
    runtime.thrusters.set_zero.assert_called_once_with()
    runtime.thrusters.shutdown.assert_called_once_with()
