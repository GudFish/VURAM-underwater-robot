"""Actual HTTP cache handoff to the ROS lap recorder, without a ROS master."""
import os
from pathlib import Path
import sys
import threading

import h5py
import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]
LEARNING_ROOT = Path(os.environ.get('AUV_LEARNING_ROOT', '~/mujoco_auv_env')).expanduser().resolve()
sys.path.insert(0, str(PACKAGE / 'scripts'))
sys.path.insert(0, str(LEARNING_ROOT))

from auv_learning.deployment.lan_client import LanClient
from auv_learning.deployment.lan_http import LanHTTPServer
from auv_learning.deployment.episode import EpisodeBundle
from auv_learning.deployment.snapshot import export_deploy, publish_deploy
from auv_learning.learning.models.encoder import HistoryEncoder
from auv_learning.learning.models.policy import LatentPolicy
from data_collection.contract import TrainingContract
from data_collection.episodes import LapRecorder, write_episode
from trajectory.figure8 import FixedFigure8
from vuram_policy.model_reload import PolicyReloader
from vuram_policy.policy_inference import PolicyInference, configure_inference_threads
from control.observation import load_spec


def test_lan_models_are_adopted_only_between_laps_and_h5_reaches_host(tmp_path):
    configure_inference_threads(1)
    contract = TrainingContract(str(LEARNING_ROOT))
    spec = load_spec(str(PACKAGE / 'config/obs_act.yaml'))
    models = tmp_path / 'host/models'
    actor, encoder = LatentPolicy(), HistoryEncoder()
    first = export_deploy(actor, encoder, False, str(models), 0, spec, contract.allocator)
    with torch.no_grad():
        actor.out.bias[0] += .5
    second = export_deploy(actor, encoder, True, str(models), 1, spec, contract.allocator)
    publish_deploy(str(models), first)
    server = LanHTTPServer(('127.0.0.1', 0), tmp_path / 'host')
    serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    serving.start()
    try:
        token = tmp_path / 'vehicle/token'
        token.parent.mkdir()
        token.write_text(server.token_file.read_text())
        client = LanClient('http://127.0.0.1:{}'.format(server.server_port),
                           tmp_path / 'vehicle/outbox', tmp_path / 'vehicle/models',
                           'auv01', token, log=lambda message: None)
        client.outbox.mkdir()
        assert client.download_once()

        def load(path):
            policy = PolicyInference(path, num_threads=1)
            policy.validate_spec(spec)
            contract.validate_policy(policy)
            policy.warmup()
            return policy

        reloader = PolicyReloader(str(client.model_root), load)
        reloader.poll_once()
        policy = reloader.take_ready()
        state = np.r_[np.zeros(6), [1., 0., 0., 0.], [0., 0., -.3]]
        curve = FixedFigure8(state, duration=.2)
        bundles = []
        for episode in range(2):
            lap = LapRecorder(contract, spec, curve, policy, episode)
            for index in range(curve.intervals + 1):
                if episode == 0 and index == 2:
                    publish_deploy(str(models), second)
                    assert client.download_once()
                    reloader.poll_once()
                    assert policy.version == 0
                lap.record(state, 100. + index * .05, 100. + index * .05,
                           lambda action: None)
            bundle, extras = lap.finish()
            bundles.append(bundle)
            source = Path(write_episode(bundle, extras, str(client.outbox)))
            assert client.upload_once() == 1 and not source.exists()
            prepared = reloader.take_ready()
            if prepared is not None:
                policy = prepared
        files = sorted((tmp_path / 'host/inbox').glob('*.h5'))
        assert len(files) == 2
        received = [EpisodeBundle.from_hdf5(str(path)) for path in files]
        assert [int(bundle.attrs['policy_version']) for bundle in received] == [0, 1]
        assert [int(bundle.attrs['iteration']) for bundle in received] == [0, 1]
        for original, remote, path in zip(bundles, received, files):
            for name in original.arrays:
                np.testing.assert_array_equal(original.arrays[name], remote.arrays[name])
            with h5py.File(path, 'r') as handle:
                assert handle['episode/dof_pwm'].shape == (5, 6)
                assert handle['episode/action_applied'][-1, 0] == 0
        np.testing.assert_array_equal(bundles[0].arrays['motor_force_pre'][-1],
                                      bundles[1].arrays['motor_force_pre'][0])
    finally:
        server.shutdown()
        serving.join(5.)
        assert not serving.is_alive()
        server.server_close()
