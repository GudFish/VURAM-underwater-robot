"""LAN mode launch/wrapper checks without ROS master or network traffic."""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import rospkg
import roslaunch
from roslaunch import substitution_args

PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def expand_launch(monkeypatch):
    paths = [str(PACKAGE.parent)] + rospkg.get_ros_paths()
    monkeypatch.setattr(substitution_args, '_rospack', rospkg.RosPack(ros_paths=paths))

    def expand(*argv):
        config = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(
            str(PACKAGE / 'launch/policy_deploy.launch'), config,
            argv=['enable_rviz:=false', 'learning_root:=/tmp/learning'] + list(argv),
            verbose=False)
        return config, {name: param.value for name, param in config.params.items()}
    return expand


def test_default_local_mode_keeps_existing_paths_and_has_no_network_node(expand_launch):
    config, params = expand_launch()
    assert params['/policy_inference/transport_mode'] == 'local'
    assert params['/policy_inference/data_dir'] == '/tmp/learning/data/vnd/inbox'
    assert params['/policy_inference/model_root'] == '/tmp/learning/data/vnd/models'
    assert params['/policy_inference/save_data'] is True
    assert not any(node.name == 'policy_lan_sync' for node in config.nodes)
    assert params['/policy_inference/ref_speed'] == 1.
    assert params['/policy_inference/figure8_ratio'] == .4
    assert '--episode_len_s 20.0' in next(
        node.args for node in config.nodes if node.name == 'policy_inference')


def test_lan_mode_runs_independent_respawning_client_with_shared_defaults(expand_launch):
    config, params = expand_launch('transport_mode:=lan')
    sync = next(node for node in config.nodes if node.name == 'policy_lan_sync')
    assert sync.type == 'lan_sync_node.py'
    assert sync.respawn and not sync.required and sync.respawn_delay == 5.
    for name, value in [('data_dir', '/tmp/learning/data/vnd/lan/outbox'),
                        ('model_root', '/tmp/learning/data/vnd/lan/models')]:
        assert params['/policy_inference/' + name] == value
        assert params['/policy_lan_sync/' + name] == value
    assert params['/policy_lan_sync/server_url'] == 'http://192.168.1.1:8765'
    assert params['/policy_lan_sync/robot_id'] == 'auv01'
    assert params['/policy_lan_sync/token_file'] == '/tmp/learning/data/vnd/lan/token'
    assert params['/policy_lan_sync/poll_interval'] == 1.
    assert params['/policy_lan_sync/download_models'] is True
    assert params['/policy_lan_sync/upload_data'] is True


def test_policy_test_mode_keeps_model_downloads_and_disables_upload(expand_launch):
    config, params = expand_launch('transport_mode:=lan', 'save_data:=false')
    assert params['/policy_inference/save_data'] is False
    assert any(node.name == 'policy_lan_sync' for node in config.nodes)
    assert params['/policy_lan_sync/upload_data'] is False
    assert params['/policy_lan_sync/download_models'] is True


def test_policy_local_test_mode_disables_recording(expand_launch):
    config, params = expand_launch('save_data:=false')
    assert params['/policy_inference/save_data'] is False
    assert not any(node.name == 'policy_lan_sync' for node in config.nodes)


def test_policy_fixed_snapshot_test_mode_needs_no_lan_client(expand_launch):
    config, params = expand_launch(
        'transport_mode:=lan', 'save_data:=false', 'model_dir:=/tmp/fixed_snapshot')
    assert params['/policy_inference/save_data'] is False
    assert not any(node.name == 'policy_lan_sync' for node in config.nodes)


@pytest.mark.parametrize('snapshot, download', [('', True), ('/tmp/fixed_snapshot', False)])
def test_lan_path_overrides_and_fixed_snapshot(expand_launch, snapshot, download):
    _, params = expand_launch(
        'transport_mode:=lan', 'data_dir:=/tmp/custom_outbox', 'model_root:=/tmp/custom_models',
        'model_dir:=' + snapshot, 'lan_server_url:=http://192.168.1.9:9010',
        'robot_id:=auv09', 'lan_token_file:=/tmp/custom_token', 'lan_poll_interval:=2.5')
    for name, value in [('data_dir', '/tmp/custom_outbox'), ('model_root', '/tmp/custom_models')]:
        assert params['/policy_inference/' + name] == params['/policy_lan_sync/' + name] == value
    assert params['/policy_inference/model_dir'] == snapshot
    assert params['/policy_lan_sync/download_models'] is download
    assert params['/policy_lan_sync/server_url'] == 'http://192.168.1.9:9010'
    assert params['/policy_lan_sync/robot_id'] == 'auv09'
    assert params['/policy_lan_sync/token_file'] == '/tmp/custom_token'
    assert params['/policy_lan_sync/poll_interval'] == 2.5


@pytest.fixture
def wrapper(monkeypatch):
    path = PACKAGE / 'scripts/data_collection/lan_sync_node.py'
    spec = importlib.util.spec_from_file_location('policy_lan_sync_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ros = SimpleNamespace(init_node=Mock(), on_shutdown=Mock(), loginfo=Mock(), logerr=Mock())
    monkeypatch.setattr(module, 'rospy', ros)
    return module, ros


def test_sync_entry_is_resolved_and_installed_from_shared_directory(monkeypatch):
    import roslib.packages
    monkeypatch.setenv('ROS_PACKAGE_PATH', str(PACKAGE.parent))
    resolved = roslib.packages.find_node('vuram_control', 'lan_sync_node.py')
    assert (PACKAGE / 'scripts/data_collection/lan_sync_node.py').resolve() in {
        Path(path).resolve() for path in resolved}
    assert 'scripts/data_collection/lan_sync_node.py' in (PACKAGE / 'CMakeLists.txt').read_text()


def test_wrapper_passes_paths_and_shutdown_to_separate_client(wrapper, monkeypatch):
    module, ros = wrapper
    params = {'~learning_root': '/tmp/learner', '~server_url': 'http://192.168.1.1:9000',
              '~data_dir': '/tmp/episodes', '~model_root': '/tmp/models', '~robot_id': 'robot2',
              '~token_file': '/tmp/token', '~poll_interval': 2., '~download_models': False,
              '~upload_data': False}
    ros.get_param = lambda name, default: params.get(name, default)
    client = Mock()

    def run(stop):
        assert not stop.is_set()
        ros.on_shutdown.call_args.args[0]()
        assert stop.is_set()
    client.run.side_effect = run
    client_type = Mock(return_value=client)
    loader = Mock(return_value=client_type)
    monkeypatch.setattr(module, 'load_client_type', loader)
    assert module.main() == 0
    loader.assert_called_once_with(Path('/tmp/learner'))
    client_type.assert_called_once_with(
        server_url='http://192.168.1.1:9000', outbox='/tmp/episodes', model_root='/tmp/models',
        robot_id='robot2', token_file='/tmp/token', poll_interval=2., download_models=False,
        upload_data=False,
        log=ros.loginfo)


def test_wrapper_rejects_invalid_mode_before_import_or_network(wrapper, monkeypatch):
    module, ros = wrapper
    ros.get_param = lambda name, default: 'wrong' if name == '~transport_mode' else default
    loader = Mock()
    monkeypatch.setattr(module, 'load_client_type', loader)
    assert module.main() == 1
    loader.assert_not_called()
    assert 'transport_mode=lan' in str(ros.logerr.call_args)


def test_wrapper_failure_exits_for_respawn_and_sets_stop(wrapper, monkeypatch):
    module, ros = wrapper
    ros.get_param = lambda name, default: default
    stops = []

    def fail(stop):
        stops.append(stop)
        raise RuntimeError('client failure')
    client = Mock()
    client.run.side_effect = fail
    monkeypatch.setattr(module, 'load_client_type', Mock(return_value=Mock(return_value=client)))
    assert module.main() == 1
    assert stops[0].is_set()
    assert 'client failure' in str(ros.logerr.call_args)


def test_wrapper_rejects_different_imported_learning_root(wrapper, tmp_path, monkeypatch):
    module, _ = wrapper
    (tmp_path / 'auv_learning').mkdir()
    package = ModuleType('auv_learning')
    package.__file__ = '/other/auv_learning/__init__.py'
    monkeypatch.setitem(sys.modules, 'auv_learning', package)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    with pytest.raises(ValueError, match='different learning_root'):
        module.load_client_type(tmp_path)


def test_wrapper_imports_client_from_requested_learning_root(wrapper, tmp_path, monkeypatch):
    module, _ = wrapper
    (tmp_path / 'auv_learning').mkdir()
    package = ModuleType('auv_learning')
    package.__file__ = str(tmp_path / 'auv_learning/__init__.py')
    client_module = ModuleType('auv_learning.deployment.lan_client')
    client_module.LanClient = object()
    monkeypatch.setitem(sys.modules, 'auv_learning', package)
    monkeypatch.setitem(sys.modules, 'auv_learning.deployment.lan_client', client_module)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    assert module.load_client_type(tmp_path) is client_module.LanClient
