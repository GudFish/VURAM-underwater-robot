"""PD launch configuration checks without running a ROS master or controller."""
from pathlib import Path

import pytest
import rospkg
import roslaunch
from roslaunch import substitution_args


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def expand_pd_launch(monkeypatch):
    paths = [str(PACKAGE.parent)] + rospkg.get_ros_paths()
    monkeypatch.setattr(substitution_args, '_rospack', rospkg.RosPack(ros_paths=paths))

    def expand(*argv):
        config = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(
            str(PACKAGE / 'launch/pd_deploy.launch'), config,
            argv=['enable_rviz:=false', 'learning_root:=/tmp/learning'] + list(argv),
            verbose=False)
        return config, {name: param.value for name, param in config.params.items()}
    return expand


def test_pd_defaults_reuse_trajectory_and_safety_without_policy(expand_pd_launch):
    config, params = expand_pd_launch()
    controller = next(node for node in config.nodes if node.name == 'pd_controller')
    assert controller.type == 'pd_controller_node.py'
    assert controller.required
    assert not any(node.name in ('policy_inference', 'policy_lan_sync', 'pd_lan_sync')
                   for node in config.nodes)
    assert params['/pd_controller/save_data'] is True
    assert params['/pd_controller/episode_len_s'] == 20.
    assert params['/pd_controller/ref_speed'] == 1.
    assert params['/pd_controller/figure8_ratio'] == .4
    assert params['/pd_controller/odom_timeout'] == .2
    assert params['/pd_controller/wait_param_list'] is True
    assert params['/pd_controller/data_dir'] == '/tmp/learning/data/vnd/inbox'
    assert params['/pd_controller/kp_xyz'] == [80., 80., 60.]
    assert params['/pd_controller/kd_xyz'] == [40., 40., 32.]
    assert params['/pd_controller/kp_att'] == [2.49, 2.60]
    assert params['/pd_controller/kd_att'] == [1.245, 1.30]
    assert not any(name.startswith('/pd_controller/model') for name in params)


def test_pd_lan_only_uploads_data(expand_pd_launch):
    config, params = expand_pd_launch('transport_mode:=lan')
    sync = next(node for node in config.nodes if node.name == 'pd_lan_sync')
    assert sync.type == 'lan_sync_node.py'
    assert sync.respawn and not sync.required
    assert params['/pd_controller/data_dir'] == '/tmp/learning/data/vnd/lan/outbox'
    assert params['/pd_lan_sync/data_dir'] == params['/pd_controller/data_dir']
    assert params['/pd_lan_sync/upload_data'] is True
    assert params['/pd_lan_sync/download_models'] is False


@pytest.mark.parametrize('transport', ['local', 'lan'])
def test_pd_test_mode_disables_recording_and_all_transport(expand_pd_launch, transport):
    config, params = expand_pd_launch('transport_mode:=' + transport, 'save_data:=false')
    assert params['/pd_controller/save_data'] is False
    assert not any(node.name == 'pd_lan_sync' for node in config.nodes)


def test_pd_custom_gains_paths_and_topics(expand_pd_launch, tmp_path):
    gains = tmp_path / 'gains.yaml'
    gains.write_text('kp_xyz: [1, 2, 3]\nkd_xyz: [4, 5, 6]\n'
                     'kp_att: [7, 8]\nkd_att: [9, 10]\n')
    _, params = expand_pd_launch(
        'pd_gains:=' + str(gains), 'data_dir:=/tmp/pd_runs',
        'dof_cmd_topic:=/test/dof', 'odom_topic:=/dvl/odom',
        'episode_len_s:=30', 'ref_speed:=0.5', 'figure8_ratio:=0.6')
    assert params['/pd_controller/kp_xyz'] == [1, 2, 3]
    assert params['/pd_controller/kd_att'] == [9, 10]
    assert params['/pd_controller/data_dir'] == '/tmp/pd_runs'
    assert params['/pd_controller/episode_len_s'] == 30
    for name in ('cmd_topic', 'ready_topic', 'enabled_topic'):
        assert params['/pd_controller/' + name] == params['/set_6dof_cmd/' + name]
    assert params['/pd_controller/cmd_topic'] == '/test/dof'
    assert params['/pd_controller/ready_topic'] == '/test/dof/ready'
    assert params['/pd_controller/check_mavros_imu'] is False
