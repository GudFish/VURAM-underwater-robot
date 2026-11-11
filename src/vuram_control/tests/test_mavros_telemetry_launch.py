"""One shared FC/bridge/telemetry stack per deployment launch."""
from pathlib import Path

import pytest
import roslaunch
import rospkg
from roslaunch import substitution_args


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture
def expand(monkeypatch):
    paths = [str(PACKAGE.parent)] + rospkg.get_ros_paths()
    monkeypatch.setattr(substitution_args, '_rospack', rospkg.RosPack(ros_paths=paths))

    def load(name, *argv):
        config = roslaunch.config.ROSLaunchConfig()
        base = (['show_rviz:=false', 'start_boundary:=false'] if name == 'AUV_ctrl_6dof.launch'
                else ['enable_rviz:=false', 'learning_root:=/tmp/learning'])
        roslaunch.xmlloader.XmlLoader().load(
            str(PACKAGE / 'launch' / name), config, argv=base+list(argv), verbose=False)
        return config, {key: value.value for key, value in config.params.items()}
    return load


@pytest.mark.parametrize('launch', ['AUV_ctrl_6dof.launch', 'policy_deploy.launch', 'pd_deploy.launch'])
def test_default_has_one_mavros_bridge_and_shared_telemetry_node(expand, launch):
    config, params = expand(launch)
    for name in ('mavros', 'set_6dof_cmd', 'mavros_telemetry'):
        assert sum(node.name == name for node in config.nodes) == 1
    assert params['/mavros_telemetry/mavros_telemetry_rate'] == 20.
    assert params['/mavros_telemetry/message_interval_service'] == '/mavros/set_message_interval'
    assert params['/mavros_telemetry/ready_topic'] == params['/set_6dof_cmd/ready_topic']
    assert params['/set_6dof_cmd/arm_vehicle'] is False
    for name in ('policy_inference', 'pd_controller'):
        for key in ('mavros_telemetry_rate', 'message_interval_service', 'configure_mavros_rates'):
            assert '/' + name + '/' + key not in params
    if launch != 'AUV_ctrl_6dof.launch':
        assert params['/set_6dof_cmd/frame_config'] == 1
        assert params['/set_6dof_cmd/acro_yaw_p'] == 4.5
        assert not any(node.name == 'boundary_loader' for node in config.nodes)


@pytest.mark.parametrize('launch', ['AUV_ctrl_6dof.launch', 'policy_deploy.launch', 'pd_deploy.launch'])
@pytest.mark.parametrize('mode, expected', [('dvl_fused', 0), ('mavros_fused', 1)])
def test_device_only_mode_skips_fc_rates_and_imu_fusion_uses_shared_setup(expand, launch, mode, expected):
    config, _ = expand(launch, 'odom_source:=dvl', 'dvl_mode:='+mode)
    assert sum(node.name == 'mavros_telemetry' for node in config.nodes) == expected
    assert sum(node.name == 'dvl_velocity_node' for node in config.nodes) == 1


def test_common_launch_owns_rate_service_and_setup_overrides(expand):
    _, params = expand('AUV_ctrl_6dof.launch', 'mavros_telemetry_rate:=30.0',
                       'message_interval_service:=/robot/set_interval',
                       'ready_topic:=/robot/ready', 'param_wait_timeout:=45',
                       'param_pull_service:=/robot/param/pull')
    assert params['/mavros_telemetry/mavros_telemetry_rate'] == 30.
    assert params['/mavros_telemetry/message_interval_service'] == '/robot/set_interval'
    assert params['/mavros_telemetry/ready_topic'] == params['/set_6dof_cmd/ready_topic'] == '/robot/ready'
    assert params['/mavros_telemetry/param_wait_timeout'] == 45.
    assert params['/mavros_telemetry/param_pull_service'] == '/robot/param/pull'


def test_enabling_parallel_dvl_does_not_publish_into_mavros_odometry(expand):
    config, params = expand('AUV_ctrl_6dof.launch', 'use_dvl:=true', 'odom_source:=mavros')
    assert params['/dvl_odometry_node/odom_topic'] == '/dvl/odom'
    assert params['/dvl_odometry_node/odom_frame'] == 'odom'
    assert params['/dvl_odometry_node/publish_tf'] is False
    assert sum(node.name == 'mavros_telemetry' for node in config.nodes) == 1


def test_common_setup_can_be_disabled_when_rates_are_managed_externally(expand):
    config, _ = expand('AUV_ctrl_6dof.launch', 'configure_mavros_rates:=false')
    assert not any(node.name == 'mavros_telemetry' for node in config.nodes)
