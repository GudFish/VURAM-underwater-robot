"""Deployment odometry selection without a ROS master or DVL connection."""
from pathlib import Path

import pytest
import rospkg
import roslaunch
from roslaunch import substitution_args


PACKAGE = Path(__file__).resolve().parents[1]


@pytest.fixture(params=[('policy_deploy.launch', 'policy_inference'),
                        ('pd_deploy.launch', 'pd_controller')],
                ids=['policy', 'pd'])
def deployment(request, monkeypatch):
    paths = [str(PACKAGE.parent)] + rospkg.get_ros_paths()
    monkeypatch.setattr(substitution_args, '_rospack', rospkg.RosPack(ros_paths=paths))
    launch_file, controller_name = request.param

    def expand(*argv):
        config = roslaunch.config.ROSLaunchConfig()
        roslaunch.xmlloader.XmlLoader().load(
            str(PACKAGE / 'launch' / launch_file), config,
            argv=['learning_root:=/tmp/learning'] + list(argv), verbose=False)
        params = {name: param.value for name, param in config.params.items()}
        return config, params

    return controller_name, expand


def dvl_nodes(config):
    return [node for node in config.nodes if node.package == 'dvl_odometry']


def node_named(config, name):
    return next(node for node in config.nodes if node.name == name)


def test_default_keeps_mavros_odometry_and_display(deployment):
    name, expand = deployment
    config, params = expand()
    prefix = '/' + name + '/'
    assert params[prefix + 'odom_source'] == 'mavros'
    assert params[prefix + 'odom_topic'] == '/mavros/local_position/odom'
    assert params[prefix + 'viz_frame_id'] == 'map'
    assert params[prefix + 'twist_in_world'] is False
    assert params[prefix + 'check_mavros_imu'] is True
    assert params[prefix + 'save_data'] is True
    assert not dvl_nodes(config)
    assert node_named(config, 'tf_baselink_map_node')
    assert 'policy_eval_ardusub.rviz' in node_named(config, 'rviz').args


def test_dvl_starts_device_fusion_and_switches_control_and_display(deployment):
    name, expand = deployment
    config, params = expand('odom_source:=dvl', 'save_data:=false')
    prefix = '/' + name + '/'
    assert params[prefix + 'odom_source'] == 'dvl'
    assert params[prefix + 'dvl_mode'] == 'dvl_fused'
    assert params[prefix + 'dvl_timeout'] == 1.
    assert params[prefix + 'odom_topic'] == '/dvl/odom'
    assert params[prefix + 'viz_frame_id'] == 'odom'
    assert params[prefix + 'twist_in_world'] is False
    assert params[prefix + 'check_mavros_imu'] is False
    assert params[prefix + 'save_data'] is False
    assert sorted(node.type for node in dvl_nodes(config)) == [
        'dvl_fusion_node.py', 'dvl_velocity_node.py']
    assert not any(node.name == 'dvl_ekf' for node in config.nodes)
    assert params['/dvl_velocity_node/ip'] == '192.168.1.95'
    assert params['/dvl_odometry_node/odom_rate'] == 20.
    assert params['/dvl_odometry_node/odom_topic'] == '/dvl/odom'
    assert params['/dvl_odometry_node/publish_tf'] is True
    assert not any(node.name == 'tf_baselink_map_node' for node in config.nodes)
    assert 'policy_eval_dvl.rviz' in node_named(config, 'rviz').args


def test_explicit_mavros_fused_preserves_imu_inputs_and_rate_checks(deployment):
    name, expand = deployment
    config, params = expand('odom_source:=dvl', 'dvl_mode:=mavros_fused')
    assert params['/' + name + '/check_mavros_imu'] is True
    assert node_named(config, 'dvl_ekf').type == 'ekf_localization_node'
    assert sorted(node.type for node in dvl_nodes(config)) == [
        'dvl_velocity_node.py', 'mavros_dvl_fusion_node.py']
    assert params['/dvl_ekf/frequency'] == 20.
    assert params['/dvl_ekf/publish_tf'] is False
    assert params['/mavros_dvl_fusion_node/odom_topic'] == '/dvl/odom'
    assert params['/mavros_dvl_fusion_node/publish_tf'] is True
    assert params['/mavros_dvl_fusion_node/imu_topic'] == params['/' + name + '/imu_topic']


@pytest.mark.parametrize('mode,driver', [
    ('dvl_raw', 'dvl_odometry_node.py'), ('dvl_fused', 'dvl_fusion_node.py')])
def test_dvl_mode_ip_rate_and_topic_are_forwarded(deployment, mode, driver):
    name, expand = deployment
    config, params = expand(
        'odom_source:=dvl', 'dvl_mode:=' + mode, 'dvl_ip:=192.168.1.96',
        'dvl_odom_rate:=30.0', 'odom_topic:=/robot/dvl/odom',
        'viz_frame_id:=robot_odom', 'enable_rviz:=false')
    assert node_named(config, 'dvl_odometry_node').type == driver
    assert params['/dvl_velocity_node/ip'] == '192.168.1.96'
    assert params['/dvl_odometry_node/odom_rate'] == 30.
    assert params['/dvl_odometry_node/odom_topic'] == '/robot/dvl/odom'
    assert params['/dvl_odometry_node/odom_frame'] == 'robot_odom'
    assert params['/' + name + '/odom_topic'] == '/robot/dvl/odom'
    assert params['/' + name + '/viz_frame_id'] == 'robot_odom'
    assert params['/' + name + '/check_mavros_imu'] is False
    assert not any(node.name == 'rviz' for node in config.nodes)


def test_mavros_fused_dvl_mode_has_one_tf_owner(deployment):
    name, expand = deployment
    config, params = expand(
        'odom_source:=dvl', 'dvl_mode:=mavros_fused', 'dvl_timeout:=1.5',
        'imu_topic:=/vehicle/imu', 'viz_frame_id:=robot_odom')
    assert node_named(config, 'mavros_dvl_fusion_node').type == 'mavros_dvl_fusion_node.py'
    assert node_named(config, 'dvl_ekf').type == 'ekf_localization_node'
    assert params['/mavros_dvl_fusion_node/publish_tf'] is True
    assert params['/dvl_ekf/publish_tf'] is False
    assert params['/mavros_dvl_fusion_node/odom_topic'] == params['/' + name + '/odom_topic']
    assert params['/mavros_dvl_fusion_node/imu_topic'] == '/vehicle/imu'
    assert params['/mavros_dvl_fusion_node/odom_frame'] == 'robot_odom'
    assert params['/dvl_ekf/world_frame'] == 'robot_odom'
    assert params['/' + name + '/dvl_timeout'] == 1.5
    assert params['/dvl_velocity_node/vel_timeout'] == 1.5
    assert params['/dvl_velocity_node/dr_timeout'] == 1.5
    assert params['/mavros_dvl_fusion_node/vel_timeout'] == 1.5
    assert not any(node.name == 'tf_baselink_map_node' for node in config.nodes)


def test_external_dvl_driver_is_not_started_twice(deployment):
    name, expand = deployment
    config, params = expand('odom_source:=dvl', 'start_dvl:=false')
    assert not dvl_nodes(config)
    assert not any(node.name == 'dvl_ekf' for node in config.nodes)
    assert params['/' + name + '/odom_topic'] == '/dvl/odom'
    assert params['/' + name + '/viz_frame_id'] == 'odom'
    assert params['/' + name + '/check_mavros_imu'] is False
    assert not any(node.name == 'tf_baselink_map_node' for node in config.nodes)
    assert 'policy_eval_dvl.rviz' in node_named(config, 'rviz').args


def test_custom_external_odometry_and_explicit_overrides_remain_supported(deployment):
    name, expand = deployment
    config, params = expand(
        'odom_topic:=/custom/odom', 'viz_frame_id:=world', 'twist_in_world:=true',
        'check_mavros_imu:=true', 'enable_rviz:=false')
    prefix = '/' + name + '/'
    assert params[prefix + 'odom_topic'] == '/custom/odom'
    assert params[prefix + 'viz_frame_id'] == 'world'
    assert params[prefix + 'twist_in_world'] is True
    assert params[prefix + 'check_mavros_imu'] is True
    assert not dvl_nodes(config)


def test_mavros_source_never_starts_dvl_even_when_start_flag_is_true(deployment):
    _, expand = deployment
    config, _ = expand('odom_source:=mavros', 'start_dvl:=true', 'enable_rviz:=false')
    assert not dvl_nodes(config)


def test_unknown_odometry_source_is_rejected_during_launch_expansion(deployment):
    _, expand = deployment
    with pytest.raises((roslaunch.xmlloader.XmlParseException,
                        substitution_args.SubstitutionException, KeyError)):
        expand('odom_source:=dvll')
