"""Real ROS/TCP/EKF regression tests, isolated from a running vehicle ROS master.

Run after catkin_make and sourcing devel/setup.bash:
  python3 -m pytest -q src/dvl_odometry/tests/test_ros_integration.py
"""
import collections
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
import pytest


def wait_for(predicate, timeout=10.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.025)
    raise AssertionError('condition not met within %.1f seconds' % timeout)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='module')
def ros(tmp_path_factory):
    directory = tmp_path_factory.mktemp('dvl_ros')
    env = os.environ.copy()
    env.update(ROS_MASTER_URI='http://127.0.0.1:%d' % free_port(), ROS_HOSTNAME='127.0.0.1',
               ROS_LOG_DIR=str(directory))
    env.pop('ROS_IP', None)
    os.environ.update({key: env[key] for key in ('ROS_MASTER_URI', 'ROS_HOSTNAME', 'ROS_LOG_DIR')})
    os.environ.pop('ROS_IP', None)
    master_log = open(directory/'master.log', 'w')
    port = env['ROS_MASTER_URI'].rsplit(':', 1)[1]
    master = subprocess.Popen(['roscore', '-p', port], env=env, stdout=master_log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    import rosgraph
    wait_for(lambda: rosgraph.is_master_online())
    import rospy
    rospy.init_node('dvl_regression', anonymous=True, disable_signals=True)
    yield rospy, env, directory
    rospy.signal_shutdown('tests complete')
    os.killpg(master.pid, signal.SIGINT)
    master.wait(timeout=15)
    master_log.close()


class Rig:
    def __init__(self, ros, mode, sim_args=(), driver_args=()):
        self.rospy, self.env, self.directory = ros
        self.messages = collections.deque(maxlen=5000)
        self.velocities = collections.deque(maxlen=5000)
        self.diagnostics = collections.deque(maxlen=5000)
        self.transforms = collections.deque(maxlen=5000)
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TwistWithCovarianceStamped
        from diagnostic_msgs.msg import DiagnosticArray
        from tf2_msgs.msg import TFMessage
        self.subscribers = [
            self.rospy.Subscriber('/dvl/odom', Odometry, self.messages.append, queue_size=100),
            self.rospy.Subscriber('/dvl/velocity', TwistWithCovarianceStamped, self.velocities.append, queue_size=100),
            self.rospy.Subscriber('/diagnostics', DiagnosticArray, self.diagnostics.append, queue_size=100),
            self.rospy.Subscriber('/tf', TFMessage, self.transforms.append, queue_size=100)]
        self.port = free_port()
        self.processes, self.logs = [], []
        self.start('sim', ['roslaunch', 'dvl_odometry', 'dvl_simulator.launch', 'publish_imu:=true',
                           'port:=%d' % self.port]+list(sim_args))
        self.start('driver', ['roslaunch', 'dvl_odometry', 'dvl_odometry.launch', 'mode:='+mode,
                              'ip:=127.0.0.1', 'port:=%d' % self.port, 'publish_tf:=true']+list(driver_args))
        wait_for(lambda: len(self.messages) >= 5, timeout=15.)

    def start(self, name, command):
        path = self.directory/('%s-%d.log' % (name, time.time_ns()))
        handle = open(path, 'w')
        self.logs.append((path, handle))
        self.processes.append(subprocess.Popen(command, env=self.env, stdout=handle,
                                                stderr=subprocess.STDOUT, start_new_session=True))

    def fault(self, name, seconds=2.):
        with socket.create_connection(('127.0.0.1', self.port), timeout=2.) as sock:
            sock.sendall((json.dumps(dict(command=name, parameters=dict(seconds=seconds)))+'\n').encode())
            time.sleep(.05)

    def statuses(self, name):
        return [status for array in list(self.diagnostics) for status in array.status if status.name == name]

    def close(self):
        for process in reversed(self.processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        for subscriber in self.subscribers:
            subscriber.unregister()
        for path, handle in self.logs:
            handle.close()
            print('\n'+str(path)+'\n'+path.read_text()[-5000:])


@pytest.fixture
def rig_factory(ros):
    rigs = []
    def factory(*args, **kwargs):
        # Construct in two stages so failed startup is also cleaned up.
        rig = Rig.__new__(Rig)
        rigs.append(rig)
        rig.__init__(ros, *args, **kwargs)
        return rig
    yield factory
    for rig in reversed(rigs):
        rig.close()


def measured_rate(messages):
    stamps = [message.header.stamp.to_sec() for message in list(messages)]
    return (len(stamps)-1)/(stamps[-1]-stamps[0])


def vector(message):
    v = message.twist.twist.linear
    return np.array([v.x, v.y, v.z])


@pytest.mark.parametrize('mode,minimum,maximum', [('dvl_raw', 4., 6.), ('dvl_fused', 18., 22.),
                                                ('mavros_fused', 18., 22.)])
def test_real_mode_rates_and_body_velocity(rig_factory, mode, minimum, maximum):
    rig = rig_factory(mode)
    time.sleep(2.2)
    assert minimum < measured_rate(rig.messages) < maximum
    assert np.linalg.norm(vector(rig.messages[-1])-[.3, 0., 0.]) < .06
    assert all(m.header.frame_id == 'odom' and m.child_frame_id == 'base_link' for m in rig.messages)
    assert len(rig.transforms) > 0


def test_velocity_cap_does_not_fabricate_samples(rig_factory):
    rig = rig_factory('dvl_fused', sim_args=['velocity_rate:=2.0'], driver_args=['velocity_rate:=20.0'])
    time.sleep(2.5)
    assert 1.7 < measured_rate(rig.velocities) < 2.3
    stamps = [m.header.stamp.to_nsec() for m in rig.velocities]
    assert len(stamps) == len(set(stamps))
    assert any('below requested cap' in s.message for s in rig.statuses('dvl/driver'))


def test_dvl_hold_reset_bad_packets_and_reconnect(rig_factory):
    rig = rig_factory('dvl_fused', sim_args=['yaw_rate:=0.0'])
    rig.fault('sim_pause_dr', 2.5)
    rig.fault('sim_pause_velocity', 2.5)
    time.sleep(1.4)
    position = rig.messages[-1].pose.pose.position.x
    time.sleep(.4)
    assert abs(rig.messages[-1].pose.pose.position.x-position) < .005
    time.sleep(1.)
    rig.fault('sim_malformed')
    old = len(rig.messages)
    wait_for(lambda: len(rig.messages) > old+5)
    from std_srvs.srv import Trigger
    reset = rig.rospy.ServiceProxy('/dvl_odometry/reset', Trigger)
    before = len(rig.messages)
    assert reset().success
    wait_for(lambda: len(rig.messages) > before+5)
    assert rig.messages[-1].pose.pose.position.x < .2
    time.sleep(2.)
    before_x = rig.messages[-1].pose.pose.position.x
    rig.fault('sim_dropout')
    time.sleep(2.)
    assert rig.messages[-1].pose.pose.position.x > before_x+.25  # no automatic reconnect reset


def test_mavros_inertial_outage_recovery_and_imu_gate(rig_factory):
    rig = rig_factory('mavros_fused', sim_args=['speed:=0.2', 'acceleration:=0.05', 'yaw_rate:=0.0'])
    time.sleep(1.)
    initial = vector(rig.messages[-1])[0]
    initial_cov = rig.messages[-1].pose.covariance[0]
    rig.fault('sim_invalid', 2.5)
    time.sleep(1.8)
    assert vector(rig.messages[-1])[0] > initial+.045
    assert rig.messages[-1].pose.covariance[0] > initial_cov
    assert any('inertial dead reckoning' in s.message for s in rig.statuses('dvl/mavros_fused'))
    time.sleep(1.3)
    wait_for(lambda: rig.statuses('dvl/mavros_fused')[-1].level == 0, timeout=3.)
    from std_srvs.srv import SetBool, Trigger
    toggle = rig.rospy.ServiceProxy('/dvl_simulator/imu_enabled', SetBool)
    assert toggle(False).success
    time.sleep(.8)
    count, tf_count = len(rig.messages), len(rig.transforms)
    held_position = rig.messages[-1].pose.pose.position.x
    time.sleep(.4)
    assert len(rig.messages) == count and len(rig.transforms) == tf_count
    assert toggle(True).success
    wait_for(lambda: len(rig.messages) > count+3)
    assert abs(rig.messages[-1].pose.pose.position.x-held_position) < .3
    reset = rig.rospy.ServiceProxy('/dvl_odometry/reset', Trigger)
    assert reset().success
    stamp = rig.rospy.Time.now()
    wait_for(lambda: rig.messages[-1].header.stamp > stamp)
    assert abs(rig.messages[-1].pose.pose.position.x) < .15


@pytest.mark.parametrize('roll,pitch', [(0., 0.), (.2, -.3)])
def test_ekf_stationary_gravity_compensation(rig_factory, roll, pitch):
    rig = rig_factory('mavros_fused', sim_args=['speed:=0.0', 'yaw_rate:=0.0',
                                              'roll:=%s' % roll, 'pitch:=%s' % pitch])
    time.sleep(2.)
    assert np.linalg.norm(vector(rig.messages[-1])) < .03
    p = rig.messages[-1].pose.pose.position
    assert np.linalg.norm([p.x, p.y, p.z]) < .08


def test_dvl_fused_low_rate_only_follows_device_reports(rig_factory):
    rig = rig_factory('dvl_fused', driver_args=['odom_rate:=2.0'])
    time.sleep(2.)
    assert 1.4 < measured_rate(rig.messages) <= 2.05
    stamps = [m.header.stamp.to_nsec() for m in rig.messages]
    assert len(stamps) == len(set(stamps))


def test_mavros_waits_for_first_valid_dvl_measurement(rig_factory):
    # Unit of motion is irrelevant: simulator water-tracking reports must never
    # initialize a bottom-relative EKF. Start normally, then restart only the
    # driver/estimator while water tracking is enabled.
    rig = rig_factory('mavros_fused')
    for process in rig.processes[1:]:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=12)
    rig.messages.clear()
    with socket.create_connection(('127.0.0.1', rig.port), timeout=2.) as sock:
        sock.sendall(b'{"command":"set_config","parameters":{"range_mode":"wt"}}\n')
        time.sleep(.1)
    rig.start('restart-water-track', ['roslaunch', 'dvl_odometry', 'dvl_odometry.launch',
                                      'mode:=mavros_fused', 'ip:=127.0.0.1',
                                      'port:=%d' % rig.port, 'reset_on_start:=false'])
    time.sleep(2.)
    assert len(rig.messages) == 0
    assert any('waiting for fresh IMU' in s.message for s in rig.statuses('dvl/mavros_fused'))
    with socket.create_connection(('127.0.0.1', rig.port), timeout=2.) as sock:
        sock.sendall(b'{"command":"set_config","parameters":{"range_mode":"auto"}}\n')
        time.sleep(.1)
    wait_for(lambda: len(rig.messages) > 5)
