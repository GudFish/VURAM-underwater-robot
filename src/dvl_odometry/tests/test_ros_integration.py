"""Real ROS/TCP/EKF regression tests, isolated from a running vehicle ROS master.

Run after catkin_make and sourcing devel/setup.bash:
  python3 -m pytest -q src/dvl_odometry/tests/test_ros_integration.py
"""
import collections
import json
import os
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
        self.poses = collections.deque(maxlen=5000)
        self.velocities = collections.deque(maxlen=5000)
        self.diagnostics = collections.deque(maxlen=5000)
        self.transforms = collections.deque(maxlen=5000)
        self.angular_valid = collections.deque(maxlen=5000)
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
        from diagnostic_msgs.msg import DiagnosticArray
        from tf2_msgs.msg import TFMessage
        from std_msgs.msg import Bool
        self.subscribers = [
            self.rospy.Subscriber('/dvl/odom', Odometry, self.messages.append, queue_size=100),
            self.rospy.Subscriber('/dvl/dead_reckoning', PoseWithCovarianceStamped,
                                  self.poses.append, queue_size=100),
            self.rospy.Subscriber('/dvl/velocity', TwistWithCovarianceStamped, self.velocities.append, queue_size=100),
            self.rospy.Subscriber('/dvl/angular_velocity_valid', Bool, self.angular_valid.append, queue_size=100),
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


def angular_vector(message):
    v = message.twist.twist.angular
    return np.array([v.x, v.y, v.z])


def angular_covariance(message):
    return np.array(message.twist.covariance).reshape(6, 6)[3:, 3:]


def angular_known(message):
    return np.max(np.diag(angular_covariance(message))) < 1.e5


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


@pytest.mark.parametrize('stamp_offset', [.005, .015])
def test_mavros_small_future_imu_continuity_and_outage_recovery(rig_factory, stamp_offset):
    # Every IMU, including the startup and recovery samples, leads ROS time.
    rig = rig_factory('mavros_fused', sim_args=['speed:=0.3', 'yaw_rate:=0.0',
                                               'imu_stamp_offset:=%s' % stamp_offset])
    wait_for(lambda: rig.statuses('dvl/mavros_fused') and
             rig.statuses('dvl/mavros_fused')[-1].level == 0)
    rig.messages.clear()
    rig.diagnostics.clear()
    time.sleep(1.5)
    assert len(rig.messages) > 20
    speeds = np.array([vector(message)[0] for message in rig.messages])
    assert np.max(np.abs(speeds-.3)) < .08
    assert len(rig.transforms) > 0
    health = rig.statuses('dvl/mavros_fused')
    assert health and all(status.level == 0 for status in health)

    from std_srvs.srv import SetBool
    toggle = rig.rospy.ServiceProxy('/dvl_simulator/imu_enabled', SetBool)
    assert toggle(False).success
    time.sleep(.8)
    count, tf_count = len(rig.messages), len(rig.transforms)
    held_position = rig.messages[-1].pose.pose.position.x
    time.sleep(.3)
    assert len(rig.messages) == count and len(rig.transforms) == tf_count
    assert toggle(True).success
    wait_for(lambda: len(rig.messages) > count+4)
    assert abs(rig.messages[-1].pose.pose.position.x-held_position) < .3
    wait_for(lambda: rig.statuses('dvl/mavros_fused')[-1].level == 0)


def test_mavros_rejects_bad_imu_stamps_without_resetting_healthy_ekf(rig_factory):
    rig = rig_factory('mavros_fused', sim_args=['speed:=0.3', 'yaw_rate:=0.0'])
    wait_for(lambda: rig.statuses('dvl/mavros_fused') and
             rig.statuses('dvl/mavros_fused')[-1].level == 0)
    from sensor_msgs.msg import Imu
    bad_imu = rig.rospy.Publisher('/mavros/imu/data', Imu, queue_size=10)
    try:
        wait_for(lambda: bad_imu.get_num_connections() > 0)
        start = rig.rospy.Time.now().to_sec()
        rig.diagnostics.clear()
        # The simulator continues sending valid IMU at 100 Hz. Alternate
        # future and late packets at 20 Hz on a second publisher, reproducing
        # isolated timestamp faults without creating an actual IMU outage.
        for index in range(60):
            message = Imu()
            message.header.stamp = rig.rospy.Time.now()+rig.rospy.Duration(2. if index % 2 else -2.)
            message.header.frame_id = 'base_link'
            message.orientation.w = 1.
            message.linear_acceleration.z = 9.80665
            message.orientation_covariance = np.diag([.03**2]*3).ravel().tolist()
            message.angular_velocity_covariance = np.diag([.01**2]*3).ravel().tolist()
            message.linear_acceleration_covariance = np.diag([.1**2]*3).ravel().tolist()
            bad_imu.publish(message)
            time.sleep(.05)
        samples = [message for message in rig.messages if message.header.stamp.to_sec() > start+.25]
        assert len(samples) > 20
        # Repeated SetPose recovery resets twist towards zero. Check motion
        # continuity instead of imposing hard scheduler or publication gaps.
        speeds = np.array([vector(message)[0] for message in samples])
        assert speeds.min() > .2
        assert np.max(np.abs(speeds-.3)) < .08
        health = rig.statuses('dvl/mavros_fused')
        assert health and all(status.level == 0 for status in health)
    finally:
        bad_imu.unregister()


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


@pytest.mark.parametrize('roll,pitch,yaw_rate', [(0., 0., .1), (.35, -.25, -.1),
                                               (-.25, .35, 2.)])
def test_dvl_fused_estimates_body_angular_velocity(rig_factory, roll, pitch, yaw_rate):
    # The last case crosses the +/-pi yaw branch in about 1.6 seconds. A direct
    # Euler derivative would spike there and a world-z rate would miss x/y.
    rig = rig_factory('dvl_fused', sim_args=['roll:=%s' % roll, 'pitch:=%s' % pitch,
                                           'yaw_rate:=%s' % yaw_rate])
    from dvl_odometry.simulation import rpy_rotation
    expected = rpy_rotation(roll, pitch, 0.).T.dot([0., 0., yaw_rate])
    wait_for(lambda: angular_known(rig.messages[-1]))
    wait_for(lambda: rig.angular_valid and rig.angular_valid[-1].data)
    start = rig.rospy.Time.now().to_sec()
    time.sleep(3.)
    samples = [message for message in rig.messages if message.header.stamp.to_sec() > start]
    assert len(samples) > 45
    assert all(angular_known(message) for message in samples)
    estimates = np.array([angular_vector(message) for message in samples])
    # Receipt stamps carry scheduler jitter; allow five percent at the fast
    # wrap test, while still detecting a frame error or Euler-wrap spike.
    assert np.max(np.linalg.norm(estimates-expected, axis=1)) < max(.03, .05*abs(yaw_rate))
    covariances = [angular_covariance(message) for message in samples]
    assert all(np.linalg.eigvalsh(covariance).min() > 0. for covariance in covariances)
    assert all(np.isfinite(covariance).all() for covariance in covariances)


def test_dvl_fused_angular_velocity_invalidates_and_recovers(rig_factory):
    rig = rig_factory('dvl_fused', sim_args=['yaw_rate:=0.1'],
                      driver_args=['dr_timeout:=0.6', 'angular_timeout:=2.0'])
    wait_for(lambda: angular_known(rig.messages[-1]))

    rig.fault('sim_pause_dr', 1.6)
    time.sleep(.15)
    assert angular_known(rig.messages[-1])
    assert np.linalg.norm(angular_vector(rig.messages[-1])-[0., 0., .1]) < .02
    wait_for(lambda: not angular_known(rig.messages[-1]), timeout=2.)
    wait_for(lambda: not rig.angular_valid[-1].data, timeout=.3)
    assert np.array_equal(angular_vector(rig.messages[-1]), np.zeros(3))
    wait_for(lambda: angular_known(rig.messages[-1]), timeout=3.)
    wait_for(lambda: rig.angular_valid[-1].data, timeout=.3)
    assert np.linalg.norm(angular_vector(rig.messages[-1])-[0., 0., .1]) < .02
    # The independently configured angular timeout (2 s) cannot bridge a DR
    # timeout (0.6 s). Recovery anchors once, then needs a second new attitude.
    pose_stamps = [pose.header.stamp.to_sec() for pose in rig.poses]
    recovery_index = next(index for index in range(1, len(pose_stamps))
                          if pose_stamps[index]-pose_stamps[index-1] > .6)
    first_recovery, second_recovery = pose_stamps[recovery_index:recovery_index+2]
    recovery_messages = [message for message in rig.messages
                         if message.header.stamp.to_sec() >= first_recovery]
    assert any(not angular_known(message) and message.header.stamp.to_sec() < second_recovery
               for message in recovery_messages)
    assert next(message.header.stamp.to_sec() for message in recovery_messages
                if angular_known(message)) >= second_recovery

    # Explicit invalidity must clear the estimate immediately rather than
    # leaving a previously valid rate alive until its freshness timeout.
    from std_msgs.msg import Bool
    invalid = rig.rospy.Publisher('/dvl/dr_valid', Bool, queue_size=1)
    try:
        wait_for(lambda: invalid.get_num_connections() > 0)
        rig.fault('sim_pause_dr', 1.0)
        invalid.publish(False)
        wait_for(lambda: not angular_known(rig.messages[-1]), timeout=.3)
        wait_for(lambda: not rig.angular_valid[-1].data, timeout=.3)
        assert np.array_equal(angular_vector(rig.messages[-1]), np.zeros(3))
        wait_for(lambda: angular_known(rig.messages[-1]), timeout=2.)
    finally:
        invalid.unregister()

    # Reset changes the DR coordinate frame. The first new pose must be an
    # unknown-rate anchor, not a difference across the frame discontinuity.
    from std_srvs.srv import Trigger
    reset = rig.rospy.ServiceProxy('/dvl_odometry/reset', Trigger)
    stamp = rig.rospy.Time.now().to_sec()
    assert reset().success
    wait_for(lambda: any(message.header.stamp.to_sec() > stamp and
                         not angular_known(message) for message in rig.messages))
    wait_for(lambda: angular_known(rig.messages[-1]))
    samples = [message for message in rig.messages if message.header.stamp.to_sec() > stamp]
    assert max(np.linalg.norm(angular_vector(message)) for message in samples) < .15

    stamp = rig.rospy.Time.now().to_sec()
    rig.fault('sim_dropout')
    wait_for(lambda: not angular_known(rig.messages[-1]), timeout=2.)
    assert np.array_equal(angular_vector(rig.messages[-1]), np.zeros(3))
    wait_for(lambda: angular_known(rig.messages[-1]), timeout=4.)
    samples = [message for message in rig.messages if message.header.stamp.to_sec() > stamp]
    assert max(np.linalg.norm(angular_vector(message)) for message in samples) < .15


def test_dvl_raw_keeps_angular_velocity_unknown(rig_factory):
    rig = rig_factory('dvl_raw', sim_args=['yaw_rate:=0.2'])
    time.sleep(.5)
    assert all(not angular_known(message) for message in rig.messages)
    assert all(np.array_equal(angular_vector(message), np.zeros(3)) for message in rig.messages)


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
