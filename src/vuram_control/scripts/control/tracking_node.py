"""ROS lifecycle and wall-clock guards for continuous real-robot collection."""
from contextlib import ExitStack
import os
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from mavros_msgs.msg import State
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from data_collection.contract import TrainingContract
from data_collection.episodes import EpisodeWriter, LapRecorder, next_iteration
from trajectory.figure8 import FixedFigure8
from control.state import PoseState
from control.dvl_health import DvlHealth
from control.control_timing import ControlTiming
from control.thruster_interface import ThrusterInterface
from control.telemetry import StreamMonitor
from control.mavros_startup import wait_for_parameter_list
from visualization.traj_visualizer import TrajVisualizer
from utils.math3d import quat_normalize


class ControlGate:
    """Latched bridge flags, FC heartbeat and both odometry freshness clocks."""

    def __init__(self, ready_required=True, odom_timeout=.2, state_timeout=3.):
        self.ready = not ready_required
        self.enabled = False
        self.state = None
        self.state_received = None
        self.odom_timeout, self.state_timeout = odom_timeout, state_timeout
        self._last_stamp, self._stamp_advanced = None, None
        self._last_received = -float('inf')
        self._lock = threading.Lock()

    def update_state(self, msg):
        self.state_received = time.monotonic()
        self.state = msg

    def reason(self, snapshot, now, ros_now, allow_time_reset=False):
        with self._lock:
            return self._reason(snapshot, now, ros_now, allow_time_reset)

    def _reason(self, snapshot, now, ros_now, allow_time_reset):
        if not self.ready:
            return 'thruster bridge is not ready'
        if not self.enabled:
            return 'thruster bridge is disabled'
        state = self.state
        if state is None or now - self.state_received > self.state_timeout:
            return 'MAVROS state heartbeat is missing/stale'
        if not state.connected or not state.armed or state.mode != 'MANUAL':
            return 'vehicle must be connected, armed and in MANUAL mode'
        if snapshot is None:
            return 'no odometry'
        physical, stamp = snapshot['state'], snapshot['time']
        if (physical.shape != (13,) or not np.isfinite(physical).all()
                or np.linalg.norm(physical[6:10]) < 1e-12 or not np.isfinite(stamp)):
            return 'invalid odometry state/quaternion/timestamp'
        # The watchdog and control loop can sample concurrently. An older
        # received snapshot is not a new callback reporting a clock rollback.
        if snapshot['received_monotonic'] >= self._last_received:
            if self._last_stamp is not None and stamp < self._last_stamp:
                if allow_time_reset:
                    # MAVROS can establish a new clock offset during startup.
                    # Stream readiness still requires a full healthy window
                    # after this reset; in-motion resets remain fatal.
                    self._last_stamp, self._stamp_advanced = stamp, now
                    self._last_received = snapshot['received_monotonic']
                return 'odometry timestamp moved backwards'
            if self._last_stamp is None or stamp > self._last_stamp:
                self._last_stamp, self._stamp_advanced = stamp, now
            self._last_received = snapshot['received_monotonic']
        receipt_age = now - snapshot['received_monotonic']
        stamp_stalled = now - self._stamp_advanced
        header_age = ros_now - stamp
        if receipt_age > self.odom_timeout:
            return 'odometry receive timeout: callback_age={:.3f}s limit={:.3f}s'.format(
                receipt_age, self.odom_timeout)
        if stamp_stalled > self.odom_timeout:
            return 'odometry timestamp stalled: no advance for {:.3f}s (callback_age={:.3f}s)'.format(
                stamp_stalled, receipt_age)
        if header_age > self.odom_timeout:
            return 'odometry header is stale: header_age={:.3f}s callback_age={:.3f}s limit={:.3f}s'.format(
                header_age, receipt_age, self.odom_timeout)
        if stamp - ros_now > self.odom_timeout:
            return 'odometry timestamp is ahead of the ROS clock: lead={:.3f}s callback_age={:.3f}s'.format(
                -header_age, receipt_age)
        return None


class CollectionWatchdog:
    """Latch a fault and neutralize even while inference/the control thread stalls.

    Publish and neutralization share a lock, so a late inference result cannot
    resume non-neutral commands after a fault has been detected.
    """

    def __init__(self, check, neutral, timeout=.1, detail=None):
        self.check, self.neutral, self.timeout = check, neutral, timeout
        self.detail = detail
        self.fault = None
        self._last_command = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='collection-watchdog', daemon=True)
        self._thread.start()

    def _validate(self):
        if self.fault is None:
            try:
                self.fault = self.check()
                if self.fault is None:
                    self.fault = self._interval_reason()
            except Exception as exc:
                self.fault = str(exc)
            if self.fault:
                if self.detail is not None:
                    try:
                        self.fault += '; ' + self.detail()
                    except Exception:
                        pass  # diagnostics must never prevent neutralization
                self.neutral()
        if self.fault:
            raise RuntimeError(self.fault)

    def _interval_reason(self):
        elapsed = time.monotonic() - self._last_command
        if elapsed > self.timeout:
            return 'control command interval exceeded {:.3f}s (actual={:.3f}s)'.format(
                self.timeout, elapsed)
        return None

    def publish(self, action, send):
        with self._lock:
            self._validate()
            send(action)
            # A slow publisher must not reset the deadline and hide a gap.
            # Validate again before accepting this send as a healthy tick.
            self._validate()
            self._last_command = time.monotonic()
            return rospy.Time.now().to_sec()

    def _run(self):
        while not self._stop.wait(min(.025, self.timeout / 4)):
            with self._lock:
                try:
                    self._validate()
                except RuntimeError:
                    return

    def close(self):
        self._stop.set()
        self._thread.join()


class StaticControllerSource:
    """A fixed controller with the same lifecycle as the policy snapshot source."""

    def __init__(self, controller):
        self.controller = controller

    def start(self):
        pass

    def take_ready(self):
        controller, self.controller = self.controller, None
        return controller

    def close(self):
        pass


def run_tracking_node(spec, episode_len_s, source_factory):
    """Run an anchored figure8; controller sources supply prepared controllers.

    Network I/O and H5 writes remain outside the control thread. Test mode
    keeps control, lap boundaries and policy updates, without creating a writer
    or accumulating episode arrays.
    """

    transport_mode = rospy.get_param('~transport_mode', 'local')
    if transport_mode not in ('local', 'lan'):
        raise ValueError('transport_mode must be local or lan')
    odom_source = rospy.get_param('~odom_source', 'mavros')
    if odom_source not in ('mavros', 'dvl'):
        raise ValueError('odom_source must be mavros or dvl')
    dvl_mode = rospy.get_param('~dvl_mode', 'dvl_fused')
    dvl_health = (DvlHealth(mode=dvl_mode, timeout=float(rospy.get_param('~dvl_timeout', 1.)),
                            now=lambda: time.monotonic()) if odom_source == 'dvl' else None)
    learning_root = os.path.expanduser(rospy.get_param('~learning_root', '~/mujoco_auv_env'))
    storage_root = os.path.join(learning_root, 'data/vnd', 'lan' if transport_mode == 'lan' else '')
    model_root = rospy.get_param('~model_root', os.path.join(storage_root, 'models'))
    data_dir = rospy.get_param('~data_dir', os.path.join(
        storage_root, 'outbox' if transport_mode == 'lan' else 'inbox'))
    save_data = rospy.get_param('~save_data', True)
    if not isinstance(save_data, bool):
        raise ValueError('save_data must be a boolean')
    contract = TrainingContract(learning_root)
    speed = float(rospy.get_param('~ref_speed', .5))
    ratio = float(rospy.get_param('~figure8_ratio', .6))
    # Validate geometry/configuration before opening any command publisher.
    FixedFigure8(np.r_[np.zeros(6), [1., 0., 0., 0.], np.zeros(3)],
                 episode_len_s, speed, ratio, spec['dt'])
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
    state_metadata = dict(odom_source=odom_source, odom_topic=odom_topic,
                          angular_velocity_source=('dvl_dr_quaternion_difference_lowpass'
                                                   if odom_source == 'dvl' and dvl_mode == 'dvl_fused'
                                                   else 'odometry_message'))
    if odom_source == 'dvl':
        state_metadata['dvl_mode'] = dvl_mode
    window = float(rospy.get_param('~odom_ready_window', 1.))
    odom_timeout = float(rospy.get_param('~odom_timeout', .2))
    if not np.isfinite(odom_timeout) or odom_timeout <= 0:
        raise ValueError('odom_timeout must be positive and finite')
    odom_stream = StreamMonitor('odometry', window=window)
    check_imu = rospy.get_param('~check_mavros_imu',
                                odom_topic.endswith('/local_position/odom') or
                                (odom_source == 'dvl' and dvl_mode == 'mavros_fused'))
    imu_stream = StreamMonitor('MAVROS IMU', window=window) if check_imu else None
    reloader = writer = thrusters = pose = viz = watchdog = None
    subscribers = []
    try:
        reloader = source_factory(contract, model_root)
        writer = (EpisodeWriter(data_dir, saved=lambda path: rospy.loginfo('saved real-robot lap: %s', path))
                  if save_data else None)
        cmd_topic = rospy.get_param('~cmd_topic', '/dof_cmd')
        thrusters = ThrusterInterface(cmd_topic)
        rospy.on_shutdown(thrusters.shutdown)
        pose = PoseState(odom_topic, twist_in_world=rospy.get_param('~twist_in_world', False),
                         on_sample=odom_stream.push)
        gate = ControlGate(ready_required=rospy.get_param('~wait_ready', True), odom_timeout=odom_timeout)
        subscribers.append(rospy.Subscriber(rospy.get_param('~state_topic', '/mavros/state'), State, gate.update_state))
        subscribers.append(rospy.Subscriber(rospy.get_param('~ready_topic', cmd_topic + '/ready'), Bool,
                             lambda msg: setattr(gate, 'ready', msg.data)))
        subscribers.append(rospy.Subscriber(rospy.get_param('~enabled_topic', cmd_topic + '/enabled'), Bool,
                             lambda msg: setattr(gate, 'enabled', msg.data)))
        if imu_stream is not None:
            subscribers.append(rospy.Subscriber(rospy.get_param('~imu_topic', '/mavros/imu/data'), Imu,
                               lambda msg: imu_stream.push(msg.header.stamp.to_sec(), time.monotonic()),
                               queue_size=10))
        if dvl_health is not None:
            for topic, callback in (
                    ('/dvl/connected', dvl_health.update_connected),
                    ('/dvl/resetting', dvl_health.update_resetting),
                    ('/dvl/velocity_valid', dvl_health.update_velocity_valid)):
                subscribers.append(rospy.Subscriber(topic, Bool,
                                   lambda msg, callback=callback: callback(msg.data), queue_size=10))
            subscribers.append(rospy.Subscriber('/dvl/velocity', TwistWithCovarianceStamped,
                               lambda msg: dvl_health.update_velocity(
                                   msg.header.stamp.to_sec(), time.monotonic()), queue_size=10))
            if dvl_mode != 'mavros_fused':
                subscribers.append(rospy.Subscriber('/dvl/dr_valid', Bool,
                                   lambda msg: dvl_health.update_dr_valid(msg.data), queue_size=10))
                subscribers.append(rospy.Subscriber('/dvl/dead_reckoning', PoseWithCovarianceStamped,
                                   lambda msg: dvl_health.update_dr(
                                       msg.header.stamp.to_sec(), time.monotonic()), queue_size=10))
            if dvl_mode == 'dvl_fused':
                subscribers.append(rospy.Subscriber('/dvl/angular_velocity_valid', Bool,
                                   lambda msg: dvl_health.update_angular_velocity_valid(msg.data), queue_size=10))
        viz = TrajVisualizer(rospy.get_param('~viz', False),
                             frame_id=rospy.get_param('~viz_frame_id', 'map'))
        if not np.isfinite(thrusters.wrench_gain):
            raise ValueError('wrench_gain must be finite')
        iteration = next_iteration(writer.data_dir) if writer is not None else 0
        reloader.start()
        while not rospy.is_shutdown() and not gate.ready:
            thrusters.set_zero()
            rospy.loginfo_throttle(5., 'collection waiting for thruster bridge ready')
            time.sleep(.05)
        if rospy.is_shutdown():
            return
        if rospy.get_param('~wait_param_list', True):
            service = rospy.get_param('~param_pull_service', '/mavros/param/pull')
            timeout = float(rospy.get_param('~param_wait_timeout', 180.))
            if not wait_for_parameter_list(service, thrusters.set_zero, timeout):
                return

        def stream_readiness(now, ros_now):
            for stream in (odom_stream, imu_stream):
                if stream is not None:
                    reason = stream.ready_reason(now, ros_now, min_rate=.95 * spec['control_rate'],
                                                 max_gap=2. * spec['dt'], timeout=odom_timeout)
                    if reason:
                        return reason
            return None

        policy = None
        while not rospy.is_shutdown():
            prepared = reloader.take_ready()
            if prepared is not None:
                policy = prepared
            snapshot = pose.step()
            now, ros_now = time.monotonic(), rospy.Time.now().to_sec()
            reason = gate.reason(snapshot, now, ros_now, allow_time_reset=True) or stream_readiness(now, ros_now)
            if reason is None and dvl_health is not None:
                reason = dvl_health.reason(now, ros_now)
            if policy is not None and reason is None:
                break
            thrusters.set_zero()
            rospy.loginfo_throttle(5., 'collection waiting: %s', reason or 'valid controller')
            if reason and 'odometry' in reason:
                rospy.loginfo_throttle(5., 'odometry readiness: %s. For MAVProxy forwarding, '
                                      'disable stream-rate overrides with: set streamrate -1',
                                      odom_stream.summary(now))
            time.sleep(.05)
        if rospy.is_shutdown():
            return

        first_snapshot = snapshot
        if dvl_health is not None:
            dvl_health.arm()
        trajectory = FixedFigure8(snapshot['state'], episode_len_s, speed, ratio, spec['dt'])
        viz.start_episode(trajectory.params, trajectory, trajectory.duration)
        lap = LapRecorder(contract, spec, trajectory, policy, iteration, thrusters.wrench_gain,
                          save_data=save_data, state_metadata=state_metadata)

        def runtime_reason(snapshot, now, ros_now):
            reason = gate.reason(snapshot, now, ros_now)
            if reason:
                return reason + '; ' + odom_stream.summary(now)
            if dvl_health is not None:
                reason = dvl_health.reason(now, ros_now)
                if reason:
                    return reason
            # Callback monitors retain short clock faults even if another
            # packet has replaced the pose by the next control/watchdog tick.
            for stream in (odom_stream, imu_stream):
                if stream is not None:
                    reason = stream.freshness_reason(now, ros_now, odom_timeout)
                    if reason:
                        return reason + '; ' + stream.summary(now)
            return None

        def check():
            if rospy.is_shutdown():
                return 'ROS shutdown'
            if writer is not None:
                writer.check()
            now, ros_now = time.monotonic(), rospy.Time.now().to_sec()
            return runtime_reason(pose.step(), now, ros_now)

        timing = ControlTiming()
        watchdog = CollectionWatchdog(check, thrusters.set_zero, detail=timing.summary)

        def publish(action):
            return watchdog.publish(action, thrusters.set_wrench_action)

        rospy.loginfo('tracking horizontal figure8: controller=%s T=%.3fs peak_speed=%.3fm/s ratio=%.3f '
                      'start=%s version=%s save_data=%s data_dir=%s',
                      getattr(policy, 'controller_type', 'policy'), trajectory.duration, speed, ratio,
                      trajectory.origin, policy.version, save_data,
                      writer.data_dir if writer is not None else '(disabled)')
        rospy.loginfo('verified stream readiness: %s', odom_stream.summary(time.monotonic()))
        while not rospy.is_shutdown():
            tick = time.monotonic()
            if watchdog.fault:
                raise RuntimeError(watchdog.fault)
            with timing.measure('state_check'):
                if writer is not None:
                    writer.check()
                snapshot = first_snapshot if first_snapshot is not None else pose.step()
                first_snapshot = None
                reason = runtime_reason(snapshot, tick, rospy.Time.now().to_sec())
                if reason:
                    raise RuntimeError(reason)
                state = snapshot['state'].copy()
                state[6:10] = quat_normalize(state[6:10])
                stamp = rospy.Time.now().to_sec()
            terminal = lap.record(state, snapshot['time'], stamp, publish, timing=timing)
            if terminal:
                with timing.measure('lap_boundary'):
                    if writer is not None:
                        writer.submit(*lap.finish())
                    iteration += 1
                    candidate = reloader.take_ready()
                    if candidate is not None:
                        policy = candidate
                        rospy.loginfo('switched to controller version %s at lap boundary', policy.version)
                    lap = LapRecorder(contract, spec, trajectory, policy, iteration, thrusters.wrench_gain,
                                      save_data=save_data, state_metadata=state_metadata)
                    viz.start_episode(trajectory.params, trajectory, trajectory.duration)
                # The terminal row is observational only. Issue the next lap's
                # first action now, using the very same measured boundary state.
                lap.record(state, snapshot['time'], stamp, publish, timing=timing)
            with timing.measure('visualization'):
                if lap.step_index % 10 == 1:
                    viz.add_actual_point(state[10:13])
            with timing.measure('wait_next_tick'):
                time.sleep(max(0., tick + spec['dt'] - time.monotonic()))
    except (Exception, KeyboardInterrupt) as exc:
        rospy.logerr('collection stopped; incomplete lap discarded: %s', exc)
    finally:
        # LIFO cleanup also drains completed laps if another cleanup raises.
        # Each resource is registered only after successful construction.
        with ExitStack() as cleanup:
            if thrusters is not None:
                cleanup.callback(thrusters.shutdown)
            if pose is not None:
                cleanup.callback(pose.sub.unregister)
            for sub in subscribers:
                cleanup.callback(sub.unregister)
            if viz is not None:
                cleanup.callback(viz.close)
            if writer is not None:
                cleanup.callback(writer.close)
            if reloader is not None:
                cleanup.callback(reloader.close)
            if thrusters is not None:
                cleanup.callback(thrusters.set_zero)
            if watchdog is not None:
                cleanup.callback(watchdog.close)
