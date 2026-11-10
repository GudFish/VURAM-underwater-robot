"""ROS lifecycle and wall-clock guards for continuous real-robot collection."""
import os
import threading
import time

import numpy as np
import rospy
from mavros_msgs.msg import State
from mavros_msgs.srv import MessageInterval, ParamPull
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool

from .collection import (EpisodeWriter, FixedFigure8, LapRecorder, PolicyReloader,
                         TrainingContract, next_iteration)
from .control_timing import ControlTiming
from .thruster_interface import ThrusterInterface
from .telemetry import StreamMonitor
from .traj_visualizer import TrajVisualizer
from .utils import quat_normalize


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


def wait_for_parameter_list(service, hold_neutral, timeout=180.):
    """Wait across MAVROS's per-call fetch timeout without starting collection.

    ParamPull can return success=False after its own 30 s wait while the same
    download continues. Non-forced retries join that download. Run only one
    RPC at a time off-thread so neutral output, shutdown and our overall wall
    timeout remain responsive even while the service call is blocked.
    """
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('param_wait_timeout must be positive and finite')
    deadline = time.monotonic() + timeout
    received = 0

    def keep_waiting():
        if rospy.is_shutdown():
            return False
        hold_neutral()
        if time.monotonic() >= deadline:
            raise RuntimeError('MAVROS parameter-list download was not confirmed within '
                               '{:.1f}s (last received {} parameters)'.format(timeout, received))
        return True

    def request(done, result):
        try:
            remaining = max(.001, deadline - time.monotonic())
            rospy.wait_for_service(service, timeout=min(5., remaining))
            result['response'] = rospy.ServiceProxy(service, ParamPull)(force_pull=False)
        except Exception as exc:
            result['error'] = exc
        finally:
            done.set()

    while keep_waiting():
        done, result = threading.Event(), {}
        threading.Thread(target=request, args=(done, result),
                         name='parameter-pull', daemon=True).start()
        while not done.wait(.1):
            if not keep_waiting():
                return False
            rospy.loginfo_throttle(5., 'waiting for MAVROS parameter-list download; holding neutral')
        if not keep_waiting():
            return False
        response = result.get('response')
        if response is not None:
            received = int(response.param_received)
            if response.success:
                rospy.loginfo('MAVROS parameter list received (%d params)', received)
                return True
            rospy.logwarn('MAVROS parameter pull has not completed (%d params received); '
                          'holding neutral and retrying without restarting download', received)
        else:
            rospy.logwarn('MAVROS parameter pull unavailable (%s); holding neutral and retrying',
                          result.get('error'))
        retry_at = time.monotonic() + 1.
        while time.monotonic() < retry_at:
            if not keep_waiting():
                return False
            time.sleep(max(0., min(.1, retry_at - time.monotonic())))
    return False


def configure_mavros_telemetry(service, rate, hold_neutral, timeout=15.):
    """Request attitude and position rates; measured readiness is checked next."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('mavros_telemetry_rate must be positive and finite')
    done, errors = threading.Event(), []

    def request():
        try:
            rospy.wait_for_service(service, timeout=5.)
            proxy = rospy.ServiceProxy(service, MessageInterval)
            # Quaternions avoid MAVROS ignoring Euler attitude after it has
            # already seen an ATTITUDE_QUATERNION from another ground station.
            for message_id in (31, 32):
                if rospy.is_shutdown():
                    break
                try:
                    response = proxy(message_id=message_id, message_rate=rate)
                    if not response.success:
                        errors.append('message {} request rejected'.format(message_id))
                except Exception as exc:
                    errors.append('message {}: {}'.format(message_id, exc))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            done.set()

    hold_neutral()
    threading.Thread(target=request, name='telemetry-rate-request', daemon=True).start()
    deadline = time.monotonic() + timeout
    while not done.wait(.1):
        if rospy.is_shutdown():
            return False
        hold_neutral()
        if time.monotonic() >= deadline:
            rospy.logwarn('MAVROS telemetry-rate request timed out; waiting for measured stream readiness')
            return False
    if rospy.is_shutdown():
        return False
    if errors:
        rospy.logwarn('MAVROS telemetry-rate setup incomplete (%s); '
                      'waiting for measured stream readiness', '; '.join(errors))
        return False
    rospy.loginfo('requested MAVROS attitude/position telemetry at %.1f Hz', rate)
    return True


def run_collection_node(model_dir, spec, episode_len_s):
    from .policy_inference import PolicyInference, PoseState, configure_inference_threads

    model_dir = rospy.get_param('~model_dir', model_dir)
    learning_root = os.path.expanduser(rospy.get_param('~learning_root', '~/mujoco_auv_env'))
    model_root = rospy.get_param('~model_root', os.path.join(learning_root, 'data/vnd/models'))
    data_dir = rospy.get_param('~data_dir', os.path.join(learning_root, 'data/vnd/inbox'))
    contract = TrainingContract(learning_root)
    speed = float(rospy.get_param('~ref_speed', .5))
    ratio = float(rospy.get_param('~figure8_ratio', .6))
    # Validate geometry/configuration before opening any command publisher.
    FixedFigure8(np.r_[np.zeros(6), [1., 0., 0., 0.], np.zeros(3)],
                 episode_len_s, speed, ratio, spec['dt'])
    device = rospy.get_param('~device', 'cpu')
    inference_threads = rospy.get_param('~inference_threads', 1)
    # This MUST run on the control thread. Configuring the background loader
    # alone leaves this thread using OpenMP's original (often all-core) budget.
    configure_inference_threads(inference_threads)

    def load_policy(path):
        policy = PolicyInference(path, device=device, num_threads=inference_threads)
        if not policy.loaded:
            raise ValueError('snapshot has no deploy.pt: {}'.format(path))
        policy.validate_spec(spec)
        contract.validate_policy(policy)
        policy.warmup()
        return policy

    reloader = PolicyReloader(model_root, load_policy, model_dir,
                              warn=lambda msg: rospy.logwarn_throttle(5., msg))
    writer = EpisodeWriter(data_dir, saved=lambda path: rospy.loginfo('saved real-robot lap: %s', path))
    cmd_topic = rospy.get_param('~cmd_topic', '/dof_cmd')
    thrusters = ThrusterInterface(cmd_topic)
    rospy.on_shutdown(thrusters.shutdown)
    odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
    window = float(rospy.get_param('~odom_ready_window', 1.))
    odom_timeout = float(rospy.get_param('~odom_timeout', .2))
    if not np.isfinite(odom_timeout) or odom_timeout <= 0:
        raise ValueError('odom_timeout must be positive and finite')
    odom_stream = StreamMonitor('odometry', window=window)
    request_rates = rospy.get_param('~configure_mavros_rates', odom_topic.endswith('/local_position/odom'))
    imu_stream = StreamMonitor('MAVROS IMU', window=window) if request_rates else None
    pose = PoseState(odom_topic, twist_in_world=rospy.get_param('~twist_in_world', False),
                     on_sample=odom_stream.push)
    gate = ControlGate(ready_required=rospy.get_param('~wait_ready', True), odom_timeout=odom_timeout)
    subscribers = [
        rospy.Subscriber(rospy.get_param('~state_topic', '/mavros/state'), State, gate.update_state),
        rospy.Subscriber(rospy.get_param('~ready_topic', cmd_topic + '/ready'), Bool,
                         lambda msg: setattr(gate, 'ready', msg.data)),
        rospy.Subscriber(rospy.get_param('~enabled_topic', cmd_topic + '/enabled'), Bool,
                         lambda msg: setattr(gate, 'enabled', msg.data)),
    ]
    if imu_stream is not None:
        subscribers.append(rospy.Subscriber(rospy.get_param('~imu_topic', '/mavros/imu/data'), Imu,
                           lambda msg: imu_stream.push(msg.header.stamp.to_sec(), time.monotonic()),
                           queue_size=10))
    viz = TrajVisualizer(rospy.get_param('~viz', False),
                         frame_id=rospy.get_param('~viz_frame_id', 'map'))
    watchdog = None
    try:
        if not np.isfinite(thrusters.wrench_gain):
            raise ValueError('wrench_gain must be finite')
        iteration = next_iteration(writer.data_dir)
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
        if request_rates:
            rate = float(rospy.get_param('~mavros_telemetry_rate', max(30., 1.5 * spec['control_rate'])))
            if not np.isfinite(rate) or rate < spec['control_rate']:
                raise ValueError('mavros_telemetry_rate must be at least the policy control rate')
            configure_mavros_telemetry(
                rospy.get_param('~message_interval_service', '/mavros/set_message_interval'),
                rate, thrusters.set_zero)

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
            if policy is not None and reason is None:
                break
            thrusters.set_zero()
            rospy.loginfo_throttle(5., 'collection waiting: %s', reason or 'valid published policy')
            if reason and 'odometry' in reason:
                rospy.loginfo_throttle(5., 'odometry readiness: %s. For MAVProxy forwarding, '
                                      'disable stream-rate overrides with: set streamrate -1',
                                      odom_stream.summary(now))
            time.sleep(.05)
        if rospy.is_shutdown():
            return

        first_snapshot = snapshot
        trajectory = FixedFigure8(snapshot['state'], episode_len_s, speed, ratio, spec['dt'])
        viz.start_episode(trajectory.params, trajectory, trajectory.duration)
        lap = LapRecorder(contract, spec, trajectory, policy, iteration, thrusters.wrench_gain)

        def runtime_reason(snapshot, now, ros_now):
            reason = gate.reason(snapshot, now, ros_now)
            if reason:
                return reason + '; ' + odom_stream.summary(now)
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
            writer.check()
            now, ros_now = time.monotonic(), rospy.Time.now().to_sec()
            return runtime_reason(pose.step(), now, ros_now)

        timing = ControlTiming()
        watchdog = CollectionWatchdog(check, thrusters.set_zero, detail=timing.summary)

        def publish(action):
            return watchdog.publish(action, thrusters.set_wrench_action)

        rospy.loginfo('collecting horizontal figure8: T=%.3fs peak_speed=%.3fm/s ratio=%.3f '
                      'start=%s policy=v%s inbox=%s', trajectory.duration, speed, ratio,
                      trajectory.origin, policy.version, writer.data_dir)
        rospy.loginfo('verified stream readiness: %s', odom_stream.summary(time.monotonic()))
        while not rospy.is_shutdown():
            tick = time.monotonic()
            if watchdog.fault:
                raise RuntimeError(watchdog.fault)
            with timing.measure('state_check'):
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
                    writer.submit(*lap.finish())
                    iteration += 1
                    candidate = reloader.take_ready()
                    if candidate is not None:
                        policy = candidate
                        rospy.loginfo('switched to policy v%s at lap boundary', policy.version)
                    lap = LapRecorder(contract, spec, trajectory, policy, iteration, thrusters.wrench_gain)
                    viz.start_episode(trajectory.params, trajectory, trajectory.duration)
                # The terminal row is observational only. Issue the next lap's
                # first action now, using the very same measured boundary state.
                lap.record(state, snapshot['time'], stamp, publish, timing=timing)
            with timing.measure('visualization'):
                if len(lap.rows) % 10 == 1:
                    viz.add_actual_point(state[10:13])
            with timing.measure('wait_next_tick'):
                time.sleep(max(0., tick + spec['dt'] - time.monotonic()))
    except (Exception, KeyboardInterrupt) as exc:
        rospy.logerr('collection stopped; incomplete lap discarded: %s', exc)
    finally:
        if watchdog is not None:
            watchdog.close()
        thrusters.set_zero()
        reloader.close()
        try:
            writer.close()
        finally:
            for sub in subscribers:
                sub.unregister()
            pose.sub.unregister()
            thrusters.shutdown()
