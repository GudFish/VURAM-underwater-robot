"""Fixed figure-eight episodes, background snapshot preparation and H5 spooling.

This module does not start ROS or MuJoCo. The training checkout supplies the
episode schema, nominal motor model and reward used by the online learner.
"""
import glob
import importlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import uuid

import numpy as np

from .spec import build_obs, build_ref11
from .thruster_interface import wrench_to_pwm
from .utils import quat_to_R, yaw_quat


class FixedFigure8:
    """One periodic, horizontal Gerono curve anchored once per launch."""

    def __init__(self, state, duration=30., speed=.5, ratio=.6, dt=.05):
        values = np.asarray([duration, speed, ratio, dt], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('figure8 duration, speed, ratio and dt must be positive and finite')
        self.intervals = int(round(duration / dt))
        if self.intervals < 1:
            raise ValueError('figure8 duration must cover at least one control interval')
        self.dt = float(dt)
        self.duration = self.intervals * self.dt
        omega = 2 * math.pi / self.duration
        ax = speed / (omega * math.hypot(1., 2 * ratio))
        heading = math.atan2(quat_to_R(state[6:10])[1, 0],
                             quat_to_R(state[6:10])[0, 0])
        rotation = heading - math.atan2(2 * ratio, 1.)
        self.basis = np.array([[math.cos(rotation), -math.sin(rotation)],
                               [math.sin(rotation), math.cos(rotation)], [0., 0.]])
        self.origin = np.asarray(state[10:13], dtype=float).copy()
        theta = np.linspace(0, 2 * math.pi, 2049)
        arc = np.hypot(ax * np.cos(theta), 2 * ax * ratio * np.cos(2 * theta))
        length = float(np.trapz(arc, theta))
        # Compatible with auv_learning.task.references.get_ref. Deliberately
        # omit terminal_deceleration_fraction: the reference never stops.
        self.params = dict(family='figure8', origin=self.origin.tolist(),
                           amplitudes=[ax, ax * ratio], omega=omega, phase=0.,
                           plane_heading=rotation, in_plane_heading=0., plane_tilt=0.,
                           duration_s=self.duration, motion_duration_s=self.duration,
                           speed=float(speed), path_length_m=length,
                           mean_speed=length / self.duration, start_anchored=True)

    def get_ref(self, params, t):
        # Modulo also gives an exact shared endpoint at the lap boundary.
        theta = params['omega'] * (float(t) % self.duration)
        ax, ay = params['amplitudes']
        omega = params['omega']
        pos = self.origin + self.basis @ [ax * math.sin(theta), ay * math.sin(2 * theta)]
        vel = self.basis @ [ax * omega * math.cos(theta), 2 * ay * omega * math.cos(2 * theta)]
        acc = self.basis @ [-ax * omega**2 * math.sin(theta), -4 * ay * omega**2 * math.sin(2 * theta)]
        yaw = math.atan2(vel[1], vel[0])
        wz = (vel[0] * acc[1] - vel[1] * acc[0]) / np.dot(vel[:2], vel[:2])
        return pos, yaw_quat(yaw), vel, float(wz)


class TrainingContract:
    """Import the learner's NumPy/HDF5 contracts without running a simulator."""

    def __init__(self, learning_root):
        root = Path(learning_root).expanduser().resolve()
        if not (root / 'auv_learning').is_dir():
            raise ValueError('learning_root must contain auv_learning: {}'.format(root))
        sys.path.insert(0, str(root))
        import auv_learning
        if Path(auv_learning.__file__).resolve().parents[1] != root:
            raise ValueError('auv_learning was already imported from a different learning_root')
        from auv_learning.deployment.episode import EpisodeBundle
        from auv_learning.dynamics.allocation import WrenchAllocator
        from auv_learning.simulation.environment import reward_fn
        from auv_learning.task.math_utils import load_yaml
        self.bundle_type = EpisodeBundle
        self.reward = reward_fn
        self.env_config = load_yaml(str(root / 'config/env_config.yaml'))
        self.reward_weights = load_yaml(str(root / 'config/task_config.yaml'))['reward_weights']
        self.allocator = WrenchAllocator.from_config(self.env_config['vehicle']['com'],
                                                    self.env_config['thruster'])

    def validate_policy(self, policy):
        """A reload must preserve the meaning of normalized actions/motor state."""
        manifest, allocator = policy.manifest, self.allocator
        expected = dict(action_scales=allocator.scales,
                        negative_action_scales=allocator.negative_scales,
                        thruster_force_range_N=[allocator.min_thrust, allocator.max_forward_thrust])
        expected_model = dict(allocator.curve.as_dict(), tau=allocator.lag_tau)
        actual_model = manifest.get('thruster_model', {})
        for name, value in expected_model.items():
            if isinstance(value, str):
                if actual_model.get(name) != value:
                    raise ValueError('snapshot thruster_model.{} differs from training config'.format(name))
            else:
                expected['thruster_model.' + name] = value
        for name, value in expected.items():
            actual = (actual_model.get(name.split('.', 1)[1])
                      if name.startswith('thruster_model.') else manifest.get(name))
            if actual is None or np.shape(actual) != np.shape(value) or not np.allclose(actual, value):
                raise ValueError('snapshot {} differs from training config'.format(name))


class PolicyReloader:
    """Prepare immutable snapshots off the control thread; adopt only on request."""

    def __init__(self, model_root, load_policy, model_dir='', warn=lambda msg: None):
        self.model_root = os.path.abspath(os.path.expanduser(model_root))
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else ''
        self.load_policy, self.warn = load_policy, warn
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prepared = None
        self._highest_version = -1
        self._attempted = None
        self._thread = None

    def poll_once(self):
        if self.model_dir:
            with self._lock:
                if self._highest_version >= 0:
                    return
            snapshot, version = self.model_dir, None
        else:
            with open(os.path.join(self.model_root, 'latest.json')) as stream:
                pointer = json.load(stream)
            version = int(pointer['version'])
            snapshot = os.path.expanduser(pointer['path'])
            if not os.path.isabs(snapshot):
                snapshot = os.path.join(self.model_root, snapshot)
            elif not os.path.isdir(snapshot):
                snapshot = os.path.join(self.model_root, os.path.basename(snapshot))
            with self._lock:
                if version <= self._highest_version:
                    return
        snapshot = os.path.abspath(snapshot)
        signature = (snapshot, version, tuple(
            (os.stat(os.path.join(snapshot, name)).st_mtime_ns,
             os.stat(os.path.join(snapshot, name)).st_size)
            for name in ('deploy.pt', 'spec.json')))
        if signature == self._attempted:
            return
        self._attempted = signature
        policy = self.load_policy(snapshot)
        if version is not None and int(policy.version) != version:
            raise ValueError('latest.json version differs from snapshot manifest')
        with self._lock:
            if int(policy.version) > self._highest_version:
                self._prepared = policy
                self._highest_version = int(policy.version)

    def take_ready(self):
        with self._lock:
            candidate, self._prepared = self._prepared, None
        return candidate

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.warn('policy preparation failed; keeping current policy: {}'.format(exc))
            self._stop.wait(1.)

    def start(self):
        self._thread = threading.Thread(target=self._run, name='policy-loader', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.)


def next_iteration(data_dir):
    """Continue numbering across the inbox and learner's archived/rejected files."""
    parent = os.path.dirname(os.path.abspath(data_dir))
    maximum = -1
    for directory in (data_dir, os.path.join(parent, 'rollouts'), os.path.join(parent, 'rejected')):
        for path in glob.glob(os.path.join(directory, 'ep_*.h5')):
            match = re.match(r'ep_(\d+)(?:_|\.h5$)', os.path.basename(path))
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def write_episode(bundle, extras, data_dir):
    """Publish a fully closed H5 atomically; temporary names are not *.h5."""
    import h5py
    final = os.path.join(data_dir, bundle.episode_id + '.h5')
    if os.path.exists(final):
        raise FileExistsError(final)
    temporary = final + '.tmp-' + uuid.uuid4().hex
    try:
        bundle.to_hdf5(temporary)
        with h5py.File(temporary, 'a') as stream:
            for name, values in extras.items():
                stream['episode'].create_dataset(name, data=values)
            stream.flush()
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, final)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return final


class EpisodeWriter:
    def __init__(self, data_dir, saved=lambda path: None):
        # Import/initialize the extension before arming can start a lap, rather
        # than doing its first import on the control path's first file export.
        importlib.import_module('h5py')
        self.data_dir = os.path.abspath(os.path.expanduser(data_dir))
        os.makedirs(self.data_dir, exist_ok=True)
        self.saved = saved
        self.error = None
        self._queue = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._run, name='episode-writer', daemon=True)
        self._thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('episode writer failed: {}'.format(self.error))

    def submit(self, bundle, extras):
        self.check()
        try:
            self._queue.put_nowait((bundle, extras))
        except queue.Full:
            raise RuntimeError('episode writer queue is full')

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    path = write_episode(item[0], item[1], self.data_dir)
                    self.saved(path)
            except Exception as exc:
                self.error = exc
            finally:
                self._queue.task_done()

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self.check()


class LapRecorder:
    """Pre-action rows plus one terminal state; nominal motors outlive each lap."""

    def __init__(self, contract, spec, trajectory, policy, iteration, wrench_gain=1.):
        self.contract, self.spec = contract, spec
        self.trajectory, self.policy = trajectory, policy
        self.iteration, self.wrench_gain = iteration, wrench_gain
        self.rows, self.extra_rows = [], []
        self.history, self.previous = [], np.zeros(5)
        self.last_pwm = np.full(6, 1500.)

    def record(self, state, odom_stamp, command_stamp, publish=None, timing=None):
        terminal = len(self.rows) == self.trajectory.intervals
        time_in_lap = len(self.rows) * self.trajectory.dt
        measure = timing.measure if timing is not None else lambda phase: nullcontext()
        with measure('reference_observation'):
            ref = build_ref11(*self.trajectory.get_ref(self.trajectory.params, time_in_lap))
            obs = build_obs(state, ref, self.previous, self.spec['obs_layout'])
            context, width = self.policy.history_shape
            history = np.zeros((context, width), np.float32)
            if self.history:
                history[-len(self.history):] = self.history
        with measure('policy_inference'):
            action = np.clip(self.policy.step(obs, history, len(self.history)), -1., 1.)
        allocator = self.contract.allocator
        with measure('allocation'):
            motor_pre = allocator.motor_force.copy()
            _, force = allocator.command(action)
            pwm = self.last_pwm.copy() if terminal else wrench_to_pwm(action, self.wrench_gain)
        if not terminal:
            if publish is None:
                raise ValueError('nonterminal row requires a command publisher')
            # Publish before advancing the *nominal* hidden state. Never claim
            # that the allocated force/PWM was a measured physical motor output.
            with measure('command_publish'):
                sent_at = publish(action)
            if sent_at is not None:
                command_stamp = sent_at
        with measure('episode_record'):
            if not terminal:
                allocator.step(action, self.trajectory.dt)
                self.last_pwm = pwm.copy()
            self.rows.append(dict(time=time_in_lap, state=np.array(state, copy=True),
                                  action_norm=action, motor_force_pre=motor_pre,
                                  thruster_cmd=allocator.normalize_force(force), ref=ref,
                                  reward=self.contract.reward(state, ref, action, self.previous,
                                                              self.contract.reward_weights),
                                  done=int(terminal)))
            self.extra_rows.append(dict(odom_stamp=[odom_stamp], command_stamp=[command_stamp],
                                        dof_pwm=pwm, action_applied=[int(not terminal)]))
            if not terminal:
                self.previous = action
                self.history.append(np.r_[state, action].astype(np.float32))
                self.history = self.history[-context:]
        return terminal

    def finish(self):
        if len(self.rows) != self.trajectory.intervals + 1:
            raise ValueError('cannot export an incomplete lap')
        allocator = self.contract.allocator
        attrs = dict(schema_version=4, observation_encoding='body_relative_tracking_v1',
                     episode_id='ep_{:06d}_ros_{}'.format(self.iteration, uuid.uuid4().hex),
                     iteration=self.iteration, mode='policy', seed=0,
                     policy_version=int(self.policy.version), dr={}, ref=self.trajectory.params,
                     reward_weights=self.contract.reward_weights, termination_reason='time_limit',
                     dt_ctrl=self.trajectory.dt, episode_len_s=self.trajectory.duration,
                     n_steps=len(self.rows), condition_name='real_robot',
                     collection_source='ros_real_robot', motor_force_source='nominal_allocator_estimate',
                     thruster_model=dict(allocator.curve.as_dict(), tau=allocator.lag_tau),
                     thruster_cmd_encoding='signed_fraction_of_directional_force_limit',
                     dof_pwm_encoding='rc_6dof_channels_pitch_roll_throttle_yaw_forward_lateral_us',
                     wrench_gain=self.wrench_gain, history_reset_at_lap=True,
                     motor_state_reset_at_lap=False)
        bundle = self.contract.bundle_type.from_rows(self.rows, attrs)
        extras = {key: np.asarray([row[key] for row in self.extra_rows])
                  for key in self.extra_rows[0]}
        return bundle, extras
