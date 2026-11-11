"""Controller-independent lap stepping, optional recording and H5 spooling."""
import glob
import importlib
from contextlib import nullcontext
import os
import queue
import re
import threading
import uuid

import numpy as np

from control.observation import build_obs, build_ref11
from control.thruster_interface import wrench_to_pwm


def next_iteration(data_dir):
    """Continue numbering across the inbox and learner's archived/rejected files."""
    parent = os.path.dirname(os.path.abspath(data_dir))
    maximum = -1
    for directory in (data_dir, os.path.join(parent, 'rollouts'), os.path.join(parent, 'rejected')):
        for path in glob.glob(os.path.join(directory, 'ep_*.h5')):
            match = re.match(r'ep_(\d+)(?:_|\.h5$)', os.path.basename(path))
            if match:
                maximum = max(maximum, int(match.group(1)))
    floor = 0
    if os.path.isfile(os.path.join(parent, 'lan', 'receipts.sqlite3')):
        # Sequential LAN -> local use must honor receipts even if their H5
        # files were archived elsewhere. Do not create a ledger in local mode.
        from auv_learning.deployment.lan_store import ledger_next_iteration
        floor = ledger_next_iteration(parent)
    return max(maximum + 1, floor)


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

    def __init__(self, contract, spec, trajectory, policy, iteration, wrench_gain=1.,
                 save_data=True, state_metadata=None):
        self.contract, self.spec = contract, spec
        self.trajectory, self.policy = trajectory, policy
        self.iteration, self.wrench_gain = iteration, wrench_gain
        self.save_data = save_data
        self.state_metadata = dict(state_metadata or {})
        self.step_index = 0
        self.rows, self.extra_rows = [], []
        self.history, self.previous = [], np.zeros(5)
        self.last_pwm = np.full(6, 1500.)

    def record(self, state, odom_stamp, command_stamp, publish=None, timing=None):
        if self.step_index > self.trajectory.intervals:
            raise ValueError('lap is already complete')
        terminal = self.step_index == self.trajectory.intervals
        time_in_lap = self.step_index * self.trajectory.dt
        measure = timing.measure if timing is not None else lambda phase: nullcontext()
        with measure('reference_observation'):
            ref = build_ref11(*self.trajectory.get_ref(self.trajectory.params, time_in_lap))
            obs = build_obs(state, ref, self.previous, self.spec['obs_layout'])
            context, width = self.policy.history_shape
            history = np.zeros((context, width), np.float32)
            if self.history:
                history[-len(self.history):] = self.history
        with measure('controller_step'):
            action = np.asarray(self.policy.step(obs, history, len(self.history)), dtype=float)
            if action.shape != (5,) or not np.isfinite(action).all():
                raise ValueError('controller must return five finite normalized wrench actions')
            action = np.clip(action, -1., 1.)
        allocator = self.contract.allocator
        with measure('allocation'):
            if self.save_data:
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
            if self.save_data:
                if not terminal:
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
            self.step_index += 1
        return terminal

    def finish(self):
        if not self.save_data:
            raise ValueError('data saving is disabled')
        if len(self.rows) != self.trajectory.intervals + 1:
            raise ValueError('cannot export an incomplete lap')
        allocator = self.contract.allocator
        attrs = dict(schema_version=4, observation_encoding='body_relative_tracking_v1',
                     episode_id='ep_{:06d}_ros_{}'.format(self.iteration, uuid.uuid4().hex),
                     iteration=self.iteration, mode=getattr(self.policy, 'controller_type', 'policy'), seed=0,
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
        attrs['controller_type'] = attrs['mode']
        if hasattr(self.policy, 'parameters'):
            attrs['controller_parameters'] = self.policy.parameters
        attrs.update(self.state_metadata)
        bundle = self.contract.bundle_type.from_rows(self.rows, attrs)
        extras = {key: np.asarray([row[key] for row in self.extra_rows])
                  for key in self.extra_rows[0]}
        return bundle, extras
