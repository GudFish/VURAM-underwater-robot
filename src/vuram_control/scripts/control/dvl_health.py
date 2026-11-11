"""Source-data health checks independent of predicted odometry publication.

A fused odometry publisher can keep advancing its output timestamps while
holding an old DVL state. These checks therefore inspect actual DVL input
samples as well as the driver's connection and validity flags.
"""
import math
import threading
import time


class DvlHealth:
    """Recoverable startup checks and latched faults after :meth:`arm`.

    ``now`` and receipt timestamps use the monotonic clock. Source timestamps
    and ``ros_now`` use the ROS clock. A true validity flag alone never counts
    as a measurement. ``mavros_fused`` uses DVL velocity with IMU attitude and
    deliberately does not require device dead-reckoning messages. Device fusion
    additionally requires a valid angular velocity estimate, so unknown zero
    values cannot enter tracking as if they were measured stationary motion.
    """

    MODES = ('dvl_raw', 'dvl_fused', 'mavros_fused')

    def __init__(self, mode='dvl_fused', timeout=1., now=time.monotonic):
        if mode not in self.MODES:
            raise ValueError('unknown DVL mode: {}'.format(mode))
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('DVL timeout must be positive and finite')
        if not callable(now):
            raise ValueError('DVL monotonic clock must be callable')
        self.mode = mode
        self._now = now
        self._lock = threading.RLock()
        self._armed = False
        self._fault = None
        self._flags = dict(connected=None, resetting=None,
                           velocity_valid=None, dr_valid=None,
                           angular_velocity_valid=None)
        self._samples = dict(velocity=None, dr=None)
        self._sample_errors = dict(velocity=None, dr=None)

    def _latch(self, reason):
        if self._armed and self._fault is None and reason is not None:
            self._fault = reason
        return self._fault or reason

    def _flag_reason(self):
        for name, desired, message in (
                ('connected', True, 'DVL disconnected'),
                ('resetting', False, 'DVL reset in progress'),
                ('velocity_valid', True, 'DVL velocity is invalid')):
            value = self._flags[name]
            if value is None:
                return 'waiting for DVL {} status'.format(name)
            if value != desired:
                return message
        if self.mode != 'mavros_fused':
            if self._flags['dr_valid'] is None:
                return 'waiting for DVL dead-reckoning validity status'
            if not self._flags['dr_valid']:
                return 'DVL dead-reckoning is invalid'
        if self.mode == 'dvl_fused':
            if self._flags['angular_velocity_valid'] is None:
                return 'waiting for DVL angular velocity validity status'
            if not self._flags['angular_velocity_valid']:
                return 'DVL angular velocity estimate is invalid'
        return None

    def _update_flag(self, name, value, invalid_reason):
        with self._lock:
            self._flags[name] = bool(value)
            invalid = bool(value) if name == 'resetting' else not bool(value)
            required = ((name != 'dr_valid' or self.mode != 'mavros_fused') and
                        (name != 'angular_velocity_valid' or self.mode == 'dvl_fused'))
            if invalid and required:
                self._latch(invalid_reason)

    def update_connected(self, value):
        self._update_flag('connected', value, 'DVL disconnected')

    def update_resetting(self, value):
        self._update_flag('resetting', value, 'DVL reset in progress')

    def update_velocity_valid(self, value):
        self._update_flag('velocity_valid', value, 'DVL velocity is invalid')

    def update_dr_valid(self, value):
        self._update_flag('dr_valid', value, 'DVL dead-reckoning is invalid')

    def update_angular_velocity_valid(self, value):
        self._update_flag('angular_velocity_valid', value,
                          'DVL angular velocity estimate is invalid')

    def _update_sample(self, name, stamp, received):
        with self._lock:
            try:
                stamp = float(stamp)
                received = float(self._now() if received is None else received)
                if not math.isfinite(stamp) or not math.isfinite(received):
                    raise ValueError('non-finite timestamp')
            except (TypeError, ValueError, OverflowError):
                error = 'DVL {} has a non-finite/invalid timestamp'.format(name)
                self._sample_errors[name] = error
                if name == 'velocity' or self.mode != 'mavros_fused':
                    self._latch(error)
                return
            self._samples[name] = (stamp, received)
            self._sample_errors[name] = None

    def update_velocity(self, stamp, received=None):
        self._update_sample('velocity', stamp, received)

    def update_dr(self, stamp, received=None):
        self._update_sample('dr', stamp, received)

    def arm(self):
        """Enable fault latching; there is intentionally no reset/resume API."""
        with self._lock:
            self._armed = True
            self._latch(self._flag_reason())
            for name in self._required_samples():
                self._latch(self._sample_errors[name])

    enable = arm

    def _required_samples(self):
        return ('velocity',) if self.mode == 'mavros_fused' else ('velocity', 'dr')

    def reason(self, now, ros_now):
        """Return a health failure or ``None``; armed failures never recover."""
        with self._lock:
            if self._fault is not None:
                return self._fault
            try:
                now, ros_now = float(now), float(ros_now)
                if not math.isfinite(now) or not math.isfinite(ros_now):
                    raise ValueError('non-finite clock')
            except (TypeError, ValueError, OverflowError):
                return self._latch('DVL health clock is non-finite/invalid')
            reason = self._flag_reason()
            if reason is not None:
                return self._latch(reason)
            # ROS epoch timestamps can round exact timeout boundaries by a
            # fraction of a microsecond; apply the same tolerance to both clocks.
            limit = self.timeout + 1e-6
            for name in self._required_samples():
                if self._sample_errors[name]:
                    return self._latch(self._sample_errors[name])
                sample = self._samples[name]
                if sample is None:
                    return self._latch('waiting for actual DVL {} sample'.format(name))
                stamp, received = sample
                receive_age, source_age = now - received, ros_now - stamp
                if receive_age > limit:
                    return self._latch('DVL {} receive timeout: {:.3f}s > {:.3f}s'.format(
                        name, receive_age, self.timeout))
                if receive_age < -limit:
                    return self._latch('DVL {} receipt timestamp is ahead of monotonic clock'.format(name))
                if source_age > limit:
                    return self._latch('DVL {} source timestamp is stale: {:.3f}s > {:.3f}s'.format(
                        name, source_age, self.timeout))
                if source_age < -limit:
                    return self._latch('DVL {} source timestamp is ahead of ROS clock'.format(name))
            return None
