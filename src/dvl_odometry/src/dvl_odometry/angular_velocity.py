"""Body angular rates derived from successive DVL dead-reckoning attitudes.

This is a finite-difference estimate, not a gyro measurement. The estimate
describes the preceding DR interval and is held only until ``timeout``. Input
quaternions use ROS xyzw order and rotate body vectors into the world frame.
Attitude covariances describe small rotations in each attitude's body frame.
"""
import math

import numpy as np

from .core import UNKNOWN_VARIANCE, covariance, rotation


def _quaternion(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError('expected a finite xyzw quaternion')
    norm = np.linalg.norm(result)
    if not math.isfinite(norm) or norm < 1e-6:
        raise ValueError('invalid quaternion norm')
    return result / norm


def _relative(previous, current):
    """Return the shortest SO(3) logarithm of previous^-1 * current."""
    pv, pw = -previous[:3], previous[3]
    cv, cw = current[:3], current[3]
    vector = pw * cv + cw * pv + np.cross(pv, cv)
    scalar = pw * cw - np.dot(pv, cv)
    if scalar < 0:
        vector, scalar = -vector, -scalar
    # At exactly pi the axis sign is unobservable; choose one deterministically.
    if abs(scalar) < 1e-14:
        for component in vector:
            if abs(component) > 1e-14:
                if component < 0:
                    vector = -vector
                break
        scalar = abs(scalar)
    length = np.linalg.norm(vector)
    if length < 1e-8:
        return 2. * vector
    angle = 2. * math.atan2(length, max(0., scalar))
    return vector * (angle / length)


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _right_jacobian_inverse(vector):
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    if angle < 1e-5:
        coefficient = 1. / 12. + angle * angle / 720.
    else:
        coefficient = (1. - .5 * angle / math.tan(.5 * angle)) / angle ** 2
    return np.eye(3) + .5 * cross + coefficient * cross.dot(cross)


class AngularVelocityEstimator:
    """Differentiate DR orientation, reject jumps, and optionally filter rates.

    ``update`` returns whether a rate could be estimated. Invalid samples clear
    the estimate; valid poses separated by an excessive gap, duplicate/backward
    time, or a jump become fresh anchors, requiring another pose before reuse.
    ``max_rate`` is the magnitude in rad/s; ``max_angle`` is the largest allowed
    inter-pose rotation in radians. These reject resets rather than clipping
    fabricated angular rates into apparently valid data.

    Filtering uses the actual DR interval. The preceding filtered vector and
    covariance are rotated into the current body frame first. Covariances use a
    conservative convex combination rather than assuming independent successive
    differences, which share an attitude sample.
    """
    def __init__(self, timeout=1., filter_tau=.2, max_rate=3., max_angle=1.2,
                 min_std=.05):
        values = (timeout, filter_tau, max_rate, max_angle, min_std)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('angular-rate settings must be finite')
        if timeout <= 0 or filter_tau < 0 or max_rate <= 0 or min_std <= 0:
            raise ValueError('timeout/rate/std must be positive; tau non-negative')
        if not 0 < max_angle <= math.pi:
            raise ValueError('max_angle must be in (0, pi]')
        self.timeout = float(timeout)
        self.filter_tau = float(filter_tau)
        self.max_rate = float(max_rate)
        self.max_angle = float(max_angle)
        self.min_variance = float(min_std) ** 2
        self.reset()

    def reset(self):
        self.stamp = None
        self.quaternion = None
        self.pose_covariance = None
        self.omega = np.zeros(3)
        self.rate_covariance = np.eye(3) * UNKNOWN_VARIANCE
        self.valid = False

    def _anchor(self, stamp, quaternion, pose_covariance):
        self.stamp = stamp
        self.quaternion = quaternion
        self.pose_covariance = pose_covariance

    def update(self, stamp, quaternion, pose_covariance=None):
        try:
            stamp = float(stamp)
            if not math.isfinite(stamp):
                raise ValueError('non-finite DR stamp')
            current = _quaternion(quaternion)
            current_cov = covariance(pose_covariance)
        except (ValueError, TypeError, OverflowError):
            self.reset()
            return False

        if self.stamp is None:
            self._anchor(stamp, current, current_cov)
            return False
        dt = stamp - self.stamp
        if dt <= 0 or dt > self.timeout:
            self.reset()
            self._anchor(stamp, current, current_cov)
            return False

        vector = _relative(self.quaternion, current)
        angle = np.linalg.norm(vector)
        if angle > self.max_angle + 1e-12 or angle / dt > self.max_rate + 1e-12:
            self.reset()
            self._anchor(stamp, current, current_cov)
            return False

        # The relative rotation's axis is invariant under that same rotation,
        # so its logarithm has identical components in the two body frames.
        raw = vector / dt
        old_to_current = rotation(current).T.dot(rotation(self.quaternion))
        delta_cov = old_to_current.dot(self.pose_covariance).dot(old_to_current.T)
        delta_cov += current_cov
        jacobian = _right_jacobian_inverse(vector)
        raw_cov = jacobian.dot(delta_cov).dot(jacobian.T) / dt ** 2
        if self.valid and self.filter_tau > 0:
            alpha = -math.expm1(-dt / self.filter_tau)
            previous = old_to_current.dot(self.omega)
            previous_cov = old_to_current.dot(self.rate_covariance).dot(old_to_current.T)
            self.omega = (1. - alpha) * previous + alpha * raw
            self.rate_covariance = (1. - alpha) * previous_cov + alpha * raw_cov
        else:
            self.omega = raw
            self.rate_covariance = raw_cov
        self.rate_covariance = .5 * (self.rate_covariance + self.rate_covariance.T)
        # A fixed floor acknowledges differentiation/model errors even if a
        # device reports zero uncertainty. Shift eigenvalues, preserving axes.
        smallest = np.linalg.eigvalsh(self.rate_covariance).min()
        self.rate_covariance += np.eye(3) * max(0., self.min_variance - smallest)
        self.valid = True
        self._anchor(stamp, current, current_cov)
        return True

    def estimate(self, now):
        """Return (body rate, covariance, valid); invalid is zero + unknown."""
        try:
            now = float(now)
            fresh = (math.isfinite(now) and self.stamp is not None and
                     0 <= now - self.stamp <= self.timeout)
        except (ValueError, TypeError, OverflowError):
            fresh = False
        if not self.valid or not fresh:
            return np.zeros(3), np.eye(3) * UNKNOWN_VARIANCE, False
        return self.omega.copy(), self.rate_covariance.copy(), True
