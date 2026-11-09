"""ROS-independent validation and bounded, piecewise DVL prediction."""
import math

import numpy as np

UNKNOWN_VARIANCE = 1e6
FLIP = np.diag([1., -1., -1.])


def finite(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('non-finite measurement')
    return result


def covariance(raw, fallback_std=0.05):
    """Never turn a missing/broken covariance into a perfect measurement."""
    try:
        matrix = finite(raw)
        if matrix.shape != (3, 3):
            raise ValueError('expected 3x3 covariance')
        matrix = (matrix + matrix.T) / 2
        if np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError('covariance is not positive semidefinite')
        return matrix + np.eye(3) * 1e-10
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return np.eye(3) * fallback_std ** 2


def frd_to_flu(vector, cov):
    return FLIP.dot(finite(vector)), FLIP.dot(cov).dot(FLIP)


def rotation(quaternion):
    x, y, z, w = finite(quaternion)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-6:
        raise ValueError('zero quaternion')
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class RateLimiter:
    def __init__(self, rate=0.):
        if not math.isfinite(rate) or rate < 0:
            raise ValueError('rate must be finite and non-negative')
        self.period = 1. / rate if rate else 0.
        self.last = None

    def allow(self, stamp):
        if self.last is None or stamp < self.last or stamp-self.last >= self.period-1e-6:
            self.last = stamp
            return True
        return False


class DvlPredictor:
    """Advance old velocity BEFORE replacing it; stop at each freshness boundary.

    A held velocity is one correlated sample: covariance grows as elapsed**2,
    not as a sum of timer dt**2 (which would depend on the output frequency).
    """
    def __init__(self, vel_timeout=1., dr_timeout=1., growth=0.01):
        self.vel_timeout = vel_timeout
        self.dr_timeout = dr_timeout
        self.growth = growth
        self.clear()

    def clear(self):
        self.position = None
        self.quaternion = None
        self.pose_cov = None
        self.dr_stamp = None
        self.dr_valid = False
        self.velocity = np.zeros(3)
        self.vel_cov = np.eye(3) * UNKNOWN_VARIANCE
        self.vel_stamp = None
        self.vel_valid = False
        self.time = None
        self.segment_start = None

    def anchor(self, stamp, position, quaternion, pose_cov):
        self.position = finite(position).copy()
        rotation(quaternion)
        self.quaternion = finite(quaternion).copy()
        self.pose_cov = finite(pose_cov).reshape(6, 6).copy()
        self.dr_stamp = self.time = self.segment_start = stamp
        self.dr_valid = True

    def set_velocity(self, stamp, vector, cov):
        self.advance(stamp)
        self.velocity = finite(vector).copy()
        self.vel_cov = finite(cov).reshape(3, 3).copy()
        self.vel_stamp = stamp
        self.vel_valid = True
        self.segment_start = stamp

    def invalidate_velocity(self, stamp):
        self.advance(stamp)
        self.vel_valid = False

    def invalidate_dr(self, stamp):
        self.advance(stamp)
        self.dr_valid = False

    def velocity_fresh(self, stamp):
        return (self.vel_valid and self.vel_stamp is not None and
                0 <= stamp-self.vel_stamp <= self.vel_timeout)

    def predicting(self, stamp):
        return (self.position is not None and self.dr_valid and
                0 <= stamp-self.dr_stamp <= self.dr_timeout and self.velocity_fresh(stamp))

    def advance(self, stamp):
        if self.time is None:
            return
        if stamp < self.time:
            return
        end = self.time
        if self.dr_valid and self.vel_valid and self.vel_stamp is not None:
            end = max(self.time, min(stamp, self.vel_stamp+self.vel_timeout,
                                      self.dr_stamp+self.dr_timeout))
        dt = end-self.time
        if dt > 0:
            rot = rotation(self.quaternion)
            self.position += rot.dot(self.velocity) * dt
            elapsed0 = max(0., self.time-self.segment_start)
            elapsed1 = max(0., end-self.segment_start)
            world_cov = rot.dot(self.vel_cov).dot(rot.T)
            # Include the attitude uncertainty when projecting body velocity.
            vx, vy, vz = self.velocity
            skew = np.array([[0., -vz, vy], [vz, 0., -vx], [-vy, vx, 0.]])
            world_cov += rot.dot(skew).dot(self.pose_cov[3:, 3:]).dot(skew.T).dot(rot.T)
            self.pose_cov[:3, :3] += world_cov * (elapsed1**2-elapsed0**2)
        # Process noise also grows while holding a stale position.
        self.pose_cov[:3, :3] += np.eye(3) * self.growth * (stamp-self.time)
        self.time = stamp
