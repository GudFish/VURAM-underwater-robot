"""Shared helpers: quaternion/rpy conversions, rotations, yaml."""
import numpy as np
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def quat_to_rpy(q):
    """Roll/pitch/yaw from xyzw quaternion. Normalizes first (guard NaN)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    x, y, z, w = q / n
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def rpy_to_R(rpy):
    """Rotation matrix body->world for roll-pitch-yaw (ZYX intrinsic, ENU)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def wrap_angle(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def world_to_body(v_world, rpy):
    """Rotate a world-frame vector into the body frame: R(q)^T * v."""
    return rpy_to_R(rpy).T @ np.asarray(v_world, dtype=np.float64)
