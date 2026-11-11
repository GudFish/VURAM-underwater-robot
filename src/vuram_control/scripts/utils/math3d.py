"""Shared helpers: quaternion math (wxyz), rotations, yaml.

wxyz 四元数函数移植自 auv_vnd/task/math_utils.py（训练口径）：
状态/参考姿态是单位四元数 (w, x, y, z)，不再是 rpy。
quat_to_rpy（ROS 消息 xyzw 顺序）仅供诊断保留。
"""
import numpy as np
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---- wxyz quaternion math (MuJoCo math_utils ports) --------------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('orientation quaternion has zero norm')
    return q / norm


def quat_conj(q):
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def quat_mul(a, b):
    w, x, y, z = np.moveaxis(np.asarray(a), -1, 0)
    W, X, Y, Z = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack((w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
                     w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W), axis=-1)


def yaw_quat(yaw):
    half = np.asarray(yaw) / 2
    zeros = np.zeros_like(half)
    return np.stack((np.cos(half), zeros, zeros, np.sin(half)), axis=-1)


def quat_to_R(q):
    w, x, y, z = np.moveaxis(quat_normalize(q), -1, 0)
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1
    ).reshape(np.shape(q)[:-1] + (3, 3))


def quat_error_rotvec(q_current, q_ref):
    """Shortest reference rotation in the current body frame, in radians."""
    error = quat_normalize(quat_mul(quat_conj(quat_normalize(q_current)),
                                    quat_normalize(q_ref)))
    error = np.where(error[..., :1] < 0, -error, error)
    vector = error[..., 1:]
    magnitude = np.linalg.norm(vector, axis=-1, keepdims=True)
    scale = np.where(magnitude > 1e-12,
                     2 * np.arctan2(magnitude, error[..., :1]) /
                     np.maximum(magnitude, 1e-12), 2.0)
    return scale * vector


def quat_tilt(q_wxyz):
    """Angle between body up and world up (MuJoCo termination tilt)."""
    R = quat_to_R(q_wxyz)
    return np.arccos(np.clip(R[..., 2, 2], -1.0, 1.0))


def world_to_body(v_world, q_wxyz):
    """Rotate a world-frame vector into the body frame: R(q)^T @ v."""
    return quat_to_R(q_wxyz).T @ np.asarray(v_world, dtype=np.float64)


# ---- legacy xyzw helpers (diagnostics only) ----------------------------------
def quat_to_rpy(q):
    """Roll/pitch/yaw from xyzw quaternion (ROS message order). Normalizes
    first (guard NaN)."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3)
    x, y, z, w = q / n
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def wrap_angle(a):
    """Wrap to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
