"""obs_act.yaml loader + obs construction（VND v4 契约，obs[21]）。

build_obs() 与训练侧（auv_learning/task/specification.py）一致，
保证喂给导出策略的观测与训练完全一致。
"""
import numpy as np

from .utils import load_yaml, quat_error_rotvec, world_to_body

OBSERVATION_ENCODING = 'body_relative_tracking_v1'


def get_observation_encoding(metadata):
    """Only current body-relative deploy snapshots are supported."""
    encoding = metadata.get('observation_encoding')
    if encoding != OBSERVATION_ENCODING:
        raise ValueError('expected observation encoding {!r}, got {!r}'.format(
            OBSERVATION_ENCODING, encoding))
    return encoding


def load_spec(path):
    return load_yaml(path)


def obs_dim_from_layout(layout):
    return sum(v['len'] for v in layout.values())


def build_obs(s, ref, a_prev, layout):
    """s[13] = [v_body6, quat_wxyz4, pos3], ref[11] = [pos3, quat_wxyz4,
    v_world3, w_z], a_prev[5] -> obs[21]."""
    s = np.asarray(s, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    a_prev = np.asarray(a_prev, dtype=np.float64)

    e_pos = world_to_body(ref[0:3] - s[10:13], s[6:10])
    v_ref = world_to_body(ref[7:10], s[6:10])
    e_att = quat_error_rotvec(s[6:10], ref[3:7])
    parts = {
        'e_pos': e_pos,
        'e_att': e_att,
        'v_body': s[0:6],
        'v_ref_lin': v_ref,
        'w_ref_z': np.asarray([ref[10]]),
        'a_prev': a_prev,
    }
    obs = np.zeros(obs_dim_from_layout(layout), dtype=np.float64)
    for name, spec in layout.items():
        obs[spec['start']:spec['start'] + spec['len']] = parts[name]
    return obs


def build_ref11(pos, quat, vel, w_z):
    """参考状态组装：[pos3, quat_wxyz4, vel3, w_z]。"""
    return np.concatenate([pos, quat, vel, [w_z]])
