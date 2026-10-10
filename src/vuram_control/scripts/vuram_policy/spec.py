"""obs_act.yaml loader + obs construction for the test node.

build_obs() is code-identical to the MuJoCo training side
(mujoco_auv_env/envs/spec.py) so the obs fed to the exported policy is
exactly what it was trained with.
"""
import numpy as np

from .utils import load_yaml, wrap_angle


def load_spec(path):
    return load_yaml(path)


def obs_dim_from_layout(layout):
    return sum(v['len'] for v in layout.values())


def build_obs(s, ref, a_prev, layout):
    """s[12] = [v_body6, rpy3, pos3], ref[10] = [pos3, rpy3, v_world3, w_z],
    a_prev[6] -> obs[22].

    Note: v_body is taken directly from s (the dynamics state) -- the caller
    is responsible for the twist_in_world rotation when building s from
    pose_gt.
    """
    s = np.asarray(s, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    a_prev = np.asarray(a_prev, dtype=np.float64)

    e_pos = ref[0:3] - s[9:12]
    e_att = wrap_angle(ref[3:6] - s[6:9])   # ref roll/pitch are 0
    parts = {
        'e_pos': e_pos,
        'e_att': e_att,
        'v_body': s[0:6],
        'v_ref_lin': ref[6:9],
        'w_ref_z': np.asarray([ref[9]]),
        'a_prev': a_prev,
    }
    obs = np.zeros(obs_dim_from_layout(layout), dtype=np.float64)
    for name, spec in layout.items():
        obs[spec['start']:spec['start'] + spec['len']] = parts[name]
    return obs
