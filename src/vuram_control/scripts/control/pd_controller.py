"""Body-frame PD tracking with the same five-axis action contract as policies.

Position gains are force per metre and force per metre/second; attitude gains
are torque per radian and torque per radian/second, in roll/yaw order. Defaults
convert the learner's warm-start gains to physical wrench gains (40 kg effective
mass and 2.49/2.60 kg m² effective roll/yaw inertia). There is no pitch command.
This module imports neither ROS nor Torch; ROS parameter access is optional.
"""
import numpy as np

from control.observation import (get_observation_encoding,
                                 obs_dim_from_layout)


_OBSERVATION_LAYOUT = {
    'e_pos': (0, 3), 'e_att': (3, 3), 'v_body': (6, 6),
    'v_ref_lin': (12, 3), 'w_ref_z': (15, 1), 'a_prev': (16, 5),
}


class PDController:
    """``step(obs21, history, valid) -> normalized [X,Y,Z,K,N]``.

    The observation builder already rotates position/velocity errors into the
    current body frame and selects the shortest quaternion attitude error.
    Normalization uses each axis's separate positive and negative capability,
    exactly as the nominal training allocator does.
    """

    history_shape = (20, 18)
    version = -1
    controller_type = 'pd'
    loaded = True

    def __init__(self, allocator, kp_xyz=(80., 80., 60.),
                 kd_xyz=(40., 40., 32.), kp_att=(2.49, 2.60),
                 kd_att=(1.245, 1.30)):
        self.kp_xyz = self._vector(kp_xyz, 3, 'kp_xyz', allow_zero=True)
        self.kd_xyz = self._vector(kd_xyz, 3, 'kd_xyz', allow_zero=True)
        self.kp_att = self._vector(kp_att, 2, 'kp_att', allow_zero=True)
        self.kd_att = self._vector(kd_att, 2, 'kd_att', allow_zero=True)
        self.scales = self._vector(allocator.scales, 5, 'action_scales')
        self.negative_scales = self._vector(
            allocator.negative_scales, 5, 'negative_action_scales')

    @staticmethod
    def _vector(value, size, name, allow_zero=False):
        vector = np.asarray(value, dtype=np.float64)
        if (vector.shape != (size,) or not np.isfinite(vector).all() or
                np.any(vector < 0 if allow_zero else vector <= 0)):
            raise ValueError('{} must contain {} finite {} values'.format(
                name, size, 'nonnegative' if allow_zero else 'positive'))
        return vector.copy()

    @classmethod
    def from_ros_params(cls, allocator, get_param=None):
        """Read private gains, optionally with an injected ROS-style getter."""
        if get_param is None:
            import rospy
            get_param = rospy.get_param
        return cls(allocator,
                   kp_xyz=get_param('~kp_xyz', [80., 80., 60.]),
                   kd_xyz=get_param('~kd_xyz', [40., 40., 32.]),
                   kp_att=get_param('~kp_att', [2.49, 2.60]),
                   kd_att=get_param('~kd_att', [1.245, 1.30]))

    @property
    def parameters(self):
        """Physical gains and units for episode metadata."""
        return dict(kp_xyz=self.kp_xyz.tolist(), kd_xyz=self.kd_xyz.tolist(),
                    kp_att=self.kp_att.tolist(), kd_att=self.kd_att.tolist(),
                    position_frame='body', attitude_axes=['roll', 'yaw'],
                    kp_xyz_units='N/m', kd_xyz_units='N/(m/s)',
                    kp_att_units='Nm/rad', kd_att_units='Nm/(rad/s)')

    def validate_spec(self, spec):
        get_observation_encoding(spec)
        layout = spec['obs_layout']
        if spec['a_dim'] != 5 or obs_dim_from_layout(layout) != 21:
            raise ValueError('PD requires obs[21] and action[5]')
        if (set(layout) != set(_OBSERVATION_LAYOUT) or any(
                not isinstance(layout[name].get(key), int) or
                isinstance(layout[name].get(key), bool) or
                layout[name][key] != expected[index]
                for name, expected in _OBSERVATION_LAYOUT.items()
                if name in layout
                for index, key in enumerate(('start', 'len')))):
            raise ValueError('PD requires the standard body-relative obs_layout')
        dt = float(spec['dt'])
        rate = float(spec['control_rate'])
        if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(rate) or rate <= 0 or
                not np.isclose(rate * dt, 1.0, rtol=1e-9, atol=0.)):
            raise ValueError('control_rate must equal 1 / dt')

    def step(self, obs, history=None, valid=0):
        observation = np.asarray(obs, dtype=np.float64)
        if observation.shape != (21,) or not np.isfinite(observation).all():
            raise ValueError('PD requires a finite observation[21]')
        force = (self.kp_xyz * observation[:3] +
                 self.kd_xyz * (observation[12:15] - observation[6:9]))
        # No reference roll rate; yaw feed-forward follows the path tangent.
        attitude_error = observation[[3, 5]]
        rate_error = np.array([-observation[9],
                               observation[15] - observation[11]])
        torque = self.kp_att * attitude_error + self.kd_att * rate_error
        wrench = np.concatenate((force, torque))
        scales = np.where(wrench >= 0, self.scales, self.negative_scales)
        return np.clip(wrench / scales, -1., 1.)
