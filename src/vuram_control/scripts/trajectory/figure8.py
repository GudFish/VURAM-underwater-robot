"""Continuous horizontal figure-eight anchored to a measured vehicle pose."""
import math

import numpy as np

from utils.math3d import quat_to_R, yaw_quat


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

