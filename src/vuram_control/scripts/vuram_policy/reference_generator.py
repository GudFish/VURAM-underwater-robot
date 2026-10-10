"""Reference trajectory families: line / lissajous / waypoints_dp.

Ported verbatim from auv_training/reference_generator.py (numpy-only).
sample_episode(rng) -> params dict.
get_ref(params, t) -> (pos[3], rpy[3], vel_world[3], w_z).
Reference roll/pitch are always 0; yaw follows the velocity heading.
Peak reference speed is capped at MAX_REF_SPEED (vehicle feasibility).

The depth band comes from work_env['z'] (defaults to the MuJoCo training
band); override it when deploying at a different operating depth (e.g. SITL
at the surface).
"""
import numpy as np

MAX_REF_SPEED = 0.5   # m/s

# default depth band for the reference trajectories (world ENU, m)
MIN_Z = -52.5
MAX_Z = -47.5


class ReferenceGenerator(object):
    def __init__(self, work_env=None):
        # work envelope for trajectory origins
        self.work_env = work_env or {
            'x': [-8.0, 8.0], 'y': [-8.0, 8.0], 'z': [MIN_Z, MAX_Z],
        }
        self.z_band = tuple(self.work_env['z'])

    def sample_episode(self, rng, start_pos=None, family=None):
        """Sample a random trajectory.

        When start_pos is given, the reference is anchored so that
        get_ref(params, 0)[0] == start_pos (used by the deployment/demo
        path, where the vehicle is NOT teleported toward the reference).
        Without start_pos the origin is sampled inside the work envelope.
        `family` forces the trajectory family ('line' | 'lissajous' |
        'waypoints_dp'); None samples uniformly.
        """
        if family is None:
            family = rng.choice(['line', 'lissajous', 'waypoints_dp'])
        else:
            family = str(family)
            if family not in ('line', 'lissajous', 'waypoints_dp'):
                raise ValueError('unknown reference family: %r' % family)
        params = {'family': family}
        if start_pos is not None:
            params['origin'] = [float(v) for v in start_pos]
            params['start_anchored'] = True

        if family == 'line':
            if start_pos is None:
                params['origin'] = self._random_origin(rng)
            params['speed'] = float(rng.uniform(0.1, 0.4))
            params['heading'] = float(rng.uniform(-np.pi, np.pi))
        elif family == 'lissajous':
            A_xy = float(rng.uniform(2.0, 6.0))
            A_xy_ratio = float(rng.uniform(0.8, 1.2))
            if start_pos is None:
                origin = self._random_origin(rng)
                z0 = origin[2]
            else:
                z0 = float(start_pos[2])
            # keep the z excursion inside the depth band: the path oscillates
            # around origin_z, which itself may sit A_z away from z0 when
            # anchored (pos(0) == start_pos), so anchor with margin/2.
            # No floor: an origin on the band edge gives a flat-z lissajous.
            z_margin = min(z0 - self.z_band[0], self.z_band[1] - z0)
            bound = (z_margin / 2.0) if start_pos is not None else z_margin
            A_z = min(float(rng.uniform(0.5, 2.0)), max(bound, 0.0))
            A = [A_xy, A_xy_ratio * A_xy, A_z]
            w = [float(rng.uniform(0.05, 0.3)) for _ in range(3)]
            peak = max(a * wi for a, wi in zip(A, w))
            if peak > MAX_REF_SPEED:      # scale down to the speed cap
                w = [wi * MAX_REF_SPEED / peak for wi in w]
            phi = [float(rng.uniform(0, 2 * np.pi)) for _ in range(3)]
            if start_pos is None:
                params['origin'] = origin
            else:
                # pos(0) = origin + A*sin(phi) == start_pos
                params['origin'] = [start_pos[i] - A[i] * np.sin(phi[i])
                                    for i in range(3)]
            params['A'] = A
            params['w'] = w
            params['phi'] = phi
        else:  # waypoints_dp: smooth cubic-Hermite waypoint path (no holds)
            n = int(rng.integers(3, 7))
            if start_pos is None:
                origin = self._random_origin(rng)
                params['origin'] = origin
            else:
                origin = [float(v) for v in start_pos]
                params['origin'] = origin
            x, y, z = origin
            waypoints = []
            p = np.array([x, y, z])
            waypoints.append([float(v) for v in p])
            for _ in range(n - 1):
                for _try in range(20):      # min 3 m separation keeps the
                    offset = np.array([rng.uniform(-8, 8),      # tangents sane
                                       rng.uniform(-8, 8),
                                       rng.uniform(-3, 3)])
                    if np.linalg.norm(offset) >= 3.0:
                        break
                p = p + offset
                waypoints.append([float(v) for v in p])
            if start_pos is not None:
                waypoints[0] = [float(v) for v in start_pos]
            for wp in waypoints:
                wp[2] = float(np.clip(wp[2], self.z_band[0], self.z_band[1]))
            params['waypoints'] = waypoints
            params['v_bar'] = float(rng.uniform(0.2, 0.5))
            params['waypoint_speed_factors'] = [float(rng.uniform(0.5, 1.5))
                                                for _ in range(n)]
        return params

    def _random_origin(self, rng):
        return [float(rng.uniform(*self.work_env['x'])),
                float(rng.uniform(*self.work_env['y'])),
                float(rng.uniform(*self.work_env['z']))]

    def get_ref(self, params, t):
        family = params['family']
        if family == 'line':
            return _ref_line(params, t)
        if family == 'lissajous':
            return _ref_lissajous(params, t)
        return _ref_waypoints(params, t, self.z_band)

    def ref_start_pose(self, params):
        pos, rpy, _, _ = self.get_ref(params, 0.0)
        return pos, rpy


def _heading_from_vel(v, w_z, yaw_prev=None):
    """Yaw from velocity direction; falls back to the previous yaw at ~0 speed."""
    speed = float(np.hypot(v[0], v[1]))
    if speed > 1e-3:
        yaw = np.arctan2(v[1], v[0])
    else:
        yaw = yaw_prev if yaw_prev is not None else 0.0
    return yaw, float(w_z)


def _ref_line(params, t):
    U = params['speed']
    psi = params['heading']
    x0, y0, z0 = params['origin']
    pos = np.array([x0 + U * np.cos(psi) * t, y0 + U * np.sin(psi) * t, z0])
    vel = np.array([U * np.cos(psi), U * np.sin(psi), 0.0])
    rpy = np.array([0.0, 0.0, psi])
    return pos, rpy, vel, 0.0


def _ref_lissajous(params, t):
    A = params['A']
    w = params['w']
    phi = params['phi']
    x0, y0, z0 = params['origin']
    s = [np.sin(w[i] * t + phi[i]) for i in range(3)]
    c = [np.cos(w[i] * t + phi[i]) for i in range(3)]
    pos = np.array([x0 + A[0] * s[0], y0 + A[1] * s[1], z0 + A[2] * s[2]])
    vel = np.array([A[0] * w[0] * c[0], A[1] * w[1] * c[1],
                    A[2] * w[2] * c[2]])
    ax = -A[0] * w[0] ** 2 * s[0]
    ay = -A[1] * w[1] ** 2 * s[1]
    v2 = vel[0] ** 2 + vel[1] ** 2
    w_z = (vel[0] * ay - vel[1] * ax) / max(v2, 1e-9)
    yaw, w_z = _heading_from_vel(vel, w_z)
    rpy = np.array([0.0, 0.0, yaw])
    return pos, rpy, vel, w_z


def _ref_waypoints(params, t, z_band):
    """Smooth cubic-Hermite waypoint path (no holds).

    Each segment runs from wp_i to wp_{i+1} with boundary velocities
    v_i = m_i * v_bar * f_i, where m_i is the (central-difference) chord
    tangent at wp_i. The velocity vector is CONTINUOUS at the waypoints
    (same v_i on both sides), never zero, and its xy-projection direction
    defines the heading - so the reference yaw is the path tangent pointing
    along the direction of travel everywhere.
    """
    waypoints = np.asarray(params['waypoints'], dtype=np.float64)
    n = len(waypoints)
    v_bar = float(params['v_bar'])
    factors = params['waypoint_speed_factors']

    # tangents (direction only), one-sided at the ends
    tangents = []
    for i in range(n):
        if i == 0:
            d = waypoints[1] - waypoints[0]
        elif i == n - 1:
            d = waypoints[-1] - waypoints[-2]
        else:
            d = waypoints[i + 1] - waypoints[i - 1]
        tangents.append(d / (np.linalg.norm(d) + 1e-12))
    vels = [tangents[i] * (v_bar * factors[i]) for i in range(n)]
    yaw_prev = float(np.arctan2(tangents[0][1], tangents[0][0]))

    t_cum = 0.0
    for i in range(n - 1):
        chord = float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        T = chord / v_bar
        if t < t_cum + T:
            tau = (t - t_cum) / T
            pos, vel, acc = _hermite(waypoints[i], waypoints[i + 1],
                                     vels[i], vels[i + 1], T, tau)
            # cubic Hermite segments can overshoot the waypoints; keep z
            # inside the depth band and drop the z reference velocity
            # where the clip engages
            z_clipped = float(np.clip(pos[2], z_band[0], z_band[1]))
            if z_clipped != pos[2]:
                pos = pos.copy()
                pos[2] = z_clipped
                vel = vel.copy()
                vel[2] = 0.0
                acc = acc.copy()
                acc[2] = 0.0
            v2 = vel[0] ** 2 + vel[1] ** 2
            w_z = (vel[0] * acc[1] - vel[1] * acc[0]) / max(v2, 1e-9)
            yaw, w_z = _heading_from_vel(vel, w_z, yaw_prev)
            rpy = np.array([0.0, 0.0, yaw])
            return pos, rpy, vel, w_z
        yaw_prev = float(np.arctan2(waypoints[i + 1][1] - waypoints[i][1],
                                    waypoints[i + 1][0] - waypoints[i][0]))
        t_cum += T
    # past the end: hold the last waypoint with the last heading
    pos = waypoints[-1]
    rpy = np.array([0.0, 0.0, yaw_prev])
    return pos, rpy, np.zeros(3), 0.0


def _hermite(p0, p1, v0, v1, T, tau):
    """Cubic Hermite segment at normalized time tau in [0,1]:
    returns (pos, vel, acc)."""
    tau = float(np.clip(tau, 0.0, 1.0))
    h00 = 2 * tau ** 3 - 3 * tau ** 2 + 1
    h10 = tau ** 3 - 2 * tau ** 2 + tau
    h01 = -2 * tau ** 3 + 3 * tau ** 2
    h11 = tau ** 3 - tau ** 2
    pos = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    dh00 = 6 * tau ** 2 - 6 * tau
    dh10 = 3 * tau ** 2 - 4 * tau + 1
    dh01 = -6 * tau ** 2 + 6 * tau
    dh11 = 3 * tau ** 2 - 2 * tau
    vel = (dh00 * p0 + dh10 * T * v0 + dh01 * p1 + dh11 * T * v1) / T
    ddh00 = 12 * tau - 6
    ddh10 = 6 * tau - 4
    ddh01 = -12 * tau + 6
    ddh11 = 6 * tau - 2
    acc = (ddh00 * p0 + ddh10 * T * v0 + ddh01 * p1 + ddh11 * T * v1) / T ** 2
    return pos, vel, acc
