# -*- coding: utf-8 -*-

import math


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mat_vec_mul_3x3(a, x):
    return [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]


def mat_add_3x3(a, b):
    return [
        [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
        [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
    ]


def inv_3x3(m):
    a = m[0][0]
    b = m[0][1]
    c = m[0][2]
    d = m[1][0]
    e = m[1][1]
    f = m[1][2]
    g = m[2][0]
    h = m[2][1]
    i = m[2][2]

    A = e * i - f * h
    B = -(d * i - f * g)
    C = d * h - e * g
    D = -(b * i - c * h)
    E = a * i - c * g
    F = -(a * h - b * g)
    G = b * f - c * e
    H = -(a * f - c * d)
    I = a * e - b * d

    det = a * A + b * B + c * C
    if abs(det) < 1e-9:
        raise ValueError("Mass matrix is singular, cannot invert.")

    inv_det = 1.0 / det
    return [
        [A * inv_det, D * inv_det, G * inv_det],
        [B * inv_det, E * inv_det, H * inv_det],
        [C * inv_det, F * inv_det, I * inv_det],
    ]


class Fossen3DOFModel:
    """
    3-DOF (surge, sway, yaw) marine craft model in body frame:
      eta = [x, y, psi]
      nu  = [u, v, r]

    Dynamics:
      M * nu_dot + C(nu)*nu + D(nu)*nu = tau
      eta_dot = J(psi) * nu
    """

    def __init__(self, p):
        # Rigid body / added mass
        self.m11 = float(p.get("m11", 40.0))
        self.m22 = float(p.get("m22", 55.0))
        self.m23 = float(p.get("m23", 0.0))
        self.m33 = float(p.get("m33", 12.0))

        # Linear damping
        self.d11 = float(p.get("d11", 20.0))
        self.d22 = float(p.get("d22", 28.0))
        self.d23 = float(p.get("d23", 0.0))
        self.d32 = float(p.get("d32", 0.0))
        self.d33 = float(p.get("d33", 8.0))

        # Quadratic damping
        self.n11 = float(p.get("n11", 18.0))
        self.n22 = float(p.get("n22", 24.0))
        self.n23 = float(p.get("n23", 0.0))
        self.n32 = float(p.get("n32", 0.0))
        self.n33 = float(p.get("n33", 6.0))

        self.M = [
            [self.m11, 0.0, 0.0],
            [0.0, self.m22, self.m23],
            [0.0, self.m23, self.m33],
        ]
        self.M_inv = inv_3x3(self.M)

    def coriolis(self, nu):
        u, v, r = nu[0], nu[1], nu[2]
        m11 = self.m11
        m22 = self.m22
        m23 = self.m23

        # 3DOF structure with general M entries used above
        return [
            [0.0, 0.0, -m22 * v - m23 * r],
            [0.0, 0.0, m11 * u],
            [m22 * v + m23 * r, -m11 * u, 0.0],
        ]

    def damping(self, nu):
        u, v, r = nu[0], nu[1], nu[2]
        du = self.d11 + self.n11 * abs(u)
        dv = self.d22 + self.n22 * abs(v)
        dr = self.d33 + self.n33 * abs(r)
        d23 = self.d23 + self.n23 * abs(r)
        d32 = self.d32 + self.n32 * abs(v)
        return [
            [du, 0.0, 0.0],
            [0.0, dv, d23],
            [0.0, d32, dr],
        ]

    @staticmethod
    def kinematics(psi, nu):
        u, v, r = nu[0], nu[1], nu[2]
        c = math.cos(psi)
        s = math.sin(psi)
        return [c * u - s * v, s * u + c * v, r]

    def nu_dot(self, nu, tau):
        c_mat = self.coriolis(nu)
        d_mat = self.damping(nu)
        cd = mat_add_3x3(c_mat, d_mat)
        cd_nu = mat_vec_mul_3x3(cd, nu)
        rhs = [tau[0] - cd_nu[0], tau[1] - cd_nu[1], tau[2] - cd_nu[2]]
        return mat_vec_mul_3x3(self.M_inv, rhs)

    def step(self, eta, nu, tau, dt):
        nud = self.nu_dot(nu, tau)
        nu_next = [
            nu[0] + nud[0] * dt,
            nu[1] + nud[1] * dt,
            nu[2] + nud[2] * dt,
        ]
        etad = self.kinematics(eta[2], nu_next)
        eta_next = [
            eta[0] + etad[0] * dt,
            eta[1] + etad[1] * dt,
            wrap_pi(eta[2] + etad[2] * dt),
        ]
        return eta_next, nu_next

