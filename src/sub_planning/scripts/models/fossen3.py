# -*- coding: utf-8 -*-

"""
功能：提供 Fossen 三自由度水下航行器动力学模型及相关矩阵运算工具函数。
输入：通过函数参数或上层调用提供动力学、噪声和状态等模型输入。
输出：返回模型计算结果，供规划器、估计器或演示脚本直接调用。
"""
import math


def wrap_pi(angle):
    """
    功能：将输入角度归一化到 [-pi, pi] 区间。
    输入：`angle` 为该函数执行所需的输入参数。
    输出：返回归一化后的弧度角。
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def mat_vec_mul_3x3(a, x):
    """
    功能：执行 3x3 矩阵与三维向量的乘法。
    输入：`a`、`x` 为该函数执行所需的输入参数。
    输出：返回矩阵与向量乘法得到的三维结果。
    """
    return [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]


def mat_add_3x3(a, b):
    """
    功能：执行两个 3x3 矩阵的逐元素相加。
    输入：`a`、`b` 为该函数执行所需的输入参数。
    输出：返回相加后的 3x3 矩阵。
    """
    return [
        [a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]],
        [a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]],
    ]


def inv_3x3(m):
    """
    功能：计算 3x3 矩阵的逆矩阵。
    输入：`m` 为该函数执行所需的输入参数。
    输出：返回输入矩阵的逆矩阵；若矩阵奇异则抛出异常。
    """
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
    功能：描述三自由度水下航行器的动力学与运动学离散更新模型。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """

    def __init__(self, p):
        # Rigid body / added mass
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
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
        """
        功能：根据当前速度计算科氏与离心项矩阵。
        输入：`nu` 为该函数执行所需的输入参数。
        输出：返回与该功能对应的计算结果。
        """
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
        """
        功能：根据当前速度计算线性与非线性阻尼矩阵。
        输入：`nu` 为该函数执行所需的输入参数。
        输出：返回与该功能对应的计算结果。
        """
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
        """
        功能：根据姿态角和体速度计算世界系运动学导数。
        输入：`psi`、`nu` 为该函数执行所需的输入参数。
        输出：返回世界系下的运动学导数。
        """
        u, v, r = nu[0], nu[1], nu[2]
        c = math.cos(psi)
        s = math.sin(psi)
        return [c * u - s * v, s * u + c * v, r]

    def nu_dot(self, nu, tau):
        """
        功能：根据动力学方程计算体速度导数。
        输入：`nu`、`tau` 为该函数执行所需的输入参数。
        输出：返回体速度导数。
        """
        c_mat = self.coriolis(nu)
        d_mat = self.damping(nu)
        cd = mat_add_3x3(c_mat, d_mat)
        cd_nu = mat_vec_mul_3x3(cd, nu)
        rhs = [tau[0] - cd_nu[0], tau[1] - cd_nu[1], tau[2] - cd_nu[2]]
        return mat_vec_mul_3x3(self.M_inv, rhs)

    def step(self, eta, nu, tau, dt):
        """
        功能：执行一次离散时间动力学步进。
        输入：`eta`、`nu`、`tau`、`dt` 为该函数执行所需的输入参数。
        输出：返回下一时刻的位姿状态和速度状态。
        """
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

