"""Deterministic 3-D trajectory shared by TCP DVL and ROS IMU simulators.

ENU/FLU kinematics, with analytic velocity/acceleration. IMU specific force
includes gravity; a stationary level IMU reports +9.80665 along FLU z.
"""
import math
import numpy as np


def trajectory(t, speed=0.3, acceleration=0., yaw_rate=0.1, roll=0., pitch=0.):
    yaw = yaw_rate*t
    v = speed+acceleration*t
    if abs(yaw_rate) < 1e-8:
        p = np.array([speed*t+0.5*acceleration*t*t, 0., 0.])
    else:
        w = yaw_rate
        p = np.array([speed*math.sin(yaw)/w + acceleration*(t*math.sin(yaw)/w+(math.cos(yaw)-1)/w**2),
                      speed*(1-math.cos(yaw))/w + acceleration*(-t*math.cos(yaw)/w+math.sin(yaw)/w**2), 0.])
    rotation = rpy_rotation(roll, pitch, yaw)
    c, s = math.cos, math.sin
    world_velocity = np.array([v*c(yaw), v*s(yaw), 0.])
    world_acceleration = np.array([acceleration*c(yaw)-v*yaw_rate*s(yaw),
                                   acceleration*s(yaw)+v*yaw_rate*c(yaw), 0.])
    velocity = rotation.T.dot(world_velocity)
    specific_force = rotation.T.dot(world_acceleration+np.array([0., 0., 9.80665]))
    omega = rotation.T.dot([0., 0., yaw_rate])
    return p, (roll, pitch, yaw), velocity, specific_force, omega


def rpy_rotation(roll, pitch, yaw):
    c, s = math.cos, math.sin
    rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
    ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
    rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
    return rz.dot(ry).dot(rx)


def matrix_rpy(matrix):
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1., 1.)))
    if abs(math.cos(pitch)) < 1e-8:
        return 0., pitch, math.atan2(-matrix[0, 1], matrix[1, 1])
    return math.atan2(matrix[2, 1], matrix[2, 2]), pitch, math.atan2(matrix[1, 0], matrix[0, 0])
