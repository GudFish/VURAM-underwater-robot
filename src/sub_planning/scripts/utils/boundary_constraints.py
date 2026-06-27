#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：提供规划边界半空间约束的加载与几何辅助函数。
输入：通过函数参数和参数服务器读取几何边界数据。
输出：返回边界几何约束或辅助计算结果。
"""
import math

import rospy


def polygon_signed_area(points):
    """
    功能：计算二维多边形的有向面积。
    输入：`points` 为该函数执行所需的输入参数。
    输出：返回多边形的有向面积值。
    """
    area = 0.0
    for idx, p0 in enumerate(points):
        p1 = points[(idx + 1) % len(points)]
        area += p0[0] * p1[1] - p1[0] * p0[1]
    return 0.5 * area


def load_boundary_halfspaces(param_name='/boundary_loader/enu_points', wait_timeout=5.0, required=False, margin=0.0):
    """
    功能：从参数服务器读取边界并转换为半空间约束。
    输入：`param_name`、`wait_timeout`、`required`、`margin` 为该函数执行所需的输入参数。
    输出：返回与该功能对应的计算结果。
    """
    deadline = rospy.Time.now() + rospy.Duration(max(wait_timeout, 0.0))
    while not rospy.is_shutdown() and not rospy.has_param(param_name) and rospy.Time.now() < deadline:
        rospy.loginfo_throttle(2.0, '[BoundaryConstraints] waiting for %s ...', param_name)
        rospy.sleep(0.1)

    if not rospy.has_param(param_name):
        msg = '[BoundaryConstraints] boundary param %s is not available.' % param_name
        if required:
            raise rospy.ROSException(msg)
        rospy.logwarn(msg + ' Boundary constraints disabled.')
        return []

    raw_points = rospy.get_param(param_name, [])
    points = []
    for raw in raw_points:
        if len(raw) < 2:
            continue
        x = float(raw[0])
        y = float(raw[1])
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))

    if len(points) < 3:
        msg = '[BoundaryConstraints] need at least 3 boundary points, got %d.' % len(points)
        if required:
            raise rospy.ROSException(msg)
        rospy.logwarn(msg + ' Boundary constraints disabled.')
        return []

    if polygon_signed_area(points) < 0.0:
        points.reverse()

    halfspaces = []
    for idx, p0 in enumerate(points):
        p1 = points[(idx + 1) % len(points)]
        ex = p1[0] - p0[0]
        ey = p1[1] - p0[1]
        length = math.hypot(ex, ey)
        if length < 1e-9:
            continue

        # For a CCW polygon, the inward normal is the left normal of each edge.
        nx = -ey / length
        ny = ex / length
        c = nx * p0[0] + ny * p0[1]
        halfspaces.append((nx, ny, c + float(margin)))

    if len(halfspaces) < 3:
        msg = '[BoundaryConstraints] valid boundary halfspaces are fewer than 3.'
        if required:
            raise rospy.ROSException(msg)
        rospy.logwarn(msg + ' Boundary constraints disabled.')
        return []

    rospy.loginfo(
        '[BoundaryConstraints] loaded %d boundary edges from %s with margin %.2f m.',
        len(halfspaces),
        param_name,
        margin,
    )
    return halfspaces
