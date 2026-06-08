#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.visualization.boundary_loader import BoundaryLoader


if __name__ == '__main__':
    try:
        BoundaryLoader()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
