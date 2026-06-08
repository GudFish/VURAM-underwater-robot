#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.control.depth_hold_velocity_mux import DepthHoldVelocityMux


if __name__ == '__main__':
    try:
        DepthHoldVelocityMux().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
