#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.planners.mpc_dynamic_target_tracker import DynamicTargetMPCTracker


if __name__ == '__main__':
    try:
        DynamicTargetMPCTracker().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
