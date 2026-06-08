#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.planners.mpc_belief_target_tracker import BeliefAwareMPCTracker


if __name__ == '__main__':
    try:
        BeliefAwareMPCTracker().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
