#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.planners.mpc_2d_planner import IPOPTMPC2DPlanner


if __name__ == '__main__':
    try:
        IPOPTMPC2DPlanner().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
