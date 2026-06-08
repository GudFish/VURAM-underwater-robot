#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.planners.dynamic_target_generator import DynamicTargetGenerator


if __name__ == '__main__':
    try:
        DynamicTargetGenerator().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
