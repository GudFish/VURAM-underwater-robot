#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.control.set_body_vel_yaw import CombinedPlanner


if __name__ == '__main__':
    try:
        CombinedPlanner().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
