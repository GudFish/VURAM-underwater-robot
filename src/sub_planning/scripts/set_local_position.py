#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.control.set_local_position import PositionController


if __name__ == '__main__':
    try:
        PositionController().run()
    except Exception as exc:
        import rospy
        if not rospy.is_shutdown():
            raise exc
