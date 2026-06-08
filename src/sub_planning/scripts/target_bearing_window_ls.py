#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.estimators.target_bearing_window_ls import TargetBearingWindowLS


if __name__ == '__main__':
    try:
        import rospy
        TargetBearingWindowLS()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
