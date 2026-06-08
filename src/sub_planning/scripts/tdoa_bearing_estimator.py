#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.sensors.tdoa_bearing_estimator import TDOABearingEstimator


if __name__ == '__main__':
    try:
        import rospy
        TDOABearingEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
