#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.estimators.target_bearing_estimator_ekf import TargetBearingEstimatorEKF


if __name__ == '__main__':
    try:
        import rospy
        TargetBearingEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
