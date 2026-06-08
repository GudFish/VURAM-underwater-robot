#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.estimators.target_state_estimator_ekf import TargetStateEstimatorEKF


if __name__ == '__main__':
    try:
        import rospy
        TargetStateEstimatorEKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
