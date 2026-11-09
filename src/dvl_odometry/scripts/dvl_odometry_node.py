#!/usr/bin/env python3
"""Publish device DR measurements at their native rate, without extrapolation."""
import rospy
from dvl_odometry.dvl_estimator import DvlEstimator

if __name__ == '__main__':
    rospy.init_node('dvl_odometry_node')
    DvlEstimator(fused=False)
    rospy.spin()
