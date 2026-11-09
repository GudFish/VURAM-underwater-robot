#!/usr/bin/env python3
"""DVL-only odometry: native position/orientation anchors plus bounded prediction."""
import rospy
from dvl_odometry.dvl_estimator import DvlEstimator

if __name__ == '__main__':
    rospy.init_node('dvl_fusion_node')
    DvlEstimator(fused=True)
    rospy.spin()
