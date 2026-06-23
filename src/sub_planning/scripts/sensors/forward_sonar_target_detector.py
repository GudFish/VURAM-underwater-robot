#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.sensors.forward_sonar_target_detector import ForwardSonarTargetDetector


if __name__ == '__main__':
    try:
        import rospy
        ForwardSonarTargetDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
