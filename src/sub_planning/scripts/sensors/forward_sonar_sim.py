#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.sensors.forward_sonar_sim import ForwardSonarSim


if __name__ == '__main__':
    try:
        import rospy
        ForwardSonarSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
