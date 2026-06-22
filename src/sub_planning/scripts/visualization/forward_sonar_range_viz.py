#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.visualization.forward_sonar_range_viz import ForwardSonarRangeViz


if __name__ == '__main__':
    try:
        import rospy
        ForwardSonarRangeViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
