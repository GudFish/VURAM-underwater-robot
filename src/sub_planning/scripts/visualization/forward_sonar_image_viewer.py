#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.visualization.forward_sonar_image_viewer import ForwardSonarImageViewer


if __name__ == '__main__':
    try:
        import rospy
        ForwardSonarImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
