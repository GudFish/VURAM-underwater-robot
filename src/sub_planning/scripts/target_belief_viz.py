#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.visualization.target_belief_viz import TargetBeliefViz


if __name__ == '__main__':
    try:
        import rospy
        TargetBeliefViz()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
