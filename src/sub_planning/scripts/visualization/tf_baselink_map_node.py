#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.visualization.tf_baselink_map_node import TFBroadcasterNode


if __name__ == '__main__':
    try:
        import rospy
        TFBroadcasterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
