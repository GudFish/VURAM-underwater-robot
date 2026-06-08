#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.sensors.acoustic_range_sensor_sim import AcousticRangeSensorSim


if __name__ == '__main__':
    try:
        import rospy
        AcousticRangeSensorSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
