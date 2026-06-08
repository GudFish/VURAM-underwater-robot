#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sub_planning.sensors.acoustic_tdoa_sensor_sim import AcousticTDOASensorSim


if __name__ == '__main__':
    try:
        import rospy
        AcousticTDOASensorSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
