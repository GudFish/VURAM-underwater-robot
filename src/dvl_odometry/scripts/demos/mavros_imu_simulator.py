#!/usr/bin/env python3
"""Run TCP DVL + optional MAVROS-style IMU against exactly the same trajectory."""
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import Imu
from std_srvs.srv import SetBool, SetBoolResponse
from tf.transformations import quaternion_from_euler

from dvl_odometry.simulation import trajectory
from dvl_odometry.simulator import DvlSimulator


class ImuSimulator:
    def __init__(self):
        self.start_time = time.time()
        self.motion = {key: float(rospy.get_param('~'+key, value)) for key, value in
                       [('speed', 0.3), ('acceleration', 0.), ('yaw_rate', 0.1), ('roll', 0.), ('pitch', 0.)]}
        self.enabled = rospy.get_param('~publish_imu', False)
        self.imu_stamp_offset = float(rospy.get_param('~imu_stamp_offset', 0.))
        self.publisher = rospy.Publisher('/mavros/imu/data', Imu, queue_size=30)
        self.frame = rospy.get_param('~robot_frame', 'base_link')
        self.simulator = DvlSimulator(host=rospy.get_param('~ip', '127.0.0.1'),
                                      port=int(rospy.get_param('~port', 16171)),
                                      velocity_rate=float(rospy.get_param('~velocity_rate', 10.)),
                                      dr_rate=float(rospy.get_param('~dr_rate', 5.)),
                                      start_time=self.start_time, **self.motion)
        threading.Thread(target=self.simulator.run, daemon=True).start()
        rospy.Service('/dvl_simulator/imu_enabled', SetBool, self.set_enabled)
        rospy.Timer(rospy.Duration(1./float(rospy.get_param('~imu_rate', 100.))), self.tick)

    def set_enabled(self, request):
        self.enabled = request.data
        return SetBoolResponse(True, 'IMU enabled' if self.enabled else 'IMU paused')

    def tick(self, _event):
        if not self.enabled:
            return
        _, rpy, _, force, omega = trajectory(max(0., time.time()-self.start_time), **self.motion)
        message = Imu()
        message.header.stamp = rospy.Time.now()+rospy.Duration(self.imu_stamp_offset)
        message.header.frame_id = self.frame
        (message.orientation.x, message.orientation.y, message.orientation.z,
         message.orientation.w) = quaternion_from_euler(*rpy)
        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = force
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = omega
        message.orientation_covariance = np.diag([0.03**2]*3).ravel().tolist()
        message.angular_velocity_covariance = np.diag([0.01**2]*3).ravel().tolist()
        message.linear_acceleration_covariance = np.diag([0.1**2]*3).ravel().tolist()
        self.publisher.publish(message)


if __name__ == '__main__':
    rospy.init_node('dvl_simulator')
    ImuSimulator()
    rospy.spin()
