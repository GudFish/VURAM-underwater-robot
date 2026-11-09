#!/usr/bin/env python3
"""Validate EKF inputs and gate public odometry/TF on live inertial data.

DVL outage: keep feeding fresh IMU; the EKF estimates velocity from acceleration.
IMU outage: hide internal predictions; on recovery restore the last public pose
before resuming, so old acceleration is never integrated across the blind gap.
"""
import copy
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse
from robot_localization.srv import SetPose

from dvl_odometry.core import finite, rotation
from dvl_odometry.ros_support import Diagnostics, OdomOutput, positive_param


class MavrosDvlFusion:
    def __init__(self):
        self.lock = threading.RLock()
        self.output = OdomOutput()
        self.imu_timeout = positive_param('imu_timeout', 0.5)
        self.vel_timeout = positive_param('vel_timeout', 1.)
        self.imu = self.velocity = self.last_output = None
        self.vel_valid = False
        self.vel_invalid_at = 0.
        self.started = False
        self.suspended = False
        self.minimum_output_stamp = 0.
        self.last_imu_stamp = None
        self.diagnostics = Diagnostics('dvl/mavros_fused')
        self.imu_pub = rospy.Publisher('/dvl/ekf/imu', Imu, queue_size=50)
        self.velocity_pub = rospy.Publisher('/dvl/ekf/velocity', TwistWithCovarianceStamped, queue_size=30)
        self.set_pose = rospy.ServiceProxy('/dvl/ekf/set_pose', SetPose)
        rospy.Subscriber(rospy.get_param('~imu_topic', '/mavros/imu/data'), Imu, self.on_imu, queue_size=50)
        rospy.Subscriber('/dvl/velocity', TwistWithCovarianceStamped, self.on_velocity, queue_size=30)
        rospy.Subscriber('/dvl/velocity_valid', Bool, self.on_validity, queue_size=20)
        rospy.Subscriber('/dvl/connected', Bool, self.on_connection, queue_size=5)
        rospy.Subscriber('/dvl/ekf/odom', Odometry, self.on_output, queue_size=5)
        rospy.Service('/dvl_odometry/reset', Trigger, self.reset)
        rospy.Timer(rospy.Duration(0.1), self.watchdog)
        rospy.Timer(rospy.Duration(1.), self.diagnose)

    def fresh(self, message, timeout):
        return message is not None and 0 <= (rospy.Time.now()-message.header.stamp).to_sec() <= timeout

    def on_validity(self, message):
        if not message.data:
            with self.lock:
                self.vel_valid = False
                self.vel_invalid_at = rospy.Time.now().to_sec()

    def on_connection(self, message):
        if not message.data:
            with self.lock:
                self.vel_valid = False
                self.vel_invalid_at = rospy.Time.now().to_sec()

    def try_start(self):
        if (not self.started and self.fresh(self.imu, self.imu_timeout) and self.vel_valid
                and self.fresh(self.velocity, self.vel_timeout)):
            # Wait for EKF subscribers, so startup measurements cannot disappear.
            if self.imu_pub.get_num_connections() and self.velocity_pub.get_num_connections():
                self.started = True
                self.imu_pub.publish(self.imu)
                self.velocity_pub.publish(self.velocity)

    def on_velocity(self, message):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        with self.lock:
            if not self.fresh(message, self.vel_timeout) or message.header.stamp.to_sec() < self.vel_invalid_at:
                return
            self.velocity, self.vel_valid = message, True
            if self.started and not self.suspended and self.fresh(self.imu, self.imu_timeout):
                self.velocity_pub.publish(message)
            else:
                self.try_start()

    def validate_imu(self, message):
        if message.header.frame_id != self.output.robot_frame:
            raise ValueError('IMU must be expressed in '+self.output.robot_frame+'; configure MAVROS imu/frame_id')
        q, a, w = message.orientation, message.linear_acceleration, message.angular_velocity
        rotation([q.x, q.y, q.z, q.w])
        finite([a.x, a.y, a.z, w.x, w.y, w.z])
        result = copy.deepcopy(message)
        norm = np.linalg.norm([q.x, q.y, q.z, q.w])
        result.orientation.x, result.orientation.y, result.orientation.z, result.orientation.w = np.array(
            [q.x, q.y, q.z, q.w])/norm
        for name, floor in [('orientation_covariance', 0.03**2),
                            ('angular_velocity_covariance', 0.01**2),
                            ('linear_acceleration_covariance', 0.1**2)]:
            raw = getattr(result, name)
            if raw[0] < 0:
                raise ValueError('IMU has unavailable '+name)
            matrix = finite(raw).reshape(3, 3)
            if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < -1e-10:
                raise ValueError('invalid IMU '+name)
            # MAVROS sensor-spec defaults can be unrealistically confident for vehicle dynamics.
            matrix += np.diag(np.maximum(0., floor-np.diag(matrix)))
            setattr(result, name, matrix.ravel().tolist())
        return result

    def on_imu(self, message):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        with self.lock:
            try:
                if not self.fresh(message, self.imu_timeout):
                    raise ValueError('stale/future IMU timestamp')
                message = self.validate_imu(message)
                stamp = message.header.stamp.to_sec()
                if self.last_imu_stamp is not None and stamp <= self.last_imu_stamp:
                    return
                self.last_imu_stamp = stamp
            except (ValueError, TypeError) as exc:
                if self.started:
                    self.suspended = True
                self.imu = None
                rospy.logwarn_throttle(2., 'DVL EKF rejected IMU: %s', exc)
                return
            if self.started and self.imu is not None and not self.fresh(self.imu, self.imu_timeout):
                self.suspended = True
            self.imu = message
            if self.started and self.suspended:
                if not self.restore_pose(zero_position=False, stamp=message.header.stamp):
                    return
                self.suspended = False
            if self.started:
                self.imu_pub.publish(message)
            else:
                self.try_start()

    def restore_pose(self, zero_position, stamp):
        if self.last_output is None and self.imu is None:
            return False
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = stamp-rospy.Duration(1e-6)
        pose.header.frame_id = self.output.odom_frame
        if self.last_output is not None:
            pose.pose = copy.deepcopy(self.last_output.pose)
            pose.pose.covariance = list(pose.pose.covariance)
        else:
            # Recovery can happen before the first public EKF frame.
            pose.pose.pose.orientation = copy.deepcopy(self.imu.orientation)
            cov = np.eye(6)*1e-6
            cov[3:, 3:] = np.asarray(self.imu.orientation_covariance).reshape(3, 3)
            pose.pose.covariance = cov.ravel().tolist()
        if zero_position:
            pose.pose.pose.position.x = pose.pose.pose.position.y = pose.pose.pose.position.z = 0.
            for index in (0, 7, 14):
                pose.pose.covariance[index] = 1e-6
        elif self.last_output is not None:
            gap = max(0., (stamp-self.last_output.header.stamp).to_sec())
            for index in (0, 7, 14):
                pose.pose.covariance[index] += gap
        try:
            rospy.wait_for_service('/dvl/ekf/set_pose', timeout=0.3)
            self.set_pose(pose)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(2., 'DVL EKF set_pose unavailable: %s', exc)
            return False
        self.minimum_output_stamp = stamp.to_sec()
        # SetPose resets velocity; it is unknown after a blind inertial gap.
        # A new DVL sample will re-constrain it, without inventing a gap displacement.
        return True

    def reset(self, _request):
        with self.lock:
            if not self.started or not self.fresh(self.imu, self.imu_timeout):
                return TriggerResponse(False, 'EKF not initialized or IMU unavailable')
            ok = self.restore_pose(True, rospy.Time.now())
            return TriggerResponse(ok, 'position reset; ENU orientation retained' if ok else 'EKF reset unavailable')

    def on_output(self, message):
        with self.lock:
            if (not self.started or self.suspended or not self.fresh(self.imu, self.imu_timeout)
                    or message.header.stamp.to_sec() < self.minimum_output_stamp):
                return
            self.last_output = copy.deepcopy(message)
            self.output.publish(copy.deepcopy(message))

    def watchdog(self, _event):
        with self.lock:
            if self.started and not self.fresh(self.imu, self.imu_timeout):
                self.suspended = True
            if not self.started:
                self.try_start()

    def diagnose(self, _event):
        with self.lock:
            if not self.started:
                level, text = 1, 'waiting for fresh IMU and first valid bottom velocity'
            elif self.suspended or not self.fresh(self.imu, self.imu_timeout):
                level, text = 2, 'IMU unavailable: public odometry and TF paused'
            elif not self.vel_valid or not self.fresh(self.velocity, self.vel_timeout):
                level, text = 1, 'DVL unavailable: inertial dead reckoning; position drift accumulates'
                rospy.logwarn_throttle(5., text)
            else:
                level, text = 0, 'MAVROS IMU + DVL velocity EKF'
            self.diagnostics.send(level, text, output_hz=20, initialized=self.started,
                                  imu_fresh=self.fresh(self.imu, self.imu_timeout),
                                  dvl_fresh=self.vel_valid and self.fresh(self.velocity, self.vel_timeout))


if __name__ == '__main__':
    rospy.init_node('mavros_dvl_fusion_node')
    MavrosDvlFusion()
    rospy.spin()
