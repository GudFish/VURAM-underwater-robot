#!/usr/bin/env python3
"""Water Linked A50 TCP JSON driver. Rates are measurement caps, never resampling."""
import copy
import math
import threading
import time
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger, TriggerResponse
from tf.transformations import quaternion_from_euler

from dvl_odometry.core import UNKNOWN_VARIANCE, RateLimiter, covariance, finite, frd_to_flu
from dvl_odometry.ros_support import Diagnostics, positive_param
from dvl_odometry.transport import JsonClient


class DvlVelocityNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.dvl_frame = rospy.get_param('~dvl_frame', 'dvl_link')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.rate = positive_param('velocity_rate', 0., allow_zero=True)
        self.limiter = RateLimiter(self.rate)
        self.vel_timeout = positive_param('vel_timeout', 1.)
        self.dr_timeout = positive_param('dr_timeout', 1.)
        self.reset_timeout = positive_param('reset_timeout', 2.)
        self.calib_timeout = positive_param('gyro_calib_timeout', 20.)
        self.fallback_std = positive_param('velocity_fallback_std', 0.05)
        self.z_std = positive_param('dr_z_std', 1.)
        self.rp_std = math.radians(positive_param('dr_rp_std', 3.))
        self.yaw_std = math.radians(positive_param('dr_yaw_std', 5.))
        self.velocity_pub = rospy.Publisher('/dvl/velocity', TwistWithCovarianceStamped, queue_size=20)
        self.raw_pub = rospy.Publisher('/dvl/velocity_raw', TwistWithCovarianceStamped, queue_size=20)
        self.pose_pub = rospy.Publisher('/dvl/dead_reckoning', PoseWithCovarianceStamped, queue_size=10)
        self.vel_valid_pub = rospy.Publisher('/dvl/velocity_valid', Bool, queue_size=10, latch=True)
        self.dr_valid_pub = rospy.Publisher('/dvl/dr_valid', Bool, queue_size=10, latch=True)
        self.connected_pub = rospy.Publisher('/dvl/connected', Bool, queue_size=2, latch=True)
        self.reset_pub = rospy.Publisher('/dvl/resetting', Bool, queue_size=10, latch=True)
        self.alt_pub = rospy.Publisher('/dvl/altitude', Range, queue_size=5)
        self.fom_pub = rospy.Publisher('/dvl/fom', Float32, queue_size=5)
        self.diagnostics = Diagnostics('dvl/driver')
        self.connected = self.vel_valid = self.dr_valid = self.resetting = False
        self.reset_until = float('inf')
        self.reset_sent = False
        self.errors = 0
        self.velocity_times = deque(maxlen=100)
        self.dr_times = deque(maxlen=50)
        self.last_vel = self.last_dr = None
        self.device_time = ''
        self.vel_valid_pub.publish(False)
        self.dr_valid_pub.publish(False)
        self.connected_pub.publish(False)
        self.reset_pub.publish(False)
        self.client = JsonClient(rospy.get_param('~ip', '192.168.1.95'),
                                 int(rospy.get_param('~port', 16171)),
                                 self.on_message, self.on_connection, self.on_error,
                                 positive_param('reconnect_min_delay', 1.),
                                 positive_param('reconnect_max_delay', 30.))
        rospy.Service('/dvl_driver/reset_dead_reckoning', Trigger, self.reset_service)
        rospy.Service('/dvl_odometry/calibrate_gyro', Trigger, self.calibrate_service)
        rospy.on_shutdown(self.client.close)
        rospy.Timer(rospy.Duration(1.), self.diagnose)
        self.client.start()

    def on_error(self, error):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        self.errors += 1
        rospy.logwarn_throttle(5., 'DVL transport/measurement error: %s', error)

    def invalidate(self):
        self.vel_valid = self.dr_valid = False
        self.vel_valid_pub.publish(False)
        self.dr_valid_pub.publish(False)

    def on_connection(self, connected):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        with self.lock:
            self.connected = connected
            self.connected_pub.publish(connected)
            self.invalidate()
            self.limiter.last = None
            if not connected:
                return
            if rospy.get_param('~apply_config', False):
                params = {}
                sound = float(rospy.get_param('~speed_of_sound', 0.))
                if sound:
                    if not 1000 <= sound <= 2000:
                        self.on_error('speed_of_sound must be 1000..2000 m/s')
                    else:
                        params['speed_of_sound'] = sound
                # Explicit apply_config permits setting the offset back to zero.
                offset = float(rospy.get_param('~mounting_rotation_offset', 0.))
                if math.isfinite(offset) and 0 <= offset <= 360:
                    params['mounting_rotation_offset'] = offset
                else:
                    self.on_error('mounting_rotation_offset must be 0..360 degrees')
                self.client.submit({'command': 'set_config', 'parameters': params}, self.reset_timeout,
                                   lambda ok, error: self.on_error(error) if not ok else None)
            if rospy.get_param('~reset_on_start', True) and not self.reset_sent:
                self.reset_sent = True
                self.begin_reset()

    def begin_reset(self):
        if self.resetting:
            return None
        self.resetting = True
        self.reset_until = float('inf')
        self.reset_pub.publish(True)
        self.invalidate()
        return self.client.submit({'command': 'reset_dead_reckoning'}, self.reset_timeout, self.reset_done)

    def reset_done(self, ok, error):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        with self.lock:
            if ok:
                # ACK precedes hardware zeroing by about 50 ms. Discard reports in flight.
                self.reset_until = time.monotonic()+0.1
            else:
                self.resetting = False
                self.reset_pub.publish(False)
                self.on_error('reset failed: '+error)

    def reset_service(self, _request):
        with self.lock:
            command = self.begin_reset()
        if command is None:
            return TriggerResponse(False, 'reset already in progress')
        command.event.wait(self.reset_timeout+0.2)
        return TriggerResponse(command.event.is_set() and command.ok,
                               command.error or ('reset acknowledged; waiting for post-reset report' if command.ok
                                                  else 'reset timeout'))

    def calibrate_service(self, _request):
        rospy.logwarn('DVL gyro calibration: keep device stationary')
        command = self.client.submit({'command': 'calibrate_gyro'}, self.calib_timeout)
        command.event.wait(self.calib_timeout+0.2)
        return TriggerResponse(command.event.is_set() and command.ok, command.error or 'calibration complete')

    def on_message(self, message):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        with self.lock:
            kind = message.get('type')
            if self.resetting:
                if time.monotonic() < self.reset_until:
                    return
                self.resetting = False
                self.reset_pub.publish(False)
            if kind in ('velocity', 'velocity_water'):
                try:
                    self.on_velocity(message)
                except (ValueError, TypeError, KeyError, OverflowError):
                    self.vel_valid = False
                    self.vel_valid_pub.publish(False)
                    raise
            elif kind == 'position_local':
                try:
                    self.on_pose(message)
                except (ValueError, TypeError, KeyError, OverflowError):
                    self.dr_valid = False
                    self.dr_valid_pub.publish(False)
                    raise

    def on_velocity(self, message):
        stamp = rospy.Time.now()
        now = stamp.to_sec()
        self.last_vel = now
        self.velocity_times.append(now)
        self.device_time = str(message.get('time_of_validity', 'unavailable'))
        self.vel_valid = message.get('velocity_valid') is True and message['type'] == 'velocity'
        self.vel_valid_pub.publish(self.vel_valid)
        if not self.vel_valid:
            return
        vector = finite([message['vx'], message['vy'], message['vz']])
        cov = covariance(message.get('covariance'), self.fallback_std)
        fom = float(message.get('fom', self.fallback_std))
        if math.isfinite(fom) and fom >= 0:
            self.fom_pub.publish(fom)
        altitude = float(message.get('altitude', -1.))
        if math.isfinite(altitude) and altitude >= 0:
            value = Range()
            value.header.stamp, value.header.frame_id = stamp, self.dvl_frame
            value.radiation_type = Range.ULTRASOUND
            value.min_range, value.max_range, value.range = 0.05, 50., altitude
            self.alt_pub.publish(value)
        if not self.limiter.allow(now):
            return
        raw = TwistWithCovarianceStamped()
        raw.header.stamp, raw.header.frame_id = stamp, self.dvl_frame
        raw.twist.twist.linear.x, raw.twist.twist.linear.y, raw.twist.twist.linear.z = vector
        block = np.eye(6) * UNKNOWN_VARIANCE
        block[:3, :3] = cov
        raw.twist.covariance = block.ravel().tolist()
        self.raw_pub.publish(raw)
        output = copy.deepcopy(raw)
        vector, cov = frd_to_flu(vector, cov)
        output.header.frame_id = self.robot_frame
        output.twist.twist.linear.x, output.twist.twist.linear.y, output.twist.twist.linear.z = vector
        block[:3, :3] = cov
        output.twist.covariance = block.ravel().tolist()
        self.velocity_pub.publish(output)

    def on_pose(self, message):
        stamp = rospy.Time.now()
        self.last_dr = stamp.to_sec()
        self.dr_times.append(self.last_dr)
        self.dr_valid = message.get('status', 0) == 0
        self.dr_valid_pub.publish(self.dr_valid)
        if not self.dr_valid:
            return
        x, y, z, roll, pitch, yaw, std = finite([message[key] for key in
                                                ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'std')])
        if std < 0:
            raise ValueError('negative position std')
        pose = PoseWithCovarianceStamped()
        pose.header.stamp, pose.header.frame_id = stamp, self.odom_frame
        pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z = x, -y, -z
        q = quaternion_from_euler(*np.radians([roll, -pitch, -yaw]))
        (pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
         pose.pose.pose.orientation.z, pose.pose.pose.orientation.w) = q
        pose.pose.covariance = np.diag([max(std**2, 1e-6), max(std**2, 1e-6),
                                       max(std**2, self.z_std**2), self.rp_std**2,
                                       self.rp_std**2, self.yaw_std**2]).ravel().tolist()
        self.pose_pub.publish(pose)

    def diagnose(self, _event):
        with self.lock:
            now = rospy.Time.now().to_sec()
            vel_age = now-self.last_vel if self.last_vel is not None else float('inf')
            dr_age = now-self.last_dr if self.last_dr is not None else float('inf')
            if vel_age > self.vel_timeout:
                self.vel_valid = False
                self.vel_valid_pub.publish(False)
            if dr_age > self.dr_timeout:
                self.dr_valid = False
                self.dr_valid_pub.publish(False)
            def rate(times):
                recent = [t for t in times if 0 <= now-t < 10.]
                return (len(recent)-1)/(recent[-1]-recent[0]) if len(recent) > 1 else 0.
            velocity_hz, dr_hz = rate(self.velocity_times), rate(self.dr_times)
            level, message = 0, 'bottom-track measurements available'
            if not self.connected:
                level, message = 2, 'DVL disconnected'
            elif self.resetting:
                level, message = 1, 'waiting for hardware reset'
            elif not self.vel_valid:
                level, message = 1, 'bottom velocity invalid/stale'
            elif self.rate and velocity_hz < 0.9*self.rate:
                level, message = 1, 'device below requested cap; publishing actual rate (no resampling)'
            self.diagnostics.send(level, message, velocity_measurement_hz=round(velocity_hz, 2),
                                  dr_measurement_hz=round(dr_hz, 2), velocity_cap_hz=self.rate,
                                  velocity_age_s=vel_age, dr_age_s=dr_age,
                                  device_time_of_validity_us=self.device_time, errors=self.errors)


if __name__ == '__main__':
    rospy.init_node('dvl_velocity_node')
    DvlVelocityNode()
    rospy.spin()
