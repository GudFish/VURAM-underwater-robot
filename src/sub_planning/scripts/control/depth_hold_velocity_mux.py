#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rospy
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry


class DepthHoldVelocityMux:
    def __init__(self):
        rospy.init_node('depth_hold_velocity_mux', anonymous=True)

        self.state_topic = rospy.get_param('~state_topic', '/mavros/state')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        self.input_vel_topic = rospy.get_param('~input_vel_topic', '/target_velocity_xy')
        self.output_vel_topic = rospy.get_param('~output_vel_topic', '/target_velocity')

        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.timeout = rospy.get_param('~timeout', 0.5)
        self.depth_kp = rospy.get_param('~depth_kp', 0.9)
        self.depth_kd = rospy.get_param('~depth_kd', 0.6)
        self.max_abs_vz = rospy.get_param('~max_abs_vz', 0.25)
        self.depth_deadband = rospy.get_param('~depth_deadband', 0.03)
        self.vz_deadband = rospy.get_param('~vz_deadband', 0.02)
        self.reset_depth_on_rearm = rospy.get_param('~reset_depth_on_rearm', True)

        self.current_state = State()
        self.have_odom = False
        self.current_z = 0.0
        self.current_vz_enu = 0.0
        self.hold_z = None
        self.last_armed = False

        self.planar_cmd = Twist()
        self.last_cmd_time = rospy.Time(0)

        self.state_sub = rospy.Subscriber(self.state_topic, State, self.state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.input_sub = rospy.Subscriber(self.input_vel_topic, Twist, self.input_vel_cb, queue_size=1)
        self.output_pub = rospy.Publisher(self.output_vel_topic, Twist, queue_size=10)

        period = 1.0 / max(self.control_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo('Depth-hold mux started. input=%s output=%s', self.input_vel_topic, self.output_vel_topic)

    @staticmethod
    def clip(value, vmin, vmax):
        return max(vmin, min(vmax, value))

    def capture_hold_depth(self, reason):
        if not self.have_odom:
            return
        self.hold_z = self.current_z
        rospy.loginfo('Depth reference latched from %s: z_ref=%.3f m', reason, self.hold_z)

    def state_cb(self, msg):
        armed_rising = (not self.last_armed) and msg.armed
        armed_falling = self.last_armed and (not msg.armed)
        self.current_state = msg

        if armed_rising:
            if self.have_odom:
                self.capture_hold_depth('arming event')
            else:
                rospy.logwarn('Armed before odom available. Depth reference will be latched on first odom.')

        if armed_falling and self.reset_depth_on_rearm:
            self.hold_z = None

        self.last_armed = msg.armed

    def odom_cb(self, msg):
        self.current_z = msg.pose.pose.position.z
        self.current_vz_enu = msg.twist.twist.linear.z
        self.have_odom = True

        if self.current_state.armed and self.hold_z is None:
            self.capture_hold_depth('first odom after arming')

    def input_vel_cb(self, msg):
        self.planar_cmd = msg
        self.last_cmd_time = rospy.Time.now()

    def compute_vz_cmd(self):
        if self.hold_z is None or not self.have_odom:
            return 0.0, 0.0

        # Odom z is ENU up-positive. BODY_NED velocity.z is down-positive.
        z_error_enu = self.hold_z - self.current_z
        if abs(z_error_enu) < self.depth_deadband and abs(self.current_vz_enu) < self.vz_deadband:
            return 0.0, z_error_enu

        vz_cmd = self.depth_kp * z_error_enu + self.depth_kd * self.current_vz_enu
        vz_cmd = self.clip(vz_cmd, -self.max_abs_vz, self.max_abs_vz)
        return vz_cmd, z_error_enu

    def timer_cb(self, _event):
        if not self.have_odom:
            rospy.logwarn_throttle(5.0, 'Depth-hold mux waiting for odom...')
            return

        now = rospy.Time.now()
        out = Twist()
        if (now - self.last_cmd_time).to_sec() <= self.timeout:
            out.linear.x = self.planar_cmd.linear.x
            out.linear.y = self.planar_cmd.linear.y
        else:
            rospy.logwarn_throttle(5.0, 'Planar velocity input timeout. Publishing zero vx, vy.')

        vz_cmd, z_error_enu = self.compute_vz_cmd()
        out.linear.z = vz_cmd
        self.output_pub.publish(out)

        rospy.loginfo_throttle(
            2.0,
            'Depth mux: in_xy=(%.2f, %.2f), out_z=%.2f, z_ref=%.2f, z=%.2f, z_err=%.2f'
            % (
                out.linear.x,
                out.linear.y,
                out.linear.z,
                self.hold_z if self.hold_z is not None else float('nan'),
                self.current_z,
                z_error_enu,
            ),
        )

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = DepthHoldVelocityMux()
        node.run()
    except rospy.ROSInterruptException:
        pass
