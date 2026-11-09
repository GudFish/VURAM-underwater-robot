"""Shared ROS output/diagnostics; no estimator owns the hardware socket."""
import math

import rospy
import tf2_ros
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header


def positive_param(name, default, allow_zero=False):
    value = float(rospy.get_param('~'+name, default))
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError('%s must be finite and %s' % (name, 'non-negative' if allow_zero else 'positive'))
    return value


class Diagnostics:
    def __init__(self, name):
        self.name = name
        self.publisher = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=5)

    def send(self, level, message, **values):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        status = DiagnosticStatus(level=level, name=self.name, message=message, hardware_id='WaterLinked A50')
        status.values = [KeyValue(str(key), str(value)) for key, value in values.items()]
        self.publisher.publish(DiagnosticArray(header=Header(stamp=rospy.Time.now()), status=[status]))


class OdomOutput:
    def __init__(self):
        from nav_msgs.msg import Odometry
        self.publisher = rospy.Publisher(rospy.get_param('~odom_topic', '/dvl/odom'), Odometry, queue_size=5)
        self.broadcaster = tf2_ros.TransformBroadcaster() if rospy.get_param('~publish_tf', False) else None
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.publish_z = rospy.get_param('~publish_z', True)

    def publish(self, message):
        if rospy.is_shutdown() or rospy.core.is_shutdown_requested():
            return
        if not self.publish_z:
            message.pose.pose.position.z = 0.
        self.publisher.publish(message)
        if self.broadcaster:
            transform = TransformStamped(header=message.header, child_frame_id=message.child_frame_id)
            transform.transform.translation.x = message.pose.pose.position.x
            transform.transform.translation.y = message.pose.pose.position.y
            transform.transform.translation.z = message.pose.pose.position.z
            transform.transform.rotation = message.pose.pose.orientation
            self.broadcaster.sendTransform(transform)
