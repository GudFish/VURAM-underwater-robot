#!/usr/bin/env python3
"""Configure shared FC telemetry independently of policy/PD control loops."""
import math
from pathlib import Path
import sys
import threading

import rospy
from std_msgs.msg import Bool

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from control.mavros_startup import configure_mavros_telemetry, wait_for_parameter_list


def run():
    rate = float(rospy.get_param('~mavros_telemetry_rate', 20.))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('mavros_telemetry_rate must be positive and finite')
    if rospy.get_param('~wait_ready', True):
        ready = threading.Event()
        subscriber = rospy.Subscriber(
            rospy.get_param('~ready_topic', '/dof_cmd/ready'), Bool,
            lambda message: ready.set() if message.data else ready.clear(), queue_size=1)
        try:
            while not rospy.is_shutdown() and not ready.wait(.1):
                rospy.loginfo_throttle(5., 'telemetry setup waiting for thruster bridge ready')
        finally:
            subscriber.unregister()
    if rospy.is_shutdown():
        return
    if rospy.get_param('~wait_param_list', True):
        if not wait_for_parameter_list(
                rospy.get_param('~param_pull_service', '/mavros/param/pull'), lambda: None,
                float(rospy.get_param('~param_wait_timeout', 180.))):
            return
    configure_mavros_telemetry(
        rospy.get_param('~message_interval_service', '/mavros/set_message_interval'),
        rate, lambda: None)
    # Keep the shared node alive after setup. Controllers independently check
    # measured stream readiness, including when the FC rejected a request.
    rospy.spin()


def main():
    rospy.init_node('mavros_telemetry', anonymous=False)
    try:
        run()
        return 0
    except (ValueError, RuntimeError, rospy.ROSException) as exc:
        rospy.logerr('MAVROS telemetry setup failed: %s', exc)
        return 1


if __name__ == '__main__':
    sys.exit(main())
