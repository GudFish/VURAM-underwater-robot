"""Shared MAVROS parameter synchronization and telemetry-rate requests.

These helpers never publish actuator commands; callers may supply a neutral
callback while waiting. Measured stream readiness belongs to the controller.
"""
import threading
import time

import numpy as np
import rospy
from mavros_msgs.srv import MessageInterval, ParamPull


def wait_for_parameter_list(service, hold_neutral, timeout=180.):
    """Wait across MAVROS's per-call fetch timeout without starting collection.

    ParamPull can return success=False after its own 30 s wait while the same
    download continues. Non-forced retries join that download. Run only one
    RPC at a time off-thread so neutral output, shutdown and our overall wall
    timeout remain responsive even while the service call is blocked.
    """
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('param_wait_timeout must be positive and finite')
    deadline = time.monotonic() + timeout
    received = 0

    def keep_waiting():
        if rospy.is_shutdown():
            return False
        hold_neutral()
        if time.monotonic() >= deadline:
            raise RuntimeError('MAVROS parameter-list download was not confirmed within '
                               '{:.1f}s (last received {} parameters)'.format(timeout, received))
        return True

    def request(done, result):
        try:
            remaining = max(.001, deadline - time.monotonic())
            rospy.wait_for_service(service, timeout=min(5., remaining))
            result['response'] = rospy.ServiceProxy(service, ParamPull)(force_pull=False)
        except Exception as exc:
            result['error'] = exc
        finally:
            done.set()

    while keep_waiting():
        done, result = threading.Event(), {}
        threading.Thread(target=request, args=(done, result),
                         name='parameter-pull', daemon=True).start()
        while not done.wait(.1):
            if not keep_waiting():
                return False
            rospy.loginfo_throttle(5., 'waiting for MAVROS parameter-list download')
        if not keep_waiting():
            return False
        response = result.get('response')
        if response is not None:
            received = int(response.param_received)
            if response.success:
                rospy.loginfo('MAVROS parameter list received (%d params)', received)
                return True
            rospy.logwarn('MAVROS parameter pull has not completed (%d params received); '
                          'retrying without restarting download', received)
        else:
            rospy.logwarn('MAVROS parameter pull unavailable (%s); retrying',
                          result.get('error'))
        retry_at = time.monotonic() + 1.
        while time.monotonic() < retry_at:
            if not keep_waiting():
                return False
            time.sleep(max(0., min(.1, retry_at - time.monotonic())))
    return False


def configure_mavros_telemetry(service, rate, hold_neutral, timeout=15.):
    """Request raw IMU, attitude and position; controllers check actual streams."""
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('mavros_telemetry_rate must be positive and finite')
    done, errors = threading.Event(), []

    def request():
        try:
            rospy.wait_for_service(service, timeout=5.)
            proxy = rospy.ServiceProxy(service, MessageInterval)
            # ArduSub's RAW_IMU supplies acceleration to MAVROS imu/data;
            # attitude messages alone leave acceleration marked unavailable.
            # Request it before attitude. Quaternions avoid MAVROS ignoring
            # Euler attitude after another ground station sent quaternions.
            for message_id in (27, 31, 32):
                if rospy.is_shutdown():
                    break
                try:
                    response = proxy(message_id=message_id, message_rate=rate)
                    if not response.success:
                        errors.append('message {} request rejected'.format(message_id))
                except Exception as exc:
                    errors.append('message {}: {}'.format(message_id, exc))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            done.set()

    hold_neutral()
    threading.Thread(target=request, name='telemetry-rate-request', daemon=True).start()
    deadline = time.monotonic() + timeout
    while not done.wait(.1):
        if rospy.is_shutdown():
            return False
        hold_neutral()
        if time.monotonic() >= deadline:
            rospy.logwarn('MAVROS telemetry-rate request timed out; waiting for measured stream readiness')
            return False
    if rospy.is_shutdown():
        return False
    if errors:
        rospy.logwarn('MAVROS telemetry-rate setup incomplete (%s); '
                      'waiting for measured stream readiness', '; '.join(errors))
        return False
    rospy.loginfo('requested MAVROS raw IMU/attitude/position telemetry at %.1f Hz', rate)
    return True
