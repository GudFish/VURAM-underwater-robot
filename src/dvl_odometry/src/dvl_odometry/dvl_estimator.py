"""ROS adapters for direct DR output and bounded DVL-only interpolation/prediction."""
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Trigger, TriggerResponse

from .core import UNKNOWN_VARIANCE, DvlPredictor, RateLimiter
from .ros_support import Diagnostics, OdomOutput, positive_param


class DvlEstimator:
    def __init__(self, fused=False):
        self.fused = fused
        self.lock = threading.RLock()
        self.output = OdomOutput()
        self.rate = positive_param('odom_rate', 20.)
        self.predictor = DvlPredictor(positive_param('vel_timeout', 1.),
                                      positive_param('dr_timeout', 1.),
                                      positive_param('cov_growth_pos', 0.01, allow_zero=True))
        self.limiter = RateLimiter(self.rate if fused and self.rate <= 5 else 0.)
        self.resetting = False
        self.ignore_before = 0.
        self.vel_invalid_at = self.dr_invalid_at = 0.
        self.diagnostics = Diagnostics('dvl/'+('dvl_fused' if fused else 'dvl_raw'))
        self.driver_reset = rospy.ServiceProxy('/dvl_driver/reset_dead_reckoning', Trigger)
        rospy.Service('/dvl_odometry/reset', Trigger, self.reset)
        rospy.Subscriber('/dvl/velocity', TwistWithCovarianceStamped, self.velocity, queue_size=30)
        rospy.Subscriber('/dvl/dead_reckoning', PoseWithCovarianceStamped, self.pose, queue_size=10)
        rospy.Subscriber('/dvl/velocity_valid', Bool, self.velocity_validity, queue_size=30)
        rospy.Subscriber('/dvl/dr_valid', Bool, self.dr_validity, queue_size=10)
        rospy.Subscriber('/dvl/connected', Bool, self.connection, queue_size=5)
        rospy.Subscriber('/dvl/resetting', Bool, self.reset_state, queue_size=10)
        if fused and self.rate > 5:
            rospy.Timer(rospy.Duration(1./self.rate), self.tick)
        elif not fused:
            rospy.loginfo('dvl_raw: odom_rate is ignored; output follows actual DR reports (~5 Hz)')
        rospy.Timer(rospy.Duration(1.), self.diagnose)

    def reset(self, request):
        try:
            return self.driver_reset(request)
        except rospy.ServiceException as exc:
            return TriggerResponse(False, str(exc))

    def reset_state(self, message):
        with self.lock:
            self.resetting = message.data
            if message.data:
                self.predictor.clear()
                self.ignore_before = rospy.Time.now().to_sec()
                self.limiter.last = None

    def connection(self, message):
        if not message.data:
            with self.lock:
                now = rospy.Time.now().to_sec()
                self.vel_invalid_at = self.dr_invalid_at = now
                self.predictor.invalidate_velocity(now)
                self.predictor.invalidate_dr(now)

    def velocity_validity(self, message):
        if not message.data:
            with self.lock:
                self.vel_invalid_at = rospy.Time.now().to_sec()
                self.predictor.invalidate_velocity(self.vel_invalid_at)

    def dr_validity(self, message):
        if not message.data:
            with self.lock:
                self.dr_invalid_at = rospy.Time.now().to_sec()
                self.predictor.invalidate_dr(self.dr_invalid_at)

    def velocity(self, message):
        with self.lock:
            stamp = message.header.stamp.to_sec()
            if self.resetting or stamp < max(self.ignore_before, self.vel_invalid_at):
                return
            v = message.twist.twist.linear
            self.predictor.set_velocity(stamp, [v.x, v.y, v.z],
                                         np.array(message.twist.covariance).reshape(6, 6)[:3, :3])

    def pose(self, message):
        with self.lock:
            stamp = message.header.stamp.to_sec()
            if self.resetting or stamp < max(self.ignore_before, self.dr_invalid_at):
                return
            p, q = message.pose.pose.position, message.pose.pose.orientation
            self.predictor.anchor(stamp, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w], message.pose.covariance)
            if not self.fused or self.rate <= 5:
                if self.limiter.allow(stamp):
                    self.publish(message.header.stamp, predict=False)

    def tick(self, event):
        with self.lock:
            if self.resetting or self.predictor.position is None:
                return
            now = rospy.Time.now()
            if now.to_sec() < self.predictor.time:
                self.predictor.clear()
                return
            self.predictor.advance(now.to_sec())
            self.publish(now, predict=True)

    def publish(self, stamp, predict):
        state = self.predictor
        message = Odometry()
        message.header.stamp, message.header.frame_id = stamp, self.output.odom_frame
        message.child_frame_id = self.output.robot_frame
        (message.pose.pose.position.x, message.pose.pose.position.y,
         message.pose.pose.position.z) = state.position
        (message.pose.pose.orientation.x, message.pose.pose.orientation.y,
         message.pose.pose.orientation.z, message.pose.pose.orientation.w) = state.quaternion
        message.pose.covariance = state.pose_cov.ravel().tolist()
        twist_cov = np.eye(6) * UNKNOWN_VARIANCE
        valid = state.velocity_fresh(stamp.to_sec())
        if predict:
            valid = valid and state.predicting(stamp.to_sec())
        if valid:
            (message.twist.twist.linear.x, message.twist.twist.linear.y,
             message.twist.twist.linear.z) = state.velocity
            twist_cov[:3, :3] = state.vel_cov
        message.twist.covariance = twist_cov.ravel().tolist()
        self.output.publish(message)

    def diagnose(self, event):
        with self.lock:
            now = rospy.Time.now().to_sec()
            state = self.predictor
            if state.position is None or self.resetting:
                level, text = 1, 'waiting for valid post-reset DVL pose'
            elif not state.dr_valid or now-state.dr_stamp > state.dr_timeout:
                level, text = 1, 'DVL pose stale/invalid; holding predicted position'
            elif self.fused and not state.velocity_fresh(now):
                level, text = 1, 'velocity stale/invalid; no extrapolation, DR corrections remain active'
            else:
                level, text = 0, 'DR anchor + bounded velocity prediction' if self.fused else 'device DR measurements'
            self.diagnostics.send(level, text, output_hz=self.rate if self.fused else 'device (~5)',
                                  prediction_frames=self.fused and self.rate > 5,
                                  vel_age_s=now-state.vel_stamp if state.vel_stamp is not None else 'none')
