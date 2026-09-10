"""Shared sonar-image obstacle cache and last-resort command safety filter."""
import math

import rospy
from std_msgs.msg import Float32MultiArray


class SonarObstacleAvoidance:
    """Consumes [count, range, bearing, radius, confidence, ...] obstacle messages."""
    def __init__(self):
        self.enabled = bool(rospy.get_param('~avoidance_enable', True))
        self.topic = rospy.get_param('~obstacle_topic', '/forward_sonar/obstacles')
        self.timeout = float(rospy.get_param('~avoidance_timeout', 0.8))
        self.max_range = float(rospy.get_param('~avoidance_range', 12.0))
        self.max_obstacles = int(rospy.get_param('~avoidance_max_obstacles', 8))
        self.max_obstacle_speed = float(rospy.get_param('~avoidance_max_obstacle_speed', 0.2))
        self.vehicle_radius = float(rospy.get_param('~avoidance_vehicle_radius', 0.45))
        self.clearance = float(rospy.get_param('~avoidance_clearance', 1.5))
        self.brake_distance = float(rospy.get_param('~avoidance_brake_distance', 3.0))
        self.blind_forward_scale = float(rospy.get_param('~avoidance_blind_forward_scale', 0.45))
        self.lateral_gain = float(rospy.get_param('~avoidance_lateral_gain', 0.45))
        self.min_confidence = float(rospy.get_param('~avoidance_min_confidence', 0.15))
        self._items = []
        self._stamp = None
        self.sub = rospy.Subscriber(self.topic, Float32MultiArray, self._cb, queue_size=1)

    def _cb(self, msg):
        data = msg.data
        count = int(round(data[0])) if data else 0
        items = []
        for idx in range(max(0, count)):
            base = 1 + idx * 5
            if base + 5 > len(data):
                break
            rng, bearing, radius, confidence = map(float, data[base:base + 4])
            # 置信度门控：滤除弱杂波回波，避免密集场景下过度保守
            if 0.0 < rng <= self.max_range and confidence >= self.min_confidence:
                items.append((rng, bearing, max(radius, 0.05), min(confidence, 1.0)))
        self._items = sorted(items, key=lambda item: item[0])[:max(self.max_obstacles, 0)]
        self._stamp = rospy.Time.now()

    def fresh(self):
        return self._stamp is not None and (rospy.Time.now() - self._stamp).to_sec() <= self.timeout

    def obstacles_world(self, x, y, yaw):
        """Return fixed-size world obstacles as (x, y, inflated_radius, enabled)."""
        age = 0.0 if self._stamp is None else max(0.0, (rospy.Time.now() - self._stamp).to_sec())
        result = []
        if self.enabled and self.fresh():
            for rng, bearing, radius, confidence in self._items:
                angle = yaw + bearing
                inflated = self.vehicle_radius + self.clearance + radius + self.max_obstacle_speed * age
                result.append((x + rng * math.cos(angle), y + rng * math.sin(angle), inflated, 1.0))
        while len(result) < max(self.max_obstacles, 0):
            result.append((0.0, 0.0, 0.0, 0.0))
        return result

    def filter_command(self, u, v, yaw, u_max, v_max):
        """Brake toward frontal hazards and steer laterally away from the closest one."""
        if not self.enabled:
            return u, v, yaw, False
        if not self.fresh():
            return min(u, max(0.0, u_max * self.blind_forward_scale)), v, yaw, True
        intervention = False
        for rng, bearing, radius, _confidence in self._items:
            frontal = max(0.0, math.cos(bearing))
            threshold = self.brake_distance + self.vehicle_radius + self.clearance + radius
            if frontal <= 0.05 or rng >= threshold:
                continue
            scale = max(0.0, min(1.0, (rng - self.vehicle_radius - self.clearance - radius) /
                                 max(self.brake_distance, 1e-3)))
            u = min(u, max(0.0, u) * scale)
            direction = -1.0 if bearing >= 0.0 else 1.0
            v = max(-v_max, min(v_max, v + direction * self.lateral_gain * (1.0 - scale)))
            yaw += direction * 0.20 * (1.0 - scale)
            intervention = True
        return u, v, yaw, intervention
