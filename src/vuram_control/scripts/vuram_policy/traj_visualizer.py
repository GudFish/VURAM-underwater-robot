"""Trajectory overlays for RViz: reference path + actual vehicle path.

Standard visualization_msgs/MarkerArray (LINE_STRIP) published on
/auv_viz/viz_markers_array; display them in RViz with the fixed frame
configured via frame_id (default 'map', the fixed frame of the
vuram_control stack). With enabled=False (enable_viz:=false) nothing is
published.

IMPORTANT (rviz convention): the RViz Marker display subscribes to
<topic> expecting visualization_msgs/Marker and to <topic>_array expecting
MarkerArray. Publishing MarkerArray on a non-_array topic is silently
rejected by the type check ("topic types do not match"), so the publisher
topic must carry the _array suffix.
"""
import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


class TrajVisualizer(object):
    NS = 'auv_viz'
    REF_ID = 0
    ACTUAL_ID = 1
    # LINE_STRIP widths [m] (marker scale.x) -- tweak here for RViz
    REF_WIDTH = 0.01
    ACTUAL_WIDTH = 0.02

    def __init__(self, enabled=False, topic='/auv_viz/viz_markers_array',
                 sample_dt=0.5, frame_id='map'):
        self.enabled = enabled
        if not enabled:
            return
        self.frame_id = frame_id
        self.pub = rospy.Publisher(topic, MarkerArray, queue_size=5)
        self.sample_dt = sample_dt
        self._ref_points = []
        self._actual_points = []
        self._timer = rospy.Timer(rospy.Duration(1.0), self._publish)

    # ---- episode lifecycle ---------------------------------------------------
    def start_episode(self, ref_params, ref_gen, episode_len_s):
        """Precompute the reference path for the whole episode."""
        if not self.enabled:
            return
        points = []
        t = 0.0
        while t <= episode_len_s:
            pos, _, _, _ = ref_gen.get_ref(ref_params, t)
            points.append(np.asarray(pos, dtype=np.float64))
            t += self.sample_dt
        self._ref_points = points
        self._actual_points = []
        self._publish(None)

    def add_actual_point(self, pos):
        if not self.enabled:
            return
        self._actual_points.append(np.asarray(pos, dtype=np.float64).copy())

    def clear(self):
        """Remove both overlays (episode end / reset)."""
        if not self.enabled:
            return
        self._ref_points = []
        self._actual_points = []
        ma = MarkerArray()
        for marker_id in (self.REF_ID, self.ACTUAL_ID):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = self.NS
            m.id = marker_id
            m.action = Marker.DELETE
            ma.markers.append(m)
        self.pub.publish(ma)

    # ---- publishing ------------------------------------------------------------
    def _publish(self, _event):
        ma = MarkerArray()
        if self._ref_points:
            ma.markers.append(self._line(self.REF_ID, self._ref_points,
                                         width=self.REF_WIDTH,
                                         color=(0.0, 0.9, 0.1, 0.9)))
        if self._actual_points:
            ma.markers.append(self._line(self.ACTUAL_ID, self._actual_points,
                                         width=self.ACTUAL_WIDTH,
                                         color=(1.0, 0.55, 0.0, 0.95)))
        if ma.markers:
            self.pub.publish(ma)

    def _line(self, marker_id, points, width, color):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = TrajVisualizer.NS
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = width
        # geometry_msgs/Quaternion defaults to (0,0,0,0), a ZERO-norm
        # quaternion: RViz's tf conversion divides by the norm and produces a
        # NaN transform, so the marker silently never renders. Set identity.
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = color
        for p in points:
            pt = Point()
            pt.x, pt.y, pt.z = p[0], p[1], p[2]
            m.points.append(pt)
        return m
