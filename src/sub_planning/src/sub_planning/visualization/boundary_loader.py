#!/usr/bin/env python3
import math
import rospy
from mavros_msgs.msg import HomePosition
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import pymap3d as pm
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev

class BoundaryLoader:
    def __init__(self):
        rospy.init_node('boundary_loader')

        # 1. 读取私有参数 ~gps_points
        self.gps_points = rospy.get_param('~gps_points', [])
        if not self.gps_points:
            rospy.logerr("Parameter '~gps_points' is empty or not set. Shutting down.")
            return
        self.sim_boundary_margin = float(rospy.get_param('~sim_boundary_margin', 6.0))
        self.sim_boundary_smoothness = float(rospy.get_param('~sim_boundary_smoothness', 2.5))
        self.sim_boundary_resolution = int(rospy.get_param('~sim_boundary_resolution', 120))
        self.sim_boundary_variation = float(rospy.get_param('~sim_boundary_variation', 0.75))
        self.sim_boundary_min_clearance_ratio = float(rospy.get_param('~sim_boundary_min_clearance_ratio', 0.28))
        self.sim_boundary_max_clearance_ratio = float(rospy.get_param('~sim_boundary_max_clearance_ratio', 0.62))
        self.plan_marker_width = float(rospy.get_param('~plan_marker_width', 0.18))
        self.plan_marker_alpha = float(rospy.get_param('~plan_marker_alpha', 0.35))
        self.sim_marker_width = float(rospy.get_param('~sim_marker_width', 0.42))
        self.sim_marker_alpha = float(rospy.get_param('~sim_marker_alpha', 0.95))

        rospy.loginfo("Loaded %d GPS boundary points.", len(self.gps_points))

        # 2. 等待飞控 home 位置
        self.home_position = None
        rospy.Subscriber('/mavros/home_position/home', HomePosition, self.home_cb)
        rospy.loginfo("Waiting for home position from FCU...")
        rate = rospy.Rate(1)
        while self.home_position is None and not rospy.is_shutdown():
            rate.sleep()

        h = self.home_position.geo
        rospy.loginfo("Home position received: lat=%.6f, lon=%.6f, alt=%.2f",
                      h.latitude, h.longitude, h.altitude)

        # 3. 将 GPS 坐标转换为 ENU 坐标
        enu_points = []
        for pt in self.gps_points:
            lat, lon = pt[0], pt[1]
            alt = pt[2] if len(pt) > 2 else h.altitude  # 使用飞控海拔作为参考
            e, n, u = pm.geodetic2enu(lat, lon, alt,
                                      h.latitude, h.longitude, h.altitude)
            enu_points.append([float(e), float(n), float(u)])

        # 4. 计算规划边界凸包（二维，忽略高度）
        ordered_enu = self.compute_convex_hull(enu_points)
        sim_boundary = self.build_sim_boundary(ordered_enu)

        # 5. 将 ENU 坐标写入参数服务器
        rospy.set_param('/boundary_loader/enu_points', ordered_enu)
        rospy.set_param('/boundary_loader/plan_enu_points', ordered_enu)
        rospy.set_param('/boundary_loader/sim_enu_points', sim_boundary)
        rospy.loginfo("Ordered ENU boundary points saved to /boundary_loader/enu_points")

        # 6. 发布 Marker 用于 Rviz 可视化
        self.marker_pub = rospy.Publisher('/boundary_loader/visualization', Marker,
                                          queue_size=10, latch=True)
        self.plan_marker = None
        self.sim_marker = None
        self.publish_boundary_markers(ordered_enu, sim_boundary)
        self.marker_timer = rospy.Timer(rospy.Duration(1.0), self.republish_boundary_markers)

        rospy.loginfo("Boundary loader finished initialization. Keeping node alive.")
        rospy.spin()

    def home_cb(self, msg):
        if self.home_position is None:
            self.home_position = msg

    def compute_convex_hull(self, points):
        """
        计算二维点的凸包，返回按逆时针顺序排列的顶点列表。
        如果点数少于3，则直接返回原列表。
        """
        if len(points) < 3:
            rospy.logwarn("Less than 3 points, cannot compute convex hull. Returning original order.")
            return points

        # 提取 x, y 坐标（忽略 z）
        xy = np.array([[p[0], p[1]] for p in points])

        try:
            hull = ConvexHull(xy)
            # hull.vertices 是按逆时针顺序的原始点索引
            ordered_indices = hull.vertices.tolist()
            # 提取对应的三维点
            ordered_points = [points[i] for i in ordered_indices]
            rospy.loginfo("Convex hull computed with %d vertices.", len(ordered_points))
            return ordered_points
        except Exception as e:
            rospy.logerr("Convex hull computation failed: %s. Returning original order.", e)
            return points

    @staticmethod
    def polygon_area(points_xy):
        area = 0.0
        num = len(points_xy)
        for idx in range(num):
            x0, y0 = points_xy[idx]
            x1, y1 = points_xy[(idx + 1) % num]
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    def outward_offset_polygon(self, points_xy, margin):
        pts = np.array(points_xy, dtype=float)
        if len(pts) < 3:
            return pts

        area = self.polygon_area(pts)
        ccw = area > 0.0
        expanded = []
        count = len(pts)
        for idx in range(count):
            prev_pt = pts[(idx - 1) % count]
            curr_pt = pts[idx]
            next_pt = pts[(idx + 1) % count]

            edge_prev = curr_pt - prev_pt
            edge_next = next_pt - curr_pt
            norm_prev = np.linalg.norm(edge_prev)
            norm_next = np.linalg.norm(edge_next)
            if norm_prev < 1.0e-6 or norm_next < 1.0e-6:
                expanded.append(curr_pt.copy())
                continue
            edge_prev /= norm_prev
            edge_next /= norm_next
            if ccw:
                n_prev = np.array([edge_prev[1], -edge_prev[0]])
                n_next = np.array([edge_next[1], -edge_next[0]])
            else:
                n_prev = np.array([-edge_prev[1], edge_prev[0]])
                n_next = np.array([-edge_next[1], edge_next[0]])

            direction = n_prev + n_next
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1.0e-6:
                direction = n_prev
                direction_norm = np.linalg.norm(direction)
            direction /= max(direction_norm, 1.0e-6)
            denom = max(np.dot(direction, n_prev), 0.2)
            expanded.append(curr_pt + direction * (margin / denom))
        return np.array(expanded, dtype=float)

    def smooth_closed_polygon(self, points_xy):
        pts = np.array(points_xy, dtype=float)
        if len(pts) < 4:
            return pts
        try:
            closed = np.vstack((pts, pts[0]))
            tck, _ = splprep([closed[:, 0], closed[:, 1]], s=max(self.sim_boundary_smoothness, 0.0), per=True)
            u_new = np.linspace(0.0, 1.0, max(self.sim_boundary_resolution, len(pts) * 4), endpoint=False)
            smoothed = np.asarray(splev(u_new, tck)).T
            return smoothed
        except Exception as exc:
            rospy.logwarn("Closed spline boundary smoothing failed: %s. Using offset polygon.", exc)
            return pts

    @staticmethod
    def ray_segment_intersection(origin, direction, p0, p1):
        ox, oy = origin
        dx, dy = direction
        sx = p1[0] - p0[0]
        sy = p1[1] - p0[1]
        denom = dx * sy - dy * sx
        if abs(denom) < 1.0e-9:
            return None
        qx = p0[0] - ox
        qy = p0[1] - oy
        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def polygon_support_radius(self, center, direction, plan_xy):
        best = None
        count = len(plan_xy)
        for idx in range(count):
            hit = self.ray_segment_intersection(center, direction, plan_xy[idx], plan_xy[(idx + 1) % count])
            if hit is None:
                continue
            if best is None or hit > best:
                best = hit
        return best

    def shoreline_profile(self, angles, plan_xy):
        seed = float(np.sum(plan_xy[:, 0] * 0.173 + plan_xy[:, 1] * 0.297))
        phase1 = math.fmod(abs(seed) * 0.37, 2.0 * math.pi)
        phase2 = math.fmod(abs(seed) * 0.63 + 0.9, 2.0 * math.pi)
        phase3 = math.fmod(abs(seed) * 0.91 + 1.7, 2.0 * math.pi)
        phase4 = math.fmod(abs(seed) * 1.19 + 2.6, 2.0 * math.pi)

        profile = (
            0.48 * np.sin(2.0 * angles + phase1)
            + 0.24 * np.sin(3.0 * angles + phase2)
            + 0.18 * np.cos(5.0 * angles + phase3)
            + 0.12 * np.sin(7.0 * angles + phase4)
        )
        for _ in range(1):
            profile = 0.25 * np.roll(profile, -1) + 0.50 * profile + 0.25 * np.roll(profile, 1)

        pmin = float(np.min(profile))
        pmax = float(np.max(profile))
        if pmax - pmin < 1.0e-6:
            return np.ones_like(profile) * 0.5
        return (profile - pmin) / (pmax - pmin)

    def build_sim_boundary(self, ordered_enu):
        plan_xy = np.array([[pt[0], pt[1]] for pt in ordered_enu], dtype=float)
        if len(plan_xy) < 3:
            return ordered_enu

        center = np.mean(plan_xy, axis=0)
        sample_count = max(self.sim_boundary_resolution, len(plan_xy) * 18)
        angles = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
        profile = self.shoreline_profile(angles, plan_xy)

        margin = max(self.sim_boundary_margin, 0.6)
        min_clearance = max(0.35, margin * self.sim_boundary_min_clearance_ratio)
        variation = max(self.sim_boundary_variation, 0.0)
        variation_amplitude = margin * 0.18 * math.log1p(variation)
        max_clearance = max(
            min_clearance + 0.25,
            margin * self.sim_boundary_max_clearance_ratio + variation_amplitude,
        )

        shoreline = []
        fallback_radius = float(np.max(np.linalg.norm(plan_xy - center, axis=1)))
        for angle, profile_value in zip(angles, profile):
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
            support = self.polygon_support_radius(center, direction, plan_xy)
            if support is None:
                support = fallback_radius
            signed_profile = 2.0 * profile_value - 1.0
            clearance = margin + signed_profile * variation_amplitude
            clearance = float(np.clip(clearance, min_clearance, max_clearance))
            shoreline.append(center + direction * (support + clearance))

        sim_xy = self.smooth_closed_polygon(np.array(shoreline, dtype=float))
        sim_z = float(np.mean([pt[2] for pt in ordered_enu])) if ordered_enu else 0.0
        sim_boundary = [[float(pt[0]), float(pt[1]), sim_z] for pt in sim_xy.tolist()]
        rospy.loginfo("Generated simulation boundary with %d vertices.", len(sim_boundary))
        return sim_boundary

    def create_boundary_marker(self, namespace, marker_id, width, rgba, enu_points):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r = rgba[0]
        marker.color.g = rgba[1]
        marker.color.b = rgba[2]
        marker.color.a = rgba[3]

        for pt in enu_points + [enu_points[0]]:
            p = Point()
            p.x, p.y, p.z = pt[0], pt[1], pt[2]
            marker.points.append(p)
        return marker

    def publish_boundary_markers(self, plan_enu_points, sim_enu_points):
        self.plan_marker = self.create_boundary_marker(
            'boundary_plan',
            0,
            self.plan_marker_width,
            (1.0, 0.2, 0.2, self.plan_marker_alpha),
            plan_enu_points,
        )
        self.sim_marker = self.create_boundary_marker(
            'boundary_sim',
            1,
            self.sim_marker_width,
            (0.45, 0.75, 1.0, self.sim_marker_alpha),
            sim_enu_points,
        )
        self.marker_pub.publish(self.plan_marker)
        self.marker_pub.publish(self.sim_marker)

        rospy.loginfo("Plan and simulation boundary markers published to /boundary_loader/visualization")

    def republish_boundary_markers(self, _event):
        if self.plan_marker is None or self.sim_marker is None:
            return
        stamp = rospy.Time.now()
        self.plan_marker.header.stamp = stamp
        self.sim_marker.header.stamp = stamp
        self.marker_pub.publish(self.plan_marker)
        self.marker_pub.publish(self.sim_marker)

if __name__ == '__main__':
    try:
        BoundaryLoader()
    except rospy.ROSInterruptException:
        pass
