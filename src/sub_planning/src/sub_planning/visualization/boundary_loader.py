#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import HomePosition
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import pymap3d as pm
import numpy as np
from scipy.spatial import ConvexHull

class BoundaryLoader:
    def __init__(self):
        rospy.init_node('boundary_loader')

        # 1. 读取私有参数 ~gps_points
        self.gps_points = rospy.get_param('~gps_points', [])
        if not self.gps_points:
            rospy.logerr("Parameter '~gps_points' is empty or not set. Shutting down.")
            return

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

        # 4. 计算凸包（二维，忽略高度）
        ordered_enu = self.compute_convex_hull(enu_points)

        # 5. 将 ENU 坐标写入参数服务器
        rospy.set_param('/boundary_loader/enu_points', ordered_enu)
        rospy.loginfo("Ordered ENU boundary points saved to /boundary_loader/enu_points")

        # 6. 发布 Marker 用于 Rviz 可视化
        self.marker_pub = rospy.Publisher('/boundary_loader/visualization', Marker,
                                          queue_size=10, latch=True)
        self.publish_boundary_marker(ordered_enu)

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

    def publish_boundary_marker(self, enu_points):
        """
        发布 LINE_STRIP 类型的 Marker，点集已按凸包顺序排列。
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "boundary"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0   # 消除四元数未初始化警告

        # 线宽与颜色设置
        marker.scale.x = 0.5               # 线宽（米）
        marker.color.r = 1.0               # 红色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 添加点序列（自动闭合，第一个点重复）
        for pt in enu_points + [enu_points[0]]:
            p = Point()
            p.x, p.y, p.z = pt[0], pt[1], pt[2]
            marker.points.append(p)

        self.marker_pub.publish(marker)
        rospy.loginfo("Boundary marker published to /boundary_loader/visualization")

if __name__ == '__main__':
    try:
        BoundaryLoader()
    except rospy.ROSInterruptException:
        pass