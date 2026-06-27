#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import struct

import numpy as np
import rospy
import rospkg
import random
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray


class ForwardSonarSim:
    def __init__(self):
        rospy.init_node('forward_sonar_sim', anonymous=True)

        self.robot_odom_topic = rospy.get_param('~robot_odom_topic', '/mavros/local_position/odom')
        self.target_pose_topic = rospy.get_param('~target_pose_topic', '/dynamic_target/pose')
        self.obstacle_marker_topic = rospy.get_param('~obstacle_marker_topic', '/dynamic_obstacles/markers')
        self.image_topic = rospy.get_param('~image_topic', '/forward_sonar/image')
        self.boundary_param = rospy.get_param('~boundary_param', '/boundary_loader/enu_points')

        self.frame_id = rospy.get_param('~frame_id', 'map')
        self.publish_rate = float(rospy.get_param('~publish_rate', 8.0))
        self.image_width = int(rospy.get_param('~image_width', 512))
        self.image_height = int(rospy.get_param('~image_height', 256))
        self.fov_deg = float(rospy.get_param('~fov_deg', 130.0))
        self.max_range = float(rospy.get_param('~max_range', 35.0))
        self.min_range = float(rospy.get_param('~min_range', 0.2))

        self.noise_mean = float(rospy.get_param('~noise_mean', 10.0))
        self.noise_std = float(rospy.get_param('~noise_std', 5.0))
        self.output_auto_gain = rospy.get_param('~output_auto_gain', True)
        self.output_gain_percentile = float(rospy.get_param('~output_gain_percentile', 99.4))
        self.output_min_high = float(rospy.get_param('~output_min_high', 20.0))
        self.publish_waiting_image = rospy.get_param('~publish_waiting_image', True)
        self.show_range_rings = rospy.get_param('~show_range_rings', False)
        self.range_ring_spacing = float(rospy.get_param('~range_ring_spacing', 5.0))
        self.range_ring_intensity = float(rospy.get_param('~range_ring_intensity', 3.0))
        self.target_strength = float(rospy.get_param('~target_strength', 210.0))
        self.boundary_strength = float(rospy.get_param('~boundary_strength', 120.0))
        self.target_radius = float(rospy.get_param('~target_radius', 0.7))
        self.target_model = rospy.get_param('~target_model', 'mesh')
        self.target_mesh_resource = rospy.get_param(
            '~target_mesh_resource',
            'package://sub_planning/meshes/AUV.STL',
        )
        self.target_mesh_scale = float(rospy.get_param('~target_mesh_scale', 1.0))
        self.target_mesh_sample_count = int(rospy.get_param('~target_mesh_sample_count', 6000))
        self.target_mesh_point_radius = float(rospy.get_param('~target_mesh_point_radius', 0.045))
        self.target_mesh_seed = int(rospy.get_param('~target_mesh_seed', 7))
        self.target_mesh_normal_power = float(rospy.get_param('~target_mesh_normal_power', 1.7))
        self.target_mesh_grazing_gain = float(rospy.get_param('~target_mesh_grazing_gain', 0.10))
        self.target_mesh_occlusion_cell = int(rospy.get_param('~target_mesh_occlusion_cell', 2))
        self.target_mesh_shadow = rospy.get_param('~target_mesh_shadow', True)
        self.target_mesh_shadow_start = float(rospy.get_param('~target_mesh_shadow_start', 0.45))
        self.target_mesh_shadow_length = float(rospy.get_param('~target_mesh_shadow_length', 2.0))
        self.enable_obstacle_echoes = rospy.get_param('~enable_obstacle_echoes', True)
        self.obstacle_strength = float(rospy.get_param('~obstacle_strength', 120.0))
        self.obstacle_sample_count = int(rospy.get_param('~obstacle_sample_count', 180))
        self.obstacle_point_radius = float(rospy.get_param('~obstacle_point_radius', 0.08))
        self.use_boundary = rospy.get_param('~use_boundary', True)
        self.boundary_ray_step_deg = float(rospy.get_param('~boundary_ray_step_deg', 1.0))

        self.have_robot = False
        self.have_target = False
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_z = 0.0
        self.robot_yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0
        self.target_mesh_points, self.target_mesh_normals, self.target_mesh_weights = self.load_target_mesh_samples()
        self.obstacle_markers = []
        self.boundary_points = []
        self.last_boundary_reload = rospy.Time(0)

        self.odom_sub = rospy.Subscriber(self.robot_odom_topic, Odometry, self.odom_cb, queue_size=1)
        self.target_sub = rospy.Subscriber(self.target_pose_topic, PoseStamped, self.target_cb, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(self.obstacle_marker_topic, MarkerArray, self.obstacle_cb, queue_size=1)
        self.image_pub = rospy.Publisher(self.image_topic, Image, queue_size=2)

        period = 1.0 / max(self.publish_rate, 1e-3)
        self.timer = rospy.Timer(rospy.Duration(period), self.timer_cb)
        rospy.loginfo(
            '[ForwardSonar] publishing image=%s fov=%.1f deg range=%.1f m',
            self.image_topic,
            self.fov_deg,
            self.max_range,
        )

    def odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id
        self.have_robot = True

    def target_cb(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.target_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.have_target = True

    def obstacle_cb(self, msg):
        self.obstacle_markers = [marker for marker in msg.markers if marker.action != Marker.DELETE]

    def resolve_package_uri(self, uri):
        if uri.startswith('package://'):
            rel = uri[len('package://'):]
            package_name, package_path = rel.split('/', 1)
            try:
                package_root = rospkg.RosPack().get_path(package_name)
            except rospkg.ResourceNotFound:
                package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            return os.path.join(package_root, package_path)
        return uri

    def load_target_mesh_samples(self):
        if self.target_model != 'mesh':
            return self.empty_mesh_samples()
        try:
            mesh_path = self.resolve_package_uri(self.target_mesh_resource)
            points, normals, weights = self.load_binary_stl_samples(mesh_path)
            if points.size == 0:
                rospy.logwarn('[ForwardSonar] mesh has no points: %s', mesh_path)
                return self.empty_mesh_samples()
            points *= self.target_mesh_scale
            rospy.loginfo('[ForwardSonar] loaded %d mesh surface samples from %s', len(points), mesh_path)
            return points, normals, weights
        except Exception as exc:
            rospy.logwarn('[ForwardSonar] failed to load target mesh %s: %s. Falling back to blob target.',
                          self.target_mesh_resource, exc)
            self.target_model = 'blob'
            return self.empty_mesh_samples()

    @staticmethod
    def empty_mesh_samples():
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
        )

    def load_binary_stl_samples(self, mesh_path):
        with open(mesh_path, 'rb') as stream:
            stream.read(80)
            triangle_count = struct.unpack('<I', stream.read(4))[0]
            raw = stream.read(triangle_count * 50)
        if len(raw) < triangle_count * 50:
            return self.empty_mesh_samples()

        vertices = np.empty((triangle_count, 3, 3), dtype=float)
        for tri_idx in range(triangle_count):
            offset = tri_idx * 50 + 12
            tri = struct.unpack('<9f', raw[offset:offset + 36])
            vertices[tri_idx] = (
                (tri[0], tri[1], tri[2]),
                (tri[3], tri[4], tri[5]),
                (tri[6], tri[7], tri[8]),
            )

        edge_a = vertices[:, 1, :] - vertices[:, 0, :]
        edge_b = vertices[:, 2, :] - vertices[:, 0, :]
        cross = np.cross(edge_a, edge_b)
        double_area = np.linalg.norm(cross, axis=1)
        valid = double_area > 1e-12
        if not np.any(valid):
            return self.empty_mesh_samples()

        vertices = vertices[valid]
        cross = cross[valid]
        area = 0.5 * double_area[valid]
        normals = cross / np.maximum(np.linalg.norm(cross, axis=1)[:, None], 1e-12)

        sample_count = max(1, min(self.target_mesh_sample_count, len(area) * 4))
        rng = np.random.default_rng(self.target_mesh_seed)
        probability = area / np.sum(area)
        tri_indices = rng.choice(len(area), size=sample_count, replace=True, p=probability)
        r1 = np.sqrt(rng.random(sample_count))
        r2 = rng.random(sample_count)
        a = 1.0 - r1
        b = r1 * (1.0 - r2)
        c = r1 * r2
        selected = vertices[tri_indices]
        points = (
            selected[:, 0, :] * a[:, None]
            + selected[:, 1, :] * b[:, None]
            + selected[:, 2, :] * c[:, None]
        )
        sample_normals = normals[tri_indices]
        weights = area[tri_indices] / np.maximum(probability[tri_indices], 1e-12)
        weights /= max(np.mean(weights), 1e-12)
        return points, sample_normals, weights

    @staticmethod
    def wrap_pi(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def world_to_body(self, wx, wy):
        dx = wx - self.robot_x
        dy = wy - self.robot_y
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def body_to_world(self, bx, by):
        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)
        return self.robot_x + c * bx - s * by, self.robot_y + s * bx + c * by

    def reload_boundary_if_needed(self):
        if not self.use_boundary:
            return
        now = rospy.Time.now()
        if (now - self.last_boundary_reload).to_sec() < 2.0:
            return
        self.last_boundary_reload = now
        if not rospy.has_param(self.boundary_param):
            return
        raw_points = rospy.get_param(self.boundary_param, [])
        points = []
        for item in raw_points:
            if len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
        if len(points) >= 3:
            self.boundary_points = points

    def make_base_image(self):
        image = np.random.normal(self.noise_mean, self.noise_std, (self.image_height, self.image_width))
        image = np.clip(image, 0.0, 255.0)

        forward_ranges = self.forward_ranges_by_row()
        attenuation = 1.0 / (1.0 + 0.045 * forward_ranges * forward_ranges)
        image *= attenuation[:, None]
        image *= self.sonar_footprint_mask()

        if self.show_range_rings:
            self.add_range_rings(image)
        return image

    def add_range_rings(self, image):
        spacing = max(self.range_ring_spacing, 1e-3)
        max_lateral = self.max_lateral_span()
        columns = np.arange(self.image_width)
        lateral = max_lateral - columns * (2.0 * max_lateral / max(self.image_width - 1, 1))
        for rng in np.arange(spacing, self.max_range + 0.01, spacing):
            valid = np.abs(lateral) <= rng * math.sin(math.radians(self.fov_deg) * 0.5)
            if not np.any(valid):
                continue
            forward_sq = rng * rng - lateral[valid] * lateral[valid]
            forward = np.sqrt(np.maximum(forward_sq, 0.0))
            rows = np.array([self.forward_to_row(value) for value in forward], dtype=np.int32)
            valid_cols = columns[valid]
            for offset, gain in ((-1, 0.35), (0, 1.0), (1, 0.35)):
                rr = rows + offset
                inside = (rr >= 0) & (rr < self.image_height)
                image[rr[inside], valid_cols[inside]] += self.range_ring_intensity * gain

    def sonar_footprint_mask(self):
        forward = self.forward_ranges_by_row()[:, None]
        max_lateral = self.max_lateral_span()
        lateral = np.linspace(max_lateral, -max_lateral, self.image_width)[None, :]
        half_fov = math.radians(self.fov_deg) * 0.5
        mask = np.abs(lateral) <= forward * math.tan(half_fov)
        return np.where(mask, 1.0, 0.08)

    def forward_ranges_by_row(self):
        return np.linspace(self.max_range, self.min_range, self.image_height)

    def max_lateral_span(self):
        return max(self.max_range * math.sin(math.radians(self.fov_deg) * 0.5), 1e-6)

    def forward_to_row(self, forward):
        t = (forward - self.min_range) / max(self.max_range - self.min_range, 1e-6)
        return int((1.0 - np.clip(t, 0.0, 1.0)) * (self.image_height - 1))

    def lateral_to_col(self, lateral):
        max_lateral = self.max_lateral_span()
        # Body +Y is vehicle-left. Image columns grow left-to-right, so negative
        # lateral/right-side returns must move toward larger column indices.
        t = (max_lateral - lateral) / (2.0 * max_lateral)
        return int(np.clip(t, 0.0, 1.0) * (self.image_width - 1))

    def in_sonar_footprint(self, forward, lateral):
        if forward < self.min_range:
            return False
        horizontal_range = math.hypot(forward, lateral)
        if horizontal_range > self.max_range:
            return False
        half_fov = math.radians(self.fov_deg) * 0.5
        bearing = math.atan2(lateral, max(forward, 1e-6))
        return abs(bearing) <= half_fov

    def add_echo_blob(self, image, forward, lateral, strength, radius_m, draw_shadow=True):
        if not self.in_sonar_footprint(forward, lateral):
            return

        row = self.forward_to_row(forward)
        col = self.lateral_to_col(lateral)
        row_sigma = max(1.4, radius_m / max(self.max_range - self.min_range, 1e-6) * self.image_height)
        col_sigma = max(2.0, radius_m / (2.0 * self.max_lateral_span()) * self.image_width)
        rr = int(max(3, row_sigma * 4.0))
        cc = int(max(4, col_sigma * 4.0))

        r0 = max(0, row - rr)
        r1 = min(self.image_height, row + rr + 1)
        c0 = max(0, col - cc)
        c1 = min(self.image_width, col + cc + 1)
        ys = np.arange(r0, r1)[:, None]
        xs = np.arange(c0, c1)[None, :]
        blob = strength * np.exp(-0.5 * (((ys - row) / row_sigma) ** 2 + ((xs - col) / col_sigma) ** 2))
        image[r0:r1, c0:c1] += blob

        if draw_shadow:
            self.add_radial_shadow(
                image,
                forward,
                lateral,
                radius_m * 4.0,
                radius_m * 9.0,
                max(0.58, 1.0 - 0.0018 * strength),
                max(1.0, col_sigma * 0.70),
            )

    def add_target_echo(self, image):
        if not self.have_target:
            return None
        if self.target_model == 'mesh' and len(self.target_mesh_points) > 0:
            return self.add_target_mesh_echo(image)
        bx, by = self.world_to_body(self.target_x, self.target_y)
        if bx <= 0.0:
            return None
        horizontal_range = math.hypot(bx, by)
        vertical_offset = self.target_z - self.robot_z
        slant_range = math.hypot(horizontal_range, vertical_offset)
        bearing = math.atan2(by, bx)
        if slant_range > self.max_range or slant_range < self.min_range:
            return None
        if abs(bearing) > math.radians(self.fov_deg) * 0.5:
            return None
        self.add_echo_blob(image, bx, by, self.target_strength, self.target_radius)
        return slant_range, bearing

    def target_mesh_points_in_body(self):
        cy = math.cos(self.target_yaw)
        sy = math.sin(self.target_yaw)
        local = self.target_mesh_points
        world_x = self.target_x + cy * local[:, 0] - sy * local[:, 1]
        world_y = self.target_y + sy * local[:, 0] + cy * local[:, 1]
        world_z = self.target_z + local[:, 2]

        dx = world_x - self.robot_x
        dy = world_y - self.robot_y
        cr = math.cos(self.robot_yaw)
        sr = math.sin(self.robot_yaw)
        forward = cr * dx + sr * dy
        lateral = -sr * dx + cr * dy
        vertical = world_z - self.robot_z
        local_normals = self.target_mesh_normals
        normal_world_x = cy * local_normals[:, 0] - sy * local_normals[:, 1]
        normal_world_y = sy * local_normals[:, 0] + cy * local_normals[:, 1]
        normal_world_z = local_normals[:, 2]
        normal_forward = cr * normal_world_x + sr * normal_world_y
        normal_lateral = -sr * normal_world_x + cr * normal_world_y
        normal_vertical = normal_world_z
        return forward, lateral, vertical, normal_forward, normal_lateral, normal_vertical

    def add_target_mesh_echo(self, image):
        forward, lateral, vertical, nf, nl, nv = self.target_mesh_points_in_body()
        horizontal_range = np.hypot(forward, lateral)
        slant_range = np.hypot(horizontal_range, vertical)
        bearing = np.arctan2(lateral, np.maximum(forward, 1e-6))
        half_fov = math.radians(self.fov_deg) * 0.5
        valid = (
            (forward > self.min_range)
            & (slant_range <= self.max_range)
            & (np.abs(bearing) <= half_fov)
        )
        if not np.any(valid):
            return None

        vf = forward[valid]
        vl = lateral[valid]
        vv = vertical[valid]
        vr = slant_range[valid]
        vnf = nf[valid]
        vnl = nl[valid]
        vnv = nv[valid]
        vweights = self.target_mesh_weights[valid]
        vbearing = bearing[valid]

        rows = np.array([self.forward_to_row(value) for value in vf], dtype=np.int32)
        cols = np.array([self.lateral_to_col(value) for value in vl], dtype=np.int32)
        coarse = max(1, self.target_mesh_occlusion_cell)
        keys = (rows // coarse) * (self.image_width // coarse + 1) + (cols // coarse)
        order = np.lexsort((vr, keys))

        visible = []
        last_key = None
        for idx in order:
            key = keys[idx]
            if key == last_key:
                continue
            visible.append(idx)
            last_key = key
        if not visible:
            return None
        visible = np.asarray(visible, dtype=np.int32)

        look_x = -vf[visible] / np.maximum(vr[visible], 1e-6)
        look_y = -vl[visible] / np.maximum(vr[visible], 1e-6)
        look_z = -vv[visible] / np.maximum(vr[visible], 1e-6)
        incidence = np.abs(vnf[visible] * look_x + vnl[visible] * look_y + vnv[visible] * look_z)
        incidence_gain = self.target_mesh_grazing_gain + (1.0 - self.target_mesh_grazing_gain) * (
            incidence ** max(self.target_mesh_normal_power, 0.1)
        )
        distance_gain = 1.0 / (1.0 + 0.018 * vr[visible] * vr[visible])
        speckle = np.random.rayleigh(scale=0.85, size=len(visible))
        strength = self.target_strength * incidence_gain * distance_gain * vweights[visible] * speckle
        strength = np.clip(strength, 0.0, self.target_strength * 1.35)

        for idx, point_strength in zip(visible, strength):
            self.add_echo_blob(image, vf[idx], vl[idx], point_strength, self.target_mesh_point_radius, draw_shadow=False)
            if self.target_mesh_shadow and point_strength > self.target_strength * 0.12:
                self.add_mesh_shadow(image, vf[idx], vl[idx], point_strength)

        nearest = visible[np.argmin(vr[visible])]
        return vr[nearest], vbearing[nearest]

    def add_obstacle_echoes(self, image):
        if not self.enable_obstacle_echoes or not self.obstacle_markers:
            return
        for marker in self.obstacle_markers:
            if marker.type not in (Marker.SPHERE, Marker.CUBE, Marker.CYLINDER):
                continue
            samples = self.sample_obstacle_marker(marker)
            if samples.size == 0:
                continue
            forward, lateral, vertical = self.points_world_to_body(samples)
            horizontal_range = np.hypot(forward, lateral)
            slant_range = np.hypot(horizontal_range, vertical)
            bearing = np.arctan2(lateral, np.maximum(forward, 1e-6))
            half_fov = math.radians(self.fov_deg) * 0.5
            valid = (
                (forward > self.min_range)
                & (slant_range <= self.max_range)
                & (np.abs(bearing) <= half_fov)
            )
            if not np.any(valid):
                continue
            vf = forward[valid]
            vl = lateral[valid]
            vr = slant_range[valid]
            order = np.argsort(vr)
            for idx in order[:max(1, min(len(order), self.obstacle_sample_count))]:
                distance_gain = 1.0 / (1.0 + 0.022 * vr[idx] * vr[idx])
                strength = self.obstacle_strength * (0.5 + 0.5 * random.random()) * distance_gain
                self.add_echo_blob(image, vf[idx], vl[idx], strength, self.obstacle_point_radius)

    def sample_obstacle_marker(self, marker):
        rng = np.random.default_rng(marker.id + 1009)
        count = max(16, self.obstacle_sample_count)
        if marker.type == Marker.SPHERE:
            local = self.sample_sphere_surface(count, marker.scale.x * 0.5, rng)
        elif marker.type == Marker.CYLINDER:
            local = self.sample_cylinder_surface(count, marker.scale.x * 0.5, marker.scale.z, rng)
        else:
            local = self.sample_box_surface(count, marker.scale.x, marker.scale.y, marker.scale.z, rng)
        return self.local_points_to_world(local, marker)

    @staticmethod
    def sample_sphere_surface(count, radius, rng):
        radius = max(float(radius), 1e-3)
        z = rng.uniform(-1.0, 1.0, count)
        theta = rng.uniform(-math.pi, math.pi, count)
        r_xy = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        return np.column_stack((radius * r_xy * np.cos(theta), radius * r_xy * np.sin(theta), radius * z))

    @staticmethod
    def sample_box_surface(count, sx, sy, sz, rng):
        half = np.array([max(float(sx), 1e-3), max(float(sy), 1e-3), max(float(sz), 1e-3)], dtype=float) * 0.5
        points = np.empty((count, 3), dtype=float)
        faces = rng.integers(0, 6, count)
        for idx, face in enumerate(faces):
            axis = face // 2
            sign = -1.0 if face % 2 == 0 else 1.0
            point = rng.uniform(-half, half)
            point[axis] = sign * half[axis]
            points[idx] = point
        return points

    @staticmethod
    def sample_cylinder_surface(count, radius, height, rng):
        radius = max(float(radius), 1e-3)
        half_h = max(float(height), 1e-3) * 0.5
        points = np.empty((count, 3), dtype=float)
        part = rng.random(count)
        theta = rng.uniform(-math.pi, math.pi, count)
        side = part < 0.7
        points[side, 0] = radius * np.cos(theta[side])
        points[side, 1] = radius * np.sin(theta[side])
        points[side, 2] = rng.uniform(-half_h, half_h, np.count_nonzero(side))
        cap = ~side
        cap_count = np.count_nonzero(cap)
        if cap_count > 0:
            cap_r = radius * np.sqrt(rng.random(cap_count))
            cap_theta = theta[cap]
            points[cap, 0] = cap_r * np.cos(cap_theta)
            points[cap, 1] = cap_r * np.sin(cap_theta)
            points[cap, 2] = np.where(rng.random(cap_count) < 0.5, -half_h, half_h)
        return points

    @staticmethod
    def marker_yaw(marker):
        q = marker.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def local_points_to_world(self, local, marker):
        yaw = self.marker_yaw(marker)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        world = np.empty_like(local)
        world[:, 0] = marker.pose.position.x + cy * local[:, 0] - sy * local[:, 1]
        world[:, 1] = marker.pose.position.y + sy * local[:, 0] + cy * local[:, 1]
        world[:, 2] = marker.pose.position.z + local[:, 2]
        return world

    def points_world_to_body(self, world):
        dx = world[:, 0] - self.robot_x
        dy = world[:, 1] - self.robot_y
        cr = math.cos(self.robot_yaw)
        sr = math.sin(self.robot_yaw)
        forward = cr * dx + sr * dy
        lateral = -sr * dx + cr * dy
        vertical = world[:, 2] - self.robot_z
        return forward, lateral, vertical

    def add_mesh_shadow(self, image, forward, lateral, strength):
        self.add_radial_shadow(
            image,
            forward,
            lateral,
            self.target_mesh_shadow_start,
            self.target_mesh_shadow_length,
            max(0.52, 1.0 - 0.0018 * strength),
            1.8,
        )

    def add_radial_shadow(self, image, forward, lateral, start_offset, length, attenuation, width_px):
        range_xy = math.hypot(forward, lateral)
        if range_xy <= self.min_range:
            return
        bearing = math.atan2(lateral, max(forward, 1e-6))
        far_range = min(self.max_range, range_xy + max(length, 0.0))
        near_range = min(far_range, range_xy + max(start_offset, 0.0))
        if far_range <= near_range:
            return

        samples = max(4, int((far_range - near_range) / max(self.max_range - self.min_range, 1e-6) * self.image_height * 1.5))
        ranges = np.linspace(near_range, far_range, samples)
        cos_b = math.cos(bearing)
        sin_b = math.sin(bearing)
        rows = np.array([self.forward_to_row(rng * cos_b) for rng in ranges], dtype=np.int32)
        cols = np.array([self.lateral_to_col(rng * sin_b) for rng in ranges], dtype=np.int32)
        width = max(1, int(round(width_px)))
        for row, col in zip(rows, cols):
            if row < 0 or row >= self.image_height or col < 0 or col >= self.image_width:
                continue
            r0 = max(0, row - width)
            r1 = min(self.image_height, row + width + 1)
            c0 = max(0, col - width)
            c1 = min(self.image_width, col + width + 1)
            image[r0:r1, c0:c1] *= attenuation

    @staticmethod
    def ray_segment_intersection(origin, direction, p0, p1):
        ox, oy = origin
        dx, dy = direction
        sx = p1[0] - p0[0]
        sy = p1[1] - p0[1]
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            return None
        qx = p0[0] - ox
        qy = p0[1] - oy
        t = (qx * sy - qy * sx) / denom
        u = (qx * dy - qy * dx) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def add_boundary_echoes(self, image):
        if len(self.boundary_points) < 3:
            return
        half_fov = math.radians(self.fov_deg) * 0.5
        step = max(math.radians(self.boundary_ray_step_deg), math.radians(0.2))
        origin = (self.robot_x, self.robot_y)
        closed = self.boundary_points + [self.boundary_points[0]]
        bearing = -half_fov
        while bearing <= half_fov + 1e-6:
            yaw = self.robot_yaw + bearing
            direction = (math.cos(yaw), math.sin(yaw))
            best = None
            for i in range(len(closed) - 1):
                hit = self.ray_segment_intersection(origin, direction, closed[i], closed[i + 1])
                if hit is None:
                    continue
                if self.min_range <= hit <= self.max_range and (best is None or hit < best):
                    best = hit
            if best is not None:
                strength = self.boundary_strength * (0.45 + 0.55 * random.random())
                self.add_echo_blob(image, best * math.cos(bearing), best * math.sin(bearing), strength, 0.25)
            bearing += step

    def publish_image(self, image, stamp):
        image_u8 = self.to_output_image(image)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height = self.image_height
        msg.width = self.image_width
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = self.image_width
        msg.data = image_u8.tobytes()
        self.image_pub.publish(msg)

    def to_output_image(self, image):
        output = np.clip(image, 0.0, None)
        if self.output_auto_gain:
            high = float(np.percentile(output, np.clip(self.output_gain_percentile, 50.0, 100.0)))
            high = max(high, self.output_min_high)
            output = output * (255.0 / high)
        return np.ascontiguousarray(np.clip(output, 0, 255).astype(np.uint8))

    def timer_cb(self, _event):
        if not self.have_robot:
            rospy.logwarn_throttle(3.0, '[ForwardSonar] waiting for robot odometry: %s', self.robot_odom_topic)
            if self.publish_waiting_image:
                self.publish_image(self.make_base_image(), rospy.Time.now())
            return

        stamp = rospy.Time.now()
        self.reload_boundary_if_needed()
        image = self.make_base_image()
        self.add_boundary_echoes(image)
        self.add_obstacle_echoes(image)
        self.add_target_echo(image)
        self.publish_image(image, stamp)


if __name__ == '__main__':
    try:
        ForwardSonarSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
