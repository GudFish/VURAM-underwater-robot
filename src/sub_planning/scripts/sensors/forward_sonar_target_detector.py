#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray

try:
    import cv2
except ImportError:
    cv2 = None


class ForwardSonarTargetDetector:
    def __init__(self):
        rospy.init_node('forward_sonar_target_detector', anonymous=True)

        self.image_topic = rospy.get_param('~image_topic', '/forward_sonar/image')
        self.bearing_topic = rospy.get_param('~bearing_topic', '/target_acoustic_bearing')
        self.measurement_topic = rospy.get_param('~measurement_topic', '/forward_sonar/measurement')
        self.candidate_topic = rospy.get_param('~candidate_topic', '/forward_sonar/candidates')
        self.scene_candidate_topic = rospy.get_param('~scene_candidate_topic', '/forward_sonar/scene_candidates')
        self.debug_topic = rospy.get_param('~debug_topic', '/forward_sonar/detection_debug')
        self.debug_image_topic = rospy.get_param('~debug_image_topic', '/forward_sonar/debug_image')

        self.fov_deg = float(rospy.get_param('~fov_deg', 130.0))
        self.max_range = float(rospy.get_param('~max_range', 30.0))
        self.min_range = float(rospy.get_param('~min_range', 0.2))
        self.threshold_percentile = float(rospy.get_param('~threshold_percentile', 99.2))
        self.min_threshold = float(rospy.get_param('~min_threshold', 40.0))
        self.min_blob_area = int(rospy.get_param('~min_blob_area', 18))
        self.publish_min_score = float(rospy.get_param('~publish_min_score', 0.15))
        self.candidate_min_score = float(rospy.get_param('~candidate_min_score', 0.08))
        self.max_candidates = int(rospy.get_param('~max_candidates', 12))
        self.publish_scene_candidates = rospy.get_param('~publish_scene_candidates', True)
        self.max_scene_candidates = int(rospy.get_param('~max_scene_candidates', 16))
        self.bearing_candidate_gate_deg = float(rospy.get_param('~bearing_candidate_gate_deg', 12.0))
        self.publish_candidates_without_bearing = rospy.get_param('~publish_candidates_without_bearing', True)
        self.publish_legacy_measurement = rospy.get_param('~publish_legacy_measurement', False)
        self.min_mean_contrast = float(rospy.get_param('~min_mean_contrast', 14.0))
        self.min_peak_delta = float(rospy.get_param('~min_peak_delta', 10.0))
        self.min_excess_score = float(rospy.get_param('~min_excess_score', 0.10))
        self.bearing_timeout = float(rospy.get_param('~bearing_timeout', 1.0))
        self.use_local_threshold = rospy.get_param('~use_local_threshold', True)
        self.local_threshold_window_px = int(rospy.get_param('~local_threshold_window_px', 31))
        self.local_threshold_std_scale = float(rospy.get_param('~local_threshold_std_scale', 1.4))
        self.local_threshold_min_delta = float(rospy.get_param('~local_threshold_min_delta', 6.0))
        self.local_threshold_global_floor_ratio = float(rospy.get_param('~local_threshold_global_floor_ratio', 0.30))
        self.suppress_boundary_like_blobs = rospy.get_param('~suppress_boundary_like_blobs', True)
        self.boundary_min_area = int(rospy.get_param('~boundary_min_area', 120))
        self.boundary_min_long_axis_px = float(rospy.get_param('~boundary_min_long_axis_px', 45.0))
        self.boundary_min_physical_length_m = float(rospy.get_param('~boundary_min_physical_length_m', 3.5))
        self.boundary_min_aspect_ratio = float(rospy.get_param('~boundary_min_aspect_ratio', 4.0))
        self.boundary_min_perimeter_area_ratio = float(rospy.get_param('~boundary_min_perimeter_area_ratio', 1.2))
        self.boundary_mask_dilate_px = int(rospy.get_param('~boundary_mask_dilate_px', 2))
        self.suppress_near_noise_blobs = rospy.get_param('~suppress_near_noise_blobs', True)
        self.near_noise_range_m = float(rospy.get_param('~near_noise_range_m', 2.5))
        self.near_noise_min_area = int(rospy.get_param('~near_noise_min_area', 32))
        self.near_noise_min_physical_length_m = float(rospy.get_param('~near_noise_min_physical_length_m', 0.45))

        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.latest_bearing = None
        self.latest_bearing_stamp = None

        if cv2 is None:
            rospy.logerr('[ForwardSonarDetector] OpenCV python module is not available.')
            rospy.signal_shutdown('cv2 import failed')
            return

        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.measurement_pub = rospy.Publisher(self.measurement_topic, Float32MultiArray, queue_size=10)
        self.candidate_pub = rospy.Publisher(self.candidate_topic, Float32MultiArray, queue_size=10)
        self.scene_candidate_pub = rospy.Publisher(self.scene_candidate_topic, Float32MultiArray, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, Float32MultiArray, queue_size=10)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=2)

        rospy.loginfo(
            '[ForwardSonarDetector] image=%s bearing=%s candidates=%s scene_candidates=%s debug_image=%s',
            self.image_topic,
            self.bearing_topic,
            self.candidate_topic,
            self.scene_candidate_topic,
            self.debug_image_topic,
        )

    @staticmethod
    def wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def bearing_cb(self, msg):
        self.latest_bearing = self.wrap_pi(float(msg.data))
        self.latest_bearing_stamp = rospy.Time.now()

    def image_cb(self, msg):
        if msg.encoding not in ('mono8', '8UC1'):
            rospy.logwarn_throttle(2.0, '[ForwardSonarDetector] unsupported encoding: %s', msg.encoding)
            return

        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        bearing = self.valid_guidance_bearing(log_warnings=False)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        footprint_mask = self.valid_mask(msg.height, msg.width)
        valid_pixels = blurred[footprint_mask]
        if valid_pixels.size == 0:
            self.publish_debug_image(msg, image)
            return

        global_threshold = max(
            self.min_threshold,
            float(np.percentile(valid_pixels, np.clip(self.threshold_percentile, 0.0, 100.0))),
        )

        raw_binary, threshold_map = self.build_detection_binary(blurred, footprint_mask, global_threshold)
        boundary_binary = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, self.kernel)
        boundary_mask, suppressed = self.extract_boundary_mask(boundary_binary, msg.height, msg.width)
        binary = np.array(raw_binary, copy=True)
        if np.any(boundary_mask):
            binary[boundary_mask] = 0
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        if np.any(boundary_mask):
            binary[boundary_mask] = 0
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        else:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)

        background_level = float(np.median(valid_pixels))
        scene_candidates, candidate_suppressed = self.extract_candidates(
            image, binary, threshold_map, background_level, None, self.max_scene_candidates
        )
        suppressed.extend(candidate_suppressed)
        if self.publish_scene_candidates:
            self.publish_candidates(scene_candidates, self.scene_candidate_pub)

        candidates = self.filter_candidates_for_pdaf(scene_candidates, bearing)
        self.publish_debug_image(msg, image, candidates, bearing, suppressed, scene_candidates)

        if not candidates:
            if bearing is None and not self.publish_candidates_without_bearing:
                rospy.logwarn_throttle(2.0, '[ForwardSonarDetector] waiting for bearing-guided acquisition.')
            return

        self.publish_candidates(candidates, self.candidate_pub)
        if self.publish_legacy_measurement:
            self.publish_legacy_best_measurement(candidates)

        debug_msg = Float32MultiArray()
        top = candidates[0]
        debug_msg.data = [
            float(len(candidates)),
            top['range'],
            top['bearing'],
            top['score'],
            top['row'],
            top['col'],
            float(top['area']),
            top['threshold'],
            top['uncertainty'],
            -999.0 if bearing is None else bearing,
            float(len(suppressed)),
        ]
        self.debug_pub.publish(debug_msg)

    def valid_guidance_bearing(self, log_warnings=True):
        if self.latest_bearing is None or self.latest_bearing_stamp is None:
            if log_warnings:
                rospy.logwarn_throttle(2.0, '[ForwardSonarDetector] waiting for acoustic bearing guidance.')
            return None

        age = (rospy.Time.now() - self.latest_bearing_stamp).to_sec()
        if age > max(self.bearing_timeout, 0.0):
            if log_warnings:
                rospy.logwarn_throttle(2.0, '[ForwardSonarDetector] acoustic bearing guidance timed out.')
            return None

        half_fov = math.radians(self.fov_deg) * 0.5
        if abs(self.latest_bearing) > half_fov:
            if log_warnings:
                rospy.logwarn_throttle(
                    2.0,
                    '[ForwardSonarDetector] acoustic bearing %.2f rad is outside sonar FOV %.2f rad.',
                    self.latest_bearing,
                    half_fov,
                )
            return None
        return self.latest_bearing

    def build_search_mask(self, height, width, footprint_mask, bearing):
        if bearing is None:
            return footprint_mask
        return footprint_mask & self.bearing_gate_mask(height, width, bearing, self.bearing_candidate_gate_deg)

    def build_detection_binary(self, blurred, mask, global_threshold):
        binary = np.zeros_like(blurred, dtype=np.uint8)
        if not self.use_local_threshold:
            binary[(blurred >= global_threshold) & mask] = 255
            threshold_map = np.full_like(blurred, global_threshold, dtype=np.float32)
            return binary, threshold_map

        window = max(int(self.local_threshold_window_px), 3)
        if window % 2 == 0:
            window += 1

        image_f = blurred.astype(np.float32)
        local_mean = cv2.GaussianBlur(image_f, (window, window), 0)
        local_sq_mean = cv2.GaussianBlur(image_f * image_f, (window, window), 0)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
        local_threshold = local_mean + self.local_threshold_min_delta + self.local_threshold_std_scale * local_std
        global_floor = max(self.min_threshold, global_threshold * self.local_threshold_global_floor_ratio)
        threshold_map = np.maximum(global_floor, local_threshold).astype(np.float32)
        binary[(image_f >= threshold_map) & mask] = 255
        return binary, threshold_map

    def extract_boundary_mask(self, binary, height, width):
        if not self.suppress_boundary_like_blobs:
            return np.zeros_like(binary, dtype=bool), []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary, dtype=bool), []

        boundary_mask = np.zeros_like(binary, dtype=np.uint8)
        suppressed = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.boundary_min_area:
                continue

            component_mask = labels == label_idx
            bbox = (
                int(stats[label_idx, cv2.CC_STAT_LEFT]),
                int(stats[label_idx, cv2.CC_STAT_TOP]),
                int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
            )
            shape = self.component_shape_metrics(component_mask, bbox, area, height, width)
            if not self.is_boundary_like_blob(area, shape):
                continue

            boundary_mask[component_mask] = 255
            suppressed.append({'bbox': bbox, 'area': area, 'shape': shape, 'reason': 'boundary_mask'})

        dilate_px = max(int(self.boundary_mask_dilate_px), 0)
        if dilate_px > 0 and np.any(boundary_mask):
            kernel_size = 2 * dilate_px + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            boundary_mask = cv2.dilate(boundary_mask, kernel, iterations=1)

        return boundary_mask.astype(bool), suppressed

    def extract_candidates(self, image, binary, threshold_map, background_level, bearing, max_candidates=None):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return [], []

        candidates = []
        suppressed = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.min_blob_area:
                continue

            component_mask = labels == label_idx
            bbox = (
                int(stats[label_idx, cv2.CC_STAT_LEFT]),
                int(stats[label_idx, cv2.CC_STAT_TOP]),
                int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
            )
            shape = self.component_shape_metrics(component_mask, bbox, area, image.shape[0], image.shape[1])
            if self.is_boundary_like_blob(area, shape):
                suppressed.append({'bbox': bbox, 'area': area, 'shape': shape, 'reason': 'boundary_candidate'})
                continue
            if self.is_near_noise_blob(shape, area):
                suppressed.append({'bbox': bbox, 'area': area, 'shape': shape, 'reason': 'near_noise'})
                continue

            pixels = image[component_mask].astype(np.float32)
            intensity_sum = float(np.sum(pixels))
            if intensity_sum <= 0.0:
                continue
            mean_intensity = intensity_sum / max(area, 1)
            peak_intensity = float(np.max(pixels))
            local_threshold = float(np.median(threshold_map[component_mask]))
            mean_contrast = mean_intensity - background_level
            excess_score = float(
                np.sum(np.maximum(pixels - background_level, 0.0)) / max(area * 255.0, 1.0)
            )
            if mean_contrast < self.min_mean_contrast:
                continue
            if peak_intensity < local_threshold + self.min_peak_delta:
                continue
            if excess_score < self.min_excess_score:
                continue

            rows, cols = np.nonzero(component_mask)
            weights = pixels + 1.0e-3
            weight_sum = float(np.sum(weights))
            row = float(np.dot(rows.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
            col = float(np.dot(cols.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
            score = float(np.clip(excess_score, 0.0, 1.0))
            if score < self.candidate_min_score:
                continue

            detection = self.row_col_to_measurement(row, col, image.shape[0], image.shape[1])
            if bearing is not None:
                bearing_error = abs(self.wrap_pi(detection['bearing'] - bearing))
                if bearing_error > math.radians(self.bearing_candidate_gate_deg):
                    continue
            else:
                bearing_error = -1.0

            uncertainty = self.estimate_candidate_uncertainty(score, mean_contrast, bearing_error)
            candidates.append(
                {
                    'label': label_idx,
                    'area': area,
                    'row': row,
                    'col': col,
                    'score': score,
                    'intensity_sum': intensity_sum,
                    'mean_contrast': float(mean_contrast),
                    'peak_intensity': peak_intensity,
                    'excess_score': excess_score,
                    'uncertainty': uncertainty,
                    'bearing_error': float(bearing_error),
                    'threshold': float(local_threshold),
                    'range': detection['range'],
                    'bearing': detection['bearing'],
                    'forward': detection['forward'],
                    'lateral': detection['lateral'],
                    'bbox': bbox,
                }
            )

        if not candidates:
            return [], suppressed

        candidates.sort(
            key=lambda item: (
                abs(item['bearing_error']) if item['bearing_error'] >= 0.0 else 0.0,
                -item['score'],
                -item['intensity_sum'],
            )
        )
        limit = self.max_candidates if max_candidates is None else max_candidates
        return candidates[:max(int(limit), 1)], suppressed

    def filter_candidates_for_pdaf(self, scene_candidates, bearing):
        if bearing is None:
            if not self.publish_candidates_without_bearing:
                return []
            return list(scene_candidates[:max(self.max_candidates, 1)])

        filtered = []
        gate = math.radians(self.bearing_candidate_gate_deg)
        for item in scene_candidates:
            bearing_error = abs(self.wrap_pi(item['bearing'] - bearing))
            if bearing_error > gate:
                continue
            candidate = dict(item)
            candidate['bearing_error'] = float(bearing_error)
            candidate['uncertainty'] = self.estimate_candidate_uncertainty(
                candidate['score'],
                candidate['mean_contrast'],
                bearing_error,
            )
            filtered.append(candidate)

        filtered.sort(
            key=lambda item: (
                item['bearing_error'],
                -item['score'],
                -item['intensity_sum'],
            )
        )
        return filtered[:max(self.max_candidates, 1)]

    def component_shape_metrics(self, component_mask, bbox, area, height, width):
        x, y, w, h = bbox
        roi = component_mask[y:y + h, x:x + w].astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0.0
        rect_aspect = 1.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))
            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            rect_aspect = max(rw, rh) / max(min(rw, rh), 1.0)

        bbox_aspect = max(w, h) / max(min(w, h), 1.0)
        long_axis = float(max(w, h))
        fill_ratio = float(area) / max(float(w * h), 1.0)
        perimeter_area_ratio = perimeter / max(float(area), 1.0)
        physical = self.component_physical_extent(component_mask, height, width)
        return {
            'bbox_aspect': float(bbox_aspect),
            'rect_aspect': float(rect_aspect),
            'long_axis': long_axis,
            'fill_ratio': fill_ratio,
            'perimeter_area_ratio': float(perimeter_area_ratio),
            'physical_length_m': physical['length_m'],
            'physical_range_span_m': physical['range_span_m'],
            'physical_lateral_span_m': physical['lateral_span_m'],
            'min_range_m': physical['min_range_m'],
        }

    def component_physical_extent(self, component_mask, height, width):
        rows, cols = np.nonzero(component_mask)
        if rows.size == 0:
            return {'length_m': 0.0, 'range_span_m': 0.0, 'lateral_span_m': 0.0, 'min_range_m': 0.0}

        rows_f = rows.astype(np.float32)
        cols_f = cols.astype(np.float32)
        forward = self.min_range + (1.0 - rows_f / max(height - 1, 1)) * (self.max_range - self.min_range)
        max_lateral = self.max_lateral_span()
        lateral = max_lateral - cols_f * (2.0 * max_lateral / max(width - 1, 1))

        range_span = float(np.max(forward) - np.min(forward))
        lateral_span = float(np.max(lateral) - np.min(lateral))
        min_range = float(np.min(np.hypot(forward, lateral)))
        coords = np.column_stack((forward, lateral)).astype(np.float32)
        if coords.shape[0] < 2:
            length_m = 0.0
        else:
            centered = coords - np.mean(coords, axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                axis = vh[0]
                projection = np.dot(centered, axis)
                length_m = float(np.max(projection) - np.min(projection))
            except np.linalg.LinAlgError:
                length_m = float(math.hypot(range_span, lateral_span))

        return {
            'length_m': float(max(length_m, 0.0)),
            'range_span_m': range_span,
            'lateral_span_m': lateral_span,
            'min_range_m': min_range,
        }

    def is_boundary_like_blob(self, area, shape):
        if not self.suppress_boundary_like_blobs:
            return False
        if area < self.boundary_min_area:
            return False

        long_enough = shape['long_axis'] >= self.boundary_min_long_axis_px
        physically_long = shape['physical_length_m'] >= self.boundary_min_physical_length_m
        elongated = max(shape['bbox_aspect'], shape['rect_aspect']) >= self.boundary_min_aspect_ratio
        arc_like = shape['perimeter_area_ratio'] >= self.boundary_min_perimeter_area_ratio
        sparse_long_region = long_enough and shape['fill_ratio'] < 0.45
        shape_like_boundary = elongated or arc_like or sparse_long_region
        return long_enough and physically_long and shape_like_boundary

    def is_near_noise_blob(self, shape, area):
        if not self.suppress_near_noise_blobs:
            return False
        if shape['min_range_m'] > self.near_noise_range_m:
            return False
        if area >= self.near_noise_min_area:
            return False
        if shape['physical_length_m'] >= self.near_noise_min_physical_length_m:
            return False
        return True

    def estimate_candidate_uncertainty(self, score, mean_contrast, bearing_error):
        score_ref = max(self.publish_min_score * 3.0, 1.0e-3)
        quality_unc = 1.0 - min(score / score_ref, 1.0)
        contrast_ref = max(self.min_mean_contrast * 3.0, 1.0e-3)
        contrast_unc = 1.0 - min(mean_contrast / contrast_ref, 1.0)
        uncertainty = 0.65 * quality_unc + 0.35 * contrast_unc

        if bearing_error < 0.0:
            uncertainty = min(1.0, uncertainty + 0.10)
        else:
            validation_gate = max(math.radians(self.bearing_candidate_gate_deg), 1.0e-3)
            bearing_unc = min(bearing_error / validation_gate, 1.0)
            uncertainty = 0.75 * uncertainty + 0.25 * bearing_unc

        return float(np.clip(uncertainty, 0.0, 1.0))

    def publish_candidates(self, candidates, publisher=None):
        if publisher is None:
            publisher = self.candidate_pub
        msg = Float32MultiArray()
        data = [float(len(candidates))]
        for item in candidates:
            data.extend(
                [
                    item['range'],
                    item['bearing'],
                    item['score'],
                    item['uncertainty'],
                    item['row'],
                    item['col'],
                    float(item['area']),
                    item['threshold'],
                ]
            )
        msg.data = data
        publisher.publish(msg)

    def publish_legacy_best_measurement(self, candidates):
        if not candidates:
            return
        best = max(candidates, key=lambda item: item['score'])
        msg = Float32MultiArray()
        msg.data = [best['range'], best['bearing'], best['score'], best['uncertainty']]
        self.measurement_pub.publish(msg)

    def publish_debug_image(self, source_msg, image, candidates=None, bearing=None, suppressed=None, scene_candidates=None):
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if bearing is not None:
            self.draw_bearing_gate(debug, bearing)
        for item in suppressed or []:
            x, y, w, h = item['bbox']
            cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
        pdaf_labels = set(item.get('label') for item in candidates or [])
        for detection in scene_candidates or []:
            if detection.get('label') in pdaf_labels:
                continue
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 220, 220), 1)
        for detection in candidates or []:
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), (0, 220, 0), 1)
            center = (int(round(detection['col'])), int(round(detection['row'])))
            cv2.circle(debug, center, 4, (255, 255, 255), 1)
            cv2.line(debug, (center[0] - 5, center[1]), (center[0] + 5, center[1]), (255, 255, 255), 1)
            cv2.line(debug, (center[0], center[1] - 5), (center[0], center[1] + 5), (255, 255, 255), 1)

        msg = Image()
        msg.header = source_msg.header
        msg.height = source_msg.height
        msg.width = source_msg.width
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = source_msg.width * 3
        msg.data = np.ascontiguousarray(debug).tobytes()
        self.debug_image_pub.publish(msg)

    def valid_mask(self, height, width):
        forward = np.linspace(self.max_range, self.min_range, height, dtype=np.float32)[:, None]
        max_lateral = self.max_lateral_span()
        lateral = np.linspace(max_lateral, -max_lateral, width, dtype=np.float32)[None, :]
        half_fov = math.radians(self.fov_deg) * 0.5
        return np.abs(lateral) <= forward * math.tan(half_fov)

    def max_lateral_span(self):
        return max(self.max_range * math.sin(math.radians(self.fov_deg) * 0.5), 1.0e-6)

    def bearing_gate_mask(self, height, width, bearing, gate_deg):
        forward = np.linspace(self.max_range, self.min_range, height, dtype=np.float32)[:, None]
        max_lateral = self.max_lateral_span()
        lateral = np.linspace(max_lateral, -max_lateral, width, dtype=np.float32)[None, :]
        image_bearing = np.arctan2(lateral, np.maximum(forward, 1.0e-6))
        delta = np.arctan2(np.sin(image_bearing - bearing), np.cos(image_bearing - bearing))
        return np.abs(delta) <= math.radians(gate_deg)

    def bearing_to_col(self, bearing, forward, width):
        lateral = forward * math.tan(bearing)
        max_lateral = self.max_lateral_span()
        t = (max_lateral - lateral) / (2.0 * max_lateral)
        return int(np.clip(t, 0.0, 1.0) * (width - 1))

    def draw_bearing_gate(self, image, bearing):
        height, width = image.shape[:2]
        gate = math.radians(self.bearing_candidate_gate_deg)
        color = (255, 220, 0) if image.ndim == 3 else 180
        for edge_bearing in (bearing - gate, bearing + gate):
            pts = []
            for row in range(0, height, max(1, height // 40)):
                forward = self.min_range + (1.0 - row / max(height - 1, 1)) * (self.max_range - self.min_range)
                col = self.bearing_to_col(edge_bearing, forward, width)
                pts.append((col, row))
            if len(pts) >= 2:
                cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, color, 1)

    def row_col_to_measurement(self, row, col, height, width):
        row = float(np.clip(row, 0.0, max(height - 1, 0)))
        col = float(np.clip(col, 0.0, max(width - 1, 0)))
        forward = self.min_range + (1.0 - row / max(height - 1, 1)) * (self.max_range - self.min_range)
        max_lateral = self.max_lateral_span()
        lateral = max_lateral - col * (2.0 * max_lateral / max(width - 1, 1))
        return {
            'forward': float(forward),
            'lateral': float(lateral),
            'bearing': float(math.atan2(lateral, max(forward, 1.0e-6))),
            'range': float(math.hypot(forward, lateral)),
        }


if __name__ == '__main__':
    try:
        ForwardSonarTargetDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
