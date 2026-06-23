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
        self.debug_topic = rospy.get_param('~debug_topic', '/forward_sonar/detection_debug')
        self.debug_image_topic = rospy.get_param('~debug_image_topic', '/forward_sonar/debug_image')

        self.fov_deg = float(rospy.get_param('~fov_deg', 130.0))
        self.max_range = float(rospy.get_param('~max_range', 30.0))
        self.min_range = float(rospy.get_param('~min_range', 0.2))
        self.threshold_percentile = float(rospy.get_param('~threshold_percentile', 99.2))
        self.min_threshold = float(rospy.get_param('~min_threshold', 40.0))
        self.min_blob_area = int(rospy.get_param('~min_blob_area', 18))
        self.track_gate_px = float(rospy.get_param('~track_gate_px', 45.0))
        self.publish_min_score = float(rospy.get_param('~publish_min_score', 0.15))
        self.min_mean_contrast = float(rospy.get_param('~min_mean_contrast', 14.0))
        self.min_peak_delta = float(rospy.get_param('~min_peak_delta', 10.0))
        self.min_excess_score = float(rospy.get_param('~min_excess_score', 0.10))
        self.min_confirm_frames = int(rospy.get_param('~min_confirm_frames', 2))
        self.max_missed_frames = int(rospy.get_param('~max_missed_frames', 3))
        self.bearing_gate_deg = float(rospy.get_param('~bearing_gate_deg', 8.0))
        self.bearing_validation_gate_deg = float(rospy.get_param('~bearing_validation_gate_deg', 14.0))
        self.bearing_timeout = float(rospy.get_param('~bearing_timeout', 1.0))
        self.allow_track_without_bearing = rospy.get_param('~allow_track_without_bearing', True)
        self.track_gate_scale = float(rospy.get_param('~track_gate_scale', 1.6))
        self.track_uncertainty_publish_max = float(rospy.get_param('~track_uncertainty_publish_max', 0.65))
        self.track_uncertainty_reset_max = float(rospy.get_param('~track_uncertainty_reset_max', 0.9))
        self.track_uncertainty_alpha = float(rospy.get_param('~track_uncertainty_alpha', 0.65))
        self.missed_uncertainty_growth = float(rospy.get_param('~missed_uncertainty_growth', 0.25))

        self.kernel = np.ones((3, 3), dtype=np.uint8)
        self.last_detection_rc = None
        self.confirmed_detection_rc = None
        self.pending_detection_rc = None
        self.pending_detection_count = 0
        self.missed_frames = 0
        self.track_uncertainty = 1.0
        self.latest_bearing = None
        self.latest_bearing_stamp = None

        if cv2 is None:
            rospy.logerr('[ForwardSonarDetector] OpenCV python module is not available.')
            rospy.signal_shutdown('cv2 import failed')
            return

        self.bearing_sub = rospy.Subscriber(self.bearing_topic, Float32, self.bearing_cb, queue_size=10)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        self.measurement_pub = rospy.Publisher(self.measurement_topic, Float32MultiArray, queue_size=10)
        self.debug_pub = rospy.Publisher(self.debug_topic, Float32MultiArray, queue_size=10)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=2)

        rospy.loginfo(
            '[ForwardSonarDetector] image=%s bearing=%s measurement=%s debug_image=%s',
            self.image_topic,
            self.bearing_topic,
            self.measurement_topic,
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
        tracking_ready = self.tracking_is_usable()
        if bearing is None and not tracking_ready:
            self.handle_missed_detection()
            self.publish_debug_image(msg, image)
            rospy.logwarn_throttle(2.0, '[ForwardSonarDetector] waiting for bearing-guided acquisition.')
            return

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        footprint_mask = self.valid_mask(msg.height, msg.width)
        mask = self.build_search_mask(msg.height, msg.width, footprint_mask, bearing, tracking_ready)
        valid_pixels = blurred[mask]
        if valid_pixels.size == 0:
            self.handle_missed_detection()
            self.publish_debug_image(msg, image)
            return

        threshold = max(
            self.min_threshold,
            float(np.percentile(valid_pixels, np.clip(self.threshold_percentile, 0.0, 100.0))),
        )

        binary = np.zeros_like(image, dtype=np.uint8)
        binary[(blurred >= threshold) & mask] = 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)

        background_level = float(np.median(valid_pixels))
        detection = self.extract_detection(image, binary, threshold, background_level)
        self.publish_debug_image(msg, image, detection, bearing)

        if detection is None:
            self.handle_missed_detection()
            return

        detection['uncertainty'] = self.estimate_detection_uncertainty(detection, bearing)
        if not self.confirm_detection(detection):
            self.handle_missed_detection(keep_pending=True)
            return

        if detection['score'] < self.publish_min_score:
            self.handle_missed_detection()
            return
        if detection['uncertainty'] > self.track_uncertainty_publish_max:
            self.handle_missed_detection(keep_pending=True)
            return

        had_confirmed_track = self.confirmed_detection_rc is not None
        self.last_detection_rc = (detection['row'], detection['col'])
        self.confirmed_detection_rc = self.last_detection_rc
        self.missed_frames = 0
        self.update_track_uncertainty(detection['uncertainty'], had_confirmed_track)
        measurement_msg = Float32MultiArray()
        measurement_msg.data = [detection['range'], detection['bearing'], detection['score'], detection['uncertainty']]
        self.measurement_pub.publish(measurement_msg)

        debug_msg = Float32MultiArray()
        debug_msg.data = [
            detection['range'],
            detection['bearing'],
            detection['score'],
            detection['row'],
            detection['col'],
            float(detection['area']),
            detection['threshold'],
            detection['uncertainty'],
            -999.0 if bearing is None else bearing,
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

    def tracking_is_usable(self):
        return (
            self.allow_track_without_bearing
            and self.confirmed_detection_rc is not None
            and self.track_uncertainty <= self.track_uncertainty_publish_max
        )

    def build_search_mask(self, height, width, footprint_mask, bearing, tracking_ready):
        if tracking_ready:
            mask = footprint_mask & self.track_gate_mask(height, width)
            if bearing is not None and self.track_uncertainty > 0.35:
                mask = mask & self.bearing_gate_mask(height, width, bearing)
            return mask
        if bearing is None:
            return np.zeros((height, width), dtype=bool)
        return footprint_mask & self.bearing_gate_mask(height, width, bearing)

    def extract_detection(self, image, binary, threshold, background_level):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return None

        candidates = []
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < self.min_blob_area:
                continue

            component_mask = labels == label_idx
            pixels = image[component_mask].astype(np.float32)
            intensity_sum = float(np.sum(pixels))
            if intensity_sum <= 0.0:
                continue
            mean_intensity = intensity_sum / max(area, 1)
            peak_intensity = float(np.max(pixels))
            mean_contrast = mean_intensity - background_level
            excess_score = float(
                np.sum(np.maximum(pixels - background_level, 0.0)) / max(area * 255.0, 1.0)
            )
            if mean_contrast < self.min_mean_contrast:
                continue
            if peak_intensity < threshold + self.min_peak_delta:
                continue
            if excess_score < self.min_excess_score:
                continue

            rows, cols = np.nonzero(component_mask)
            weights = pixels + 1.0e-3
            weight_sum = float(np.sum(weights))
            row = float(np.dot(rows.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
            col = float(np.dot(cols.astype(np.float32), weights) / max(weight_sum, 1.0e-6))
            score = float(np.clip(excess_score, 0.0, 1.0))
            detection = self.row_col_to_measurement(row, col, image.shape[0], image.shape[1])
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
                    'threshold': float(threshold),
                    'range': detection['range'],
                    'bearing': detection['bearing'],
                    'forward': detection['forward'],
                    'lateral': detection['lateral'],
                    'bbox': (
                        int(stats[label_idx, cv2.CC_STAT_LEFT]),
                        int(stats[label_idx, cv2.CC_STAT_TOP]),
                        int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                        int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
                    ),
                }
            )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item['intensity_sum'], reverse=True)
        reference_rc = self.confirmed_detection_rc if self.confirmed_detection_rc is not None else self.last_detection_rc
        if reference_rc is not None:
            lr, lc = reference_rc
            gated = [
                item
                for item in candidates
                if math.hypot(item['row'] - lr, item['col'] - lc) <= self.track_gate_px
            ]
            if gated:
                return gated[0]
        return candidates[0]

    def estimate_detection_uncertainty(self, detection, bearing):
        score_ref = max(self.publish_min_score * 3.0, 1.0e-3)
        quality_unc = 1.0 - min(detection['score'] / score_ref, 1.0)
        contrast_ref = max(self.min_mean_contrast * 3.0, 1.0e-3)
        contrast_unc = 1.0 - min(detection['mean_contrast'] / contrast_ref, 1.0)
        uncertainty = 0.65 * quality_unc + 0.35 * contrast_unc

        if self.confirmed_detection_rc is not None:
            dist = math.hypot(
                detection['row'] - self.confirmed_detection_rc[0],
                detection['col'] - self.confirmed_detection_rc[1],
            )
            gate_px = max(self.track_gate_px * self.track_gate_scale, 1.0)
            spatial_unc = min(dist / gate_px, 1.0)
            uncertainty = 0.7 * uncertainty + 0.3 * spatial_unc

        if bearing is None:
            uncertainty = min(1.0, uncertainty + 0.10)
        else:
            delta = abs(self.wrap_pi(detection['bearing'] - bearing))
            validation_gate = max(math.radians(self.bearing_validation_gate_deg), 1.0e-3)
            bearing_unc = min(delta / validation_gate, 1.0)
            uncertainty = 0.75 * uncertainty + 0.25 * bearing_unc
            if delta > validation_gate and self.confirmed_detection_rc is None:
                uncertainty = 1.0

        return float(np.clip(uncertainty, 0.0, 1.0))

    def update_track_uncertainty(self, detection_uncertainty, had_confirmed_track):
        if not had_confirmed_track:
            self.track_uncertainty = detection_uncertainty
            return
        alpha = float(np.clip(self.track_uncertainty_alpha, 0.0, 1.0))
        self.track_uncertainty = alpha * self.track_uncertainty + (1.0 - alpha) * detection_uncertainty

    def confirm_detection(self, detection):
        current_rc = (detection['row'], detection['col'])

        if self.confirmed_detection_rc is not None:
            if math.hypot(
                current_rc[0] - self.confirmed_detection_rc[0],
                current_rc[1] - self.confirmed_detection_rc[1],
            ) <= self.track_gate_px:
                self.pending_detection_rc = current_rc
                self.pending_detection_count = self.min_confirm_frames
                return True

        if self.pending_detection_rc is None:
            self.pending_detection_rc = current_rc
            self.pending_detection_count = 1
            return self.pending_detection_count >= self.min_confirm_frames

        if math.hypot(
            current_rc[0] - self.pending_detection_rc[0],
            current_rc[1] - self.pending_detection_rc[1],
        ) <= self.track_gate_px:
            self.pending_detection_count += 1
            self.pending_detection_rc = current_rc
        else:
            self.pending_detection_rc = current_rc
            self.pending_detection_count = 1

        return self.pending_detection_count >= self.min_confirm_frames

    def handle_missed_detection(self, keep_pending=False):
        self.missed_frames += 1
        self.track_uncertainty = min(1.0, self.track_uncertainty + self.missed_uncertainty_growth)
        if not keep_pending:
            self.pending_detection_rc = None
            self.pending_detection_count = 0
        if self.missed_frames >= self.max_missed_frames or self.track_uncertainty >= self.track_uncertainty_reset_max:
            self.last_detection_rc = None
            self.confirmed_detection_rc = None
            self.track_uncertainty = 1.0
            if not keep_pending:
                self.pending_detection_rc = None
                self.pending_detection_count = 0

    def publish_debug_image(self, source_msg, image, detection=None, bearing=None):
        debug = np.array(image, copy=True)
        if bearing is not None:
            self.draw_bearing_gate(debug, bearing)
        if detection is not None:
            x, y, w, h = detection['bbox']
            cv2.rectangle(debug, (x, y), (x + w - 1, y + h - 1), 220, 1)
            center = (int(round(detection['col'])), int(round(detection['row'])))
            cv2.circle(debug, center, 4, 255, 1)
            cv2.line(debug, (center[0] - 5, center[1]), (center[0] + 5, center[1]), 255, 1)
            cv2.line(debug, (center[0], center[1] - 5), (center[0], center[1] + 5), 255, 1)

        msg = Image()
        msg.header = source_msg.header
        msg.height = source_msg.height
        msg.width = source_msg.width
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = source_msg.width
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

    def bearing_gate_mask(self, height, width, bearing):
        forward = np.linspace(self.max_range, self.min_range, height, dtype=np.float32)[:, None]
        max_lateral = self.max_lateral_span()
        lateral = np.linspace(max_lateral, -max_lateral, width, dtype=np.float32)[None, :]
        image_bearing = np.arctan2(lateral, np.maximum(forward, 1.0e-6))
        delta = np.arctan2(np.sin(image_bearing - bearing), np.cos(image_bearing - bearing))
        return np.abs(delta) <= math.radians(self.bearing_gate_deg)

    def track_gate_mask(self, height, width):
        if self.confirmed_detection_rc is None:
            return np.zeros((height, width), dtype=bool)
        radius = max(self.track_gate_px * self.track_gate_scale * (1.0 + self.track_uncertainty), 1.0)
        rows, cols = np.ogrid[:height, :width]
        dr = rows - self.confirmed_detection_rc[0]
        dc = cols - self.confirmed_detection_rc[1]
        return (dr * dr + dc * dc) <= radius * radius

    def bearing_to_col(self, bearing, forward, width):
        lateral = forward * math.tan(bearing)
        max_lateral = self.max_lateral_span()
        t = (max_lateral - lateral) / (2.0 * max_lateral)
        return int(np.clip(t, 0.0, 1.0) * (width - 1))

    def draw_bearing_gate(self, image, bearing):
        height, width = image.shape[:2]
        gate = math.radians(self.bearing_gate_deg)
        for edge_bearing in (bearing - gate, bearing + gate):
            pts = []
            for row in range(0, height, max(1, height // 40)):
                forward = self.min_range + (1.0 - row / max(height - 1, 1)) * (self.max_range - self.min_range)
                col = self.bearing_to_col(edge_bearing, forward, width)
                pts.append((col, row))
            if len(pts) >= 2:
                cv2.polylines(image, [np.array(pts, dtype=np.int32)], False, 180, 1)

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
