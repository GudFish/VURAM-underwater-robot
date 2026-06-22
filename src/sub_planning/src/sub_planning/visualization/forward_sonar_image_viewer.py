#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    cv2 = None


class ForwardSonarImageViewer:
    def __init__(self):
        rospy.init_node('forward_sonar_image_viewer', anonymous=True)
        self.image_topic = rospy.get_param('~image_topic', '/forward_sonar/image')
        self.window_name = rospy.get_param('~window_name', 'Forward Looking Sonar')
        self.colormap = rospy.get_param('~colormap', True)
        self.display_scale = float(rospy.get_param('~display_scale', 1.4))

        if cv2 is None:
            rospy.logerr('[ForwardSonarViewer] OpenCV python module is not available.')
            rospy.signal_shutdown('cv2 import failed')
            return

        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo('[ForwardSonarViewer] showing %s in window "%s"', self.image_topic, self.window_name)

    def image_cb(self, msg):
        if msg.encoding not in ('mono8', '8UC1'):
            rospy.logwarn_throttle(2.0, '[ForwardSonarViewer] unsupported encoding: %s', msg.encoding)
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
        display = image
        if self.colormap:
            display = cv2.applyColorMap(display, cv2.COLORMAP_OCEAN)
        if abs(self.display_scale - 1.0) > 1e-3:
            display = cv2.resize(display, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    def shutdown(self):
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        ForwardSonarImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
