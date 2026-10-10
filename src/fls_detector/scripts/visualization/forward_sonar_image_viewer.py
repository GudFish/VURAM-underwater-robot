#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：订阅前视声呐图像并以窗口形式显示，便于快速观察仿真或检测结果。
输入：通过 ROS 参数以及状态、边界、图像或置信度话题接收可视化输入。
输出：发布 RViz 标记、TF 或显示结果，辅助观察系统运行状态。
"""
import numpy as np
import rospy
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    cv2 = None


class ForwardSonarImageViewer:
    """
    功能：显示订阅到的前视声呐图像并处理窗口关闭逻辑。
    输入：构造阶段通常无显式位置参数，主要依赖 ROS 参数、订阅话题和内部成员状态。
    输出：初始化并维护该对象的运行状态，供后续回调、计算或消息发布流程使用。
    """
    def __init__(self):
        """
        功能：初始化当前对象，读取参数、创建订阅发布器并准备运行时状态。
        输入：无显式构造参数；通过 ROS 参数、话题配置和成员变量完成初始化。
        输出：无显式返回；完成对象初始化、资源创建与成员状态准备。
        """
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
        """
        功能：处理图像消息并执行检测或显示逻辑。
        输入：`msg` 为对应订阅话题的 ROS 消息；其余输入依赖当前对象缓存状态。
        输出：无显式返回；通过更新内部状态、发布消息或执行副作用完成功能。
        """
        if msg.encoding in ('mono8', '8UC1'):
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            display = image
            if self.colormap:
                display = cv2.applyColorMap(display, cv2.COLORMAP_OCEAN)
        elif msg.encoding == 'bgr8':
            display = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        elif msg.encoding == 'rgb8':
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            rospy.logwarn_throttle(2.0, '[ForwardSonarViewer] unsupported encoding: %s', msg.encoding)
            return

        if abs(self.display_scale - 1.0) > 1e-3:
            display = cv2.resize(display, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    def shutdown(self):
        """
        功能：执行可视化关闭时的资源清理。
        输入：无显式输入参数；主要依赖对象当前状态、成员配置或全局上下文。
        输出：无显式返回；通过更新内部状态或维护对象成员完成功能。
        """
        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        ForwardSonarImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
