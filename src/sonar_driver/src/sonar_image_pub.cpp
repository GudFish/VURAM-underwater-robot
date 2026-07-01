#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QTimer>

#include "forwardlookingsonarsdk.h"

/**
 * 功能：真实前视声呐 ROS 驱动节点。
 * 使用厂商 SDK 连接声呐硬件，将声呐图像转换为 sensor_msgs/Image 发布到 ROS 话题。
 *
 * 注意事项：
 *   1. 该 SDK 编译为 ARM aarch64，需要在 ARM64 平台（如 Jetson）上编译运行。
 *   2. SDK 依赖 Qt5 的 GUI 模块，使用 QT_QPA_PLATFORM=offscreen 可实现无屏运行。
 *   3. 运行前确保声呐与主机网络连通，setting.ini 配置正确。
 *   4. 图像通过 Sonarsurface::m_pData 获取，row-major 排列（从近到远，从左到右）。
 */

int main(int argc, char** argv)
{
    // ========================================================================
    // 1. Qt 应用初始化（SDK 内部使用 QObject/QWidget，需要 QApplication）
    //    设置 QT_QPA_PLATFORM=offscreen 可在无显示器环境下运行
    // ========================================================================
    QApplication app(argc, argv);

    // ========================================================================
    // 2. ROS 节点初始化
    // ========================================================================
    ros::init(argc, argv, "sonar_driver");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // 读取 ROS 参数
    std::string image_topic = pnh.param<std::string>("image_topic", "/forward_sonar/image");
    std::string frame_id    = pnh.param<std::string>("frame_id", "sonar_frame");
    double      publish_rate = pnh.param("publish_rate", 10.0);
    bool        flip_horizontal = pnh.param("flip_horizontal", false);
    bool        flip_vertical   = pnh.param("flip_vertical", false);

    // ========================================================================
    // 3. 初始化 SDK 并连接声呐
    // ========================================================================
    ForwardlookingSonarSDK sonar;

    if (!sonar.init()) {
        ROS_ERROR("[sonar_driver] SDK init failed! Check setting.ini and SDK environment.");
        return -1;
    }
    ROS_INFO("[sonar_driver] SDK initialized.");

    if (!sonar.connectSonar()) {
        ROS_ERROR("[sonar_driver] Connect sonar failed! Check network and sonar power.");
        return -1;
    }
    ROS_INFO("[sonar_driver] Sonar connected.");

    // 设置声呐参数
    double range_m = pnh.param("range", 30.0);
    int    gain    = pnh.param("gain", 50);
    int    mode    = pnh.param("mode", 1);  // 1=LF(750kHz), 2=HF(1.2MHz)

    sonar.changeRange(range_m);
    sonar.changeGain(gain);
    sonar.changeSonarMode(mode);
    ROS_INFO("[sonar_driver] Range=%.1fm, Gain=%d%%, Mode=%d", range_m, gain, mode);

    // ========================================================================
    // 4. 图像发布器
    // ========================================================================
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>(image_topic, 2);

    // ========================================================================
    // 5. 使用 QTimer 驱动主循环（兼容 Qt 事件处理 + ROS spin）
    //    SDK 内部的网络数据接收依赖 Qt 事件循环
    // ========================================================================
    QTimer timer;
    QObject::connect(&timer, &QTimer::timeout, [&]() {
        ros::spinOnce();

        // 检查 Sonarsurface 中是否有新数据
        if (sonar.pSonarSurface == nullptr) {
            ROS_WARN_THROTTLE(2.0, "[sonar_driver] Sonar surface not ready.");
            return;
        }

        Sonarsurface* surface = sonar.pSonarSurface;
        int n_ranges = static_cast<int>(surface->m_nRngs);
        int n_beams  = static_cast<int>(surface->m_nBrgs);

        if (n_ranges <= 0 || n_beams <= 0 || surface->m_pData == nullptr) {
            ROS_WARN_THROTTLE(2.0, "[sonar_driver] No valid sonar image data yet "
                                    "(ranges=%d, beams=%d, data=%p).",
                              n_ranges, n_beams, static_cast<void*>(surface->m_pData));
            return;
        }

        // Sonarsurface::m_pData 排列: row-major, 行=距离(range), 列=方位(beam)
        // 行顺序为从近到远 (near → far)，列顺序为从左到右 (port → starboard)
        cv::Mat raw(n_ranges, n_beams, CV_8UC1, surface->m_pData);

        // 注意：forward_sonar_target_detector.py 中图像的约定为：
        //   - 顶行 = 最远距离 (max_range)
        //   - 底行 = 最近距离 (min_range)
        // 如果声呐数据是从近到远排列，需要翻转行
        cv::Mat image;
        cv::flip(raw, image, 0);  // 垂直翻转，使顶行=远场

        if (flip_horizontal) {
            cv::flip(image, image, 1);
        }
        if (flip_vertical) {
            cv::flip(image, image, 0);
        }

        // 转换为 ROS 图像消息并发布
        sensor_msgs::ImagePtr msg =
            cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::MONO8, image)
                .toImageMsg();
        msg->header.stamp  = ros::Time::now();
        msg->header.frame_id = frame_id;
        image_pub.publish(msg);

        // 重置新数据标志位
        surface->m_newImgData = false;
    });

    timer.start(static_cast<int>(1000.0 / publish_rate));  // ms
    ROS_INFO("[sonar_driver] Publishing sonar images on topic '%s' at %.1f Hz.",
             image_topic.c_str(), publish_rate);

    // ========================================================================
    // 6. 进入 Qt 事件循环
    // ========================================================================
    int ret = app.exec();

    // ========================================================================
    // 7. 清理
    // ========================================================================
    sonar.disconnectSonar();
    ROS_INFO("[sonar_driver] Sonar disconnected. Node exiting.");
    return ret;
}
