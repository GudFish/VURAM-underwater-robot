#ifndef SONAR_IMAGE_PUB_H
#define SONAR_IMAGE_PUB_H

#include <QObject>
#include <QTimer>
#include <QMap>
#include <QString>
#include <vector>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "clientctrl.h"
#include "osstatusrx.h"
#include "bhsonar.h"

class SonarImagePublisher : public QObject
{
    Q_OBJECT

public:
    explicit SonarImagePublisher(QObject* parent = nullptr);
    ~SonarImagePublisher();

    bool init();

public slots:
    void onRosSpinTimer();

private slots:
    void onSonarStatus(BHSonarStatusMsg osm, quint16 valid, quint16 invalid);
    void onNewReturnFire(BufferEntry* pEntry);
    void onSocketDisconnected();
    void onSocketTimeout();
    void onSocketReconnected();
    void onConnectionFailed(QString error);
    void onDiscoveryTimeout();
    void onReconnectTimer();

private:
    void loadParameters();
    void sendFireCommand();
    void connectToSonar(const QString& ip, uint32_t deviceId);
    void publishImage(BufferEntry* pEntry);
    void startDiscovery();
    void startReconnect();
    void buildCartesianMap(uint16_t nRanges, uint16_t nBeams,
                           const short* pBrgs, double rangeResolution,
                           double maxRange);
    void publishCartesian(const cv::Mat& rawImage);

    // ---- ROS ----
    ros::NodeHandle              nh_;
    ros::NodeHandle              pnh_;
    image_transport::ImageTransport* it_;
    image_transport::Publisher   image_pub_;
    image_transport::Publisher   cartesian_pub_;

    // ---- SDK components ----
    ClientCtrl*    sonar_client_;
    OsStatusRx*    sonar_status_;

    // ---- Qt timers ----
    QTimer*        ros_spin_timer_;
    QTimer*        discovery_timer_;
    QTimer*        reconnect_timer_;

    // ---- Connection state ----
    bool           connected_;
    bool           discovery_done_;
    bool           auto_discovery_mode_;

    // ---- Discovered sonars (deviceId -> statusMsg) ----
    QMap<uint32_t, BHSonarStatusMsg> discovered_sonars_;

    // ---- ROS parameters (matching launch file defaults) ----
    std::string    image_topic_;
    std::string    cartesian_topic_;
    std::string    frame_id_;
    double         publish_rate_;

    // ---- Cartesian output parameters ----
    int            output_width_;
    int            output_height_;
    double         output_min_range_;

    // ---- SDK configuration parameters ----
    std::string    sonar_ip_;
    double         range_;
    int            gain_;
    int            mode_;           // 1=LF(750kHz), 2=HF(1.2MHz)
    bool           gain_assist_;
    int            gamma_;
    int            net_speed_limit_;
    double         speed_of_sound_;
    double         salinity_;
    int            discovery_timeout_;
    bool           flip_horizontal_;
    bool           flip_vertical_;

    // ---- Reconnect state ----
    int            reconnect_attempts_;
    static const int MAX_RECONNECT_ATTEMPTS = 10;

    // ---- Cartesian remap LUT (computed once, reused) ----
    cv::Mat        cart_map_x_;
    cv::Mat        cart_map_y_;
    bool           cart_map_ready_;
    short          last_nbeams_;       // detect config changes
    uint16_t       last_nranges_;
    double         last_range_res_;

    // ---- Pre-allocated image message ----
    sensor_msgs::ImagePtr img_msg_;
};

#endif  // SONAR_IMAGE_PUB_H
