/**
 * @file sonar_image_pub.cpp
 * @brief ROS node for publishing forward-looking sonar images using the
 *        BHSonar / ForwardlookingSonarSDK.
 *
 * Connects to a multibeam imaging sonar via TCP, receives ping data, converts
 * the raw sonar image to sensor_msgs::Image (MONO8), and publishes on
 * /forward_sonar/image (configurable).
 */

#include "sonar_image_pub.h"

#include <ros/package.h>

#include <QCoreApplication>
#include <QString>
#include <QMutexLocker>

#include "utils.h"

// ============================================================================
//  Constructor / Destructor
// ============================================================================

SonarImagePublisher::SonarImagePublisher(QObject* parent)
    : QObject(parent)
    , nh_()
    , pnh_("~")
    , it_(nullptr)
    , sonar_client_(nullptr)
    , sonar_status_(nullptr)
    , ros_spin_timer_(nullptr)
    , discovery_timer_(nullptr)
    , reconnect_timer_(nullptr)
    , connected_(false)
    , discovery_done_(false)
    , auto_discovery_mode_(false)
    , reconnect_attempts_(0)
    , cart_map_ready_(false)
    , last_nbeams_(0)
    , last_nranges_(0)
    , last_range_res_(0.0)
{
    loadParameters();

    it_ = new image_transport::ImageTransport(nh_);
    image_pub_ = it_->advertise(image_topic_, 1);

    // Pre-allocate reusable Image message header
    img_msg_ = boost::make_shared<sensor_msgs::Image>();
}

SonarImagePublisher::~SonarImagePublisher()
{
    ROS_INFO("SonarImagePublisher: shutting down");

    if (ros_spin_timer_) {
        ros_spin_timer_->stop();
    }
    if (discovery_timer_) {
        discovery_timer_->stop();
    }
    if (reconnect_timer_) {
        reconnect_timer_->stop();
    }

    if (sonar_client_) {
        if (sonar_client_->IsOpen()) {
            sonar_client_->Disconnect();
        }
    }

    delete it_;
    it_ = nullptr;
}

// ============================================================================
//  init() — SDK component creation & signal wiring
// ============================================================================

bool SonarImagePublisher::init()
{
    ROS_INFO("SonarImagePublisher: initializing...");

    // 1. Create SDK components
    sonar_client_ = new ClientCtrl();
    sonar_status_ = new OsStatusRx();

    // 2. Wire signals from ReadThread (cross-thread, Qt auto-queues them)
    connect(&sonar_client_->readData, &ReadThread::NewReturnFire,
            this, &SonarImagePublisher::onNewReturnFire);
    connect(&sonar_client_->readData, &ReadThread::socketDisconnect,
            this, &SonarImagePublisher::onSocketDisconnected);
    connect(&sonar_client_->readData, &ReadThread::socketTimeout,
            this, &SonarImagePublisher::onSocketTimeout);
    connect(&sonar_client_->readData, &ReadThread::socketReconnected,
            this, &SonarImagePublisher::onSocketReconnected);
    connect(&sonar_client_->readData, &ReadThread::NotifyConnectionFailed,
            this, &SonarImagePublisher::onConnectionFailed);

    // 3. Wire signal from OsStatusRx (UDP status)
    connect(sonar_status_, &OsStatusRx::NewStatusMsg,
            this, &SonarImagePublisher::onSonarStatus);

    // 4. Create Cartesian image publisher
    cartesian_pub_ = it_->advertise(cartesian_topic_, 1);

    // 5. ROS spin timer (10 ms = 100 Hz)
    ros_spin_timer_ = new QTimer(this);
    connect(ros_spin_timer_, &QTimer::timeout,
            this, &SonarImagePublisher::onRosSpinTimer);
    ros_spin_timer_->start(10);

    // 6. Reconnect timer (not started until disconnection)
    reconnect_timer_ = new QTimer(this);
    reconnect_timer_->setSingleShot(true);
    connect(reconnect_timer_, &QTimer::timeout,
            this, &SonarImagePublisher::onReconnectTimer);

    // 7. Decide connection strategy
    if (!sonar_ip_.empty()) {
        // Direct-connect mode
        auto_discovery_mode_ = false;
        ROS_INFO("SonarImagePublisher: direct-connect mode, IP = %s",
                 sonar_ip_.c_str());
        connectToSonar(QString::fromStdString(sonar_ip_), 0);
    } else {
        // Auto-discovery mode
        auto_discovery_mode_ = true;
        ROS_INFO("SonarImagePublisher: auto-discovery mode, timeout = %d s",
                 discovery_timeout_);
        startDiscovery();
    }

    ROS_INFO("SonarImagePublisher: initialization complete");
    return true;
}

// ============================================================================
//  Parameter loading
// ============================================================================

void SonarImagePublisher::loadParameters()
{
    pnh_.param<std::string>("image_topic",      image_topic_,      "/forward_sonar/image_raw");
    pnh_.param<std::string>("cartesian_topic",  cartesian_topic_,  "/forward_sonar/image");
    pnh_.param<std::string>("frame_id",          frame_id_,         "sonar_frame");
    pnh_.param<double>     ("publish_rate",      publish_rate_,     10.0);
    pnh_.param<std::string>("sonar_ip",          sonar_ip_,         "");
    pnh_.param<double>     ("range",             range_,            1.0);
    pnh_.param<int>        ("gain",              gain_,             20);
    pnh_.param<int>        ("mode",              mode_,             2);    // 1=LF, 2=HF
    pnh_.param<bool>       ("gain_assist",       gain_assist_,      true);
    pnh_.param<int>        ("gamma",             gamma_,            150);
    pnh_.param<int>        ("net_speed_limit",   net_speed_limit_,  100);
    pnh_.param<double>     ("speed_of_sound",    speed_of_sound_,   1510.0);
    pnh_.param<double>     ("salinity",          salinity_,         0.0);
    pnh_.param<int>        ("discovery_timeout", discovery_timeout_, 10);
    pnh_.param<bool>       ("flip_horizontal",   flip_horizontal_,  false);
    pnh_.param<bool>       ("flip_vertical",     flip_vertical_,    false);
    pnh_.param<int>        ("output_width",      output_width_,     512);
    pnh_.param<int>        ("output_height",     output_height_,    256);
    pnh_.param<double>     ("output_min_range",  output_min_range_, 0.2);

    ROS_INFO("SonarImagePublisher: parameters loaded");
    ROS_INFO("  image_topic    = %s",   image_topic_.c_str());
    ROS_INFO("  cartesian_topic= %s",   cartesian_topic_.c_str());
    ROS_INFO("  frame_id       = %s",   frame_id_.c_str());
    ROS_INFO("  sonar_ip       = %s",   sonar_ip_.empty() ? "(auto-discovery)" : sonar_ip_.c_str());
    ROS_INFO("  range          = %.1f", range_);
    ROS_INFO("  gain           = %d",   gain_);
    ROS_INFO("  mode           = %d (%s)", mode_, (mode_ == 1 ? "LF/750kHz" : "HF/1.2MHz"));
    ROS_INFO("  gain_assist    = %s",   gain_assist_ ? "true" : "false");
    ROS_INFO("  gamma          = %d",   gamma_);
    ROS_INFO("  net_speed_limit= %d",   net_speed_limit_);
    ROS_INFO("  speed_of_sound = %.1f", speed_of_sound_);
    ROS_INFO("  salinity       = %.1f", salinity_);
    ROS_INFO("  disc_timeout   = %d s", discovery_timeout_);
    ROS_INFO("  flip_h         = %s",   flip_horizontal_ ? "true" : "false");
    ROS_INFO("  flip_v         = %s",   flip_vertical_ ? "true" : "false");
    ROS_INFO("  output_size    = %dx%d", output_width_, output_height_);
    ROS_INFO("  output_min_rng = %.1f", output_min_range_);
}

// ============================================================================
//  ROS Spin Timer
// ============================================================================

void SonarImagePublisher::onRosSpinTimer()
{
    if (ros::ok()) {
        ros::spinOnce();
    } else {
        ROS_INFO("SonarImagePublisher: ROS shutdown, quitting Qt loop");
        if (sonar_client_ && sonar_client_->IsOpen()) {
            sonar_client_->Disconnect();
        }
        QCoreApplication::quit();
    }
}

// ============================================================================
//  Connection helpers
// ============================================================================

void SonarImagePublisher::connectToSonar(const QString& ip, uint32_t deviceId)
{
    if (connected_) {
        ROS_WARN("SonarImagePublisher: already connected, skipping");
        return;
    }

    ROS_INFO("SonarImagePublisher: connecting to sonar at %s (deviceId=%u)...",
             ip.toStdString().c_str(), deviceId);

    sonar_client_->hostname  = ip;
    sonar_client_->mask      = "255.255.255.0";
    sonar_client_->DevieceId = static_cast<uint16_t>(deviceId & 0xFFFF);

    if (sonar_client_->Connect()) {
        connected_ = true;
        reconnect_attempts_ = 0;
        ROS_INFO("SonarImagePublisher: TCP connection established");

        // Send initial fire command to start pinging
        sendFireCommand();
    } else {
        ROS_ERROR("SonarImagePublisher: failed to connect to %s",
                  ip.toStdString().c_str());
        connected_ = false;

        if (auto_discovery_mode_) {
            // Retry discovery
            discovery_done_ = false;
            startDiscovery();
        } else {
            // Direct mode: schedule reconnect
            startReconnect();
        }
    }
}

void SonarImagePublisher::sendFireCommand()
{
    if (!sonar_client_ || !sonar_client_->IsOpen()) {
        ROS_WARN_THROTTLE(5.0, "SonarImagePublisher: cannot send Fire — not connected");
        return;
    }

    sonar_client_->Fire(
        mode_,                 // 1=LF(750kHz), 2=HF(1.2MHz)
        range_,                // range in meters
        static_cast<double>(gain_),  // gain 0-100
        speed_of_sound_,       // m/s
        salinity_,             // ppt
        gain_assist_,          // gain assist on/off
        static_cast<uint8_t>(gamma_),
        static_cast<uint8_t>(net_speed_limit_)
    );
}

void SonarImagePublisher::startDiscovery()
{
    // Start listening for UDP status messages
    sonar_status_->ReadDatagrams();

    // Set up discovery timeout
    if (!discovery_timer_) {
        discovery_timer_ = new QTimer(this);
        discovery_timer_->setSingleShot(true);
        connect(discovery_timer_, &QTimer::timeout,
                this, &SonarImagePublisher::onDiscoveryTimeout);
    }

    discovery_done_ = false;
    discovered_sonars_.clear();
    discovery_timer_->start(discovery_timeout_ * 1000);

    ROS_INFO("SonarImagePublisher: listening for sonar status messages "
             "(timeout=%ds)...", discovery_timeout_);
}

void SonarImagePublisher::startReconnect()
{
    if (reconnect_attempts_ >= MAX_RECONNECT_ATTEMPTS) {
        ROS_ERROR("SonarImagePublisher: max reconnect attempts (%d) reached, "
                  "giving up", MAX_RECONNECT_ATTEMPTS);
        return;
    }

    // Exponential backoff: 1s, 2s, 4s, 8s, ... capped at 30s
    int delay_ms = std::min(1000 * (1 << reconnect_attempts_), 30000);
    reconnect_attempts_++;

    ROS_WARN("SonarImagePublisher: reconnect attempt %d/%d in %d ms",
             reconnect_attempts_, MAX_RECONNECT_ATTEMPTS, delay_ms);

    reconnect_timer_->start(delay_ms);
}

// ============================================================================
//  UDP Sonar Status (discovery)
// ============================================================================

void SonarImagePublisher::onSonarStatus(BHSonarStatusMsg osm,
                                         quint16 valid, quint16 invalid)
{
    Q_UNUSED(valid);
    Q_UNUSED(invalid);

    QString ipStr = Utils::FormatIpAddress(osm.ipAddr);

    ROS_INFO_THROTTLE(10.0,
        "SonarImagePublisher: discovered sonar — deviceId=%u partNumber=%u IP=%s",
        osm.deviceId, osm.partNumber, ipStr.toStdString().c_str());

    // Store the discovered sonar
    discovered_sonars_[osm.deviceId] = osm;

    // If in direct-connect mode and we get a matching IP, connect immediately
    if (!auto_discovery_mode_ && !connected_ && !discovery_done_) {
        if (ipStr.toStdString() == sonar_ip_) {
            discovery_done_ = true;
            connectToSonar(ipStr, osm.deviceId);
        }
    }
}

void SonarImagePublisher::onDiscoveryTimeout()
{
    if (discovery_done_ || connected_) {
        return;
    }

    if (discovered_sonars_.isEmpty()) {
        ROS_WARN("SonarImagePublisher: no sonar discovered within %d s, "
                 "will keep listening...", discovery_timeout_);
        // Restart discovery to keep listening
        discovered_sonars_.clear();
        discovery_timer_->start(discovery_timeout_ * 1000);
        return;
    }

    // Pick the first discovered sonar
    BHSonarStatusMsg osm = discovered_sonars_.first();
    QString ipStr = Utils::FormatIpAddress(osm.ipAddr);

    ROS_INFO("SonarImagePublisher: discovery complete — selecting sonar at %s "
             "(deviceId=%u, partNumber=%u)",
             ipStr.toStdString().c_str(), osm.deviceId, osm.partNumber);

    discovery_done_ = true;
    connectToSonar(ipStr, osm.deviceId);
}

void SonarImagePublisher::onReconnectTimer()
{
    if (connected_) {
        return;
    }

    if (!sonar_ip_.empty()) {
        // Direct mode: retry with known IP
        connectToSonar(QString::fromStdString(sonar_ip_), 0);
    } else if (!discovered_sonars_.isEmpty()) {
        // Auto-discovery: use last known sonar
        BHSonarStatusMsg osm = discovered_sonars_.first();
        QString ipStr = Utils::FormatIpAddress(osm.ipAddr);
        connectToSonar(ipStr, osm.deviceId);
    } else {
        // Nothing to reconnect to, restart discovery
        startDiscovery();
    }
}

// ============================================================================
//  Core: New Return Fire (sonar image data)
// ============================================================================

void SonarImagePublisher::onNewReturnFire(BufferEntry* pEntry)
{
    if (!pEntry) {
        ROS_WARN_THROTTLE(1.0, "SonarImagePublisher: null BufferEntry");
        return;
    }

    // Lock the entry mutex for consistent read
    QMutexLocker locker(&pEntry->mutex);

    // --- Extract image metadata ---
    uint16_t nRanges = 0;
    uint16_t nBeams  = 0;
    DataSizeType dataSize = dataSize8Bit;

    // The 'simple' flag indicates v1 simple ping result format
    // rfm fields are the most reliable for image dimensions
    if (pEntry->rfm.nRanges > 0 && pEntry->rfm.nBeams > 0) {
        nRanges  = pEntry->rfm.nRanges;
        nBeams   = pEntry->rfm.nBeams;
        dataSize = pEntry->rfm.dataSize;
    } else if (pEntry->rfm2.nRanges > 0 && pEntry->rfm2.nBeams > 0) {
        nRanges  = pEntry->rfm2.nRanges;
        nBeams   = pEntry->rfm2.nBeams;
        dataSize = pEntry->rfm2.dataSize;
    } else {
        ROS_WARN_THROTTLE(1.0, "SonarImagePublisher: no valid image dimensions "
                          "in BufferEntry");
        locker.unlock();
        sendFireCommand();  // Keep sonar alive
        return;
    }

    // --- Validate dimensions ---
    if (nRanges == 0 || nBeams == 0 ||
        nRanges > 4096 || nBeams > 2048) {
        ROS_WARN_THROTTLE(1.0, "SonarImagePublisher: invalid image dimensions "
                          "(%u x %u)", nRanges, nBeams);
        locker.unlock();
        sendFireCommand();
        return;
    }

    // --- Validate image data pointer ---
    if (!pEntry->pImage) {
        ROS_WARN_THROTTLE(1.0, "SonarImagePublisher: null pImage in BufferEntry");
        locker.unlock();
        sendFireCommand();
        return;
    }

    // --- Determine image size and type ---
    size_t expectedSize = static_cast<size_t>(nRanges) * nBeams;
    int cvType = CV_8UC1;

    if (dataSize == dataSize16Bit) {
        expectedSize *= 2;
        cvType = CV_16UC1;
    }
    // dataSize24Bit and dataSize32Bit are not handled — unlikely for image data

    // --- Deep-copy image data from the ring buffer ---
    // CRITICAL: BufferEntry belongs to ReadThread's ring buffer.
    // We must COPY the data, not wrap the pointer.
    std::vector<uchar> imgBuffer(expectedSize);
    memcpy(imgBuffer.data(), pEntry->pImage, expectedSize);

    locker.unlock();  // Done with the shared buffer entry

    // --- Build OpenCV Mat wrapping the copied buffer ---
    cv::Mat img(nRanges, nBeams, cvType, imgBuffer.data());

    // If 16-bit, convert to 8-bit for ROS MONO8 publishing
    if (dataSize == dataSize16Bit) {
        cv::Mat img8;
        img.convertTo(img8, CV_8UC1, 1.0 / 256.0);  // scale 16-bit → 8-bit
        img = img8;  // img8 owns its data, img now refers to it
    } else {
        // Clone to own the data (img currently points into imgBuffer which is stack-local)
        img = img.clone();
    }

    // --- Apply flips ---
    if (flip_horizontal_) {
        cv::flip(img, img, 1);  // left-right
    }
    if (flip_vertical_) {
        cv::flip(img, img, 0);  // up-down
    }

    // --- Build and publish ROS Image message ---
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp    = ros::Time::now();
    cv_img.header.frame_id = frame_id_;
    cv_img.encoding        = sensor_msgs::image_encodings::MONO8;
    cv_img.image           = img;

    image_pub_.publish(cv_img.toImageMsg());

    // --- Build Cartesian map & publish transformed image ---
    double rangeResolution = 0.0;
    short* pBrgs = pEntry->pBrgs;  // bearing array (0.01 deg, nBeams entries)
    if (pEntry->simple) {
        rangeResolution = pEntry->rfm.rangeResolution;
    } else {
        rangeResolution = pEntry->rfm2.rangeResolution;
    }

    if (pBrgs && nBeams > 0 && rangeResolution > 0.0) {
        // Rebuild map if sonar config changed
        if (!cart_map_ready_ ||
            nBeams != static_cast<uint16_t>(last_nbeams_) ||
            nRanges != last_nranges_ ||
            std::abs(rangeResolution - last_range_res_) > 1e-6) {
            buildCartesianMap(nRanges, nBeams, pBrgs, rangeResolution, range_);
        }
        publishCartesian(img);
    }

    // --- Send next fire command to keep sonar alive ---
    sendFireCommand();
}

// ============================================================================
//  Connection state change handlers
// ============================================================================

void SonarImagePublisher::onSocketDisconnected()
{
    ROS_WARN("SonarImagePublisher: socket disconnected");
    connected_ = false;

    if (!auto_discovery_mode_) {
        startReconnect();
    } else {
        discovery_done_ = false;
        startDiscovery();
    }
}

void SonarImagePublisher::onSocketTimeout()
{
    ROS_WARN("SonarImagePublisher: socket timeout");
    connected_ = false;

    if (!auto_discovery_mode_) {
        startReconnect();
    } else {
        discovery_done_ = false;
        startDiscovery();
    }
}

void SonarImagePublisher::onSocketReconnected()
{
    ROS_INFO("SonarImagePublisher: socket reconnected");
    connected_ = true;
    reconnect_attempts_ = 0;

    // Re-send configuration and fire command
    sendFireCommand();
}

void SonarImagePublisher::onConnectionFailed(QString error)
{
    ROS_ERROR("SonarImagePublisher: connection failed — %s",
              error.toStdString().c_str());
    connected_ = false;

    if (!auto_discovery_mode_) {
        startReconnect();
    } else {
        discovery_done_ = false;
        // Keep listening for new sonar broadcasts
        if (!discovery_timer_ || !discovery_timer_->isActive()) {
            startDiscovery();
        }
    }
}

// ============================================================================
//  Polar-to-Cartesian Transformation
// ============================================================================

void SonarImagePublisher::buildCartesianMap(uint16_t nRanges, uint16_t nBeams,
                                             const short* pBrgs,
                                             double rangeResolution,
                                             double maxRange)
{
    int W = output_width_;
    int H = output_height_;
    double minRange = output_min_range_;

    // Compute FOV from the bearing array (in degrees)
    double fovHalf = std::abs(static_cast<double>(pBrgs[nBeams - 1] - pBrgs[0]))
                     / 200.0;  // 0.01 deg units → deg, then half
    double fovDeg = fovHalf * 2.0;

    // lateral span at max range (constant, matching simulation)
    double lateralMax = maxRange * std::sin(fovHalf * M_PI / 180.0);

    ROS_INFO("SonarImagePublisher: building Cartesian map %dx%d, "
             "fov=%.1f deg, max_range=%.1f m, nBeams=%d, nRanges=%d",
             W, H, fovDeg, maxRange, nBeams, nRanges);

    cart_map_x_ = cv::Mat(H, W, CV_32FC1);
    cart_map_y_ = cv::Mat(H, W, CV_32FC1);

    for (int row = 0; row < H; ++row) {
        // forward: maxRange (top/row 0) → minRange (bottom/row H-1)
        double forward = maxRange -
            (static_cast<double>(row) / (H - 1)) * (maxRange - minRange);

        for (int col = 0; col < W; ++col) {
            // lateral: +lateralMax (left/col 0) → -lateralMax (right/col W-1)
            double lateral = lateralMax -
                (static_cast<double>(col) / (W - 1)) * 2.0 * lateralMax;

            // Check if within fan footprint
            double maxLateralAtRange = forward * std::tan(fovHalf * M_PI / 180.0);
            if (std::abs(lateral) > maxLateralAtRange) {
                cart_map_x_.at<float>(row, col) = -1.0f;
                cart_map_y_.at<float>(row, col) = -1.0f;
                continue;
            }

            // Convert Cartesian → polar
            double rng = std::hypot(forward, lateral);
            double bearingDeg = std::atan2(lateral, forward) * 180.0 / M_PI;

            // Map bearing to beam index
            // pBrgs is in 0.01 deg units, find the nearest beam
            short bearingRaw = static_cast<short>(bearingDeg * 100.0);
            float beamIdx = 0.0f;
            if (bearingRaw <= pBrgs[0]) {
                beamIdx = 0.0f;
            } else if (bearingRaw >= pBrgs[nBeams - 1]) {
                beamIdx = static_cast<float>(nBeams - 1);
            } else {
                // Linear search for nearest bearing
                for (int b = 0; b < static_cast<int>(nBeams) - 1; ++b) {
                    if (bearingRaw >= pBrgs[b] && bearingRaw <= pBrgs[b + 1]) {
                        // Linear interpolation between beams
                        float t = static_cast<float>(bearingRaw - pBrgs[b]) /
                                  static_cast<float>(pBrgs[b + 1] - pBrgs[b]);
                        beamIdx = static_cast<float>(b) + t;
                        break;
                    }
                }
            }

            // Map range to range index
            // Raw image: row 0 = nearest range, row N-1 = farthest range
            // rangeResolution is meters per range line
            float rangeIdx = static_cast<float>(rng / rangeResolution);

            cart_map_x_.at<float>(row, col) = beamIdx;
            cart_map_y_.at<float>(row, col) = rangeIdx;
        }
    }

    cart_map_ready_ = true;
    last_nbeams_ = static_cast<short>(nBeams);
    last_nranges_ = nRanges;
    last_range_res_ = rangeResolution;

    ROS_INFO("SonarImagePublisher: Cartesian map ready");
}

void SonarImagePublisher::publishCartesian(const cv::Mat& rawImage)
{
    if (!cart_map_ready_) return;
    if (cart_map_x_.empty() || cart_map_y_.empty()) return;

    cv::Mat cartesian(output_height_, output_width_, CV_8UC1, cv::Scalar(0));

    // cv::remap: map_x = source column (beam), map_y = source row (range)
    cv::remap(rawImage, cartesian, cart_map_x_, cart_map_y_,
              cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    // --- Build and publish ROS Image message ---
    cv_bridge::CvImage cv_img;
    cv_img.header.stamp    = ros::Time::now();
    cv_img.header.frame_id = frame_id_;
    cv_img.encoding        = sensor_msgs::image_encodings::MONO8;
    cv_img.image           = cartesian;

    cartesian_pub_.publish(cv_img.toImageMsg());
}

// ============================================================================
//  main()
// ============================================================================

int main(int argc, char** argv)
{
    // 1. ROS init (must be before QCoreApplication)
    ros::init(argc, argv, "sonar_image_pub",
              ros::init_options::NoSigintHandler);

    // 2. Qt event loop (QCoreApplication — no GUI needed)
    QCoreApplication app(argc, argv);

    // 3. Create and initialize the publisher node
    SonarImagePublisher node;
    if (!node.init()) {
        ROS_FATAL("SonarImagePublisher: init() failed");
        return 1;
    }

    ROS_INFO("Sonar image publisher node started — entering Qt event loop");

    // 4. Enter Qt event loop
    //    - ReadThread signals → our image processing slots
    //    - OsStatusRx UDP datagrams → our status slots
    //    - ros_spin_timer_ → ros::spinOnce()
    //    - discovery_timer_ → auto-discovery timeout
    //    - reconnect_timer_ → delayed reconnect attempts
    int ret = app.exec();

    ros::shutdown();
    return ret;
}
