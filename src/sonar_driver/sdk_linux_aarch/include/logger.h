#ifndef LOGGER_H
#define LOGGER_H

#include <QObject>
#include <QFile>

// 声纳发来的前置信息
struct LogHeader
{
public:
  unsigned       fileHeader;   // 4字节标头标识类型
  unsigned       sizeHeader;   // 当前结构体大小
  char           source[16];   // 最多12个字符的源标识符
  unsigned short version;      // 文件版本类型
  unsigned short encryption;   // 加密样式 (0 = none)
  qint64         key;          // 尽可能保存加密密钥 (0 otherwise)
  double         time;         // 文件创建时间
};

struct LogItem
{
public:
  unsigned       itemHeader;   // 4字节标头标识类型
  unsigned       sizeHeader;   // 当前结构体大小
  unsigned short type;         // 所含数据类型的标识
  unsigned short version;      // 数据类型的版本
  double         time;         // 文件创建时间
  unsigned short compression;  // 压缩类型 0 = none, 1 = qCompress
  unsigned       originalSize; // 压缩前有效数据的大小
  unsigned       payloadSize;  // 压缩后有效数据大小
};

// 日志的当前状态
enum logState
{
  notLogging,
  opening,
  closing,
  logging
};

// 识别可能的有效数据类型-记录器可以接受任何有效数据
// 但是此记录列表可用于分配类型id以避免冲突
enum RecordTypes
{
    rt_settings          = 1,             // SettingsLogger包
    rt_serialPort        = 2,             // 原始串行字符串——版本包含端口号
    rt_oculusSonar       = 10,            // 原始oculus声纳数据
    rt_blueviewSonar     = 11,            // 原始Blueview数据日志图像
    rt_rawVideo          = 12,            // 原始视频日志
    rt_h264Video         = 13,            // H264压缩视频日志
    rt_apBattery         = 14,            // ApBattery结构体
    rt_apMissionProgress = 15,            // ApMissionProgress结构体
    rt_nortekDVL         = 16,            // 完整的Nortek DVL结构体
    rt_apNavData         = 17,            // ApNavData结构体
    rt_apDvlData         = 18,            // ApDvlData结构体
    rt_apAhrsData        = 19,            // ApAhrsData结构体
    rt_apSonarHeader     = 20,            // ApSonarHeader数据头
    rt_rawSonarImage     = 21,            // 原始声纳图像
    rt_ahrsMtData2       = 22,            // XSens MtData2数据
    rt_apVehicleInfo     = 23,            // Artemis ApVehicleInfo结构体
    rt_apMarker          = 24,            // ApMarker结构体
    rt_apGeoImageHeader  = 25,            // ApGeoImageHeader
    rt_apGeoImageData    = 26,            // 图像的ApGeoImage数据
    rt_sbgData           = 30,            // SBG导航数据信息
    rt_ocViewInfo		 = 500			  // Oculus视图信息
};
class Logger : public QObject
{
    Q_OBJECT
public:
    explicit Logger(QObject *parent = 0);
    ~Logger();

    //---------------------------函数------------------------------
    void      OpenFile(QString filename);
    void      SetupEncryption(quint16 encryption, quint64 key);
    logState GetLogState();
    bool      LogIsActive();
    //------------------------------------------------------------

    //---------------------------变量------------------------------
    QString   m_logDir;        // 日志目录
    uint32_t  m_logMaxSize;    // 日志文件的最大大小
    uint32_t  m_logCurrMaxSize;
    QString   m_ext;           // 日志文件的拓展名
    QString   m_fileName;      // 正记录日志的日志文件名
    QFile     m_file;          // 文件
    logState  m_state;         // 日志的状态
    quint16   m_encryption;    // 使用的加密级别
    quint64   m_key;           // 要使用的加密密钥
    unsigned  m_nRecords;      // 当前文件中记录的记录数
    quint64   m_loggedSize;    // 记录到当前记录的字节数

    quint64   m_maxRecords;    // 打开新文件前记录的最大记录数
    quint64   m_maxSize;       // 打开新文件前记录的最大字节数

    static const unsigned s_fileHeader;
    static const unsigned s_itemHeader;
    static char     s_source[16];
    //------------------------------------------------------------

signals:

public slots:
    void OpenLog();
    void CloseLog();
    void SetLogDirectory(QString dir);
    void SetMaxLogSize(uint32_t size);
    void LogData(unsigned short type, unsigned short version, bool compress, unsigned size, unsigned char* pData);

};

#endif // LOGGER_H
