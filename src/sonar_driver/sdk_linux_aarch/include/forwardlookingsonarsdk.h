#ifndef FORWARDLOOKINGSONARSDK_H
#define FORWARDLOOKINGSONARSDK_H

#include <QObject>
#include "sonarwidget.h"
#include "osstatusrx.h"
#include "clientctrl.h"
#include "logger.h"
#include "player.h"
#include "sonarsurface.h"
#include "forwardlookingsonarsdk_global.h"
#include "bhsonar.h"
#include <QFileDialog>
#include <QSettings>
#include <QListWidgetItem>
#include <QDesktopWidget>
#include <QWaitCondition>
#include <QNetworkInterface>
#include "utils.h"
#include <QApplication>



class FORWARDLOOKINGSONARSDK_EXPORT ForwardlookingSonarSDK: public QObject
{
    Q_OBJECT
public:
    explicit ForwardlookingSonarSDK(QObject *parent = nullptr);
    ~ForwardlookingSonarSDK();

    //---------------------------标志位--------------------------------------
    bool ifinit;                            // 初始化标志位
    bool ifconnect;                         // 连接标志位
    bool ifmeasure;                         // 测量标志位
    bool ifsave;                            // 保存标志位
    bool enableConnect;                     // 连接准许标志位
    bool ifplayback;                        // 回放标志位
    //----------------------------------------------------------------------
    //---------------------------变量----------------------------------------
    // 可连接声纳数
    QMap<uint32_t, BHSonarStatusMsg>     m_sonarList;              // 可用声纳列表
    QMap<uint32_t, QDateTime>            m_statusMessageTimes;     // 状态信息时间
    QMutex                               m_statusMessageTimesLock; // 状态信息时间线程锁
    QMutex                               m_sonarLock;              // 可用声纳列表锁
    BHSonarInfo		                     *m_pSonarInfo;            // 当前声纳信息
    BHSonarPartNumberType		         m_partNumber;             // 声纳型号信息

    QString  logDir;                        // 数据保存位置
    int      logMaxSize;                    // 数据保存最大值
    int      waterType;                     // 环境水质类型
    double   speedOfSound;                  // 环境声速
    double   salinity;                      // 环境盐度值

    int      demandMode;                    // 发送给声纳的模式命令:1(750kHz)、2(1.2MHz)
    int      demandGain;                    // 发送给声纳的增益命令:0-100(%)
    double   demandRange;                   // 发送给声纳的量程命令

    bool     gainAssist;                    // 辅助增益
    uint8_t  gammaCorrection;               // gamma调节:0-255
    uint8_t  netSpeedLimit;                 // 最大网速:10-100

    bool     upDownState;                   // 上下翻转状态
    bool     leftRightState;                // 左右翻转状态

    int      paletteIndex;                  // 颜色版索引： 0-5，默认为1

    bool	 m_timeout;                     // 重连超时标志位

    int      nEntries;                      // 当前回放文件总帧数
    QString  replayFile;                    // 当前回放文件名
    quint64* pEntries;                      // 文件回放位置表
    int      index;                         // 当前回放位置

    ApSonarDataHeader sonarReplay;          // 声纳头数据
    BufferEntry       entry;                // 最新读取的数据
    QTimer            replay;               // 回放定时器
    bool              useRawSonar;          // 声纳数据格式标志位
    QDateTime         payloadDateTime;      // 回放文件时间戳

    int               playSpeed;            // 回放速度
    QString           lastDir;              // 最新回放文件路径
    eDisplayMode      m_displayMode;        // 模式
    //----------------------------------------------------------------------
    //---------------------------对象----------------------------------------
    SonarWidget    *fanDisplay;             // 雷达图显示窗口
    Sonarsurface   *pSonarSurface;          // 声纳图
    ClientCtrl     *sonarClient;            // 网络客户端控制
    OsStatusRx     *sonarStatus;            // UDP雷达状态
    Logger         *logger;                 // 数据记录对象
    Player         *player;                 // 数据回放对象
    //----------------------------------------------------------------------
    //---------------------------函数----------------------------------------
    bool init();                                      // 声纳状态初始化
    bool connectSonar();                              // 连接声纳
    bool disconnectSonar();                           // 断开连接

    bool changeSaveLocation(QString location);        // 改变数据保存位置
    bool changeMaxsize(int size);                     // 改变数据最大保存值（MB）

    bool changeWatertype(int type);                   // 改变水环境类型
    bool changeSpeedofSound(double speed);            // 改变声速

    bool changeSonarMode(int val);                    // 改变声纳频率
    bool changeRange(double val);                     // 改变声纳量程
    bool changeGain(int val);                         // 改变声纳增益

    bool startSaveData();                             // 开始保存数据
    bool stopSaveData();                              // 停止保存数据

    bool startMeasure();                              // 开始测量模式
    bool stopMeasure();                               // 停止测量模式

    bool flipUpDown(bool flip);                       // 上下翻转
    bool flipLeftRight(bool flip);                    // 左右翻转

    bool palSelected(int pal);                        // 设置调色板

    bool changeGainAssist(bool val);                  // 设置辅助增益

    bool changeGammaCorrection(int val);              // 设置gamma调节

    bool changeNetSpeedLimit(int val);                // 设置最大网速

    bool startPlayBack();                             // 开始回放
    bool stopPlayBack();                              // 停止回放

    bool startVideo();                                // 开始播放
    bool stopVideo();                                 // 暂停播放

    bool enableLoopPlay(bool enable);                 // 循环播放设置

    bool setPlayBackSpeed(int speed);                 // 设置播放速度:1、2、4、8

    bool setPlayLocation(int location);               // 设置回放位置

    bool drawRada();                                  // 显示声纳图

    void UpdateSonarInfo(BHSonarPartNumberType pn);   // 更新声纳信息
    void Connect();                                   // 连接
    void Disconnect();                                // 断开连接
    void OpenFileEx(QString fileName);                // 打开文件
    void OpenFile(bool checked);
    void SetDisplayMode(eDisplayMode displayMode);
    //----------------------------------------------------------------------

public slots:
    //---------------------------槽函数---------------------------------------
    // 响应从UDP Socket获取的声纳状态信息
    void NewStatusMsg(BHSonarStatusMsg osm, quint16 valid, quint16 invalid);
    // 获取声纳传来的信息
    void NewReturnFire(BufferEntry* pEntry);
    // 响应获取的回放数据
    void OnNewPayload(unsigned short type, unsigned short version, double time, unsigned payloadSize, quint8* pPayload);
    // 发送简单指令信号
    void FireSonar();
    // 客户端连接超时
    void SocketTimeout();
    // 客户端重连
    void SocketReconnecting();
    // 断开客户端连接
    void SocketDisconnected();
    // 回放下一个
    void PlayNext();
    // 停止回放
    void StopReplay();
    // 开始回放
    void StartReplay();
};

#endif // FORWARDLOOKINGSONARSDK_H
