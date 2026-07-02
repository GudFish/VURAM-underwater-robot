#ifndef CLIENTCTRL_H
#define CLIENTCTRL_H

#include <QObject>
#include <QMutex>
#include <QThread>
#include <QWaitCondition>
#include <QAbstractSocket>
#include <QTimer>
#include <stdint.h>
#include <QByteArray>
#include "bhsonar.h"

#define BUFFER_SIZE 10

class QTcpSocket;

// 该类包含返回消息和嵌入图像
class  BufferEntry
{
public:
    BufferEntry();
    ~BufferEntry();

    // Methods
    void AddRawToEntry(char* pData, quint64 nData);
    void ProcessRaw(char* pData);

    // Data
    BHSonarSimplePingResult  rfm;         // 返回的消息
    BHSonarReturnFireMessage rff;
    BHSonarSimplePingResult2 rfm2;
    uchar*                   pImage;      // 图像数据
    short*                   pBrgs;       //
    QMutex                   mutex;       // 缓存锁

    uint16_t				 version;     // 消息版本标志位
    quint8*                  pRaw;        // 原始数据
    quint32                  rawSize;     // 原始数据记录的大小

    bool					 simple;      // 简单消息标志位
};

class ClientCtrl;

// 该类用于为客户端从网络读取数据的线程
class ReadThread : public QThread
{
    Q_OBJECT

public:
    ReadThread();
    ~ReadThread();

    void run();

    // 启动线程
    void Startup();
    // 如果正在运行，则关闭运行标志并等待线程退出
    void Shutdown();
    // 线程活跃性安全测试
    bool IsActive();
    // 线程活跃性安全设置
    void SetActive(bool activee);
    // 处理rx缓冲区的内容
    void ProcessRxBuffer();
    // 处理完整的有效数据
    void ProcessPayload(char* pData, quint64 nData);

signals:
    void Msg(QString msg);
    void NewReturnFire(BufferEntry* pEntry);
    void NewUserConfig(UserConfig config);
    void NotifyConnectionFailed(QString error);

    void socketTimeout();
    void socketReconnected();
    void socketDisconnect();

public:

    // Data
    ClientCtrl*   pClient;   // 指向父客户端的返回指针
    bool          active;    // 当前是否处于活动状态
    QMutex        mutex;     // 保护锁
    QMutex        sending;   // 保护锁

    QString       hostname;  // 声纳的主机名/地址
    quint16       port;      // 声纳通信端口（目前已固定）
    qint32        nFlushes;  // rx缓冲区必须被刷新的次数

    // 原始接收缓冲区
    char*         pRxBuffer; // 用于接收数据的rx缓冲区
    qint32        nRxMax;    // rx缓冲区的最大大小
    qint32        nRxIn;     // 缓冲区中当前未处理的数据量

    // 消息的接收缓冲区
    BufferEntry   osBuffer[BUFFER_SIZE];
    unsigned      osInject;   // 下一次注入数据的位置

    // 原始发送缓冲区
    QTcpSocket*   pSocket;
    char*         pToSend;
    qint64        nToSend;


    QByteArrayList sendBuffer;
    qint64		   sendCount;

private slots:

    // void socketError(QAbstractSocket::SocketError error);
    void socketDisconnected();
};

// 该类用于与声纳通信
class ClientCtrl : public QObject
{
    Q_OBJECT
public:
    ClientCtrl();
    ~ClientCtrl();

    // 尝试连接到当前主机和端口号
    bool Connect();
    // 如果端口打开，则断开连接
    bool Disconnect();
    // 当前客户端是否打开
    bool IsOpen();
    // 将数据写入Socket
    void WriteToDataSocket(char* pData, quint16 length);
    // 向声纳发送指令
    void Fire(int mode, double range, double gain, double speedOfSound, double salinity, bool gainAssist, uint8_t gammaCorrection, uint8_t netSpeedLimit);
    void DummyMessage();

    void WriteToNetSocket(char* pData, quint16 length);
    void WriteUserConfig(uint32_t ipAddress, uint32_t ipMask, bool dhcpEnable);
    // 查询用户配置
    bool RequestUserConfig();

    bool WaitForReadOrTimeout(uint32_t ms);
    // 交换32位数
    uint32_t bswap_32(uint32_t x);

public:
    QString           hostname;     // 声纳的主机名/地址
    QString           mask;
    uint16_t          DevieceId;    // 设备ID

    QMutex            lock;
    QWaitCondition    wait;
    QMutex	          memLock;

    bool              received;

    QTimer            timeout;

    QByteArray	      readBuffer;

    UserConfig        config;       // 声纳用户配置

    ReadThread        readData;     // 用于读取数据的工作线程
    ReadThread        netsetData;   // 用于发送数据的工作线程

signals:
    void waitSignal();
    void NewUserConfig();

};

#endif // CLIENTCTRL_H
