#ifndef PLAYER_H
#define PLAYER_H

#include <QObject>
#include <QFile>

class Player : public QObject
{
    Q_OBJECT
public:
    Player();
    ~Player();

    // ----------------------------函数-------------------------------------
    // 读取文件，执行任何所需的解压缩或解密，并将有效数据作为信号发出
    bool OpenFile(QString filee);
    // 关闭文件
    void CloseFile();
    // 改变文件读写位置
    void Seek(quint64 pos);
    // 读取失败，清空缓存
    bool ReadFailed(QString reason);
    // 打开一个文件并查找给定的文件位置-推出一条记录
    bool OpenFileAt(QString filee, quint64 pos);
    int  CreateTypeIndex(QString file, int type, quint64** nEntries);
    // ---------------------------------------------------------------------

    // ----------------------------变量-------------------------------------
    QFile   file;            // 回放文件
    QString fileName;        // 最新打开的文件名
    quint64 fileSize;        // 读取的文件大小
    quint64 totalBytes;      // 到目前为止从文件读取的数据量
    quint8* pPayloadBuffer;  // 数据加载的字节缓冲区
    bool    repeat;          // 重复标志位
    // ---------------------------------------------------------------------

signals:
    void NewPayload(unsigned short type, unsigned short version, double time, unsigned payloadSize, quint8* pPayload);
    void NewPayloadPos(unsigned short type, unsigned short version, double time, unsigned payloadSize, quint8* pPayload, quint64 pos);
    void EndOfFile();

public slots:
    bool ReadNextItem();
    bool PeekNextItem(double *time);
    bool ReadNextItemOfType(quint16 type);
};

#endif // PLAYER_H
