#ifndef SETTINGSCTRLS_H
#define SETTINGSCTRLS_H

#include <QObject>

class SettingsCtrls : public QObject
{
    Q_OBJECT

public:
    explicit SettingsCtrls(QObject *parent = nullptr);
    ~SettingsCtrls();

    // 日志路径
    QString logDir;

    // 日志大小
    uint32_t logSize;

    // 设置日志路径
    void SelectLogDirectory();

    // 设置日志文件最大大小
    void SetLogMaxSize(int logSize);

    // 录屏编码
    QString videoEncode;

    // 录屏后缀
    QString videoName;

    // 改变日志路径
    void NewLogDirectory(QString dir);

    // 改变最大日志文件大小
    void MaxLogSize(uint32_t size);

    // 改变录屏编码
    void ChangeVideoEncode(QString encode);

    // 改变录屏后缀
    void ChangeVideoName(QString filename);

};

#endif // SETTINGSCTRLS_H
