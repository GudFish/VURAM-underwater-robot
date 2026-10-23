#ifndef SCREENRECORDER_H
#define SCREENRECORDER_H

#include <QObject>
#include <QScreen>
#include <QTimer>
#include <QProcess>
#include <QImage>

class ScreenRecorder : public QObject
{
    Q_OBJECT
public:
    explicit ScreenRecorder(QObject *parent = nullptr);
    ~ScreenRecorder();

    void startRecording(const QString &outputFile, int fps, QString videoEncode);
    void stopRecording();

signals:
    void error(const QString &message);

private slots:
    void captureFrame();

private:
    QScreen *m_screen;
    QTimer *m_timer;
    QProcess *m_ffmpeg;
    int m_fps;
    bool m_isRecording;

};

#endif // SCREENRECORDER_H


