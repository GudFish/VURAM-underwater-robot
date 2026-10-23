#pragma once

#include <QDateTime>
#include <QObject>
#include <QMetaObject>
#include "bhsonar.h"

class QUdpSocket;

// Stores the last status message of a sonat witha given id
class COsStatusSonar : public QObject
{
  Q_OBJECT

public:

  COsStatusSonar();

  unsigned        m_id;              // The id of the sonar
  BHSonarStatusMsg m_osm;             // The last status message associated with this sonar
  QDateTime       m_lastMsgTime;     // The time of the last message
};


// OsStatusRx - a listening socket for oculus status messages

class OsStatusRx : public QObject
{
    Q_OBJECT

public:
    OsStatusRx();
    ~OsStatusRx();

    void ReadDatagrams();

signals:
    void NewStatusMsg(BHSonarStatusMsg osm, quint16 valid, quint16 invalid);

public:
    quint16     m_port;       // Port to listen on
    quint16     m_valid;      // Number of valid status messages
    quint16     m_invalid;    // Number of invalid status messages
    QUdpSocket* m_listener;   // Listening socket for status messages
};


