#pragma once

#include <QObject>
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QVector2D>
#include <QOpenGLShaderProgram>
#include <QDateTime>
#include <QPainter>
#include <QFont>
#include <QWindow>
#include <QStylePainter>
#include <QLabel>
#include "forwardlookingsonarsdk_global.h"

class Surface;

class FORWARDLOOKINGSONARSDK_EXPORT SonarWidget : public QOpenGLWidget
{
    Q_OBJECT
public:
    SonarWidget(QWidget* pParent = 0);
    virtual ~SonarWidget();

    // -----------------------------覆写------------------------------------
    // OpenGL绘制事件
    void paintGL()                                            Q_DECL_OVERRIDE;
    // OpenGL初始化事件
    void initializeGL()                                       Q_DECL_OVERRIDE;
    // OpenGL调整大小事件
    void resizeGL(int w, int h)                               Q_DECL_OVERRIDE;

    void mouseMoveEvent(QMouseEvent* pEvent)                  Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent* pEvent)                 Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent* pEvent)               Q_DECL_OVERRIDE;
    void mouseDoubleClickEvent(QMouseEvent* pEvent)           Q_DECL_OVERRIDE;
    void wheelEvent(QWheelEvent* pEvent)                      Q_DECL_OVERRIDE;

    void paintEvent(QPaintEvent *event)                       Q_DECL_OVERRIDE;

    // -----------------------------变量------------------------------------
    QLabel m_version;
    Surface* m_pSurface;

    // -----------------------------函数------------------------------------
    void RmglSetupSurface(Surface* pSurface);

    void showBranding(bool);

    Surface* RmglGetSurface() { return m_pSurface; }

public slots:
    void OnUpdate();
    void SetCursor(const QCursor& cursor);

};

//#endif // SONARWIDGET_H
