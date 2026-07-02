#ifndef SONARWIDGETUTILS_H
#define SONARWIDGETUTILS_H

#include <QMatrix4x4>

// 为声纳二维正交窗口添加某些方面工具

// 声纳图像朝向枚举
enum Aspect
{
  none,
  maintainX,
  maintainY,
  maintainXY,
};

class SonarWidgetUtils
{
public:
    SonarWidgetUtils();

    // ------------------------------变量-----------------------------------------
    QMatrix4x4   projection;           // 当前播放项目
    float        originX;              // 原始X值
    float        originY;              // 原始Y值

    Aspect       aspect;               // 朝向

    float        vpLeft;               // 视图左侧
    float        vpRight;              // 视图右侧
    float        vpTop;                // 视图顶端
    float        vpBottom;             // 视图底部
    float        rotation;             // 视图旋转

    float        vpLeftActual;         // 朝向控制后实际视图左侧
    float        vpRightActual;        // 朝向控制后实际视图右侧
    float        vpTopActual;          // 朝向控制后实际视图顶部
    float        vpBottomActual;       // 朝向控制后实际视图底部

    int          width;                // 此正交图的宽度（像素）
    int          height;               // 此正交图的高度（像素）
    float        ppu;                  // 每单位像素数
    float        ppuY;                 // 每单位Y像素
    // --------------------------------------------------------------------------

    // ------------------------------函数-----------------------------------------
    // 根据所需空间配置投影
    void rmgl2dViewport(int w, int h);
    // 设置此显示器的投影
    void rmgl2dProjection(float left, float right, float bottom, float top, Aspect aspect = maintainXY);
    // 根据当前投影将XY转换为屏幕坐标
    void XyToScreen(float x, float y, float& sx, float& sy);
    // 根据当前投影将屏幕坐标转换为XY
    void ScreenToXy(float sx, float sy, float& x, float& y);
    // --------------------------------------------------------------------------
};

#endif // SONARWIDGETUTILS_H
