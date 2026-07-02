#ifndef SONARSURFACE_H
#define SONARSURFACE_H

#include "surface.h"

// 在扇形图上显示声纳数据
class Sonarsurface: public Surface
{
    Q_OBJECT
public:
    Sonarsurface();
    ~Sonarsurface();

    // --------------------------覆写-----------------------------------
    // 渲染场景
    void Render()                                 Q_DECL_OVERRIDE;
    // 初始化OpenGL环境
    void OnCreate()                               Q_DECL_OVERRIDE;
    // 更新声纳图显示
    void OnResize(int w, int h)                   Q_DECL_OVERRIDE;
    // 鼠标移动事件
    void mouseMoveEvent(QMouseEvent* pEvent)      Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent* pEvent)     Q_DECL_OVERRIDE;
    void mouseReleaseEvent(QMouseEvent* pEvent)   Q_DECL_OVERRIDE;

    // -------------------------函数-------------------------------------
    // 创建一个空白声纳图
    void InitTextures();
    // 渲染一个背景图
    void RenderBackground();
    // 渲染声纳图像
    void RenderImg();
    // 使用预先制作的RGB数据渲染图像
    void RenderImgRgb();
    // 绘制网格线
    void RenderGrid();
    // 绘制测量线
    void RenderMeasureLine();
    // 渲染品牌
    void RenderBranding();
    // 基于给定量程和方位数组计算网格线顶点数组
    void CalcGrid();
    // 根据当前缓冲的数据重新计算图像
    void Recalculate();
    // 添加数据到图像
    void AddDataToImg();
    // 添加RGB数据到图像
    void AddRgbDataToImg();
    // 判断是否顺时针
    bool AreClockwise(QPoint ct, float radius, float angle, QPoint pt);
    // 判断是否在某区域内
    bool IsInsideSector(QPoint pt, QPoint ct, float radius, float angle1, float angle2);
    // 判断是否在扇形图像内
    bool IsInsideFan(QPoint pos);
    // 渲染左侧品牌
    void SetLeftBranding(QString file);
    // 渲染右侧品牌
    void SetRightBranding(QString file);
    // 切换网格线显示
    void ToggleGridLines();

public slots:
    // 更新扇形图像缓存
    void UpdateFan(double rng, int nBrgs, short* pBrgs, bool updateProjection = false);
    // 根据新数据更新图像
    void UpdateImg(int nRngs, int nBrgs, uchar* pData);
    // 根据新16位数据更新图像
    void UpdateImg16(int nRngs, int nBrgs, quint16* pData);

public:
    // ----------------------------变量-----------------------------------
    unsigned m_textureId;    // 图像纹理ID
    unsigned m_nBrgs;        // 方位分辨率
    unsigned m_nRngs;        // 量程分辨率
    quint8   m_nBits;        // 本地图像中存储的位数
    double   m_range;
    bool     m_freqChange;
    double   m_left;         // 扇形图左侧的角度
    double   m_right;        // 扇形图右侧的角度
    bool     m_dwGrid;       // 网格显示切换
    bool     m_dwGridText;   // 网格文本显示切换
    float    m_mkr;          // 坐标轴网格标记频率
    int      m_nArcs;        // 要渲染的弧数
    bool     m_flipY;        // 翻转（纹理的）垂直轴
    bool     m_flipX;        // 翻转（纹理的）水平轴
    bool     m_headDown;     // 反转扇形图显示的方向以指向下方
    bool     m_newImgData;   // 新图像刷新标志位

    float*   m_pImgVbo;      // 图像的顶点缓冲区对象
    float*   m_pGridVbo;     // 网格的顶点缓冲区对象
    short*   m_pBrgs;        // 方位表
    uchar*   m_pData;        // 图像的最新数据
    uchar*   m_pRgbData;     // 用于此图像的RGB数据
    bool     m_useRgb;       // 使用RGB图像而不是灰度

    int      m_width;        // 图像宽度
    int      m_height;       // 图像高度

    bool     m_disconnected;          // 声纳连接标志位
    bool     m_measureEnable;         // 测量准许位
    bool     m_measuring;             // 测量中标志位
    bool     m_showLastMeasurement;

    float    m_measureStartX;         // 测量起始点X
    float    m_measureStartY;         // 测量起始点Y
    float    m_measureEndX;           // 测量终止点X
    float    m_measureEndY;           // 测量终止点Y

    bool     m_wasInsideFan;          // 鼠标在扇形图判断位

    bool	 m_showGrid;              // 显示网格标志位


    RmGlTextBuffer m_gridText;        // 网格文本缓冲区

    RmGlTextBuffer m_measureText;     // 测量文本缓冲区

signals:
    void MouseInfo(float dist, float angle, float x, float y);

    void MouseEnter();
    void MouseLeave();
};

#endif // SONARSURFACE_H
