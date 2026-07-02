#ifndef SURFACE_H
#define SURFACE_H

#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QVector2D>
#include <QOpenGLShaderProgram>
#include <QDateTime>
#include <QPainter>
#include <QFont>
#include <QWindow>

#include "sonarwidgetutils.h"

// 内置着色器程序列表
enum eProgram : int
{
    pgmSolid,
    pgmTexture,
    pgmPalette,
    pgmAlpha,
    pgmLuminance,
    pgmKey
};

// 文本对齐选项
enum eTextAlign : unsigned
{
    taLeft     = 0x01,
    taHCentre  = 0x02,
    taRight    = 0x04,
    taTop      = 0x08,
    taVCentre  = 0x10,
    taBottom   = 0x20,
    taFlipX    = 0x40,
    taFlipY    = 0x80
};

// 存储字体图集的单个字符信息
class GlyphInfo
{
public:
    char  c;       // 字符
    int   width;   // 宽度
    int   height;  // 高度
    float s0;      // 与图集一致的纹理s0
    float s1;      // 与图集一致的纹理s1
    float t0;      // 与图集一致的纹理t0
    float t1;      // 与图集一致的纹理t1
};

#define GLYPH_BASE 32
#define N_GLYPHS   94


// ----------------------------------------------------------------------------
// 用于存储文本信息的缓冲区-这可以添加到Surface类中并由其呈现
class RmGlTextBuffer
{
public:
    RmGlTextBuffer();
    ~RmGlTextBuffer();

    // 函数
    void ClearBuffer();

    // 变量
    float* m_pVbo;     // 文本VBO
    int    m_nChars;   // 缓冲区中的字符数
};

class Surface: public QObject, public QOpenGLFunctions, public SonarWidgetUtils
{
    Q_OBJECT
public:
    Surface();
    virtual ~Surface();

    // ---------------------------------覆写--------------------------------------------
    virtual void Render() = 0;
    virtual void OnResize(int w, int h) = 0;
    // 初始化opengl参数
    virtual void OnCreate();

    // 这些虚拟函数允许基于widget的系统传递鼠标事件
    virtual void mouseMoveEvent(QMouseEvent* pEvent)         { Q_UNUSED(pEvent) }
    virtual void mousePressEvent(QMouseEvent* pEvent)        { Q_UNUSED(pEvent) }
    virtual void mouseReleaseEvent(QMouseEvent* pEvent)      { Q_UNUSED(pEvent) }
    virtual void mouseDoubleClickEvent(QMouseEvent* pEvent)  { Q_UNUSED(pEvent) }
    virtual void wheelEvent(QWheelEvent *pEvent)             { Q_UNUSED(pEvent) }

    // ----------------------------------函数--------------------------------------------
    // 从给定的字符串资源编译着色器
    int  RmglCompileShader(int type, QString shaderSource);
    // 将顶点着色器和片段着色器链接到程序中并返回ID
    int  RmglLinkProgram(int vertex, int fragment);
    // 构建着色器程序并返回ID（错误时为0）
    int  RmglBuildProgram(QString vertexSource, QString fragmnetSource);
    // 创建内部着色器
    void RmglInitShaders();
    // 加载并设置调色板着色器的调色板纹理
    void RmglInitPalette();
    // 创建字体纹理并设置默认字体
    void RmglInitFont();
    // 创建标记纹理并从标准资源加载
    void RmglInitMarkers();
    // 加载并设置字体渲染器的纹理
    void RmglSetFont(QFont font);
    // 设置背景纹理
    void RmglSetBackground(QString background);
    // 为适当的着色器配置均匀的颜色
    void RmglSetColour(QColor colour);
    // 将当前程序设置为给定程序
    void RmglSetPgm(eProgram pgm);
    // 使用texId和program=pgm渲染画板（带纹理坐标）
    void RmglRenderVbo(float* pVbo, unsigned texId, eProgram pgm);
    // 将文本添加到文本缓冲区
    void RmglAddText(RmGlTextBuffer& buffer, float x, float y, float rot, QString text, unsigned ta = (taLeft | taBottom));
    // 渲染文本缓冲区的内容
    void RmglRenderText(RmGlTextBuffer& buffer);
    // 渲染背景图像
    void RmglRenderBackground(QRect rect);
    // 渲染前景图像
    void RmglRenderOverlay();
    // 将给定的图像资源加载到纹理中
    bool RmglLoadImageToTexture(QString imgName, unsigned& texId);
    // 将标记渲染到屏幕上
    void RmglRenderMarker(float x, float y, float rot, float scale, unsigned marker, QColor colour, QString text);
    // 渲染实线
    void RmglRenderLine(float x0, float y0, float x1, float y1, QColor colour);
    // 渲染实心多段线
    void RmglRenderPolyline(float* pPts, int nPts, QColor colour);
    // 将实心矩形渲染到给定的像素区域
    void RmglRenderPixelRect(int x0, int y0, int x1, int y1, QColor colour);
    // 渲染左侧品牌
    void RmglRenderLeftBranding();
    // 渲染右侧品牌
    void RmglRenderRightBranding();
    // 设置左侧品牌
    void RmglSetLeftBranding(QString image);
    // 设置右侧品牌
    void RmglSetRightBranding(QString image);
    // 显示品牌
    void RmglShowBranding(bool enable);

    // -----------------------------------变量-----------------------------------------
    QColor    m_clearColour;             // 使用的纯色
    QColor    m_colour;                  // 当前纯色（由solid和Alpha使用）
    QColor    m_colKey;                  // 当前按键颜色（按键使用）
    QFont     m_font;                    // 用于背景文本的字体
    QString   m_background;              // 背景资源的名称
    int       m_palIndex;                // 调色板纹理使用的调色板索引
    unsigned  m_palTexId;                // 调色板纹理ID
    unsigned  m_backgroundId;            // 背景纹理ID
    unsigned  m_overlayId;               // 前景纹理ID
    int       m_nPalettes;               // 调色板数量（根据图像高度/10得出）
    int       m_pgmId;                   // 当前选中渲染程序ID

    QString   m_leftBranding;
    unsigned  m_leftBrandingId;
    QString	  m_rightBranding;
    unsigned  m_rightBrandingId;

    QRect	  m_leftRect;
    QRect	  m_rightRect;

    bool	  m_showBranding;            // 显示品牌标志位

    // 文本生成
    unsigned  m_fontTexId;               // 字体纹理ID
    GlyphInfo m_fontInfo[N_GLYPHS];      // 字体图集信息

    // 着色器
    int m_pgmTexture;                    // 基础纹理
    int m_pgmPalette;                    // 使用调色板索引对纹理进行调色
    int m_pgmSolid;                      // 纯色渲染
    int m_pgmAlpha;                      // 纹理用作alpha值（由字体使用）
    int m_pgmLuminance;                  // 纹理（alpha设置为亮度）
    int m_pgmKey;                        // colour key

    // 标记
    unsigned m_arrowId;                  // 箭头标记
    unsigned m_squareId;                 // 方框标记
    unsigned m_circleId;                 // 圆圈标记
    unsigned m_diamondId;                // 钻石标记

signals:
    void Update();
    void SetCursor(const QCursor& cursor);
};

#endif // SURFACE_H
