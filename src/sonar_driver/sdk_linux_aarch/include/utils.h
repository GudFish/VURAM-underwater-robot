#ifndef UTILS_H
#define UTILS_H

#include <QtMath>
#include <QString>

#define MAX(a, b) (a > b ? a : b)
#define MIN(a, b) (a < b ? a : b)

// 一些有用的静态工具函数
class Utils
{
public:
    // ------------------------------变量---------------------------------------------
    static double   markers[6];
    static const double threshold;          // 阈值
    static const double r2d;                // 弧度转角度参数
    static const double d2r;                // 角度转弧度参数
    static const double pi;                 // 圆周率
    static const double pi_2;               // 圆周率/2
    static const double pi3_2;              // 圆周率 * 1.5
    static const double pi_4;               // 圆周率/4
    static const double pi2;                // 圆周率 * 2
    static const double root2;              // 根号2
    static const double root2_2;            // 根号2/2
    static const double e;                  // 自然指数

    // ------------------------------------------------------------------------------

    // ------------------------------函数---------------------------------------------
    // 将uint32 IP地址格式化为字符串
    static QString FormatIpAddress(uint32_t ipAddr)
    {
        uchar ip1 = (uchar)(ipAddr & 0xff);
        uchar ip2 = (uchar)((ipAddr & 0xff00) >> 8);
        uchar ip3 = (uchar)((ipAddr & 0xff0000) >> 16);
        uchar ip4 = (uchar)((ipAddr & 0xff000000) >> 24);

        return QString::number(ip1) + "." + QString::number(ip2) + "." + QString::number(ip3) + "." + QString::number(ip4);
    }

    // 对于16位数，高八位低八位互换
    static quint16 SwapWord(quint16 w)
    {
      return ((w & 0xff) << 8) + ((w & 0xff00) >> 8);
    }

    // 对于32位数，颠倒位数
    static quint32 SwapDWord(quint32 w)
    {
      quint32 out;
      quint8* bytes = (quint8*)&out;
      quint8* dword = (quint8*)&w;

      bytes[0] = dword[3];
      bytes[1] = dword[2];
      bytes[2] = dword[1];
      bytes[3] = dword[0];

      return out;
    }

    // 颠倒float数据
    static float SwapFloat(float w)
    {
      float out;
      quint8* bytes = (quint8*)&out;
      quint8* dword = (quint8*)&w;

      bytes[0] = dword[3];
      bytes[1] = dword[2];
      bytes[2] = dword[1];
      bytes[3] = dword[0];

      return out;
    }

    // 返回等比例插值的值
    static double Interpolate (double x0, double x1, double x, double y0, double y1)
    {
      double d = x1 - x0;

      if (d)
        return y0 + ( (x - x0) / d ) * (y1 - y0);
      else
        return y0;
    }

    // 返回给定值的平方
    static unsigned Pow2Up(unsigned v)
    {
      v--;
      v |= v >> 1;
      v |= v >> 2;
      v |= v >> 4;
      v |= v >> 8;
      v |= v >> 16;
      v++;

      return v;
    }

    // 获取距离定义数组中最靠近的数（或它的数十倍）
    static double NearestMkr(double marker);
    // 获取给定数在数组中（或其10倍数）的排序后一位
    static double NextMkr(double marker);
    // 获取给定数在数组中（或其10倍数）的排序前一位
    static double PrevMkr(double marker);
    // 返回两个经纬度之间的距离
    static double LL2LLRng(double lat1, double long1, double lat2, double long2);
    // 返回两个经纬度之间的方位角
    static double LL2LLBrg(double lat1, double long1, double lat2, double long2);
    static double CrossTrack(double lat1, double long1, double lat2, double long2, double atLat, double atLong);
    static double AlongTrack(double lat1, double long1, double lat2, double long2, double atLat, double atLong);

    static double BrgOffsetDegrees(double centre, double reference);

    // 弧度转角度
    static double R2D (double r) { return r * r2d; }
    // 角度转弧度
    static double D2R (double d) { return d * d2r; }
    // 四舍五入
    static double Rnd (double x) { return floor (x + 0.5); }

    static double ClipUp (double x, double y) { return ceil(x / y) * y; }
    static double ClipDn (double x, double y) { return floor(x / y) * y; }
    static double ClipRnd (double x, double y) { return Rnd(x / y) * y; }
    static double ClipRng (double x, double a, double b) { return (Max ((a), Min ((b), (x)))); }

    // 平方
    static double Sqr (double x) { return x * x; }
    // 立方
    static double Cube (double x) { return x * x * x; }
    // 三个数取平方后相加，并开方
    static double Hypot3 (double x, double y, double z) { return sqrt(Sqr(x) + Sqr(y) + Sqr(z)); }

    // 偶数判断
    static bool   Even (double x) { return ( ((double)x / 2.0) - floor((double)x / 2.0) == 0.0 ? true  : false ); }
    // 奇数判断
    static bool   Odd (double x)  { return ( ((double)x / 2.0) - floor((double)x / 2.0) == 0.0 ? false : true );  }
    // 是否整除判断
    static bool   ExactDivisible (double x, double y) { return ( ((double)x / (double)y) - floor((double)x / (double)y) == 0.0 ? true : false ); }

    // 后数比前数大与阈值
    static bool   LessThanT (double x, double y) { return (((y - x) > threshold)  ? 1 : 0); }
    // 前数比后数大与阈值
    static bool   MoreThanT (double x, double y) { return (((x - y) > threshold)  ? 1 : 0); }
    // 两数差小于阈值，小于阈值认为两数相等
    static bool   EqualT (double x, double y)    { return ((fabs(y - x) < threshold)  ? 1 : 0); }
    // 两数差小于给定的阈值，小于阈值认为两数相等
    static bool   EqualT (double x, double y, double t)    { return ((fabs(y - x) < t)  ? 1 : 0); }
    // 取余，若余数和被除数相差很小，认为余数为0
    static double FModT (double x, double y)     { double z = fmod(x, y); return (EqualT(z, y) ? 0.0 : z); }
    // 求两数算术平方根，过小则认为0
    static double HypotT(double x, double y)     { return (EqualT(x, 0.0) && EqualT(y, 0.0) ? 0 : sqrt(x * x + y * y)); }

    static double HeadAngleD (double x) { return ( fmod(720.0 - (x) + 90.0, 360.0) ); }
    static double HeadAngleR (double x) { return ( fmod((4 * pi) - (x) + pi_2, pi2) ); }
    static double AngleHeadD (double x) { return HeadAngleD(x); }
    static double AngleHeadR (double x) { return HeadAngleR(x); }

    // 取较大
    static double Max (double x, double y) { if (x > y) return x; else return y; }
    // 取较小
    static double Min (double x, double y) { if (x > y) return y; else return x; }
    // 求平均
    static double Mid (double x, double y) { return (x + y) / 2.0; }
    static double ACosh(double x) { return log(x + sqrt((x * x - 1))); }
    static double ATan2(double y, double x) { return (x == 0.0) ? ((y > 0.0) ? Utils::pi_2 : -Utils::pi_2) : atan2(y, x); }
    // 算术平方根
    static double Hypot(double x, double y) { return sqrt(x * x + y * y); }

    // ------------------------------------------------------------------------------
};

#endif // UTILS_H
