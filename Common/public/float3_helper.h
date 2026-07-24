#pragma once
#include <cmath> 
#ifdef __CUDACC__
#include <vector_types.h>
// CUDA环境保留原生修饰符
#define CUDA_HOST_DEV __host__ __device__
#else
// CPU环境清空修饰符，消除未定义标识符报错
#define CUDA_HOST_DEV
// 自定义CPU float3
struct float3
{
    float x, y, z;
    constexpr float3() : x(0.f), y(0.f), z(0.f) {}
    constexpr float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
#endif

// 向量减法
CUDA_HOST_DEV inline float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// 向量加法
CUDA_HOST_DEV inline float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 向量 * 标量
CUDA_HOST_DEV inline float3 operator*(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// 标量 * 向量
CUDA_HOST_DEV inline float3 operator*(float s, const float3& a)
{
    return a * s;
}

// 点积
CUDA_HOST_DEV inline float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_HOST_DEV inline bool RayPlaneIntersect(
    const float3& S,
    const float3& V,
    const float3& plane_normal,
    float plane_D,
    float3& outHit
)
{
    // 射线方向向量
    float3 dir = V - S;
    // 分母：法向量与射线方向点积
    float denom = dot(plane_normal, dir);

    // 射线平行于平面，无交点
    if (fabsf(denom) < 1e-8f)
        return false;

    // 平面方程推导 t = -(n·S + D) / (n·dir)
    float dotNS = dot(plane_normal, S);
    float numer = -(dotNS + plane_D);
    float t = numer / denom;

    // t < 0 交点在射线源后方，无效
    if (t < -1e-6f)
        return false;

    // 计算交点坐标
    outHit = S + dir * t;
    return true;
}

// 输入：交点hit、当前旋转角theta、几何参数
CUDA_HOST_DEV inline void WorldHitToUV(
    const float3& hit,
    float theta,
    float SDD, float du, float dv,
    int nDetU, int nDetV,
    float& out_u, float& out_v
)
{
    // 1. 计算局部u物理长度
    float u_local = hit.x * sinf(theta) - hit.y * cosf(theta); // make sure the u direction is clockwise direction along z direction
    // 2. 计算局部v物理长度（z直接对应）
    float v_local = hit.z;

    // 转浮点像素坐标，中心为0
    out_u = u_local / du;
    out_v = v_local / dv;
}

/// @brief 双线性插值读取滤波后投影值
/// @param proj 输入投影数组（CPU std::vector / GPU device float*）
/// @param view 当前视角
/// @param u 浮点U像素坐标（中心0）
/// @param v 浮点V像素坐标（中心0）
/// @param nDetU/nDetV 探测器像素尺寸
CUDA_HOST_DEV inline float BilinearInterp(
    const float* proj,
    int view,
    float u, float v,
    int nDetU, int nDetV
)
{
    // 探测器半宽
    const float halfU = nDetU * 0.5f;
    const float halfV = nDetV * 0.5f;

    // 边界判断，越界返回0
    if (u < -halfU || u > halfU || v < -halfV || v > halfV)
        return 0.f;

    // 浮点uv转左下角整数像素下标
    int u0 = static_cast<int>(floorf(u));
    int v0 = static_cast<int>(floorf(v));
    int u1 = u0 + 1;
    int v1 = v0 + 1;

    // 计算插值权重
    float alpha = u - u0;
    float beta = v - v0;
    float w00 = (1.f - alpha) * (1.f - beta);
    float w10 = alpha * (1.f - beta);
    float w01 = (1.f - alpha) * beta;
    float w11 = alpha * beta;

    // 当前视角数据基地址
    int view_base = view * nDetU * nDetV;
    // 四个像素内存偏移
    int idx00 = view_base + v0 * nDetU + u0;
    int idx10 = view_base + v0 * nDetU + u1;
    int idx01 = view_base + v1 * nDetU + u0;
    int idx11 = view_base + v1 * nDetU + u1;

    // 加权求和
    float val = w00 * proj[idx00] + w10 * proj[idx10]
        + w01 * proj[idx01] + w11 * proj[idx11];
    return val;
}