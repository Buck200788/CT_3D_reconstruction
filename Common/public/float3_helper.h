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
    const float3& src,
    const int nDetU,
    const int nDetV,
    float theta, float SDD,
    float du, float dv,
    float& out_u, float& out_v
)
{
    // 计算当前角度探测器中心
    float cosT = cosf(theta);
    float sinT = sinf(theta);
    float3 Cdet = src - SDD * make_float3(cosT, sinT, 0.f);
    // hit相对探测器中心的偏移向量
    float3 delta = hit - Cdet;
    // U轴投影
    float u_local = -(delta.x * sinT - delta.y * cosT);
    // V轴相对探测器中心Z偏移
    float v_local = delta.z;
    out_u = u_local / du + 0.5f * nDetU - 0.5f;
    out_v = v_local / dv + 0.5f * nDetV - 0.5f;
}

/// @brief 双线性插值读取滤波后投影值
/// @param proj 输入投影数组（CPU std::vector / GPU device float*）
/// @param view 当前视角
/// @param u 浮点U像素坐标（中心0）
/// @param v 浮点V像素坐标（中心0）
/// @param nDetU/nDetV 探测器像素尺寸
CUDA_HOST_DEV inline float BilinearInterp(
    const float* proj,
    float u, float v,
    int nDetU, int nDetV
)
{
    const float eps = 1e-5f;
    if (u<0 || u>nDetU - 1 ||
        v<0 || v>nDetV - 1)
        return 0;

    // 浮点uv转左下角整数像素下标
    int u0 = static_cast<int>(floorf(u));
    int v0 = static_cast<int>(floorf(v));
    //int u1 = u0 + 1;
    //int v1 = v0 + 1;

    const int u1 =
        (u0 + 1 < nDetU) ? u0 + 1 : u0;

    const int v1 =
        (v0 + 1 < nDetV) ? v0 + 1 : v0;

    // 计算插值权重
    float alpha = u - u0;
    float beta = v - v0;
    float w00 = (1.f - alpha) * (1.f - beta);
    float w10 = alpha * (1.f - beta);
    float w01 = (1.f - alpha) * beta;
    float w11 = alpha * beta;

    // 四个像素内存偏移
    int idx00 = v0 * nDetU + u0;
    int idx10 = v0 * nDetU + u1;
    int idx01 = v1 * nDetU + u0;
    int idx11 = v1 * nDetU + u1;

    // 加权求和
    float val = w00 * proj[idx00] + w10 * proj[idx10]
        + w01 * proj[idx01] + w11 * proj[idx11];
    return val;
}

CUDA_HOST_DEV inline bool VoxelToFlatDetectorUV(
    float x, float y, float z,
    float beta, float source_z,
    float SID, float SDD,
    float du, float dv,
    int nDetU, int nDetV,
    float& u_index, float& v_index,
    float& source_to_voxel_radial)
{
    const float c = cosf(beta);
    const float s = sinf(beta);

    // Distance from source to the voxel plane measured along the central ray.
    const float den = SID - x * c - y * s;
    if (den <= 1e-6f) return false;

    const float magnification = SDD / den;
    const float u_mm = magnification * (-x * s + y * c);
    const float v_mm = magnification * (z - source_z);

    u_index = u_mm / du + 0.5f * static_cast<float>(nDetU) - 0.5f;
    v_index = v_mm / dv + 0.5f * static_cast<float>(nDetV) - 0.5f;
    source_to_voxel_radial = den;
    return true;
}

CUDA_HOST_DEV inline bool VoxelToEquiangularDetectorUV(
    float x, float y, float z,
    float beta, float source_z,
    float SID, float SDD,
    float dgamma, float dv,
    int nDetU, int nDetV,
    float& u_index,
    float& v_index,
    float& source_to_voxel_xy_sq
)
{
    const float c = cosf(beta);
    const float s = sinf(beta);
    /*
     * 当前坐标约定与原来的平板函数一致：
     *
     * e_w = (-cos(beta), -sin(beta), 0)
     *       中心射线方向
     *
     * e_u = ( sin(beta), -cos(beta), 0)
     *       探测器水平方向
     */

     // 源到体素在中心射线方向上的分量
    const float den = SID - x * c - y * s;

    if (den <= 1.0e-6f)
        return false;

    const float transverse = -x * s + y * c;

    const float L2 =den * den +transverse * transverse;

    if (L2 <= 1.0e-12f)
        return false;
    const float L = sqrtf(L2);
    const float gamma = atan2f(transverse, den);

    const float v_detector =SDD * (z - source_z) / L;
    const float center_u =0.5f *static_cast<float>(nDetU) - 0.5f;
    const float center_v =0.5f *static_cast<float>(nDetV) -0.5f;
    u_index = gamma / dgamma +center_u;
    v_index = v_detector / dv +center_v;
    source_to_voxel_xy_sq = L2;
    return true;
}

