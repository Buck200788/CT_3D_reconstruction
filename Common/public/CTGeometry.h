#pragma once
// CT锥束螺旋通用几何参数
#include <string>
struct CTGeometry
{
    // 探测器
    int nDetU = 64;
    int nDetV = 64;
    float du = 0.8f;
    float dv = 0.8f;

    float SDD = 1000.f;
    float SID = 600.f;
    float pitch = 1.0f;

    // 投影角度
    int nViews = 360;
    float angleStep = 2.f * 3.1415926535f / 360.f;

    // 重建体素
    int nx = 64;
    int ny = 64;
    int nz = 64;
    float dx = 1.f;
    float dy = 1.f;
    float dz = 1.f;
    float zStart = -64.f;
};

struct recon_para
{
    std::string filter_name;
    int cuda_device;
};