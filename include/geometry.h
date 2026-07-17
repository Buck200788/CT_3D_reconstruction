#pragma once
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct CTGeometry
{
    int nDetU, nDetV;
    float du, dv;
    float SDD, SID;
    float pitch;
    int nViews;
    float angleStep;

    int nx, ny, nz;
    float dx, dy, dz;
    float zStart;
};
