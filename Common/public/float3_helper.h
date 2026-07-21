#pragma once
#ifdef __CUDACC__
#include <vector_types.h>
#else
struct float3
{
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};
inline float3 make_float3(float x, float y, float z)
{
    return float3(x, y, z);
}
#endif