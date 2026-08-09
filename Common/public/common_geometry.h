#pragma once

#include "float3_helper.h"

struct DetectorElement
{
    float3 center;
    float3 u_direction;
    float3 v_direction;
    int nDetU;
    int nDetV;
    float du;
    float dv;
};

struct BeamSurfaceGeometry
{
    int nSource{ 0 };
    float3* source_position{ nullptr };

    int nDetector{ 0 };
    DetectorElement* detector{ nullptr };
    bool follow_source_transform{ false };

    ~BeamSurfaceGeometry()
    {
        delete[] detector;
        delete[] source_position;
    }
    BeamSurfaceGeometry(const BeamSurfaceGeometry&) = delete;
    BeamSurfaceGeometry& operator=(const BeamSurfaceGeometry&) = delete;
};

CUDA_HOST_DEV inline float3 rotateAroundAxis(
    const float3& v,
    const float3& axis,
    float cosTheta,
    float sinTheta)
{
    return v * cosTheta + cross(axis, v) * sinTheta + axis * (dot(axis, v) * (1.0f - cosTheta));
}

CUDA_HOST_DEV inline float3 perpendicularDirection(const float3& v)
{
    const float ax = fabsf(v.x);
    const float ay = fabsf(v.y);
    const float az = fabsf(v.z);

    float3 helper;

    if (ax <= ay && ax <= az)
    {
        helper = make_float3(1.0f, 0.0f, 0.0f);
    }
    else if (ay <= ax && ay <= az)
    {
        helper = make_float3(0.0f, 1.0f, 0.0f);
    }
    else
    {
        helper = make_float3(0.0f, 0.0f, 1.0f);
    }
    return normalize(cross(v, helper));
}

CUDA_HOST_DEV inline void transformDetectorElement(const float3& src1, const float3& src2, const DetectorElement& de_input, DetectorElement& de_output)
{
    constexpr float EPS = 1e-6f;
    const DetectorElement input = de_input;

    const float len1 = sqrtf(dot(src1, src1));
    const float len2 = sqrtf(dot(src2, src2));

    if (len1 < EPS || len2 < EPS)
    {
        de_output = input;
        return;
    }

    const float scale = len2 / len1;
    const float3 dir1 = src1 * (1.0f / len1);
    const float3 dir2 = src2 * (1.0f / len2);

    float cosTheta = dot(dir1, dir2);
    cosTheta = fmaxf(-1.0f, fminf(1.0f, cosTheta));

    de_output.nDetU = input.nDetU;
    de_output.nDetV = input.nDetV;
    de_output.du = input.du;
    de_output.dv = input.dv;

    if (cosTheta > 1.0f - EPS)
    {
        de_output.center = input.center * scale;
        de_output.u_direction = input.u_direction;
        de_output.v_direction = input.v_direction;
        return;
    }

    float3 axis;
    float sinTheta;
    if (cosTheta < -1.0f + EPS)
    {
        axis = perpendicularDirection(dir1);
        cosTheta = -1.0f;
        sinTheta = 0.0f;
    }
    else
    {
        const float3 c = cross(dir1, dir2);
        sinTheta = sqrtf(dot(c, c));
        axis = c * (1.0f / sinTheta);
    }

    const float3 scaledCenter = input.center * scale;
    de_output.center = rotateAroundAxis(scaledCenter, axis, cosTheta, sinTheta);
    de_output.u_direction = rotateAroundAxis(input.u_direction, axis, cosTheta, sinTheta);
    de_output.v_direction = rotateAroundAxis(input.v_direction, axis, cosTheta, sinTheta);

    de_output.u_direction = normalize(de_output.u_direction);
    de_output.v_direction = normalize(de_output.v_direction);
}