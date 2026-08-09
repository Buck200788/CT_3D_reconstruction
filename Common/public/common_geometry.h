#pragma once

#include "float3_helper.h"

struct DetectorElement
{
    float3 center;
    float3 norm_direction; // point to the source
    int nDetU;
    int nDetV;
    float du;
    float dv;
};

struct MeasurementGeometry
{
    float3 source;
    int nDetector;
    DetectorElement* detector;

};