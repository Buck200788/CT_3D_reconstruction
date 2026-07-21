#pragma once
#include "ReconBase.h"

// 检测本机是否存在可用NVIDIA CUDA设备
bool HasAvailableCudaDevice();
// 释放cudart动态库句柄
void ReleaseCudaRuntimeHandle();