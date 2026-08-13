#pragma once
#include "ReconBase.h"

// check whether has the CUDA device
bool HasAvailableCudaDevice();
// free the handle
void ReleaseCudaRuntimeHandle();