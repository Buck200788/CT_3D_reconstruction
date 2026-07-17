#include "../include/device_detect.h"

#ifndef __CUDACC__
typedef int cudaError_t;
#define cudaSuccess 0
#endif

typedef cudaError_t(__cdecl* PFN_cudaGetDeviceCount)(int*);

static HMODULE g_cudaModule = nullptr;
static PFN_cudaGetDeviceCount g_pfnCudaGetDeviceCount = nullptr;

static bool InitCudaRuntime()
{
    if (g_cudaModule)
        return true;
    g_cudaModule = LoadLibraryA("cudart120.dll");
    if (!g_cudaModule)
        return false;
    g_pfnCudaGetDeviceCount = (PFN_cudaGetDeviceCount)GetProcAddress(g_cudaModule, "cudaGetDeviceCount");
    return g_pfnCudaGetDeviceCount != nullptr;
}

bool HasCUDADevice()
{
    if (!InitCudaRuntime())
        return false;
    int devCnt = 0;
    cudaError_t err = g_pfnCudaGetDeviceCount(&devCnt);
    return (err == cudaSuccess && devCnt > 0);
}

void ReleaseCudaRuntime()
{
    if (g_cudaModule)
    {
        FreeLibrary(g_cudaModule);
        g_cudaModule = nullptr;
        g_pfnCudaGetDeviceCount = nullptr;
    }
}