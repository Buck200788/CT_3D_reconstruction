#include "device_detect.h"
#include <windows.h>

#ifndef __CUDACC__
typedef int cudaError_t;
#define cudaSuccess 0
#endif
typedef cudaError_t(__cdecl* PFN_cudaGetDeviceCount)(int*);

static HMODULE g_cudaLib = nullptr;
static PFN_cudaGetDeviceCount g_pfnGetDevCount = nullptr;

static const char* cudartList[] = {
    "cudart64_101.dll","cudart64_12.dll",
    "cudart122.dll","cudart121.dll","cudart120.dll",
    "cudart118.dll","cudart117.dll","cudart116.dll",
    "cudart115.dll","cudart114.dll","cudart113.dll",
    "cudart112.dll","cudart111.dll","cudart110.dll",
    nullptr
};

static bool InitCudaDll()
{
    if (g_cudaLib) return true;
    for (int i = 0; cudartList[i]; ++i)
    {
        HMODULE h = LoadLibraryA(cudartList[i]);
        if (!h) continue;
        auto f = (PFN_cudaGetDeviceCount)GetProcAddress(h, "cudaGetDeviceCount");
        if (f)
        {
            g_cudaLib = h;
            g_pfnGetDevCount = f;
            return true;
        }
        FreeLibrary(h);
    }
    return false;
}

bool HasAvailableCudaDevice()
{
    if (!InitCudaDll()) return false;
    int cnt = 0;
    auto err = g_pfnGetDevCount(&cnt);
    return err == cudaSuccess && cnt > 0;
}

void ReleaseCudaRuntimeHandle()
{
    if (g_cudaLib)
    {
        FreeLibrary(g_cudaLib);
        g_cudaLib = nullptr;
        g_pfnGetDevCount = nullptr;
    }
}