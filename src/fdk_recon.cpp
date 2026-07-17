#include "../include/fdk_recon.h"
#include "../include/device_detect.h"

FDKRecon::FDKRecon(const CTGeometry& geo)
    : BaseRecon(geo)
{}

bool FDKRecon::HasCudaHardware() const
{
    return HasCUDADevice();
}


void FDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& volume)
{
    bool gpuAvailable = HasCudaHardware();
    if (gpuAvailable)
    {
        RunGPU_CUDA(proj, volume);
        return;
    }
    RunCPU_OMP(proj, volume);
}

#ifndef ENABLE_CUDA
void FDKRecon::RunGPU_CUDA(const std::vector<float>& hProj, std::vector<float>& hVol)
{
    return;
}
#endif