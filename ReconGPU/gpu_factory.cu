#include "../Common/public/ReconBase.h"
#include "../Common/public/ReconAlgoEnum.h"
#include "gpu_fdk.cuh"
#include "gpu_katsevich.cuh"

extern "C" RECON_API BaseRecon* CreateGpuRecon(ReconAlgorithm algo, const CTGeometry& geo)
{
    switch (algo)
    {
    case ReconAlgorithm::FDK:
        return new GpuFDKRecon(geo);
    case ReconAlgorithm::Katsevich:
        return new GpuKatsevichRecon(geo);
    default: return nullptr;
    }
}

extern "C" RECON_API void DestroyReconInstance(BaseRecon* ptr)
{
    if (ptr) delete ptr;
}