#include "../Common/public/ReconBase.h"
#include "../Common/public/ReconAlgoEnum.h"
#include "gpu_fdk.cuh"
#include "gpu_katsevich.cuh"

extern "C" RECON_API BaseRecon* CreateGpuRecon(ReconAlgorithm algo, const CTGeometry& geo, const recon_para& rec_p)
{
    switch (algo)
    {
    case ReconAlgorithm::FDK:
        return new GpuFDKRecon(geo, rec_p);
    case ReconAlgorithm::Katsevich:
        return new GpuKatsevichRecon(geo, rec_p);
    default: return nullptr;
    }
}

extern "C" RECON_API void DestroyReconInstance(BaseRecon* ptr)
{
    if (ptr) delete ptr;
}