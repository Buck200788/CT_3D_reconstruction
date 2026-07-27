#include "../Common/public/ReconBase.h"
#include "../Common/public/ReconAlgoEnum.h"
#include "cpu_fdk.h"
#include "cpu_katsevich.h"

extern "C" RECON_API BaseRecon* CreateCpuRecon(ReconAlgorithm algo, const CTGeometry& geo, const recon_para& rec_p)
{
    switch (algo)
    {
    case ReconAlgorithm::FDK:
        return new CpuFDKRecon(geo, rec_p);
    case ReconAlgorithm::Katsevich:
        return new CpuKatsevichRecon(geo, rec_p);
    default: return nullptr;
    }
}

extern "C" RECON_API void DestroyReconInstance(BaseRecon* ptr)
{
    if (ptr) delete ptr;
}