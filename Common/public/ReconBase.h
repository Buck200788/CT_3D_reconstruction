#pragma once
#include "CTGeometry.h"
#include "ReconAlgoEnum.h"
#include <vector>
#include <string>

#if defined(_WIN32)
#   if defined(RECON_DLL_EXPORTS)
#       define RECON_API __declspec(dllexport)
#   else
#       define RECON_API __declspec(dllimport)
#   endif
#else
#   define RECON_API
#endif

class RECON_API BaseRecon
{
protected:
    CTGeometry m_geo;
    recon_para rec_p;
public:
    explicit BaseRecon(const CTGeometry& geo, const recon_para& recp) : m_geo(geo), rec_p(recp) {}
    virtual ~BaseRecon() = default;

    virtual void Reconstruct(const std::vector<float>& projData, std::vector<float>& volumeOut) = 0;
};

extern "C" RECON_API BaseRecon* CreateCpuRecon(ReconAlgorithm algo, const CTGeometry& geo, const recon_para& rec_p);
extern "C" RECON_API void DestroyReconInstance(BaseRecon* ptr);

extern "C" RECON_API BaseRecon* CreateGpuRecon(ReconAlgorithm algo, const CTGeometry& geo, const recon_para& rec_p);
