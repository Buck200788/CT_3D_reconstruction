#pragma once
#include "geometry.h"
#include <vector>
#include "device_detect.h"

#ifdef CTRECONLIB_EXPORTS
#define RECON_API __declspec(dllexport)
#else
#define RECON_API __declspec(dllimport)
#endif

enum class ReconAlgoType
{
    FDK,
    Katsevich
};

class RECON_API BaseRecon
{
protected:
    CTGeometry m_geo;
public:
    BaseRecon(const CTGeometry& geo);
    virtual ~BaseRecon();

    virtual void Reconstruct(const std::vector<float>& proj, std::vector<float>& volume) = 0;
};

RECON_API BaseRecon* CreateRecon(ReconAlgoType type, const CTGeometry& geo);
RECON_API void DestroyRecon(BaseRecon* pRecon);