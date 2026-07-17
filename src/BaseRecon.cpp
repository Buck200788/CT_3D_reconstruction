#include "../include/BaseRecon.h"
#include "../include/fdk_recon.h"

BaseRecon::BaseRecon(const CTGeometry& geo)
    : m_geo(geo)
{}

BaseRecon::~BaseRecon() = default;

BaseRecon* CreateRecon(ReconAlgoType type, const CTGeometry& geo)
{
    switch (type)
    {
    case ReconAlgoType::FDK:
        return new FDKRecon(geo);
    //case ReconAlgoType::Katsevich:
    //    return new KatsevichRecon(geo);
    default:
        return nullptr;
    }
}

void DestroyRecon(BaseRecon* pRecon)
{
    if (pRecon) delete pRecon;
    ReleaseCudaRuntime();
}