#include "cpu_katsevich.h"
#include <omp.h>
CpuKatsevichRecon::CpuKatsevichRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void CpuKatsevichRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    std::vector<float> filt = proj;
    FilterProj(proj, filt);
    BackProjectOMP(filt, vol);
}

void CpuKatsevichRecon::FilterProj(const std::vector<float>& in, std::vector<float>& out)
{
    // Katsevich滤波预留
}

void CpuKatsevichRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{
}