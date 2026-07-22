#include "cpu_fdk.h"
#include <omp.h>
CpuFDKRecon::CpuFDKRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void CpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    BackProjectOMP(proj, vol);
}

void CpuFDKRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{

}