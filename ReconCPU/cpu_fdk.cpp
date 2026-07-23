#include "cpu_fdk.h"
#include <omp.h>
#include <fstream>
CpuFDKRecon::CpuFDKRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void CpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    Filter filt(m_geo.nDetU, Str2FilterType("ram-lak"), 1.0);
    std::vector<float> filter = filt.GetFilter();
    //std::ofstream outs("E:\\test.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(filter.data()), sizeof(float) * filter.size());
    //outs.close();
    
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    BackProjectOMP(proj, vol);
}

void CpuFDKRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{

}