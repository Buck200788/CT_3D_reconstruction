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
    const auto& g = m_geo;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int thr = omp_get_num_threads();
        int zblk = g.nz / thr;
        int zs = tid * zblk;
        int ze = (tid == thr - 1) ? g.nz : (tid + 1) * zblk;
        for (int z = zs; z < ze; z++)
        {
            for (int y = 0; y < g.ny; y++)
            {
                for (int x = 0; x < g.nx; x++)
                {
                    // FDK计算预留
                }
            }
        }
    }
}