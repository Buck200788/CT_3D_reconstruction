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
                    // Katsevich反投影预留
                }
            }
        }
    }
}