#include "cpu_fdk.h"
#include <thread>
#include <fstream>
#include <execution>
CpuFDKRecon::CpuFDKRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void CpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    Filter filt(m_geo.nDetU, Str2FilterType("ram-lak"), 1.0);
    std::vector<float> filter = filt.GetFilter();
    //std::ofstream outs("E:\\test.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(filter.data()), sizeof(float) * filter.size());
    //outs.close();
    unsigned int max_thread_num = std::thread::hardware_concurrency();
    std::cout << "maximum thread number: " << max_thread_num << std::endl;

    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    BackProjectOMP(proj, vol);
}

void CpuFDKRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{
    int total_voxels = vol.size();
    std::vector<int> indices(total_voxels);
    std::iota(indices.begin(), indices.end(), 0);

    // 多线程并行遍历所有体素，等价 omp parallel for
    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&](int idx)
        {
            // 这里填入你原本的单个体素反投影计算逻辑
            float sum_val = 0.f;
            // ...投影累加计算
            vol[idx] = sum_val;
        });
}