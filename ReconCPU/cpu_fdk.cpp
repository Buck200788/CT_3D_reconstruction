#include "cpu_fdk.h"
#include <thread>
#include <fstream>
#include <execution>

CpuFDKRecon::CpuFDKRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void CpuFDKRecon::ParallelPreprocessProj(std::vector<float>& filter_geom_filted, const std::vector<float>& filter, unsigned int max_thread_num) const
{
    const int total_view = m_geo.nViews;
    if (total_view <= 0 || max_thread_num <= 0)
    {
        std::cout << "m_geo.nViews: " << total_view << ", thread num invalid" << std::endl;
        return;
    }

    unsigned int thread_cnt = std::min(max_thread_num, (unsigned int)total_view);
    thread_cnt = std::max(thread_cnt, 1U);

    std::vector<std::thread> workers;
    workers.reserve(thread_cnt);

    int block = total_view / thread_cnt;
    int remain = total_view % thread_cnt;

    for (unsigned int tid = 0; tid < thread_cnt; tid++)
    {
        int start = tid * block;
        int end = start + block;
        if (tid < remain)
            end += 1;

        workers.emplace_back([this, &filter_geom_filted, &filter, start, end]()
            {
                const int SDD = m_geo.SDD;
                const int nDetU = m_geo.nDetU;
                const int nDetV = m_geo.nDetV;
                const float du = m_geo.du;
                const float dv = m_geo.dv;

                const float D = static_cast<float>(SDD);
                const float D2 = D * D;
                const float u_len = du * nDetU;
                const float v_len = dv * nDetV;

                const int filter_len = static_cast<int>(filter.size());
                const int half_win = filter_len / 2;

                for (int view = start; view < end; view++)
                {
                    for (int iv = 0; iv < nDetV; iv++)
                    {
                        // geometry weighted
                        float v = -0.5f * v_len + (iv + 0.5f) * dv;
                        for (int iu = 0; iu < nDetU; iu++)
                        {
                            float u = -0.5f * u_len + (iu + 0.5f) * du;
                            int offset = view * nDetV * nDetU + iv * nDetU + iu;
                            filter_geom_filted[offset] *= (D / std::sqrt(D2 + u * u + v * v));
                        }

                        // 2. U axis conv filter
                        const int base = view * nDetV * nDetU + iv * nDetU;
                        std::vector<float> temp_u(nDetU, 0.f);
                        for (int iu = 0; iu < nDetU; iu++)
                        {
                            float sum_conv = 0.f;
                            for (int iu1 = 0; iu1 < filter_len; iu1++)
                            {
                                int iu_shift = iu - half_win + iu1;
                                if (iu_shift >= 0 && iu_shift < nDetU)
                                {
                                    int pix_idx = base + iu_shift;
                                    sum_conv += filter[iu1] * filter_geom_filted[pix_idx];
                                }
                            }
                            temp_u[iu] = sum_conv;
                        }
                        // write back the results
                        for (int iu = 0; iu < nDetU; iu++)
                        {
                            filter_geom_filted[base + iu] = temp_u[iu];
                        }
                    }
                }
            });
    }

    // wait till all the thread done
    for (auto& t : workers)
    {
        if (t.joinable())
            t.join();
    }
}

void CpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    Filter filt(m_geo.nDetU, Str2FilterType("ram-lak"), 1.0);
    std::vector<float> filter = filt.GetFilter();
    std::cout << "filter length: " << filter.size() << std::endl;
    //std::ofstream outs("E:\\test.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(filter.data()), sizeof(float) * filter.size());
    //outs.close();

    std::vector<float> proj_geom_filted(proj);

    unsigned int max_thread_num = std::thread::hardware_concurrency();
    std::cout << "maximum thread number: " << max_thread_num << std::endl;

    ParallelPreprocessProj(proj_geom_filted, filter, max_thread_num);

    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    BackProjectOMP(proj_geom_filted, vol);
}


void CpuFDKRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{
    int total_voxels = vol.size();
    
}