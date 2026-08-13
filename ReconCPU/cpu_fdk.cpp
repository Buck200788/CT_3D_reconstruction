#include "cpu_fdk.h"
#include <thread>
#include <fstream>
#include <execution>

CpuFDKRecon::CpuFDKRecon(const CTGeometry& geo, const recon_para& recp) : BaseRecon(geo, recp) {}


void CpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    Filter filt(m_geo.nDetU, Str2FilterType(rec_p.filter_name), m_geo.du, m_geo.scan_type);
    std::vector<float> filter = filt.GetFilter();
    std::cout << "filter length: " << filter.size() << std::endl;
    //std::ofstream outs("E:\\CFiles\\CTReconTest\\debug\\filter.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(filter.data()), sizeof(float) * filter.size());
    //outs.close();

    //for (size_t i = 0; i < filter.size(); i++)printf("%d %f\n", i, filter[i]);

    std::vector<float> proj_geom_filted(proj);

    ParallelPreprocessProj(proj_geom_filted, filter);

    //std::ofstream outs("E:\\CFiles\\CTReconTest\\debug\\data1_512x128x1600.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(proj_geom_filted.data()), sizeof(float) * proj_geom_filted.size());
    //outs.close();
    //return;

    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    BackProjectOMP(proj_geom_filted, vol);
}


void CpuFDKRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{
    int total_voxels = vol.size();

    //for (size_t i = 0; i < proj.size(); i++)
    //{
    //    if (!std::isfinite(proj[i]))
    //    {
    //        printf("proj has inf£¬index %zu£¬value=%e\n", i, proj[i]);
    //        break;
    //    }
    //}
    //std::ofstream outs("D:\\code\\C\\CTRecon\\debug\\data_512x128x1600.raw", std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(proj.data()), sizeof(float) * proj.size());
    //outs.close();
    //return;

    unsigned int max_thread_num = std::thread::hardware_concurrency();
    unsigned int thread_cnt = std::min(max_thread_num, (unsigned int)this->m_geo.nz);
    thread_cnt = std::max(thread_cnt, 1U);

    std::vector<std::thread> workers;
    workers.reserve(thread_cnt);

    int block = this->m_geo.nz / thread_cnt;
    int remain = this->m_geo.nz % thread_cnt;
    for (unsigned int tid = 0; tid < thread_cnt; tid++) {
        int start = tid * block + (int)fmin(tid, remain);
        int end = start + block + (tid < remain ? 1 : 0);

        workers.emplace_back([this, &proj, &vol, start, end]()
            {
                const float PI = std::acosf(-1.0f);
                const float dz_per_view = m_geo.pitch / (2.f * PI / std::fabs(m_geo.angleStep));
                //printf("dz_per_view: %f\n", dz_per_view);
                const int nViews_half_loop = static_cast<int>(std::round(PI / std::fabs(m_geo.angleStep)));
                //printf("%f %f %d\n", PI, m_geo.angleStep, nViews_half_loop);
                const int det_offset_per_view = m_geo.nDetU * m_geo.nDetV;
                const int scan_type = m_geo.scan_type;
                for (int iz = start; iz < end; iz++) {
                    float z_pos = (-0.5f * m_geo.nz + iz + 0.5) * m_geo.dz;
                    float z_pos_fview = (z_pos - m_geo.zStart) / dz_per_view;
                    int z_pos_iview = static_cast<int>(roundf(z_pos_fview));
                    //printf("%4d %4d %4d %4d %f\n", start, end, z_pos_iview , nViews_half_loop, z_pos);
                    for (int iview = z_pos_iview - nViews_half_loop; iview < z_pos_iview + nViews_half_loop; iview++) {
                        if (iview < 0 || iview >= m_geo.nViews)continue;
                        float angle = iview * m_geo.angleStep;
                        float ray_dire_x = std::cos(angle);
                        float ray_dire_y = std::sin(angle);

                        float sx = m_geo.SID * ray_dire_x;
                        float sy = m_geo.SID * ray_dire_y;
                        float sz = iview * dz_per_view + m_geo.zStart;
                        //float3 src = make_float3(sx, sy, sz);
                        //float3 plane_norm = make_float3(-ray_dire_x, -ray_dire_y, 0.f);
                        //const float plane_D = m_geo.SID - m_geo.SDD;
                        //float3 hits = make_float3(0.f, 0.f, 0.f);

                        int det_offset = iview * det_offset_per_view;
                        int vox_offset = iz * m_geo.ny * m_geo.nx;
                        for (int iy = 0; iy < m_geo.ny; iy++) {
                            float y_pos = (-0.5 * m_geo.ny + 0.5 + iy) * m_geo.dy;
                            for (int ix = 0; ix < m_geo.nx; ix++) {
                                float x_pos = (-0.5 * m_geo.nx + 0.5 + ix) * m_geo.dx;

                                if (scan_type == 0)
                                {
                                    float u, v, den;
                                    if (!VoxelToFlatDetectorUV(x_pos, y_pos, z_pos, angle, sz,
                                        m_geo.SID, m_geo.SDD, m_geo.du, m_geo.dv,
                                        m_geo.nDetU, m_geo.nDetV, m_geo.detectorVCenterOffsetPix, u, v, den)) continue;

                                    const float q = BilinearInterp(proj.data() + det_offset, u, v, m_geo.nDetU, m_geo.nDetV);
                                    const float w = (m_geo.SID * m_geo.SID) / (den * den);
                                    vol[vox_offset + iy * m_geo.nx + ix] += 0.5f * w * q * std::fabs(m_geo.angleStep);
                                }
                                else if (scan_type == 1) {
                                    float u = 0.0f;
                                    float v = 0.0f;
                                    float horizontal_distance_sq = 0.0f;
                                    float den;
                                    if (!VoxelToEquiangularDetectorUV(x_pos, y_pos, z_pos, angle, sz,
                                        m_geo.SID, m_geo.SDD, m_geo.du, m_geo.dv,
                                        m_geo.nDetU, m_geo.nDetV, m_geo.detectorVCenterOffsetPix, u, v, horizontal_distance_sq,den)) continue;
                                    float q = BilinearInterp(proj.data() + det_offset, u, v, m_geo.nDetU, m_geo.nDetV);
                                    vol[vox_offset + iy * m_geo.nx + ix] += 0.5f* q / horizontal_distance_sq * std::fabs(m_geo.angleStep);
                                }



                                ///////  the code below also gives the right algorithm  ///////
                                //float3 V = make_float3(x_pos, y_pos, z_pos);
                                //bool if_cross = RayPlaneIntersect(src, V, plane_norm, plane_D, hits);
                                //
                                //if (!if_cross)continue;
                                //float iu, iv;
                                //WorldHitToUV(hits, src,m_geo.nDetU, m_geo.nDetV, angle, m_geo.SDD, m_geo.du, m_geo.dv, iu, iv);
                                //float value = BilinearInterp(proj.data()+ det_offset, iu, iv, m_geo.nDetU, m_geo.nDetV);
                                //
                                //float U = m_geo.SID - (x_pos * std::cosf(angle) + y_pos * std::sinf(angle));
                                //vol[vox_offset + iy * m_geo.nx + ix] += 0.5 * m_geo.SID * m_geo.SID /(U*U) * value* std::fabs(m_geo.angleStep);
                            }
                        }
                    }
                    
                }
            }
        );
        
    }
    for (auto& t : workers) {
        if (t.joinable())
            t.join();
    }
}

void CpuFDKRecon::ParallelPreprocessProj(std::vector<float>& filter_geom_filted, const std::vector<float>& filter) const
{
    unsigned int max_thread_num = std::thread::hardware_concurrency();
    std::cout << "maximum thread number: " << max_thread_num << std::endl;
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
        int start = tid * block + (int)fmin(tid, remain);
        int end = start + block + (tid < remain ? 1 : 0);
        //printf("start, end: %d %d\n", start, end);

        workers.emplace_back([this, &filter_geom_filted, &filter, start, end]()
            {
                const float SDD = m_geo.SDD;
                const float SID = m_geo.SID;
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

                const float v_center = 0.5f * static_cast<float>(nDetV - 1);

                for (int view = start; view < end; view++)
                {
                    for (int iv = 0; iv < nDetV; iv++)
                    {
                        const float v = (static_cast<float>(iv) - v_center - m_geo.detectorVCenterOffsetPix) * dv;
                        for (int iu = 0; iu < nDetU; iu++)
                        {
                            float u = -0.5f * u_len + (iu + 0.5f) * du;
                            int offset = view * nDetV * nDetU + iv * nDetU + iu;
                            //if (view == 319 && iv == 87) printf("BB %d  %f %f\n", iu, filter_geom_filted[offset], D / std::sqrt(D2 + u * u + v * v));
                            float tmp;
                            if (m_geo.scan_type == 0) {
                                tmp = D / std::sqrt(D2 + u * u + v * v);
                            }
                            else if (m_geo.scan_type == 1) {
                                float gamma = u;
                                float cos_tau = D / std::sqrt(D2 + v * v);
                                tmp = std::cos(gamma) * cos_tau * SID;
                            }

                            //filter_geom_filted[offset] *= (D / std::sqrt(D2 + u * u + v * v));
                            filter_geom_filted[offset] *= tmp;

                            //if (view == 319 && iv == 87) printf("BB  %f\n", filter_geom_filted[offset]);
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

                            //if (view == 319 && iv == 87) printf("AA  %f\n", sum_conv);
                        }
                        // write back the results
                        for (int iu = 0; iu < nDetU; iu++)
                        {
                            filter_geom_filted[base + iu] = temp_u[iu] * m_geo.du;
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
