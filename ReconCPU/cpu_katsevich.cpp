#include "cpu_katsevich.h"
#include <thread>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
CpuKatsevichRecon::CpuKatsevichRecon(const CTGeometry& geo, const recon_para& recp) : BaseRecon(geo, recp) {}

float CpuKatsevichRecon::PsiOverTanPsi(float psi)
{
    const float a = std::fabs(psi);
    if (a < 1.0e-4f)
    {
        const float psi2 = psi * psi;
        return 1.0f- psi2 / 3.0f- psi2 * psi2 / 45.0f;
    }
    return psi / std::tan(psi);
}

void CpuKatsevichRecon::calculate_kLines()
{
    const float D = m_geo.SDD;
    const float P = m_geo.pitch;
    const float R = m_geo.SID;
    const float PI = std::acos(-1.f);

    if (D <= 0.0f || R <= 0.0f || m_geo.dv <= 0.0f || m_geo.nDetU < 2)
    {
        throw std::invalid_argument("Invalid geometry for K-line calculation");
    }
    if (std::fabs(P) < 1.0e-12f)
    {
        throw std::invalid_argument("Helical pitch must be non-zero");
    }

    const float u_edge = 0.5f * static_cast<float>(m_geo.nDetU) * m_geo.du;
    const float alpha_m = std::atan(u_edge / D);
    const float psi_min = -0.5f * PI - alpha_m;
    const float psi_max = 0.5f * PI + alpha_m;

    const float u_min = -u_edge;
    const float psi_test = psi_max;

    const float sin_psi = std::sin(psi_test);
    const float cos_psi = std::cos(psi_test);
    const float cot_psi = cos_psi / sin_psi;
    const float csc2_psi = 1.0f / (sin_psi * sin_psi);
    const float kappa_scale = D * P / (2.0f * PI * R);
    const float dv_dpsi = kappa_scale * (1.0f + (u_min / D) * (cot_psi - psi_test * csc2_psi));

    if (std::fabs(dv_dpsi) < 1.0e-12f)
    {
        throw std::runtime_error("Invalid K-line sampling derivative");
    }

    float target_dpsi = m_geo.dv / std::fabs(dv_dpsi);
    int nPsi = static_cast<int>(std::ceil((psi_max - psi_min) / target_dpsi)) + 1;
    if ((nPsi & 1) == 0) nPsi += 1;
    const float dPsi = (psi_max - psi_min) / static_cast<float>(nPsi - 1);

    m_k_lines.assign(nPsi * m_geo.nDetU, 0);
    m_inverse_Psi_index.assign(m_geo.nDetV * m_geo.nDetU, -1.f);

    const float u_center = 0.5f * (m_geo.nDetU - 1);
    const float v_center = 0.5f * (m_geo.nDetV - 1);
    for (int ipsi = 0; ipsi < nPsi; ipsi++) {
        const float psi = psi_min + ipsi * dPsi;
        const float q = PsiOverTanPsi(psi);
        for (int iu = 0; iu < m_geo.nDetU; iu++) {
            const float u = (static_cast<float>(iu) - u_center) * m_geo.du;
            const float v_kappa =kappa_scale *(psi+ q * u / D);
            const float v_index = v_kappa / m_geo.dv + v_center;
            m_k_lines[ipsi * m_geo.nDetU + iu] = v_index;
        }
    }
    m_nPsi = nPsi;
    m_psiMin = psi_min;
    m_dPsi = dPsi;

    const int nDetU = m_geo.nDetU;
    const int nDetV = m_geo.nDetV;

    for (int iu = 0; iu < nDetU; ++iu)
    {
        for (int iv = 0; iv < nDetV; ++iv)
        {
            const float target_v =static_cast<float>(iv);
            const int inverse_index =iv * nDetU + iu;
            float best_psi_index = -1.f;
            float best_abs_psi = std::numeric_limits<float>::infinity();
            for (int ipsi = 0; ipsi < nPsi - 1; ++ipsi)
            {
                int offset = ipsi * nDetU;
                float v0 = m_k_lines[offset+iu];
                float v1 = m_k_lines[offset+nDetU+iu];

                const float lower = std::min(v0, v1);
                const float upper = std::max(v0, v1);

                if (target_v < lower || target_v > upper)
                    continue;

                const float denominator = v1 - v0;

                if (std::fabs(denominator) < 1.0e-12f)
                    continue;

                const float t = (target_v - v0) / denominator;
                const float p = static_cast<float>(ipsi) + t;
                const float psi = psi_min + p * dPsi;
                const float abs_psi = std::fabs(psi);
                if (abs_psi < best_abs_psi)
                {
                    best_abs_psi = abs_psi;
                    best_psi_index = p;
                }
            }
            m_inverse_Psi_index[inverse_index] = best_psi_index;
        }
    }
}

void CpuKatsevichRecon::construct_hilbert_kernel()
{
    const float PI = std::acos(-1.f);
    const float twoOverPI = 2.f / PI;

    int kernel_len = m_geo.nDetU * 2 - 1;
    m_hilbert_kernel.assign(kernel_len, 0.f);
    for (size_t i = m_geo.nDetU; i < kernel_len; i+=2) {
        m_hilbert_kernel[i] = twoOverPI / (i - m_geo.nDetU + 1) / m_geo.du;
        if (i > m_geo.nDetU - 1) {
            m_hilbert_kernel[kernel_len - 1 - i] = -m_hilbert_kernel[i];
        }
    }
}

void CpuKatsevichRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    std::vector<float> filt = proj;
    calculate_kLines();
    construct_hilbert_kernel();
    FilterProj(proj, filt);
    BackProjectOMP(filt, vol);
}

void CpuKatsevichRecon::FilterProj(const std::vector<float>& in, std::vector<float>& out)
{
    out.assign(in.size(), 0.0f);
    if (m_geo.nViews < 3 || m_geo.nDetU < 3 || m_geo.nDetV < 3)
        throw std::invalid_argument("nViews, nDetU and nDetV must all be at least 3");
    if (m_geo.SDD <= 0 || m_geo.du <= 0 ||  m_geo.dv <= 0 || std::fabs(m_geo.angleStep) < 1.0e-12f)
    {
        throw std::invalid_argument("Invalid geometry parameters");
    }

    unsigned int max_thread_num = std::thread::hardware_concurrency();
    unsigned int valid_views = (unsigned int)this->m_geo.nViews - 2;
    unsigned int thread_cnt = std::min(max_thread_num, valid_views);
    thread_cnt = std::max(thread_cnt, 1U);
    std::vector<std::thread> workers;
    workers.reserve(thread_cnt);
    int block = valid_views / thread_cnt;
    int remain = valid_views % thread_cnt;
    for (unsigned int tid = 0; tid < thread_cnt; tid++) {
        int start = tid * block + std::min(static_cast<int>(tid), remain) + 1;
        int end = start + block + (tid < remain ? 1 : 0);
        
        workers.emplace_back([this, &in, &out, start, end]() {
            const float D = this->m_geo.SDD;
            const float D2 = D*D;
            const float dbeta = this->m_geo.angleStep;
            const float du = this->m_geo.du;
            const float dv = this->m_geo.dv;
            const int nDetU = this->m_geo.nDetU;
            const int nDetV = this->m_geo.nDetV;
            const int view_offset = nDetU * nDetV;
            const float inv2DBeta = 1.0f / (2.0f * dbeta);
            const float inv2Du = 1.0f / (2.0f * du);
            const float inv2Dv = 1.0f / (2.0f * dv);
            const float u_center= 0.5f * static_cast<float>(nDetU - 1);
            const float v_center= 0.5f * static_cast<float>(nDetV - 1);

            const int nPsi = this->m_nPsi;
            const float m_psiMin = this->m_psiMin;
            const float m_dPsi = this->m_dPsi;
            const std::vector<float>& kernel = this->m_hilbert_kernel;
            const int kernel_size = kernel.size();
            std::vector<float>G1(view_offset, 0.f);
            std::vector<float>G2(nPsi * nDetU, 0.f);
            std::vector<float>G3(nPsi * nDetU, 0.f);
            
            for (int view = start; view < end; view++)
            {
                int current_view_offset = view * view_offset;     
                std::fill(G1.begin(), G1.end(), 0.0f);
                std::fill(G2.begin(), G2.end(), 0.0f);
                std::fill(G3.begin(), G3.end(), 0.0f);
                for (int iv = 1; iv < nDetV-1; iv++) {
                    int current_v_offset = iv * nDetU;
                    float v = (static_cast<float>(iv) - v_center) * dv;
                    for (int iu = 1; iu < nDetU - 1; iu++) {
                        float dgDbeta = (in[current_view_offset + view_offset + current_v_offset + iu] - in[current_view_offset - view_offset + current_v_offset  + iu]) * inv2DBeta;
                        float dgDv = (in[current_view_offset + current_v_offset + nDetU + iu] - in[current_view_offset + current_v_offset - nDetU + iu]) * inv2Dv;
                        float dgDu = (in[current_view_offset + current_v_offset + iu + 1] - in[current_view_offset + current_v_offset + iu - 1]) * inv2Du;
                        float u = (static_cast<float>(iu) - u_center) * du;
                        float g1= dgDbeta + (D2 + u * u) / D * dgDu + (u * v) / D * dgDv;
                        G1[current_v_offset + iu] = D / std::sqrt(D2 + u * u + v * v) * g1;
                    }
                }

                for (int ipsi = 0; ipsi < nPsi; ipsi++) {
                    const int psi_offset = ipsi * nDetU;
                    for (int iu = 0; iu < nDetU; iu++) {
                        int g2_idx = psi_offset + iu;
                        float v = m_k_lines[g2_idx];
                        if (v < 0.0f || v > static_cast<float>(nDetV - 1))
                        {
                            G2[g2_idx] = 0.0f;
                            continue;
                        }

                        int v0 = static_cast<int>(floorf(v));
                        const int v1= (v0 + 1 < nDetV) ? v0 + 1 : v0;
                        float w1 = v - v0;
                        int g10_idx = v0 * nDetU + iu;
                        int g11_idx = v1 * nDetU + iu;
                        G2[g2_idx] = (1.f - w1) * G1[g10_idx] + w1 * G1[g11_idx];
                    }
                }

                const int half = kernel_size / 2;
                for (int ipsi = 0; ipsi < nPsi; ipsi++) {
                    const int psi_offset = ipsi * nDetU;
                    for (int iu = 0; iu < nDetU; iu++) {
                        int g3_idx = ipsi * nDetU + iu;
                        float s = 0.f;
                        for (int iu1 = 0; iu1 < kernel_size; iu1++) {
                            int idx1 = iu + half - iu1;
                            if (idx1 < 0 || idx1 >= nDetU)continue;
                            s += G2[psi_offset + idx1] * kernel[iu1];
                        }
                        G3[g3_idx] = s*du;
                    }
                }

                for (int iv = 1; iv < nDetV - 1; iv++) {
                    int current_v_offset = iv * nDetU;
                    for (int iu = 1; iu < nDetU - 1; iu++) {
                        int out_index = current_view_offset + current_v_offset + iu;
                        float psi0_f = m_inverse_Psi_index[current_v_offset + iu];
                        if (psi0_f < 0.0f || psi0_f > static_cast<float>(nPsi - 1))
                        {
                            out[out_index] = 0.0f;
                            continue;
                        }
                        int psi0 = static_cast<int>(floorf(psi0_f));
                        psi0 = std::min(psi0, nPsi - 1);
                        const int psi1 = (psi0 + 1 < nPsi) ? psi0 + 1 : psi0;
                        float w1 = psi0_f - psi0;
                        int g20_idx = psi0 * nDetU + iu;
                        int g21_idx = psi1 * nDetU + iu;
                        out[out_index] = (1.f - w1) * G3[g20_idx] + w1 * G3[g21_idx];
                    }
                }
            }

            });
    }
    for (auto& t : workers)
    {
        if (t.joinable())
            t.join();
    }


}

void CpuKatsevichRecon::BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol)
{

}