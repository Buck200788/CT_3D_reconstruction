#include "cpu_katsevich.h"
#include <thread>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

#include <atomic>
#include <chrono>
#include <mutex>

namespace
{
    constexpr float GEOM_EPS = 1.0e-12f;

    inline float SignedPitchPerBeta(
        float pitch,
        float angle_step)
    {
        return pitch * std::fabs(angle_step) / angle_step;
    }

    inline float HelixSlopePerBeta(
        float pitch,
        float angle_step)
    {
        const float PI = std::acos(-1.0f);
        return SignedPitchPerBeta(pitch, angle_step) / (2.0f * PI);
    }

    inline float HelixDzPerView(
        float pitch,
        float angle_step)
    {
        const float PI = std::acos(-1.0f);
        return pitch * std::fabs(angle_step) / (2.0f * PI);
    }
}

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

void CpuKatsevichRecon::calculate_kLines_equal_angle()
{
    const float D = m_geo.SDD;
    const float P = SignedPitchPerBeta(m_geo.pitch, m_geo.angleStep);
    const float R = m_geo.SID;
    const float PI = std::acos(-1.f);
    const float dAlpha = m_geo.du;
    const int nAlpha = m_geo.nDetU;
    const float dw = m_geo.dv;
    const int nDetV = m_geo.nDetV;
    const float alpha_center= 0.5f * static_cast<float>(nAlpha - 1);
    const float w_center= 0.5f * static_cast<float>(nDetV - 1);

    float alphaM = 0.5 * nAlpha * dAlpha;
    const float psiMax = 0.5 * PI + alphaM;
    const float psiMin = -1.f* psiMax;
    float alpha_edge = -alphaM;

    const float sin_psi = std::sin(psiMax);
    const float cos_psi = std::cos(psiMax);
    const float cot_psi = cos_psi / sin_psi;
    const float csc2_psi = 1.0f / (sin_psi * sin_psi);
    const float kappa_scale = D * P / (2.0f * PI * R);
    const float q_prime =cot_psi - psiMax * csc2_psi;

    const float max_dwdpsi = std::fabs(kappa_scale * (std::cos(alpha_edge) + std::sin(alpha_edge) * q_prime));
    const float A = 0.5f * PI + alphaM;
    int M = static_cast<int>(std::ceil(A * max_dwdpsi / dw));
    M = std::max(M, 1);
    const int nPsi = 2 * M + 1;
    const float dPsi = A / static_cast<float>(M);

    m_nPsi = nPsi;
    m_psiMin = psiMin;
    m_dPsi = dPsi;

    m_k_lines.assign(nPsi*nAlpha, 0.f);

    std::vector<float> sinAlpha(nAlpha);
    std::vector<float> cosAlpha(nAlpha);

    for (int ia = 0; ia < nAlpha; ++ia)
    {
        const float alpha = (static_cast<float>(ia) - alpha_center) * dAlpha;
        sinAlpha[ia] = std::sin(alpha);
        cosAlpha[ia] = std::cos(alpha);
    }

    for (int ipsi = 0; ipsi < nPsi; ipsi++) {
        const float psi = psiMin + ipsi * dPsi;
        const float q = PsiOverTanPsi(psi);
        for (int ialpha = 0; ialpha < nAlpha; ialpha++) {
            const float alpha = (static_cast<float>(ialpha) - alpha_center) * dAlpha;
            const float w_kappa = kappa_scale * (psi * cosAlpha[ialpha] + q * sinAlpha[ialpha]);
            const float w_index =w_kappa / dw + w_center + m_geo.detectorVCenterOffsetPix;
            m_k_lines[static_cast<size_t>(ipsi) * nAlpha + ialpha] = w_index;
        }
    }

}

void CpuKatsevichRecon::calculate_inverse_Psi_index()
{
    const int nPsi = m_nPsi;
    const float dPsi = m_dPsi;
    const float psi_min = m_psiMin;
    const int nDetU = m_geo.nDetU;
    const int nDetV = m_geo.nDetV;

    m_inverse_Psi_index.assign(m_geo.nDetV * m_geo.nDetU, -1.f);

    for (int iu = 0; iu < nDetU; ++iu)
    {
        for (int iv = 0; iv < nDetV; ++iv)
        {
            const float target_v = static_cast<float>(iv);
            const int inverse_index = iv * nDetU + iu;
            float best_psi_index = -1.f;
            float best_abs_psi = std::numeric_limits<float>::infinity();
            for (int ipsi = 0; ipsi < nPsi - 1; ++ipsi)
            {
                int offset = ipsi * nDetU;
                float v0 = m_k_lines[offset + iu];
                float v1 = m_k_lines[offset + nDetU + iu];

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

void CpuKatsevichRecon::calculate_kLines()
{
    const float D = m_geo.SDD;
    const float P = SignedPitchPerBeta(m_geo.pitch, m_geo.angleStep);
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

    const float half_x = 0.5f * m_geo.nx * m_geo.dx;
    const float half_y = 0.5f * m_geo.ny * m_geo.dy;

    const float u_edge = 0.5f * static_cast<float>(m_geo.nDetU) * m_geo.du;

    const float fov_radius = std::sqrt(half_x * half_x + half_y * half_y);
    if (fov_radius >= m_geo.SID)
        throw std::invalid_argument("FOV exceeds helical radius");
    const float alpha_fov = std::asin(fov_radius / m_geo.SID);
    const float alpha_detector = std::atan(u_edge / m_geo.SDD);
    if (alpha_detector < alpha_fov)        throw std::runtime_error("Detector does not cover reconstruction FOV");
    const float alpha_m = alpha_fov;
    
    //const float alpha_m = std::atan(u_edge / D);
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

    const float u_center = 0.5f * (m_geo.nDetU - 1);
    const float v_center = 0.5f * (m_geo.nDetV - 1);
    for (int ipsi = 0; ipsi < nPsi; ipsi++) {
        const float psi = psi_min + ipsi * dPsi;
        const float q = PsiOverTanPsi(psi);
        for (int iu = 0; iu < m_geo.nDetU; iu++) {
            const float u = (static_cast<float>(iu) - u_center) * m_geo.du;
            const float v_kappa =kappa_scale *(psi+ q * u / D);
            const float v_index = v_kappa / m_geo.dv + v_center+ m_geo.detectorVCenterOffsetPix;
            m_k_lines[ipsi * m_geo.nDetU + iu] = v_index;
        }
    }
    m_nPsi = nPsi;
    m_psiMin = psi_min;
    m_dPsi = dPsi;
}

void CpuKatsevichRecon::construct_hilbert_kernel()
{
    const float PI = std::acos(-1.f);
    const float twoOverPI = 2.f / PI;

    int kernel_len = m_geo.nDetU * 2 - 1;
    m_hilbert_kernel.assign(kernel_len, 0.f);
    for (size_t i = m_geo.nDetU; i < kernel_len; i+=2) {
        if(m_geo.scan_type==0)
            m_hilbert_kernel[i] = twoOverPI / (i - m_geo.nDetU + 1) / m_geo.du;
        else if(m_geo.scan_type==1)
            m_hilbert_kernel[i] = twoOverPI / std::sin((i - m_geo.nDetU + 1) * m_geo.du);
        if (i > m_geo.nDetU - 1) {
            m_hilbert_kernel[kernel_len - 1 - i] = -m_hilbert_kernel[i];
        }
    }
    
}

void CpuKatsevichRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    if (m_geo.scan_type != 0 && m_geo.scan_type != 1)
    {
        throw std::invalid_argument("Unsupported detector scan type");
    }
    if (std::fabs(m_geo.angleStep) < GEOM_EPS)        throw std::invalid_argument("angleStep must be non-zero");

    if (std::fabs(m_geo.pitch) < GEOM_EPS)
        throw std::invalid_argument("Helical pitch must be non-zero");
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);
    std::vector<float> filt = proj;
    if (m_geo.scan_type == 0)
        calculate_kLines();
    else if (m_geo.scan_type == 1)
        calculate_kLines_equal_angle();
    printf("calculating the Klines done\n");
    calculate_inverse_Psi_index();
    printf("calculating inverse Klines done\n");
    construct_hilbert_kernel();
    printf("calculating Hilbert kernel done\n");
    FilterProj(proj, filt);
    printf("pre-filter done, starting build PI LUT\n");
    
    //std::ofstream outs("D:\\data1_6144x5040.raw",std::ios::binary);
    //outs.write(reinterpret_cast<const char*>(filt.data()), sizeof(float) * filt.size());
    //outs.close();
    //return;
    build_PI_LUT();
    printf("bulid pi LUT done, starting backprojection\n");
    BackProject(filt, vol);
    printf("projection done\n");
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
    printf("filter process using thread num: %d\n", thread_cnt);
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

            std::vector<float>cosAlpha(nDetU, 1.f);
            if (m_geo.scan_type == 1) {
                for (int iu = 0; iu < nDetU; ++iu)
                {
                    const float alpha = (static_cast<float>(iu) - u_center) * du;
                    cosAlpha[iu] = std::cos(alpha);
                }
            }
            
            for (int view = start; view < end; view++)
            {
                int current_view_offset = view * view_offset;     
                std::fill(G1.begin(), G1.end(), 0.0f);
                std::fill(G2.begin(), G2.end(), 0.0f);
                std::fill(G3.begin(), G3.end(), 0.0f);
                for (int iv = 1; iv < nDetV-1; iv++) {
                    int current_v_offset = iv * nDetU;
                    float v = (static_cast<float>(iv) - v_center - m_geo.detectorVCenterOffsetPix) * dv;
                    for (int iu = 1; iu < nDetU - 1; iu++) {
                        float dgDbeta = (in[current_view_offset + view_offset + current_v_offset + iu] - in[current_view_offset - view_offset + current_v_offset  + iu]) * inv2DBeta;
                        float dgDv = (in[current_view_offset + current_v_offset + nDetU + iu] - in[current_view_offset + current_v_offset - nDetU + iu]) * inv2Dv;
                        float dgDu = (in[current_view_offset + current_v_offset + iu + 1] - in[current_view_offset + current_v_offset + iu - 1]) * inv2Du;
                        float u = (static_cast<float>(iu) - u_center) * du;
                        if (m_geo.scan_type == 0) {
                            float g1 = dgDbeta + (D2 + u * u) / D * dgDu + (u * v) / D * dgDv;
                            G1[current_v_offset + iu] = D / std::sqrt(D2 + u * u + v * v) * g1;
                        }
                        else if (m_geo.scan_type == 1) {
                            float derivative = dgDbeta + dgDu;
                            //if(derivative!=0)printf("%f %f\n", dgDbeta, dgDu);
                            G1[current_v_offset + iu] = D / std::sqrt(D2 + v * v) * derivative;
                            //if (G1[current_v_offset + iu] != 0)printf("%f\n", G1[current_v_offset + iu]);
                        }
                        
                    }
                }

                //for (int i = 0; i < G1.size(); i++) if(G1[i]!=0) printf("%d, %f\n", i, G1[i]);
                //continue;

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
                        if(m_geo.scan_type==1)out[out_index] *= cosAlpha[iu];
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

bool CpuKatsevichRecon::calculate_PI_line(const float R, const float h, const float z0, float x, float y, float z, float& beta_b, float &belta_t)
{
    const float radius2 = x * x + y * y;

    if (radius2 >= R * R)
        return false;

    const float PI = std::acos(-1.f);
    if (std::fabs(h) < 1.0e-12f) {
        throw std::invalid_argument("pitch should not be too small");
        return false;
    }
    
    auto evaluate = [&](float beta_m)
    {
        float sin_beta = std::sin(beta_m);
        float cos_beta = std::cos(beta_m);
        float cos_delta = (x * cos_beta + y * sin_beta) / R;
        cos_delta = std::clamp(cos_delta, -1.f, 1.f);
        float delta_m = std::acos(cos_delta);
        float sin_delta = std::sin(delta_m);
        if (std::fabs(sin_delta) < 1.0e-7f)
            return std::numeric_limits<float>::quiet_NaN();
        float mu = (-x * sin_beta + y * cos_beta) / (R * sin_delta);
        return z0 + h * (beta_m + mu * delta_m) - z;
    };

    float beta_z = (z - z0) / h;
    float lo = beta_z - PI;
    float hi = beta_z + PI;
    float f_lo = evaluate(lo);
    float f_hi = evaluate(hi);

    if (!std::isfinite(f_lo) || !std::isfinite(f_hi)) return false;
    if (f_lo * f_hi > 0) return false;
    for (size_t i = 0; i < 30; i++) {
        float mid = 0.5 * (lo + hi);
        float f_mid = evaluate(mid);
        if (!std::isfinite(f_mid))
            return false;
        if ((f_mid <= 0 && f_lo >= 0) || (f_mid >= 0 && f_lo <= 0)) {
            hi = mid;
            f_hi = f_mid;
        }
        else {
            lo = mid;
            f_lo = f_mid;
        }
    }
    float beta = 0.5 * (lo + hi);

    float sin_beta = std::sin(beta);
    float cos_beta = std::cos(beta);
    float cos_delta = (x * cos_beta + y * sin_beta) / R;
    cos_delta = std::clamp(cos_delta, -1.f, 1.f);
    float delta = std::acos(cos_delta);
    beta_b = beta - delta;
    belta_t = beta + delta;
    return true;
}

void CpuKatsevichRecon::build_PI_LUT()
{
    const float PI = std::acos(-1.f);

    const double R = static_cast<double>(m_geo.SID);
    const double P_beta = static_cast<double>(SignedPitchPerBeta(m_geo.pitch, m_geo.angleStep));

    const double h = P_beta / (2.0 * PI);
    const double pitch_abs = std::fabs(static_cast<double>(m_geo.pitch));

    int n_PiLines_per_pitch = 2 * std::ceil(pitch_abs / m_geo.dz);
    n_PiLines_per_pitch = std::max(n_PiLines_per_pitch, 2);
    m_nPILines_per_pitch = n_PiLines_per_pitch;
    unsigned int total_cnt = n_PiLines_per_pitch * m_geo.nx * m_geo.ny;
    const float invalid =  std::numeric_limits<float>::quiet_NaN();
    m_pi_LUT.assign(total_cnt*2, invalid);

    unsigned int max_thread_num = std::thread::hardware_concurrency();
    unsigned int thread_cnt = std::min(max_thread_num, total_cnt);
    thread_cnt = std::max(thread_cnt, 1U);
    printf("calculate PI LUT process using thread num: %d\n", thread_cnt);
    std::vector<std::thread> workers;
    workers.reserve(thread_cnt);
    int block = total_cnt / thread_cnt;
    int remain = total_cnt % thread_cnt;
    for (unsigned int tid = 0; tid < thread_cnt; tid++) {
        int start = tid * block + std::min(static_cast<int>(tid), remain);
        int end = start + block + (tid < remain ? 1 : 0);

        workers.emplace_back([this, n_PiLines_per_pitch, R, h, start, end]()
            {
                const int nx = this->m_geo.nx;
                const int ny = this->m_geo.ny;
                const float dx = this->m_geo.dx;
                const float dy = this->m_geo.dy;
                const int n_per_slice =  nx* ny;
                const float y_center= (-0.5 * ny + 0.5) * dy;
                const float x_center= (-0.5 * nx + 0.5) * dx;
                const float z0 = this->m_geo.zStart;
                std::vector<float>& pi_LUT = this->m_pi_LUT;
                const float PI = std::acos(-1.f);

                for (size_t iv = start; iv < end; iv++)
                {
                    const int iz = iv / n_per_slice;
                    const int iy = iv % n_per_slice / nx;
                    const int ix = iv % n_per_slice % nx;
                    const double phase = 2*PI * static_cast<double>(iz) / static_cast<double>(n_PiLines_per_pitch);
                    const float z = z0 + h*phase;
                    const float y = y_center + iy * dy;
                    const float x = x_center + ix * dx;
                    float beta_b, beta_t;
                    if (!calculate_PI_line(R, h, z0, x,y,z, beta_b, beta_t))
                        continue;
                    pi_LUT[iv * 2] = 0.5f * (beta_b + beta_t)- static_cast<float>(phase);
                    pi_LUT[iv * 2 + 1] = 0.5f * (beta_t - beta_b);
                }
            });
    }
    for (auto& t : workers) {
        if (t.joinable())
            t.join();
    }
}

bool CpuKatsevichRecon::calculate_pi(int ix, int iy, float z, float& beta_b, float& beta_t)
{
    const float PI = std::acos(-1.f);
    const float TWO_PI = 2 * PI;
    const double P_beta = static_cast<double>(SignedPitchPerBeta(m_geo.pitch, m_geo.angleStep));
    const double h = P_beta / (2.0 * PI);
    const float z0 = m_geo.zStart;
    const double beta_z = (static_cast<double>(z) - z0) / h;
    double phase = beta_z - TWO_PI * std::floor(beta_z / TWO_PI);
    double idx = phase * m_nPILines_per_pitch / TWO_PI;
    int i0 = static_cast<int>(std::floor(idx));
    double t = idx - i0;
    i0 %= m_nPILines_per_pitch;
    const int i1 = (i0 + 1) % m_nPILines_per_pitch;
    
    const int nx = this->m_geo.nx;
    const int ny = this->m_geo.ny;
    const float dx = this->m_geo.dx;
    const float dy = this->m_geo.dy;
    const int n_per_slice = nx * ny;
    int idx_0 = i0 * n_per_slice + iy * nx + ix;
    int idx_1 = i1 * n_per_slice + iy * nx + ix;
    const float midRelative = t * m_pi_LUT[2 * idx_1] + (1.f - t) * m_pi_LUT[2 * idx_0];
    const float delta = t * m_pi_LUT[2 * idx_1 + 1] + (1.f - t) * m_pi_LUT[2 * idx_0 + 1];
    const float betaMid = static_cast<float>(beta_z) + midRelative;
    beta_b = betaMid - delta;
    beta_t = betaMid + delta;
    return beta_b < beta_t;
}

void CpuKatsevichRecon::BackProject(const std::vector<float>& proj, std::vector<float>& vol)
{
    int total_voxels = vol.size();

    unsigned int max_thread_num = std::thread::hardware_concurrency();
    unsigned int thread_cnt = std::min(max_thread_num, (unsigned int)this->m_geo.nz);
    thread_cnt = std::max(thread_cnt, 1U);
    printf("BackProject process using thread num: %d\n", thread_cnt);
    std::vector<std::thread> workers;
    workers.reserve(thread_cnt);

    int block = this->m_geo.nz / thread_cnt;
    int remain = this->m_geo.nz % thread_cnt;

    std::atomic<int> finished_slices{ 0 };
    std::mutex print_mutex;
    const auto backproject_start =
        std::chrono::steady_clock::now();
    for (unsigned int tid = 0; tid < thread_cnt; tid++) {
        int start = tid * block + (int)fmin(tid, remain);
        int end = start + block + (tid < remain ? 1 : 0);

        workers.emplace_back([this, &proj, &vol, start, end, &finished_slices, &print_mutex, backproject_start]()
            {
                const float PI = std::acosf(-1.0f);
                //const float dz_per_view = HelixDzPerView(m_geo.pitch, m_geo.angleStep);
                const float h_beta = HelixSlopePerBeta(m_geo.pitch, m_geo.angleStep);
                const int det_offset_per_view = m_geo.nDetU * m_geo.nDetV;
                const int scan_type = m_geo.scan_type;
                for (int iz = start; iz < end; iz++) {
                    float z_pos = (-0.5f * m_geo.nz + iz + 0.5) * m_geo.dz;
                    for (int iy = 0; iy < m_geo.ny; iy++) {
                        float y_pos = (-0.5 * m_geo.ny + 0.5 + iy) * m_geo.dy;
                        for (int ix = 0; ix < m_geo.nx; ix++) {
                            float x_pos = (-0.5 * m_geo.nx + 0.5 + ix) * m_geo.dx;
                            float beta_b, beta_t;
                            //if(!calculate_PI_line(this->m_geo.SID, this->m_geo.pitch/(2.f*PI), this->m_geo.zStart, x_pos, y_pos, z_pos, beta_b, beta_t))
                            //    continue;
                            if (!calculate_pi(ix, iy, z_pos, beta_b, beta_t))continue;
                            float index_b = beta_b / this->m_geo.angleStep;
                            float index_t = beta_t / this->m_geo.angleStep;
                            float index_min = std::min(index_b, index_t);
                            float index_max = std::max(index_b, index_t);
                            int view_b = static_cast<int>(std::ceil(index_min));
                            int view_t = static_cast<int>(std::floor(index_max));
                            view_b = std::max(view_b, 1);
                            view_t = std::min(view_t, m_geo.nViews - 2);

                            if (view_b > view_t)
                                continue;
                            for (int iview = view_b; iview <= view_t; iview++) {
                                if (iview < 1 || iview >= m_geo.nViews-1)continue;
                                float angle = iview * m_geo.angleStep;
                                float ray_dire_x = std::cos(angle);
                                float ray_dire_y = std::sin(angle);

                                float sx = m_geo.SID * ray_dire_x;
                                float sy = m_geo.SID * ray_dire_y;
                                float sz = m_geo.zStart + h_beta*angle;
                                int det_offset = iview * det_offset_per_view;
                                int vox_offset = iz * m_geo.ny * m_geo.nx;
                                float u, v, den;
                                if (scan_type == 0)
                                {    
                                    if (!VoxelToFlatDetectorUV(x_pos, y_pos, z_pos, angle, sz,
                                        m_geo.SID, m_geo.SDD, m_geo.du, m_geo.dv,
                                        m_geo.nDetU, m_geo.nDetV, m_geo.detectorVCenterOffsetPix, u, v, den)) continue;
                                }
                                else if (scan_type == 1) {
                                    float source_to_voxel_xy_sq;
                                    if (!VoxelToEquiangularDetectorUV(x_pos, y_pos, z_pos, angle, sz,
                                        m_geo.SID, m_geo.SDD, m_geo.du, m_geo.dv,
                                        m_geo.nDetU, m_geo.nDetV, m_geo.detectorVCenterOffsetPix, u, v, source_to_voxel_xy_sq,den)) continue;
                                }
                                const float q = BilinearInterp(proj.data() + det_offset, u, v, m_geo.nDetU, m_geo.nDetV);
                                const float w = 1.f / (2.f * PI * den);
                                vol[vox_offset + iy * m_geo.nx + ix] += w * q * std::fabs(m_geo.angleStep);
                            }
                        }
                    }
                    const int done = finished_slices.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 5 == 0 || done == m_geo.nz)
                    {
                        std::cout << "BackProject progress: " << done << " / " << m_geo.nz << '\n';
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