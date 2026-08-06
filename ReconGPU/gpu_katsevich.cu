#include "gpu_katsevich.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include "float3_helper.h"

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


__global__ void geom_filter(float* dProj, float* dG1, CTGeometry geo)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nDetU = geo.nDetU;
    const int nDetV = geo.nDetV;
    const int pixels_per_view = nDetU * nDetV;
    const int total_size = pixels_per_view * geo.nViews;
    if (idx >= total_size) return;
    const int view_idx = idx / pixels_per_view;
    const int idx_currentView = idx % pixels_per_view;
    const int v_idx = idx_currentView / nDetU;
    const int u_idx = idx_currentView % nDetU;

    const float dbeta = geo.angleStep;
    const float du = geo.du;
    const float dv = geo.dv;
    const float inv2DBeta = 1.0f / (2.0f * dbeta);
    const float inv2Du = 1.0f / (2.0f * du);
    const float inv2Dv = 1.0f / (2.0f * dv);

    const float D = geo.SDD;
    const float D2 = D * D;

    const float u_center = 0.5f * static_cast<float>(nDetU - 1);
    const float v_center = 0.5f * static_cast<float>(nDetV - 1);

    if (view_idx == 0 || view_idx == geo.nViews - 1 || v_idx == 0 || v_idx == nDetV - 1 || u_idx == 0 || u_idx == nDetU - 1)return;

    const int view1_index = idx - pixels_per_view;
    const int view2_index = idx + pixels_per_view;
    const int v1_index = idx - nDetU;
    const int v2_index = idx + nDetU;
    const int u1_index = idx - 1;
    const int u2_index = idx + 1;

    float dgDbeta = (dProj[view2_index] - dProj[view1_index]) * inv2DBeta;
    float dgDv = (dProj[v2_index] - dProj[v1_index]) * inv2Dv;
    float dgDu = (dProj[u2_index] - dProj[u1_index]) * inv2Du;
    float u = (static_cast<float>(u_idx) - u_center) * du;
    float v = (static_cast<float>(v_idx) - v_center) * dv;
    float g1 = dgDbeta + (D2 + u * u) / D * dgDu + (u * v) / D * dgDv;
    dG1[idx] = D / std::sqrt(D2 + u * u + v * v) * g1;
}

__global__ void cal_G2(float* dG1, float* dG2, float* d_k_lines, CTGeometry geo, int nPsi, float psi_min, float dPsi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nDetU = geo.nDetU;
    const int nDetV = geo.nDetV;
    const int pixels_per_view = nDetU * nPsi;
    const int total_size = pixels_per_view * geo.nViews;
    if (idx >= total_size) return;

    const int view_idx = idx / pixels_per_view;
    const int idx_currentView = idx % pixels_per_view;
    const int psi_idx = idx_currentView / nDetU;
    const int u_idx = idx_currentView % nDetU;
    float v_idx = d_k_lines[idx_currentView];

    int v0 = static_cast<int>(floorf(v_idx));
    if (v0 < 0.0f || v0 > nDetV - 1)
    {
        dG2[idx] = 0.0f;
        return;
    }

    int v1= (v0 + 1 < nDetV) ? v0 + 1 : v0;
    float w1 = v_idx - v0;

    int g1_view_offset = view_idx*nDetU * nDetV;
    int g10_idx = g1_view_offset+ v0 * nDetU + u_idx;
    int g11_idx = g1_view_offset+v1 * nDetU + u_idx;
    
    dG2[idx]= (1.f - w1) * dG1[g10_idx] + w1 * dG1[g11_idx];
}

__global__ void cal_G3_filter(float* dG2, float* dG3, float* d_filter, CTGeometry geo, int filter_len, int nPsi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nDetU = geo.nDetU;
    const int pixels_per_view = nDetU * nPsi;
    const int total_size = pixels_per_view * geo.nViews;
    if (idx >= total_size) return;

    const int idx_currentView = idx % pixels_per_view;
    const int psi_idx = idx_currentView / nDetU;
    const int u_idx = idx_currentView % nDetU;

    const int half_win = filter_len / 2;
    float sum_conv = 0.0f;
    for (int iu1 = 0; iu1 < filter_len; ++iu1)
    {
        const int iu_shift = u_idx + half_win - iu1;
        if (iu_shift >= 0 && iu_shift < nDetU)
        {
            sum_conv += d_filter[iu1] * dG2[idx + half_win - iu1];
        }
    }

    dG3[idx] = sum_conv * geo.du;
}

__global__ void cal_filtedProj(float* dG3, float* d_filtedProj, float* d_inverse_Psi_idx, CTGeometry geo, int nPsi)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nDetU = geo.nDetU;
    const int nDetV = geo.nDetV;
    const int pixels_per_view = nDetU * nDetV;
    const int total_size = pixels_per_view * geo.nViews;
    if (idx >= total_size) return;
    const int view_idx = idx / pixels_per_view;
    const int idx_currentView = idx % pixels_per_view;
    const int v_idx = idx_currentView / nDetU;
    const int u_idx = idx_currentView % nDetU;

    if (view_idx == 0 || view_idx == geo.nViews - 1 || u_idx == 0 || u_idx == nDetU - 1 || v_idx == 0 || v_idx == nDetV - 1)
    {
        d_filtedProj[idx] = 0.0f;
        return;
    }

    float psi_idx = d_inverse_Psi_idx[idx_currentView];
    if (psi_idx < 0.0f || psi_idx > static_cast<float>(nPsi - 1))
    {
        d_filtedProj[idx] = 0.f;
        return;
    }
    int psi0 = static_cast<int>(floorf(psi_idx));
    psi0 = min(psi0, nPsi - 1);
    const int psi1 = (psi0 + 1 < nPsi) ? psi0 + 1 : psi0;
    float w1 = psi_idx - psi0;
    int psi_view_offset = view_idx*nDetU * nPsi;
    int g30_idx = psi_view_offset+psi0 * nDetU + u_idx;
    int g31_idx = psi_view_offset+psi1 * nDetU + u_idx;
    d_filtedProj[idx] = (1.f - w1) * dG3[g30_idx] + w1 * dG3[g31_idx];
}

inline __device__ float float_clamp(float x, float minv, float maxv)
{
    return fmax(minv, fmin(x, maxv));
}

__device__ float evaluate(float beta_m, float x, float y, float z, float R, float z0, float h)
{
    float sin_beta = sin(beta_m);
    float cos_beta = cos(beta_m);
    float cos_delta = (x * cos_beta + y * sin_beta) / R;
    cos_delta = float_clamp(cos_delta, -1.f, 1.f);
    float delta_m = acos(cos_delta);
    float sin_delta = sin(delta_m);
    float mu = (-x * sin_beta + y * cos_beta) / (R * sin_delta);
    return z0 + h * (beta_m + mu * delta_m) - z;
}

__device__ bool calculate_PI_line(const float R, const float h, const float z0, float x, float y, float z, float& beta_b, float& belta_t)
{
    const float PI = acos(-1.f);
    if (fabs(h) < 1.0e-12f) return false;
    float beta_z = (z - z0) / h;
    float lo = beta_z - PI;
    float hi = beta_z + PI;
    float f_lo = evaluate(lo,x,y,z,R,z0,h);
    float f_hi = evaluate(hi, x, y, z, R, z0, h);

    if (!isfinite(f_lo) || !isfinite(f_hi)) return false;
    if (f_lo * f_hi > 0) return false;
    for (size_t i = 0; i < 40; i++) {
        float mid = 0.5 * (lo + hi);
        float f_mid = evaluate(mid, x, y, z, R, z0, h);
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

    float sin_beta = sin(beta);
    float cos_beta = cos(beta);
    float cos_delta = (x * cos_beta + y * sin_beta) / R;
    cos_delta = float_clamp(cos_delta, -1.f, 1.f);
    float delta = acos(cos_delta);
    beta_b = beta - delta;
    belta_t = beta + delta;
    return true;
}

__global__ void BackProjKern(float* dProj, float* dVol, CTGeometry geo)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nx = geo.nx;
    const int ny = geo.ny;
    const int nz = geo.nz;
    const float dx = geo.dx;
    const float dy = geo.dy;
    const float dz = geo.dz;
    const int nVox_per_z = nx * ny;
    const int total_size = nVox_per_z *nz;
    if (idx >= total_size) return;

    const float PI = acos(-1.f);
    
    const int iz = idx / nVox_per_z;
    const int idx_current_z = idx % nVox_per_z;
    const int iy = idx_current_z / nx;
    const int ix = idx_current_z % nx;

    float z_pos = (-0.5f * nz + iz + 0.5) * dz;
    float y_pos = (-0.5f * ny + iy + 0.5) * dy;
    float x_pos = (-0.5f * nx + ix + 0.5) * dx;

    float beta_b, beta_t;
    if (!calculate_PI_line(geo.SID, geo.pitch / (2.f * PI), geo.zStart, x_pos, y_pos, z_pos, beta_b, beta_t)) return;
    float index_b = beta_b / geo.angleStep;
    float index_t = beta_t / geo.angleStep;
    float index_min = min(index_b, index_t);
    float index_max = max(index_b, index_t);
    int view_b = static_cast<int>(ceil(index_min));
    int view_t = static_cast<int>(floor(index_max));
    const float dz_per_view = geo.pitch / (2.f * PI / geo.angleStep);
    const int det_offset_per_view = geo.nDetU * geo.nDetV;
    const int scan_type = geo.scan_type;
    for (int iview = view_b; iview <= view_t; iview++) {
        if (iview < 1 || iview >= geo.nViews - 1)continue;
        float angle = iview * geo.angleStep;
        float ray_dire_x = cos(angle);
        float ray_dire_y = sin(angle);

        float sx = geo.SID * ray_dire_x;
        float sy = geo.SID * ray_dire_y;
        float sz = iview * dz_per_view + geo.zStart;
        int det_offset = iview * det_offset_per_view;
        if (scan_type == 0)
        {
            float u, v, den;
            if (!VoxelToFlatDetectorUV(x_pos, y_pos, z_pos, angle, sz,
                geo.SID, geo.SDD, geo.du, geo.dv,
                geo.nDetU, geo.nDetV, geo.detectorVCenterOffsetPix, u, v, den)) continue;
            const float q = BilinearInterp(dProj + det_offset, u, v, geo.nDetU, geo.nDetV);
            const float w = 1.f / (2.f * PI * den);
            dVol[idx] += w * q * std::fabs(geo.angleStep);
        }
        else if (scan_type == 1) {

        }
    }
}

float GpuKatsevichRecon::PsiOverTanPsi(float psi)
{
    const float a = std::fabs(psi);
    if (a < 1.0e-4f)
    {
        const float psi2 = psi * psi;
        return 1.0f - psi2 / 3.0f - psi2 * psi2 / 45.0f;
    }
    return psi / std::tan(psi);
}
void GpuKatsevichRecon::calculate_kLines()
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
            const float v_kappa = kappa_scale * (psi + q * u / D);
            const float v_index = v_kappa / m_geo.dv + v_center + m_geo.detectorVCenterOffsetPix;
            m_k_lines[ipsi * m_geo.nDetU + iu] = v_index;
        }
    }
    m_nPsi = nPsi;
    m_psiMin = psi_min;
    m_dPsi = dPsi;
}
void GpuKatsevichRecon::calculate_kLines_equal_angle()
{
    const float D = m_geo.SDD;
    const float P = SignedPitchPerBeta(m_geo.pitch, m_geo.angleStep);
    const float R = m_geo.SID;
    const float PI = std::acos(-1.f);
    const float dAlpha = m_geo.du;
    const int nAlpha = m_geo.nDetU;
    const float dw = m_geo.dv;
    const int nDetV = m_geo.nDetV;
    const float alpha_center = 0.5f * static_cast<float>(nAlpha - 1);
    const float w_center = 0.5f * static_cast<float>(nDetV - 1);

    float alphaM = 0.5 * nAlpha * dAlpha;
    const float psiMax = 0.5 * PI + alphaM;
    const float psiMin = -1.f * psiMax;
    float alpha_edge = -alphaM;

    const float sin_psi = std::sin(psiMax);
    const float cos_psi = std::cos(psiMax);
    const float cot_psi = cos_psi / sin_psi;
    const float csc2_psi = 1.0f / (sin_psi * sin_psi);
    const float kappa_scale = D * P / (2.0f * PI * R);
    const float q_prime = cot_psi - psiMax * csc2_psi;

    const float max_dwdpsi = std::fabs(kappa_scale * (std::cos(alpha_edge) + std::sin(alpha_edge) * q_prime));
    const float A = 0.5f * PI + alphaM;
    int M = static_cast<int>(std::ceil(A * max_dwdpsi / dw));
    M = std::max(M, 1);
    const int nPsi = 2 * M + 1;
    const float dPsi = A / static_cast<float>(M);

    m_nPsi = nPsi;
    m_psiMin = psiMin;
    m_dPsi = dPsi;

    m_k_lines.assign(nPsi * nAlpha, 0.f);

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
            const float w_index = w_kappa / dw + w_center + m_geo.detectorVCenterOffsetPix;
            m_k_lines[static_cast<size_t>(ipsi) * nAlpha + ialpha] = w_index;
        }
    }

}
void GpuKatsevichRecon::calculate_inverse_Psi_index()
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

void GpuKatsevichRecon::construct_hilbert_kernel()
{
    const float PI = std::acos(-1.f);
    const float twoOverPI = 2.f / PI;

    int kernel_len = m_geo.nDetU * 2 - 1;
    m_hilbert_kernel.assign(kernel_len, 0.f);
    for (size_t i = m_geo.nDetU; i < kernel_len; i += 2) {
        m_hilbert_kernel[i] = twoOverPI / (i - m_geo.nDetU + 1) / m_geo.du;
        if (i > m_geo.nDetU - 1) {
            m_hilbert_kernel[kernel_len - 1 - i] = -m_hilbert_kernel[i];
        }
    }
}

GpuKatsevichRecon::GpuKatsevichRecon(const CTGeometry& geo, const recon_para& recp) : BaseRecon(geo, recp) {}

void GpuKatsevichRecon::cleanup()
{
    if (d_k_lines) cudaFree(d_k_lines);
    if (d_inverse_Psi_index) cudaFree(d_inverse_Psi_index);
    if (d_hilbert_kernel) cudaFree(d_hilbert_kernel);
    if (d_proj) cudaFree(d_proj);
    if (d_G1) cudaFree(d_G1);
    if (d_G2) cudaFree(d_G2);
    if (d_G3) cudaFree(d_G3);
    if (d_filtedProj) cudaFree(d_filtedProj);
    if (d_vol) cudaFree(d_vol);

    d_k_lines = nullptr;
    d_inverse_Psi_index = nullptr;
    d_hilbert_kernel = nullptr;
    d_proj = nullptr;
    d_G1 = nullptr;
    d_G2 = nullptr;
    d_G3 = nullptr;
    d_filtedProj = nullptr;
    d_vol = nullptr;
}

void GpuKatsevichRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int pSize = proj.size();
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(vSize, 0.f);

    if (m_geo.scan_type != 0 && m_geo.scan_type != 1)
    {
        throw std::invalid_argument("Unsupported detector scan type");
    }
    if (std::fabs(m_geo.angleStep) < GEOM_EPS) throw std::invalid_argument("angleStep must be non-zero");

    if (std::fabs(m_geo.pitch) < GEOM_EPS)
        throw std::invalid_argument("Helical pitch must be non-zero");
    int size = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(size, 0.f);

    if (m_geo.scan_type == 0)
        calculate_kLines();
    else if (m_geo.scan_type == 1)
        calculate_kLines_equal_angle();
    printf("calculating the Klines done\n");
    calculate_inverse_Psi_index();
    printf("calculating inverse Klines done\n");
    construct_hilbert_kernel();
    printf("calculating Hilbert kernel done\n");

    int kline_size = m_nPsi * m_geo.nDetU;
    int psi_total_size = kline_size * m_geo.nViews;
    int inverse_size = m_geo.nDetU * m_geo.nDetV;

    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(rec_p.cuda_device);
    if (cuda_status != cudaSuccess) { printf("cuda set device failed!!!"); return; }

    cuda_status = cudaMalloc(&d_k_lines, kline_size * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_k_lines failed!!!"); return; }
    cuda_status = cudaMemcpy(d_k_lines, m_k_lines.data(), kline_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy d_k_lines failed!!!"); return; }

    cuda_status = cudaMalloc(&d_inverse_Psi_index, inverse_size * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_inverse_Psi_index failed!!!"); return; }
    cuda_status = cudaMemcpy(d_inverse_Psi_index, m_inverse_Psi_index.data(), inverse_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy d_inverse_Psi_index failed!!!"); return; }

    cuda_status = cudaMalloc(&d_hilbert_kernel, m_hilbert_kernel.size() * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_hilbert_kernel failed!!!"); return; }
    cuda_status = cudaMemcpy(d_hilbert_kernel, m_hilbert_kernel.data(), m_hilbert_kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy d_hilbert_kernel failed!!!"); return; }

    cuda_status = cudaMalloc(&d_G1, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_G1 failed!!!"); return; }
    cuda_status = cudaMalloc(&d_G2, psi_total_size * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_G2 failed!!!"); return; }
    cuda_status = cudaMalloc(&d_G3, psi_total_size * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_G3 failed!!!"); return; }
    cuda_status = cudaMalloc(&d_filtedProj, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc vs failed!!!"); return; }

    cuda_status = cudaMalloc(&d_proj, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_proj failed!!!"); return; }
    cuda_status = cudaMemcpy(d_proj, proj.data(), pSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy d_proj failed!!!"); return; }

    cuda_status = cudaMalloc(&d_vol, vSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_vol failed!!!"); return; }

    cudaMemset(d_G1, 0, pSize * sizeof(float));
    cudaMemset(d_G2, 0, psi_total_size * sizeof(float));
    cudaMemset(d_G3, 0, psi_total_size * sizeof(float));
    cudaMemset(d_filtedProj, 0, pSize * sizeof(float));
    cudaMemset(d_vol, 0, vSize * sizeof(float));

    if (!FilterKernel(d_proj))
    {
        cleanup();
        throw std::runtime_error("CUDA filtering failed");
    }

    if (!BackProjKernel(d_filtedProj, d_vol))
    {
        cleanup();
        throw std::runtime_error("CUDA backprojection failed");
    }

    cuda_status = cudaMemcpy(vol.data(), d_vol, vSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy vol failed!!!"); return; }

    cleanup();    
}

bool GpuKatsevichRecon::FilterKernel(float* dProj)
{
    int pTotal = m_geo.nDetU * m_geo.nDetV * m_geo.nViews;
    const int blockSize = 256;
    dim3 block(blockSize, 1, 1);
    dim3 grid((pTotal + blockSize - 1) / blockSize, 1, 1);
    geom_filter <<<grid, block>>>(dProj, d_G1,m_geo);
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr << "geom_filter cudaDeviceSynchronize failed: " 
            << cudaGetErrorName(cuda_status) << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    int psi_Total = m_geo.nDetU * m_nPsi * m_geo.nViews;
    dim3 grid1((psi_Total + blockSize - 1) / blockSize, 1, 1);
    cal_G2 <<< grid1, block >> > (d_G1, d_G2, d_k_lines, m_geo, m_nPsi, m_psiMin, m_dPsi);
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr << "cal_G2 cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status) << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    cal_G3_filter << < grid1, block >> > (d_G2, d_G3, d_hilbert_kernel, m_geo, m_hilbert_kernel.size(), m_nPsi);
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr << "cal_G3_filter cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status) << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }

    cal_filtedProj << <grid, block >> > (d_G3,d_filtedProj,d_inverse_Psi_index,m_geo,m_nPsi);
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr << "cal_filtedProj cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status) << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }
    return true;
}

bool GpuKatsevichRecon::BackProjKernel(float* dProj, float* dVol)
{
    int vTotal = m_geo.nx * m_geo.ny * m_geo.nz;
    dim3 block(256);
    dim3 grid((vTotal + 255) / 256);
    BackProjKern<<<grid, block>>>(dProj, dVol, m_geo);

    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr << "BackProjKern cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status) << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status) << std::endl;
        return false;
    }
    return true;
}