#pragma once
#include "../Common/public/ReconBase.h"
class GpuKatsevichRecon : public BaseRecon
{
public:
    explicit GpuKatsevichRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    bool FilterKernel(float* dProj);
    bool BackProjKernel(float* dProj, float* dVol);

    float PsiOverTanPsi(float psi);
    void calculate_kLines();
    void calculate_kLines_equal_angle();
    void calculate_inverse_Psi_index();
    void construct_hilbert_kernel();
    bool build_PI_LUT();

    void cleanup();

    int m_nPsi = 0;
    float m_psiMin = 0.0f;
    float m_dPsi = 0.0f;
    int m_nPILines_per_pitch = 0;
    std::vector<float> m_k_lines;
    std::vector<float> m_inverse_Psi_index;
    std::vector<float> m_hilbert_kernel;

    float* d_k_lines = nullptr;
    float* d_inverse_Psi_index = nullptr;
    float* d_hilbert_kernel = nullptr;

    float* d_proj = nullptr;
    float* d_G1 = nullptr;
    float* d_G2 = nullptr;
    float* d_G3 = nullptr;
    float* d_filtedProj = nullptr;
    float* d_vol = nullptr;
    float* d_pi_LUT = nullptr;
};