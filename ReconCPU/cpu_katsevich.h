#pragma once
#include "../Common/public/ReconBase.h"
#include "float3_helper.h"
class CpuKatsevichRecon : public BaseRecon
{
public:
    explicit CpuKatsevichRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void FilterProj(const std::vector<float>& in, std::vector<float>& out);
    void BackProject(const std::vector<float>& proj, std::vector<float>& vol);
    float PsiOverTanPsi(float psi);
    void calculate_kLines();
    void construct_hilbert_kernel();
    bool calculate_PI_line(const float R, const float h, const float z0, float x, float y, float z, float& beta_b, float& belta_t);
    
    void calculate_kLines_equal_angle();
    void calculate_inverse_Psi_index();

    int m_nPsi = 0;
    float m_psiMin = 0.0f;
    float m_dPsi = 0.0f;
    std::vector<float> m_k_lines;
    std::vector<float> m_inverse_Psi_index;
    std::vector<float> m_hilbert_kernel;
};