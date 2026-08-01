#pragma once
#include "../Common/public/ReconBase.h"
class CpuKatsevichRecon : public BaseRecon
{
public:
    explicit CpuKatsevichRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void FilterProj(const std::vector<float>& in, std::vector<float>& out);
    void BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol);
    float PsiOverTanPsi(float psi);
    void calculate_kLines();
    void construct_hilbert_kernel();

    int m_nPsi = 0;
    float m_psiMin = 0.0f;
    float m_dPsi = 0.0f;
    std::vector<float> m_k_lines;
    std::vector<float> m_inverse_Psi_index;
    std::vector<float> m_hilbert_kernel;

};