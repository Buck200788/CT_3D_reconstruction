#pragma once
#include "../Common/public/ReconBase.h"
#include "../Common/public/filter.h"
class GpuFDKRecon : public BaseRecon
{
public:
    explicit GpuFDKRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void LaunchKernel(float* dProj, float* dVol);
    float* d_proj = nullptr;
    float* d_proj_geom_filtered = nullptr;
    float* h_proj_geom_filtered = nullptr;
    float* d_vox = nullptr;
    float* d_filter=nullptr;
};