#pragma once
#include "../Common/public/ReconBase.h"
class GpuIterativehRecon : public BaseRecon
{
public:
    explicit GpuIterativehRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
    //void set_geom(float* geo) override;
private:
    void cleanup();
    void forward_projection(float* );
    float* d_proj = nullptr;
    float* d_FP = nullptr;
    float* d_vol = nullptr;
};