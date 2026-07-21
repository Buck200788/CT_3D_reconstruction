#pragma once
#include "../Common/public/ReconBase.h"
class GpuKatsevichRecon : public BaseRecon
{
public:
    explicit GpuKatsevichRecon(const CTGeometry& geo);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void FilterKernel(float* dProj);
    void BackProjKernel(float* dProj, float* dVol);
};