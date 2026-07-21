#pragma once
#include "../Common/public/ReconBase.h"
class GpuFDKRecon : public BaseRecon
{
public:
    explicit GpuFDKRecon(const CTGeometry& geo);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void LaunchKernel(float* dProj, float* dVol);
};