#pragma once
#include "../Common/public/ReconBase.h"
#include "filter.h"
#include "float3_helper.h"
#include <iostream>
class CpuFDKRecon : public BaseRecon
{
public:
    explicit CpuFDKRecon(const CTGeometry& geo, const recon_para& recp);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol);
    void ParallelPreprocessProj(std::vector<float>& proj_buf, const std::vector<float>& filter) const;
    //bool GetDetectorUV(int view, float vx, float vy, float vz, float& out_u, float& out_v) const;
};