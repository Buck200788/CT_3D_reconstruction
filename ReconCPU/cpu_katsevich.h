#pragma once
#include "../Common/public/ReconBase.h"
class CpuKatsevichRecon : public BaseRecon
{
public:
    explicit CpuKatsevichRecon(const CTGeometry& geo);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void FilterProj(const std::vector<float>& in, std::vector<float>& out);
    void BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol);
};