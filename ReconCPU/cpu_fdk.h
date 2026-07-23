#pragma once
#include "../Common/public/ReconBase.h"
#include "filter.h"
#include <iostream>
class CpuFDKRecon : public BaseRecon
{
public:
    explicit CpuFDKRecon(const CTGeometry& geo);
    void Reconstruct(const std::vector<float>& proj, std::vector<float>& vol) override;
private:
    void BackProjectOMP(const std::vector<float>& proj, std::vector<float>& vol);

};