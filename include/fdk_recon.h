#pragma once
#include "BaseRecon.h"
#include <vector>

class FDKRecon : public BaseRecon
{
public:
    explicit FDKRecon(const CTGeometry& geo);
    virtual ~FDKRecon() override = default;

    virtual void Reconstruct(const std::vector<float>& proj, std::vector<float>& volume) override;

private:
    void RunCPU_OMP(const std::vector<float>& proj, std::vector<float>& volume);
    void RunGPU_CUDA(const std::vector<float>& proj, std::vector<float>& volume);
    bool HasCudaHardware() const;
};