#pragma once
#include "CTGeometry.h"
#include "ReconAlgoEnum.h"
#include <vector>

// Windows DLL导出宏
#if defined(_WIN32)
#   if defined(RECON_DLL_EXPORTS)
#       define RECON_API __declspec(dllexport)
#   else
#       define RECON_API __declspec(dllimport)
#   endif
#else
#   define RECON_API
#endif

// 重建统一抽象基类，CPU/GPU完全兼容接口
class RECON_API BaseRecon
{
protected:
    CTGeometry m_geo;
public:
    explicit BaseRecon(const CTGeometry& geo) : m_geo(geo) {}
    virtual ~BaseRecon() = default;

    // 纯虚重建接口
    virtual void Reconstruct(const std::vector<float>& projData, std::vector<float>& volumeOut) = 0;
};

// 工厂函数声明
extern "C" RECON_API BaseRecon* CreateCpuRecon(ReconAlgorithm algo, const CTGeometry& geo);
extern "C" RECON_API void DestroyReconInstance(BaseRecon* ptr);

extern "C" RECON_API BaseRecon* CreateGpuRecon(ReconAlgorithm algo, const CTGeometry& geo);
