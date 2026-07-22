#include "CTGeometry.h"
#include "ReconBase.h"
#include "ReconAlgoEnum.h"
#include "device_detect.h"
#include <iostream>
#include <vector>
#include <windows.h>

#include "FFT1D.h"

int test_FFT1D()
{
    FFT1D fftSolver;
    std::vector<double> sig = { 1, 2, 3, 4, 5, 6, 7, 8 };

    // 正变换FFT
    auto fftRes = fftSolver.fft(sig);
    std::cout << "FFT结果：\n";
    for (auto& v : fftRes)
        std::cout << v << " ";
    std::cout << "\n";

    std::cout << "abs(fft)：\n";
    for (auto& v : fftRes)
        std::cout << std::abs(v) << " ";
    std::cout << "\n";

    // 逆变换IFFT还原时域
    auto ifftRes = fftSolver.ifft(fftRes);
    std::cout << "IFFT还原实部：\n";
    for (auto& v : ifftRes)
        std::cout << v.real() << " ";

    return 0;
}

typedef BaseRecon* (*PFN_CreateCpu)(ReconAlgorithm, const CTGeometry&);
typedef BaseRecon* (*PFN_CreateGpu)(ReconAlgorithm, const CTGeometry&);
typedef void (*PFN_Destroy)(BaseRecon*);

int main()
{
    test_FFT1D();
    return 0;
    CTGeometry geo;
    ReconAlgorithm algo = ReconAlgorithm::FDK;
    std::vector<float> proj(geo.nDetU * geo.nDetV * geo.nViews, 1.f);
    std::vector<float> volume;

    BaseRecon* recon = nullptr;
    HMODULE hDll = nullptr;
    PFN_Destroy fnDel = nullptr;

    if (false && HasAvailableCudaDevice())
    {
        std::cout << "Try GPU mode\n";
        hDll = LoadLibraryA("ReconGPU.dll");
        if (hDll)
        {
            auto fnCreate = (PFN_CreateGpu)GetProcAddress(hDll, "CreateGpuRecon");
            fnDel = (PFN_Destroy)GetProcAddress(hDll, "DestroyReconInstance");
            if (fnCreate && fnDel) recon = fnCreate(algo, geo);
        }
    }

    if (!recon)
    {
        std::cout << "Use CPU OMP mode\n";
        hDll = LoadLibraryA("ReconCPU.dll");
        if (hDll)
        {
            auto fnCreate = (PFN_CreateCpu)GetProcAddress(hDll, "CreateCpuRecon");
            fnDel = (PFN_Destroy)GetProcAddress(hDll, "DestroyReconInstance");
            if (fnCreate && fnDel) recon = fnCreate(algo, geo);
        }
    }

    if (!recon)
    {
        std::cerr << "Load library failed!\n";
        goto clean;
    }

    recon->Reconstruct(proj, volume);
    //std::cout << "Reconstruct done, vol size:" << volume.size() << "\n";
    fnDel(recon);

clean:
    if (hDll) FreeLibrary(hDll);
    ReleaseCudaRuntimeHandle();
    std::cout << "Press enter exit...\n";
    std::cin.get();
    return 0;
}