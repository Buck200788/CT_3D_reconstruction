#include "CTGeometry.h"
#include "ReconBase.h"
#include "ReconAlgoEnum.h"
#include "device_detect.h"
#include <iostream>
#include <vector>
#include <windows.h>

typedef BaseRecon* (*PFN_CreateCpu)(ReconAlgorithm, const CTGeometry&);
typedef BaseRecon* (*PFN_CreateGpu)(ReconAlgorithm, const CTGeometry&);
typedef void (*PFN_Destroy)(BaseRecon*);

int main()
{
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
    std::cout << "Reconstruct done, vol size:" << volume.size() << "\n";
    fnDel(recon);

clean:
    if (hDll) FreeLibrary(hDll);
    ReleaseCudaRuntimeHandle();
    std::cout << "Press enter exit...\n";
    std::cin.get();
    return 0;
}