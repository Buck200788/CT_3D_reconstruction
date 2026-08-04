#include "CTGeometry.h"
#include "ReconBase.h"
#include "ReconAlgoEnum.h"
#include "device_detect.h"
#include <iostream>
#include <vector>
#include <windows.h>
#include <fstream>
#include "FFT1D.h"

#include <chrono>
using namespace std::chrono;


int test_FFT1D()
{
    FFT1D fftSolver;
    std::vector<double> sig = { 1, 2, 3, 4, 5, 6, 7, 1 };

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

typedef BaseRecon* (*PFN_CreateCpu)(ReconAlgorithm, const CTGeometry&, const recon_para&);
typedef BaseRecon* (*PFN_CreateGpu)(ReconAlgorithm, const CTGeometry&, const recon_para&);
typedef void (*PFN_Destroy)(BaseRecon*);

int main()
{
    //test_FFT1D();
    //return 0;
    CTGeometry geo;
    recon_para rec_p;

    //geo.nDetU = 512;
    //geo.nDetV = 128;
    //geo.du = 1.0f;
    //geo.dv = 1.0f;
    //geo.SDD = 1000.f;
    //geo.SID = 500.f;
    //geo.pitch = 36.f;
    //geo.nViews = 1600;
    //geo.angleStep = 1.f / 180.f * std::acosf(-1.f);
    //geo.nx = 256;
    //geo.ny = 256;
    //geo.nz = 128;
    //geo.dx = 1.f;
    //geo.dy = 1.f;
    //geo.dz = 1.f;
    //geo.zStart = -80.f;
    //geo.scan_type = 0;
    //
    //rec_p.filter_name = "ram-lak";
    //rec_p.cuda_device = 0;

    //geo.nDetU = 512;
    //geo.nDetV = 12;
    //geo.du = 0.002569f; // d_angle
    //geo.dv = 10.f; // d_det
    //geo.SDD = 1180.f;
    //geo.SID = 710.;
    //geo.pitch = -120.f;
    //geo.nViews = 5040;
    //geo.angleStep = 0.5f / 180.f * std::acosf(-1.f);
    //geo.nx = 512;
    //geo.ny = 512;
    //geo.nz = 700;
    //geo.dx = 1.f;
    //geo.dy = 1.f;
    //geo.dz = 1.0f;
    //geo.zStart = 350.f;
    //geo.scan_type = 1;
    //
    //rec_p.filter_name = "ram-lak";
    //rec_p.cuda_device = 0;


    geo.nDetU = 512;
    geo.nDetV = 12;
    geo.du = 0.002569f; // d_angle
    geo.dv = 10.0f; // d_det
    geo.SDD = 1180.f;
    geo.SID = 710.;
    geo.pitch = -120.f;
    geo.nViews = 4320;
    geo.angleStep = 0.5f / 180.f * std::acosf(-1.f);
    geo.nx = 512;
    geo.ny = 512;
    geo.nz = 700;
    geo.dx = 1.f;
    geo.dy = 1.f;
    geo.dz = 1.f;
    geo.zStart = 350.f;
    geo.scan_type = 1;
    
    rec_p.filter_name = "ram-lak";
    rec_p.cuda_device = 0;

    //ReconAlgorithm algo = ReconAlgorithm::FDK;

    ReconAlgorithm algo = ReconAlgorithm::Katsevich;
    //ReconAlgorithm algo = ReconAlgorithm::FDK;
    std::vector<float> proj(geo.nDetU * geo.nDetV * geo.nViews, 1.f);
    std::vector<float> volume(geo.nx * geo.ny * geo.nz, 0.f);

    auto t0 = high_resolution_clock::now();


    //std::ifstream ins("D:\\bagC_512x128x1600.raw", std::ios::binary | std::ios::in);
    std::ifstream ins("D:\\bag_interp_3_6144x4320.raw", std::ios::binary | std::ios::in);
    //std::ifstream ins("D:\\bag_interp_1_6144x5040.raw", std::ios::binary | std::ios::in);
    if (ins.is_open()) {
        ins.read(reinterpret_cast<char*>(proj.data()), sizeof(float) * proj.size());
        ins.close();
    }

    auto t1 = high_resolution_clock::now();
    auto cost_ms = duration_cast<milliseconds>(t1 - t0);
    float cost_s = duration<float>(t1 - t0).count();
    printf("read projection：%lld ms，%.3f s\n", cost_ms.count(), cost_s);
    t0 = t1;

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
            if (fnCreate && fnDel) recon = fnCreate(algo, geo, rec_p);
        }
    }

    if (!recon)
    {
        std::cout << "Use CPU parallel mode\n";
        hDll = LoadLibraryA("ReconCPU.dll");
        if (hDll)
        {
            auto fnCreate = (PFN_CreateCpu)GetProcAddress(hDll, "CreateCpuRecon");
            fnDel = (PFN_Destroy)GetProcAddress(hDll, "DestroyReconInstance");
            if (fnCreate && fnDel) recon = fnCreate(algo, geo, rec_p);
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
    t1 = high_resolution_clock::now();
    cost_ms = duration_cast<milliseconds>(t1 - t0);
    cost_s = duration<float>(t1 - t0).count();
    printf("reconstruction：%lld ms，%.3f s\n", cost_ms.count(), cost_s);
    t0 = t1;

    char wf[256] = {};
    sprintf(wf, "D:\\recon_%dx%dx%d.raw", geo.nx, geo.ny, geo.nz);
    std::ofstream outs(wf, std::ios::binary | std::ios::out);
    if (outs.is_open()) {
        outs.write(reinterpret_cast<char*>(volume.data()), sizeof(float) * volume.size());
        outs.close();
    }

    t1 = high_resolution_clock::now();
    cost_ms = duration_cast<milliseconds>(t1 - t0);
    cost_s = duration<float>(t1 - t0).count();
    printf("writing recon file：%lld ms，%.3f s\n", cost_ms.count(), cost_s);
    t0 = t1;

    if (hDll) FreeLibrary(hDll);
    ReleaseCudaRuntimeHandle();
    std::cout << "Press enter exit...\n";
    //std::cin.get();
    return 0;
}