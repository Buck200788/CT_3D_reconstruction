#include "gpu_iterative.cuh"
#include <cuda_runtime.h>

GpuIterativehRecon::GpuIterativehRecon(const CTGeometry& geo, const recon_para& recp) : BaseRecon(geo, recp) {}

void GpuIterativehRecon::cleanup()
{
    if (d_proj) cudaFree(d_proj);
    if (d_vol) cudaFree(d_vol);

    d_proj = nullptr;
    d_vol = nullptr;
}

void GpuIterativehRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int pSize = proj.size();
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(vSize, 0.f);

    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(rec_p.cuda_device);
    if (cuda_status != cudaSuccess) { printf("cuda set device failed!!!"); return; }

    cuda_status = cudaMalloc(&d_proj, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_proj failed!!!"); return; }
    cuda_status = cudaMemcpy(d_proj, proj.data(), pSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cudaMemcpy d_proj failed!!!"); return; }
    cuda_status = cudaMalloc(&d_vol, vSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_vol failed!!!"); return; }
    cuda_status = cudaMemset(d_vol, 0, vSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMemSet d_vol failed!!!"); return; }

    cuda_status = cudaMalloc(&d_FP, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMalloc d_delta failed!!!"); return; }
    cuda_status = cudaMemset(d_FP, 0, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cudaMemSet d_delta failed!!!"); return; }


}