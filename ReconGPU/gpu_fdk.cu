#include "gpu_fdk.cuh"
#include <cuda_runtime.h>

__global__ void FDKKernel(float* dProj, float* dVol, CTGeometry geo)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    int volSize = geo.nx * geo.ny * geo.nz;
    if (idx >= volSize) return;
    // GPU FDK计算预留
}

GpuFDKRecon::GpuFDKRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void GpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int pSize = proj.size();
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(vSize, 0.f);
    float *dP = nullptr, *dV = nullptr;
    cudaMalloc(&dP, pSize * sizeof(float));
    cudaMalloc(&dV, vSize * sizeof(float));
    cudaMemcpy(dP, proj.data(), pSize*sizeof(float), cudaMemcpyHostToDevice);
    LaunchKernel(dP, dV);
    cudaMemcpy(vol.data(), dV, vSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dP);
    cudaFree(dV);
}

void GpuFDKRecon::LaunchKernel(float* dProj, float* dVol)
{
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    dim3 block(256);
    dim3 grid((vSize + 255) / 256);
    FDKKernel<<<grid, block>>>(dProj, dVol, m_geo);
}