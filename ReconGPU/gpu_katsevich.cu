#include "gpu_katsevich.cuh"
#include <cuda_runtime.h>

__global__ void FilterKern(float* dProj, CTGeometry g)
{
    // Katsevich滤波GPU预留
}
__global__ void BackProjKern(float* dProj, float* dVol, CTGeometry g)
{
    // Katsevich反投影GPU预留
}

GpuKatsevichRecon::GpuKatsevichRecon(const CTGeometry& geo) : BaseRecon(geo) {}

void GpuKatsevichRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    int pSize = proj.size();
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(vSize, 0.f);
    float *dP = nullptr, *dV = nullptr;
    cudaMalloc(&dP, pSize * sizeof(float));
    cudaMalloc(&dV, vSize * sizeof(float));
    cudaMemcpy(dP, proj.data(), pSize*sizeof(float), cudaMemcpyHostToDevice);
    FilterKernel(dP);
    BackProjKernel(dP, dV);
    cudaMemcpy(vol.data(), dV, vSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dP);
    cudaFree(dV);
}

void GpuKatsevichRecon::FilterKernel(float* dProj)
{
    int pTotal = m_geo.nDetU * m_geo.nDetV * m_geo.nViews;
    dim3 block(256);
    dim3 grid((pTotal + 255) / 256);
    FilterKern<<<grid, block>>>(dProj, m_geo);
}

void GpuKatsevichRecon::BackProjKernel(float* dProj, float* dVol)
{
    int vTotal = m_geo.nx * m_geo.ny * m_geo.nz;
    dim3 block(256);
    dim3 grid((vTotal + 255) / 256);
    BackProjKern<<<grid, block>>>(dProj, dVol, m_geo);
}