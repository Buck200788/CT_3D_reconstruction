#include "gpu_fdk.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

__global__ void FDKKernel(float* dProj, float* dVol, CTGeometry geo)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    int volSize = geo.nx * geo.ny * geo.nz;
    if (idx >= volSize) return;
    // GPU FDK计算预留
}

__global__ void proj_geom_filted(const float* dProj, float* d_proj_geom_filtered, CTGeometry geo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = geo.nDetU * geo.nDetV * geo.nViews;
    if (idx >= total_size) return;
    int nDetU = geo.nDetU;
    int nDetV = geo.nDetV;
    float du = geo.du;
    float dv = geo.dv;
    float u_len = du * nDetU;
    float v_len = dv * nDetV;
    float D = geo.SDD;

    int idx_per_view = idx % (nDetU * nDetV);
    int iu = idx_per_view % nDetU;
    int iv = idx_per_view / nDetU;
    float u = -0.5f * u_len + (iu + 0.5f) * du;
    float v = -0.5f * v_len + (iv + 0.5f) * dv;
    float weight = D / sqrt(D*D + u * u + v * v);
    d_proj_geom_filtered[idx] = dProj[idx] *weight;
}

__global__ void proj_filtered(float* d_proj, const float* d_proj_geom_filtered, const float* d_filter, int filter_len, CTGeometry geo)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int nDetU = geo.nDetU;
    const int nDetV = geo.nDetV;
    const int pixels_per_view = nDetU * nDetV;
    const int total_size = pixels_per_view * geo.nViews;

    if (idx >= total_size)
        return;

    const int idx_per_view = idx % pixels_per_view;
    const int iu = idx_per_view % nDetU;
    const int iv = idx_per_view / nDetU;
    const int view = idx / pixels_per_view;

    const int half_win = filter_len / 2;

    float sum_conv = 0.0f;

    for (int iu1 = 0; iu1 < filter_len; ++iu1)
    {
        const int iu_shift = iu - half_win + iu1;

        if (iu_shift >= 0 && iu_shift < nDetU)
        {
            sum_conv += d_filter[iu1] * d_proj_geom_filtered[idx - half_win + iu1];
        }
    }

    d_proj[idx] = sum_conv * geo.du;
}

GpuFDKRecon::GpuFDKRecon(const CTGeometry& geo, const recon_para& recp) : BaseRecon(geo, recp) {
}

void GpuFDKRecon::Reconstruct(const std::vector<float>& proj, std::vector<float>& vol)
{
    Filter filt(m_geo.nDetU, Str2FilterType(rec_p.filter_name), m_geo.du);
    std::vector<float> filter = filt.GetFilter();
    std::cout << "gpu filter length: " << filter.size() << std::endl;

    int pSize = proj.size();
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    vol.assign(vSize, 0.f);

    h_proj_geom_filtered = new float[pSize];

    cudaError_t cuda_status;
    cuda_status = cudaSetDevice(rec_p.cuda_device);
    if (cuda_status != cudaSuccess) { printf("cuda set device failed!!!"); return; }
    
    cuda_status=cudaMalloc(&d_proj, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cuda malloc d_proj failed!!!"); return; }
    cuda_status = cudaMemcpy(d_proj, proj.data(), pSize * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cuda memcpy d_proj failed!!!"); return; }

    cuda_status = cudaMalloc(&d_proj_geom_filtered, pSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cuda malloc d_proj_geom_filtered failed!!!"); return; }

    cuda_status = cudaMalloc(&d_filter, filter.size() * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cuda malloc d_filter failed!!!"); return; }
    cuda_status = cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) { printf("cuda memcpy d_filter failed!!!"); return; }

    const int blockSize = 256;
    dim3 block(blockSize, 1, 1);
    dim3 grid((pSize + blockSize - 1) / blockSize, 1, 1);

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        std::cerr
            << "proj_geom_filted before launch failed: "
            << cudaGetErrorName(cuda_status)
            << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status)
            << std::endl;

        return;
    }

    proj_geom_filted << <grid, block >> > (d_proj, d_proj_geom_filtered, m_geo);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        std::cerr
            << "proj_geom_filted launch failed: "
            << cudaGetErrorName(cuda_status)
            << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status)
            << std::endl;

        return;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr
            << "cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status)
            << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status)
            << std::endl;

        return;
    }

    proj_filtered << <grid, block >> > (d_proj, d_proj_geom_filtered, d_filter, filter.size(), m_geo);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        std::cerr
            << "proj_filtered launch failed: "
            << cudaGetErrorName(cuda_status)
            << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status)
            << std::endl;

        return;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess)
    {
        std::cerr
            << "cudaDeviceSynchronize failed: "
            << cudaGetErrorName(cuda_status)
            << " (" << static_cast<int>(cuda_status) << "): "
            << cudaGetErrorString(cuda_status)
            << std::endl;

        return;
    }
    
    cuda_status = cudaMemcpy(h_proj_geom_filtered, d_proj, pSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        printf("cuda memcpy h_proj_geom_filtered failed!!!");
        return;
    }
    //std::ofstream outs("E:\\CFiles\\CTReconTest\\debug\\debug_projFiltered_512x128x1600.raw",std::ios::binary);
    //if (outs.is_open()) {
    //    std::cout << "writing to file" << std::endl;
    //    outs.write(reinterpret_cast<const char*>(h_proj_geom_filtered), sizeof(float) * pSize);
    //    outs.close();
    //}
    //else {
    //    std::cout << "open file failed" << std::endl;
    //}
    

    cuda_status=cudaMalloc(&d_vox, vSize * sizeof(float));
    if (cuda_status != cudaSuccess) { printf("cuda malloc d_vox failed!!!"); return; }

    LaunchKernel(d_proj, d_vox);
    cuda_status=cudaMemcpy(vol.data(), d_vox, vSize*sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) { printf("cuda memcpy vox failed!!!"); return; }

    cuda_status=cudaFree(d_proj);
    if (cuda_status != cudaSuccess) { printf("cuda free d_proj failed!!!"); return; }
    cuda_status=cudaFree(d_vox);
    if (cuda_status != cudaSuccess) { printf("cuda free d_vox failed!!!"); return; }
    cuda_status = cudaFree(d_filter);
    if (cuda_status != cudaSuccess) { printf("cuda free d_filter failed!!!"); return; }
    cuda_status = cudaFree(d_proj_geom_filtered);
    if (cuda_status != cudaSuccess) { printf("cuda free d_proj_geom_filtered failed!!!"); return; }
    if (h_proj_geom_filtered != nullptr)delete[]h_proj_geom_filtered;
}

void GpuFDKRecon::LaunchKernel(float* dProj, float* dVol)
{
    int vSize = m_geo.nx * m_geo.ny * m_geo.nz;
    const int blockSize = 256;
    dim3 block(blockSize, 1, 1);
    dim3 grid((vSize + blockSize - 1) / blockSize, 1, 1);
    FDKKernel<<<grid, block>>>(dProj, dVol, m_geo);
}