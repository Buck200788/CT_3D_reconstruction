#include "../include/fdk_recon.h"

#ifdef ENABLE_CUDA
void FDKRecon::RunGPU_CUDA(const std::vector<float>& hProj, std::vector<float>& hVol)
{
	return; 
}
#endif