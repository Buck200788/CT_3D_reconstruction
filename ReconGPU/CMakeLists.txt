file(GLOB GPU_HEADERS *.cuh)
file(GLOB GPU_SRCS *.cu)
add_library(ReconGPU SHARED ${GPU_HEADERS} ${GPU_SRCS})

target_compile_definitions(ReconGPU PRIVATE RECON_DLL_EXPORTS)
target_include_directories(ReconGPU PRIVATE ${CMAKE_SOURCE_DIR}/Common/public)
target_link_libraries(ReconGPU PRIVATE Common cudart)
set_target_properties(ReconGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# CUDA 12.2+ 用17，低版本 fallback 14
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.2)
    set_target_properties(ReconGPU PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED OFF
    )
else()
    set_target_properties(ReconGPU PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
    )
endif()