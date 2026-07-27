# Include ordinary CUDA helper headers such as float3.h as well as .cuh files.
file(GLOB_RECURSE GPU_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.inl"
)

file(GLOB_RECURSE GPU_CUDA_SRCS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

file(GLOB_RECURSE GPU_HOST_SRCS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

if(NOT GPU_CUDA_SRCS)
    message(FATAL_ERROR "ReconGPU contains no .cu source files")
endif()

# Make the generated Visual Studio project treat every .cu source as CUDA.
# The .cu extension is normally enough, but this keeps the intent explicit.
set_source_files_properties(${GPU_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

add_library(ReconGPU SHARED
    ${GPU_HEADERS}
    ${GPU_CUDA_SRCS}
    ${GPU_HOST_SRCS}
)

target_compile_definitions(ReconGPU PRIVATE RECON_DLL_EXPORTS)

target_include_directories(ReconGPU
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(ReconGPU
    PRIVATE
        Common
)

set_target_properties(ReconGPU PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF

    CUDA_RUNTIME_LIBRARY "${CTRECON_CUDA_RUNTIME_LIBRARY}"

    # Retain relocatable device code and device linking for future non-inline
    # __device__ functions implemented in different .cu translation units.
    # A header-only float3.h does not itself require these settings, but they
    # support the future multi-file CUDA structure.
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON

    POSITION_INDEPENDENT_CODE ON
)

# Only set CUDA_ARCHITECTURES when a value is available. On CMake 3.24+ the
# default from the root file is 'native'; on older CMake versions an empty value
# lets CMake/NVCC use their configured default instead of hard-coding one GPU.
if(CTRECON_CUDA_ARCHITECTURES)
    set_property(TARGET ReconGPU PROPERTY
        CUDA_ARCHITECTURES "${CTRECON_CUDA_ARCHITECTURES}")
endif()

# Do not use TARGET_RUNTIME_DLLS in an unconditional POST_BUILD copy command.
# The list may legitimately be empty, and then `cmake -E copy_if_different` is
# invoked without a source file and MSBuild reports MSB3073. CUDA runtime
# deployment should be handled by the system CUDA installation or by a separate
# install/package step.
