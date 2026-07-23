cmake_minimum_required(VERSION 3.22)
project(CTReconMain LANGUAGES C CXX)

# 启用C++17（std::execution必须）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 统一输出目录，所有dll/lib/exe集中存放
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Windows静态VC运行库，发布无需额外vcrt dll（保留原有MT）
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # 关键：开启MSVC STL并行算法支持
    add_compile_definitions(_ENABLE_PARALLEL_ALGORITHMS)
endif()

# ========== 1. 必编译公共静态库 ==========
add_subdirectory(Common)
# ========== 2. 必编译CPU重建模块 ==========
add_subdirectory(ReconCPU)
# ========== 3. 检测CUDA，有则编译GPU模块 ==========
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler detected, build ReconGPU module")
    enable_language(CUDA)
    add_subdirectory(ReconGPU)
else()
    message(WARNING "CUDA compiler not found, skip ReconGPU module")
endif()
# ========== 4. 测试Demo程序 ==========
add_subdirectory(Demo)