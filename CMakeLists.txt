cmake_minimum_required(VERSION 3.22)

# Keep CUDA optional so the same source tree can also be configured on a
# machine without a CUDA Toolkit.
project(CTReconMain LANGUAGES C CXX)

option(CTRECON_BUILD_CUDA "Build the CUDA reconstruction module when CUDA is available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep executables and shared libraries together. This is especially useful for
# the plugin-style ReconCPU/ReconGPU DLL layout used by Demo.
foreach(_cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}
        "${CMAKE_BINARY_DIR}/bin/${_cfg}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper}
        "${CMAKE_BINARY_DIR}/bin/${_cfg}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg_upper}
        "${CMAKE_BINARY_DIR}/lib/${_cfg}")
endforeach()
unset(_cfg)
unset(_cfg_upper)

# The EXE and DLLs exchange STL objects such as std::vector. Use the shared MSVC
# runtime so allocations are handled by the same CRT across module boundaries.
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    add_compile_definitions(_ENABLE_PARALLEL_ALGORITHMS)
endif()

add_subdirectory(Common)
add_subdirectory(ReconCPU)

set(CTRECON_HAS_CUDA OFF)
if(CTRECON_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        # Shared is the safest default for a CUDA DLL/plugin. It is a cache
        # setting, so another platform may select Static in CMake GUI without
        # editing the CMakeLists files.
        set(CTRECON_CUDA_RUNTIME_LIBRARY "Shared" CACHE STRING
            "CUDA runtime linkage: Shared or Static")
        set_property(CACHE CTRECON_CUDA_RUNTIME_LIBRARY
            PROPERTY STRINGS Shared Static)

        # Do not hard-code a GPU generation or a CUDA Toolkit version.
        # With CMake 3.24+, 'native' detects the GPU of the build machine.
        # A distribution build can change this cache value to all-major or to a
        # semicolon-separated list such as 75;86;89.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set(CTRECON_CUDA_ARCHITECTURES "native" CACHE STRING
                "CUDA architectures: native, all, all-major, or a list such as 75;86")
            set_property(CACHE CTRECON_CUDA_ARCHITECTURES
                PROPERTY STRINGS native all all-major)
        else()
            set(CTRECON_CUDA_ARCHITECTURES "" CACHE STRING
                "CUDA architectures; empty uses the CUDA compiler/CMake default")
        endif()

        add_subdirectory(ReconGPU)
        set(CTRECON_HAS_CUDA ON)

        message(STATUS "CUDA compiler          : ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA compiler version  : ${CMAKE_CUDA_COMPILER_VERSION}")
        message(STATUS "CUDA Toolkit version   : ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA runtime linkage   : ${CTRECON_CUDA_RUNTIME_LIBRARY}")
        if(CTRECON_CUDA_ARCHITECTURES)
            message(STATUS "CUDA architectures     : ${CTRECON_CUDA_ARCHITECTURES}")
        else()
            message(STATUS "CUDA architectures     : compiler/CMake default")
        endif()
    else()
        message(WARNING
            "No CUDA compiler was detected. ReconGPU will not be built; "
            "Common, ReconCPU, and Demo will still be generated.")
    endif()
endif()

add_subdirectory(Demo)
