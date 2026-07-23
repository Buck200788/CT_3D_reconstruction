file(GLOB CPU_HEADERS *.h)
file(GLOB CPU_SRCS *.cpp)
add_library(ReconCPU SHARED ${CPU_HEADERS} ${CPU_SRCS})
target_compile_definitions(ReconCPU PRIVATE RECON_DLL_EXPORTS)
target_include_directories(ReconCPU PRIVATE ${CMAKE_SOURCE_DIR}/Common/public)
# 仅链接Common，移除OpenMP::OpenMP_CXX
target_link_libraries(ReconCPU PRIVATE Common)