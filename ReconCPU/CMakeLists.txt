file(GLOB_RECURSE CPU_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

file(GLOB_RECURSE CPU_SRCS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
)

add_library(ReconCPU SHARED
    ${CPU_HEADERS}
    ${CPU_SRCS}
)

target_compile_definitions(ReconCPU PRIVATE RECON_DLL_EXPORTS)

target_link_libraries(ReconCPU
    PRIVATE
        Common
)
