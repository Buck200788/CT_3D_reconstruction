add_executable(Demo main.cpp)

target_link_libraries(Demo
    PRIVATE
        Common
)

# Demo loads the reconstruction modules dynamically. These dependencies ensure
# the current libraries are built before Demo without forcing static linkage.
add_dependencies(Demo ReconCPU)
if(TARGET ReconGPU)
    add_dependencies(Demo ReconGPU)
endif()
