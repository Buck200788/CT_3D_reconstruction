add_library(Common STATIC
	public/filter.h
    src/filter.cpp
    src/device_detect.cpp
	public/device_detect.h
    public/CTGeometry.h
    public/ReconBase.h
    public/ReconAlgoEnum.h
    public/float3_helper.h
    public/FFT1D.h
    src/FFT1D.cpp
)

target_include_directories(Common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/public
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 全局OpenMP支持
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Common PUBLIC OpenMP::OpenMP_CXX)
endif()