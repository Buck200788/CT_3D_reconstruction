file(GLOB HEADERS public/*.h)
file(GLOB SRCS src/*.cpp)
add_library(Common STATIC ${HEADERS} ${SRCS})
target_include_directories(Common
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/public
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# 删掉下面所有OpenMP代码！
# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#    target_link_libraries(Common PUBLIC OpenMP::OpenMP_CXX)
# endif()