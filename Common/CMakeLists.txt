file(GLOB_RECURSE COMMON_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/public/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/public/*.hpp"
)

file(GLOB_RECURSE COMMON_SRCS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

add_library(Common STATIC
    ${COMMON_HEADERS}
    ${COMMON_SRCS}
)

# Common is linked into shared libraries on Windows and Unix-like systems.
set_target_properties(Common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(Common
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/public"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
