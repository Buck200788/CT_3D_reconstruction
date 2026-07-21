add_executable(Demo main.cpp)
target_include_directories(Demo PRIVATE ${CMAKE_SOURCE_DIR}/Common/public)
target_link_libraries(Demo PRIVATE Common)