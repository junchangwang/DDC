cmake_minimum_required(VERSION 3.14)
project(combit_eval CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for accurate benchmarks
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(COMBIT_DEBUG "Enable detailed debug timing output" OFF)

set(COMBIT_COMMON_OPTIONS
    -Wall -Wextra -Wpedantic
    -mavx512f -mavx512bw -mavx512vbmi2
    $<$<CONFIG:Release>:-O2>
)

# combit_eval: release build without debug timing
add_executable(combit_eval main.cpp src/combit.cpp src/and.cpp src/or.cpp src/xor.cpp)
target_include_directories(combit_eval PRIVATE include)
target_compile_options(combit_eval PRIVATE ${COMBIT_COMMON_OPTIONS})

# combit_debug: build with detailed debug timing output
add_executable(combit_debug main.cpp src/combit.cpp src/and.cpp src/or.cpp src/xor.cpp)
target_include_directories(combit_debug PRIVATE include)
target_compile_definitions(combit_debug PRIVATE COMBIT_DEBUG)
target_compile_options(combit_debug PRIVATE ${COMBIT_COMMON_OPTIONS})
