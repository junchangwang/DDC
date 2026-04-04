cmake_minimum_required(VERSION 3.14)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(combit LANGUAGES CXX)
endif()

add_library(combit
    src/combit.cpp
    src/and.cpp
    src/or.cpp
    src/xor.cpp
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(combit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ComBit uses AVX-512 VBMI2 intrinsics for expand-load operations.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vbmi2" COMBIT_HAS_AVX512_VBMI2)
if(COMBIT_HAS_AVX512_VBMI2)
    target_compile_options(combit PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

option(COMBIT_DEBUG "Enable detailed debug timing output" OFF)
if(COMBIT_DEBUG)
    target_compile_definitions(combit PRIVATE COMBIT_DEBUG)
endif()

# Standalone evaluation tool (correctness tests + benchmarks) — Release
add_executable(combit_eval src/combit_eval.cpp)
target_link_libraries(combit_eval PRIVATE combit)
target_include_directories(combit_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(combit_eval PRIVATE -O2 -DNDEBUG)
if(COMBIT_HAS_AVX512_VBMI2)
    target_compile_options(combit_eval PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

# Debug evaluation tool with COMBIT_DEBUG timing output
add_library(combit_debug
    src/combit.cpp
    src/and.cpp
    src/or.cpp
    src/xor.cpp
)
target_include_directories(combit_debug
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(combit_debug PRIVATE COMBIT_DEBUG)
target_compile_options(combit_debug PRIVATE -g -O0)
if(COMBIT_HAS_AVX512_VBMI2)
    target_compile_options(combit_debug PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

add_executable(combit_eval_debug src/combit_eval.cpp)
target_link_libraries(combit_eval_debug PRIVATE combit_debug)
target_include_directories(combit_eval_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(combit_eval_debug PRIVATE COMBIT_DEBUG)
target_compile_options(combit_eval_debug PRIVATE -g -O0)
if(COMBIT_HAS_AVX512_VBMI2)
    target_compile_options(combit_eval_debug PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()
