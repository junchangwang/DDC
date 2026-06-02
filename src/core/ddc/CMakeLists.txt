cmake_minimum_required(VERSION 3.14)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ddc LANGUAGES CXX)
endif()

add_library(ddc
    src/ddc.cpp
    src/and.cpp
    src/or.cpp
    src/xor.cpp
    src/not.cpp
    src/ddc_n.cpp
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(ddc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# DDC uses AVX-512 VBMI2 intrinsics for expand-load operations.
# We need both compiler support AND runtime CPU support, so use try_run
# instead of check_cxx_compiler_flag (which only checks the compiler).
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512bw -mavx512vbmi2")
check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m512i v = _mm512_setzero_si512();
    __mmask64 m = 0xFF;
    unsigned char buf[64] = {};
    v = _mm512_mask_expandloadu_epi8(v, m, buf);
    return 0;
}
" DDC_HAS_AVX512_VBMI2)
unset(CMAKE_REQUIRED_FLAGS)
target_compile_options(ddc PRIVATE -O3)
if(DDC_HAS_AVX512_VBMI2)
    target_compile_options(ddc PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

option(DDC_DEBUG "Enable detailed debug timing output" OFF)
if(DDC_DEBUG)
    target_compile_definitions(ddc PRIVATE DDC_DEBUG)
endif()

# Standalone evaluation tool (correctness tests + benchmarks) — Release
add_executable(ddc_eval src/ddc_eval.cpp)
target_link_libraries(ddc_eval PRIVATE ddc)
target_include_directories(ddc_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(ddc_eval PRIVATE -O3 -DNDEBUG)
if(DDC_HAS_AVX512_VBMI2)
    target_compile_options(ddc_eval PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

# Debug evaluation tool with DDC_DEBUG timing output
add_library(ddc_debug
    src/ddc.cpp
    src/and.cpp
    src/or.cpp
    src/xor.cpp
    src/not.cpp
    src/ddc_n.cpp
)
target_include_directories(ddc_debug
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(ddc_debug PRIVATE DDC_DEBUG)
target_compile_options(ddc_debug PRIVATE -g -O0)
if(DDC_HAS_AVX512_VBMI2)
    target_compile_options(ddc_debug PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()

add_executable(ddc_eval_debug src/ddc_eval.cpp)
target_link_libraries(ddc_eval_debug PRIVATE ddc_debug)
target_include_directories(ddc_eval_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(ddc_eval_debug PRIVATE DDC_DEBUG)
target_compile_options(ddc_eval_debug PRIVATE -g -O0)
if(DDC_HAS_AVX512_VBMI2)
    target_compile_options(ddc_eval_debug PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
endif()
