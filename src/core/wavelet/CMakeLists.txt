cmake_minimum_required(VERSION 3.14)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(wavelet LANGUAGES CXX)
endif()

# RowSet set-operation kernels (one op per translation unit, like src/core/ddc).
# BitVector and WaveletMatrix are header-only (include/), used by consumers directly.
add_library(wavelet
    src/rowset.cpp
    src/and.cpp
    src/or.cpp
    src/xor.cpp
    src/not.cpp
    src/andnot.cpp
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(wavelet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# AVX-512 F/BW/DQ + VPOPCNTDQ for rank/select popcount and the set-op kernels.
# Require both compiler AND runtime support (try_run), mirroring src/core/ddc.
include(CheckCXXSourceRuns)
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512vpopcntdq")
check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m512i v = _mm512_set1_epi64(1);
    v = _mm512_popcnt_epi64(v);
    return (int)_mm512_reduce_add_epi64(v) - 8;
}
" WAVELET_HAS_AVX512)
unset(CMAKE_REQUIRED_FLAGS)

target_compile_options(wavelet PRIVATE -O3)
if(WAVELET_HAS_AVX512)
    target_compile_options(wavelet PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vpopcntdq)
endif()

# Standalone correctness + benchmark tool.
add_executable(wavelet_test test/test_main.cpp)
target_link_libraries(wavelet_test PRIVATE wavelet)
target_include_directories(wavelet_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(wavelet_test PRIVATE -O3)
if(WAVELET_HAS_AVX512)
    target_compile_options(wavelet_test PRIVATE -mavx512f -mavx512bw -mavx512dq -mavx512vpopcntdq)
endif()
