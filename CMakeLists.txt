cmake_minimum_required(VERSION 3.14)
project(combit-bitmap-index LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required system libraries
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# 1. Compile ComBit library
add_subdirectory(src/core/combit)

# 2. Compile Bitset core library
add_subdirectory(src/core/bitset)

# 3. Demo executable
add_executable(combit_demo src/main.cpp)
target_link_libraries(combit_demo PRIVATE combit)

# ==========================================
# Benchmark Framework & Backends
# ==========================================

set(FASTBIT_SOURCES
    src/core/fastbit/bitvector.cpp
    src/core/fastbit/bitvector16.cpp
    src/core/fastbit/bitvector8.cpp
    src/core/fastbit/fileManager.cpp
    src/core/fastbit/resource.cpp
    src/core/fastbit/util.cpp
    src/core/fastbit/array_t.cpp
    src/core/utils/util.cpp
)

# Bring back combit_backend.cpp
set(BENCHMARK_SOURCES
    src/benchmark/benchmark_main.cpp
    src/benchmark/uti.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/benchmark/backends/ewah/ewah_backend.cpp
    src/benchmark/backends/Concise/concise_backend.cpp
    src/benchmark/backends/bitset/bitset_backend.cpp
    src/benchmark/backends/bitset_avx512/bitset_avx512_backend.cpp
    src/core/croaring/roaring.c
)

add_executable(benchmark_app ${BENCHMARK_SOURCES} ${FASTBIT_SOURCES})

# Add src/core/combit/include to search path
target_include_directories(benchmark_app PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)

# ComBit's AVX-512 VBMI2 expand-load needs these flags
target_compile_options(benchmark_app PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)

target_link_libraries(benchmark_app PRIVATE combit bitset_core Threads::Threads atomic Boost::program_options)

# ----------------------------------------------------------------
# transition_bench — CR container conversion vs ComBit headline benchmark
# ----------------------------------------------------------------
add_executable(transition_bench
    src/benchmark/transition_bench.cpp
    ${FASTBIT_SOURCES}
    src/core/croaring/roaring.c
)
target_include_directories(transition_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(transition_bench PRIVATE -O3 -g -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(transition_bench PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# Testing (Google Test) — disable with -DBUILD_TESTS=OFF
# ==========================================
option(BUILD_TESTS "Build Google Test targets (requires internet)" ON)

if(BUILD_TESTS)
enable_testing()
find_package(GTest REQUIRED)

# Core library tests
add_executable(combit_tests
    tests/test_combit.cpp
)
target_include_directories(combit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(combit_tests PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(combit_tests PRIVATE combit GTest::gtest_main)

# Backend tests (serialization, logical ops)
add_executable(combit_backend_tests
    tests/test_backend_ops.cpp
    src/benchmark/backends/combit/combit_backend.cpp
)
target_include_directories(combit_backend_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
)
target_link_libraries(combit_backend_tests PRIVATE combit GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(combit_tests)
gtest_discover_tests(combit_backend_tests)

# Unified 3-backend correctness tests (WAH + CRoaring + ComBit)
set(ALL_BACKEND_SOURCES
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/benchmark/backends/ewah/ewah_backend.cpp
    src/benchmark/backends/Concise/concise_backend.cpp
    src/benchmark/backends/bitset/bitset_backend.cpp
    src/benchmark/backends/bitset_avx512/bitset_avx512_backend.cpp
    src/core/croaring/roaring.c
)

add_executable(all_backend_tests
    tests/test_all_backends.cpp
    ${ALL_BACKEND_SOURCES}
    ${FASTBIT_SOURCES}
)
target_include_directories(all_backend_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)
target_link_libraries(all_backend_tests PRIVATE combit bitset_core Threads::Threads atomic Boost::program_options GTest::gtest_main)
gtest_discover_tests(all_backend_tests)

# Raw .bm file integration tests (reads gen_test_bitmaps.sh output)
add_executable(bm_file_tests
    tests/test_bm_files.cpp
    ${ALL_BACKEND_SOURCES}
    ${FASTBIT_SOURCES}
)
target_include_directories(bm_file_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)
target_link_libraries(bm_file_tests PRIVATE combit bitset_core Threads::Threads atomic Boost::program_options GTest::gtest_main)
gtest_discover_tests(bm_file_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

endif() # BUILD_TESTS

# ==========================================
# gen_bitmap: dataset + bitmap generator (all algorithms)
# ==========================================
add_executable(gen_bitmap
    util/gen_bitmap.cpp
    ${FASTBIT_SOURCES}
    src/core/croaring/roaring.c
)
target_include_directories(gen_bitmap PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/util
)
target_link_libraries(gen_bitmap PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# bench_L_sizes: analytical L2/L3/L4/L5 size measurement
# ==========================================
add_executable(bench_L_sizes
    src/benchmark/bench_L_sizes.cpp
)
target_include_directories(bench_L_sizes PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_link_libraries(bench_L_sizes PRIVATE combit Threads::Threads atomic)

# ==========================================
# bench_L_ops: real depth-2/3/4/5 compress + AND/OR/XOR timing
# Uses src/core/combit/src/combit_n.cpp — a SCALAR depth-parameterised
# implementation isolated from the existing AVX-512 ComBit code (kept
# fair across all four depths by avoiding SIMD micro-optimisations).
# ==========================================
add_executable(bench_L_ops
    src/benchmark/bench_L_ops.cpp
)
target_include_directories(bench_L_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(bench_L_ops PRIVATE -O3)
target_link_libraries(bench_L_ops PRIVATE combit Threads::Threads atomic)

# ==========================================
# bench_bypass: bypass-ablation at the default depth=4 — OR/AND/COMP
# timing under 3 configs (all / no_L4 / no_L3) of the seg_op_l4 walker.
# ==========================================
add_executable(bench_bypass
    src/benchmark/bench_bypass.cpp
)
target_include_directories(bench_bypass PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(bench_bypass PRIVATE -O3)
target_link_libraries(bench_bypass PRIVATE combit Threads::Threads atomic)

# ==========================================
# bench_L_run_bar: one-shot COMP throughput at a single cardinality
# on run-length clustered bitmaps (TPC-H Q6-aligned).  Produces the bar
# chart .dat for tools/hierarchy_depth_runlength_bar.gnuplot.
# ==========================================
add_executable(bench_L_run_bar
    src/benchmark/bench_L_run_bar.cpp
)
target_include_directories(bench_L_run_bar PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(bench_L_run_bar PRIVATE -O3)
target_link_libraries(bench_L_run_bar PRIVATE combit Threads::Threads atomic)

# ==========================================
# bench_seg: segment-size ablation — size + AND/OR/NOT/COMP timing
# at (cardinality × segment_bits) for the Figure 11 ablation study.
# ==========================================
add_executable(bench_seg
    src/benchmark/bench_seg.cpp
)
target_include_directories(bench_seg PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
target_compile_options(bench_seg PRIVATE -O3)
target_link_libraries(bench_seg PRIVATE combit Threads::Threads atomic)

# ==========================================
# verify_bitmap: bitmap correctness verification
# ==========================================
add_executable(verify_bitmap
    util/verify_bitmap.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    ${FASTBIT_SOURCES}
    src/core/croaring/roaring.c
)
target_include_directories(verify_bitmap PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/util
)
target_link_libraries(verify_bitmap PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# TPC-H Q6 Benchmark: ComBit vs WAH
# ==========================================
set(TPCH_Q6_SOURCES
    src/benchmark/tpch_q6_benchmark.cpp
    src/benchmark/uti.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
)

add_executable(tpch_q6_benchmark ${TPCH_Q6_SOURCES} ${FASTBIT_SOURCES})
target_include_directories(tpch_q6_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)
target_compile_options(tpch_q6_benchmark PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(tpch_q6_benchmark PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# Compress TPC-H Q6 bitmaps: raw → ComBit + WAH
# ==========================================
set(COMPRESS_Q6_SOURCES
    src/benchmark/compress_tpch_q6.cpp
    src/benchmark/uti.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/ewah/ewah_backend.cpp
    src/core/croaring/roaring.c
)

add_executable(compress_tpch_q6 ${COMPRESS_Q6_SOURCES} ${FASTBIT_SOURCES})
target_include_directories(compress_tpch_q6 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)
target_compile_options(compress_tpch_q6 PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(compress_tpch_q6 PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# Compress TPC-H Q1 bitmaps: raw → ComBit + WAH + CRoaring + EWAH
# ==========================================
set(COMPRESS_Q1_SOURCES
    src/benchmark/compress_tpch_q1.cpp
    src/benchmark/uti.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/ewah/ewah_backend.cpp
    src/core/croaring/roaring.c
)

add_executable(compress_tpch_q1 ${COMPRESS_Q1_SOURCES} ${FASTBIT_SOURCES})
target_include_directories(compress_tpch_q1 PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
)
target_compile_options(compress_tpch_q1 PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(compress_tpch_q1 PRIVATE combit Threads::Threads atomic Boost::program_options)

# ==========================================
# Generic raw .bm → 4-format compressor (walks any subdir layout).
# Used by queries whose bitmap layout doesn't fit Q1/Q6's hardcoded
# returnflag/linestatus/shipdate (e.g. Q3's join_result/0.bm).
# ==========================================
add_executable(compress_any
    util/compress_any.cpp
    src/core/croaring/roaring.c
    ${FASTBIT_SOURCES}
)
target_include_directories(compress_any PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
    ${CMAKE_SOURCE_DIR}/src/core/bitset/include
    ${CMAKE_SOURCE_DIR}/src/core/croaring
    ${CMAKE_SOURCE_DIR}/src/core/ewah
)
target_compile_options(compress_any PRIVATE -mavx512f -mavx512bw -mavx512vbmi2)
target_link_libraries(compress_any PRIVATE combit Threads::Threads atomic Boost::program_options)