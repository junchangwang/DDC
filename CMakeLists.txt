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

# AVX-512 uses function-level target attributes; no global SIMD flags needed

# Bring back combit library link
target_link_libraries(benchmark_app PRIVATE combit bitset_core Threads::Threads atomic Boost::program_options)

# ==========================================
# Testing (Google Test) — disable with -DBUILD_TESTS=OFF
# ==========================================
option(BUILD_TESTS "Build Google Test targets (requires internet)" ON)

if(BUILD_TESTS)
enable_testing()
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Core library tests
add_executable(combit_tests
    tests/test_bitmap_vector.cpp
    tests/test_combit_decoder.cpp
    tests/test_combit_encoder.cpp
    tests/test_bitmap_index.cpp
    tests/test_compression.cpp
)
target_include_directories(combit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)
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
# inspect_croaring: CRoaring container type diagnostic
# ==========================================
add_executable(inspect_croaring
    util/inspect_croaring.cpp
    src/core/croaring/roaring.c
)
target_include_directories(inspect_croaring PRIVATE
    ${CMAKE_SOURCE_DIR}/src/core
)
target_link_libraries(inspect_croaring PRIVATE Threads::Threads)