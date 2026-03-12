cmake_minimum_required(VERSION 3.14)
project(combit-bitmap-index LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- ComBit library (subdirectory) ---
add_subdirectory(src/combit)

# --- Main demo executable ---
add_executable(combit_demo src/main.cpp)
target_link_libraries(combit_demo PRIVATE combit)

# --- Tests (using GoogleTest) ---
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(combit_tests
    tests/test_combit_encoder.cpp
    tests/test_combit_decoder.cpp
    tests/test_bitmap_vector.cpp
)
target_link_libraries(combit_tests PRIVATE combit GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(combit_tests)

# Benchmark Framework & Backends
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

set(FASTBIT_SOURCES
    src/fastbit/bitvector.cpp
    src/fastbit/bitvector16.cpp
    src/fastbit/bitvector8.cpp
    src/fastbit/fileManager.cpp
    src/fastbit/resource.cpp
    src/fastbit/util.cpp
    src/fastbit/array_t.cpp
    src/utils/util.cpp
)

set(BENCHMARK_SOURCES
    src/benchmark/benchmark_main.cpp
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/croaring/roaring.c
)

add_executable(benchmark_app ${BENCHMARK_SOURCES} ${FASTBIT_SOURCES})

target_include_directories(benchmark_app PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/src/combit/include
)

target_link_libraries(benchmark_app PRIVATE combit Threads::Threads atomic Boost::program_options)