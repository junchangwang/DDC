cmake_minimum_required(VERSION 3.14)
project(combit-bitmap-index LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required system libraries
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# 1. Compile ComBit library
add_subdirectory(src/core/combit)

# 2. Demo executable
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
    src/benchmark/backends/wah/wah_backend.cpp
    src/benchmark/backends/croaring/croaring_backend.cpp
    src/benchmark/backends/combit/combit_backend.cpp
    src/core/croaring/roaring.c
)

add_executable(benchmark_app ${BENCHMARK_SOURCES} ${FASTBIT_SOURCES})

# Add src/core/combit/include to search path
target_include_directories(benchmark_app PRIVATE 
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/core/combit/include
)

# Bring back combit library link
target_link_libraries(benchmark_app PRIVATE combit Threads::Threads atomic Boost::program_options)