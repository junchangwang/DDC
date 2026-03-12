cmake_minimum_required(VERSION 3.14)
project(combit-bitmap-index LANGUAGES CXX)

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