add_library(bitset_core
    src/bitset_vector.cpp
    src/segmented_bitset.cpp
)

target_include_directories(bitset_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# AVX-512 paths use function-level target attributes; no global SIMD flags needed
set_target_properties(bitset_core PROPERTIES CXX_STANDARD 17)
