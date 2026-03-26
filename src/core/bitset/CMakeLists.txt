add_library(bitset_core
    src/bitset_vector.cpp
)

target_include_directories(bitset_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# AVX2 as baseline; AVX-512 paths use function-level target attributes
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" BITSET_HAS_AVX2)
if(BITSET_HAS_AVX2)
    target_compile_options(bitset_core PRIVATE -mavx2)
endif()
