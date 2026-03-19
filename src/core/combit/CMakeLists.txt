add_library(combit
    src/types.cpp
    src/combit_encoder.cpp
    src/combit_decoder.cpp
    src/bitmap_vector.cpp
)

target_include_directories(combit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Enable SIMD intrinsics. We compile with -mavx2 as the baseline (safe for
# runtime), and use function-level __attribute__((target("avx512f,..."))) for
# AVX-512 paths so the compiler can emit those instructions only in the
# explicitly annotated functions.  Runtime dispatch ensures AVX-512 code is
# never executed on CPUs that lack it.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMBIT_HAS_AVX2)
if(COMBIT_HAS_AVX2)
    target_compile_options(combit PRIVATE -mavx2)
endif()

# Check if the compiler understands AVX-512 intrinsics (for the target-attr functions)
check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vpopcntdq" COMBIT_HAS_AVX512_COMPILER)
if(COMBIT_HAS_AVX512_COMPILER)
    target_compile_definitions(combit PUBLIC COMBIT_COMPILER_SUPPORTS_AVX512=1)
else()
    target_compile_definitions(combit PUBLIC COMBIT_COMPILER_SUPPORTS_AVX512=0)
endif()