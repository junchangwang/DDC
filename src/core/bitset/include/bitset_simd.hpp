#ifndef BITSET_SIMD_HPP
#define BITSET_SIMD_HPP

/// SIMD acceleration utilities for uncompressed bitset operations.
///
/// Runtime dispatch: AVX-512 → AVX2 → scalar.
/// All public functions work on arrays of uint64_t words.

#include <cstdint>
#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define BITSET_IS_X64 1
#else
#define BITSET_IS_X64 0
#endif

#if BITSET_IS_X64
#include <immintrin.h>
#include <cpuid.h>
#endif

namespace bitset {
namespace simd {

// -----------------------------------------------------------------------
// Runtime ISA detection
// -----------------------------------------------------------------------
enum SupportFlags : int {
    BITSET_NONE   = 0,
    BITSET_AVX2   = 1,
    BITSET_AVX512 = 2,  // requires F + BW
    BITSET_AVX512_VPOPCNTDQ = 4,
};

inline int detect_hardware() {
#if !BITSET_IS_X64
    return BITSET_NONE;
#else
    int flags = BITSET_NONE;
    unsigned int eax, ebx, ecx, edx;

    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return flags;

    __cpuid(1, eax, ebx, ecx, edx);
    bool osxsave = (ecx >> 27) & 1;
    if (!osxsave) return flags;

    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    bool avx_os = (xcr0_lo & 0x06) == 0x06;
    if (!avx_os) return flags;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    bool has_avx2 = (ebx >> 5) & 1;
    if (has_avx2) flags |= BITSET_AVX2;

    bool avx512_os = (xcr0_lo & 0xE0) == 0xE0;
    bool has_avx512f  = (ebx >> 16) & 1;
    bool has_avx512bw = (ebx >> 30) & 1;
    if (avx512_os && has_avx512f && has_avx512bw)
        flags |= BITSET_AVX512;

    bool has_avx512vpopcntdq = (ecx >> 14) & 1;
    if ((flags & BITSET_AVX512) && has_avx512vpopcntdq)
        flags |= BITSET_AVX512_VPOPCNTDQ;

    return flags;
#endif
}

inline int hardware_support() {
    static int cached = detect_hardware();
    return cached;
}

// -----------------------------------------------------------------------
// AVX-512 paths
// -----------------------------------------------------------------------
#if BITSET_IS_X64

__attribute__((target("avx512f")))
inline void words_or_avx512(const uint64_t* __restrict__ a,
                            const uint64_t* __restrict__ b,
                            uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        _mm512_storeu_si512(dst + i, _mm512_or_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] | b[i];
}

__attribute__((target("avx512f")))
inline void words_and_avx512(const uint64_t* __restrict__ a,
                             const uint64_t* __restrict__ b,
                             uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        _mm512_storeu_si512(dst + i, _mm512_and_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] & b[i];
}

__attribute__((target("avx512f")))
inline void words_xor_avx512(const uint64_t* __restrict__ a,
                             const uint64_t* __restrict__ b,
                             uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] ^ b[i];
}

__attribute__((target("avx512f")))
inline void words_andnot_avx512(const uint64_t* __restrict__ a,
                                const uint64_t* __restrict__ b,
                                uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        __m512i vb = _mm512_loadu_si512(b + i);
        _mm512_storeu_si512(dst + i, _mm512_andnot_si512(vb, va));
    }
    for (; i < n; ++i) dst[i] = a[i] & ~b[i];
}

__attribute__((target("avx512f,avx512vpopcntdq")))
inline uint64_t words_popcount_avx512(const uint64_t* data, size_t n)
{
    __m512i total = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i v = _mm512_loadu_si512(data + i);
        total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
    }
    uint64_t buf[8];
    _mm512_storeu_si512(buf, total);
    uint64_t sum = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < n; ++i) sum += __builtin_popcountll(data[i]);
    return sum;
}

// -----------------------------------------------------------------------
// AVX2 paths
// -----------------------------------------------------------------------

__attribute__((target("avx2")))
inline void words_or_avx2(const uint64_t* __restrict__ a,
                          const uint64_t* __restrict__ b,
                          uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_or_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] | b[i];
}

__attribute__((target("avx2")))
inline void words_and_avx2(const uint64_t* __restrict__ a,
                           const uint64_t* __restrict__ b,
                           uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_and_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] & b[i];
}

__attribute__((target("avx2")))
inline void words_xor_avx2(const uint64_t* __restrict__ a,
                           const uint64_t* __restrict__ b,
                           uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] ^ b[i];
}

__attribute__((target("avx2")))
inline void words_andnot_avx2(const uint64_t* __restrict__ a,
                              const uint64_t* __restrict__ b,
                              uint64_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_andnot_si256(vb, va));
    }
    for (; i < n; ++i) dst[i] = a[i] & ~b[i];
}

__attribute__((target("avx2")))
inline uint64_t words_popcount_avx2(const uint64_t* data, size_t n)
{
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    size_t total_bytes = n * sizeof(uint64_t);
    size_t i = 0;
    uint64_t sum = 0;

    while (i + 32 <= total_bytes) {
        __m256i local = _mm256_setzero_si256();
        size_t limit = i + 31 * 32;
        if (limit > total_bytes) limit = total_bytes & ~31ULL;
        for (; i + 32 <= limit; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
            __m256i lo = _mm256_and_si256(v, low_mask);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, lo));
            local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, hi));
        }
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    uint64_t buf[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), acc);
    sum += buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < total_bytes; ++i) sum += __builtin_popcount(p[i]);
    return sum;
}

#endif // BITSET_IS_X64

// -----------------------------------------------------------------------
// Scalar paths (always available)
// Annotated to prevent compiler auto-vectorization so that "Plain"
// benchmarks reflect true scalar performance.
// -----------------------------------------------------------------------

__attribute__((optimize("no-tree-vectorize")))
inline void words_or_scalar(const uint64_t* __restrict__ a,
                            const uint64_t* __restrict__ b,
                            uint64_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] | b[i];
}

__attribute__((optimize("no-tree-vectorize")))
inline void words_and_scalar(const uint64_t* __restrict__ a,
                             const uint64_t* __restrict__ b,
                             uint64_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] & b[i];
}

__attribute__((optimize("no-tree-vectorize")))
inline void words_xor_scalar(const uint64_t* __restrict__ a,
                             const uint64_t* __restrict__ b,
                             uint64_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] ^ b[i];
}

__attribute__((optimize("no-tree-vectorize")))
inline void words_andnot_scalar(const uint64_t* __restrict__ a,
                                const uint64_t* __restrict__ b,
                                uint64_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] & ~b[i];
}

__attribute__((optimize("no-tree-vectorize")))
inline uint64_t words_popcount_scalar(const uint64_t* data, size_t n)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += __builtin_popcountll(data[i]);
    return sum;
}

// -----------------------------------------------------------------------
// Dispatch functions: SIMD-accelerated (best available ISA at runtime)
// -----------------------------------------------------------------------

inline void words_or_simd(const uint64_t* a, const uint64_t* b, uint64_t* dst, size_t n)
{
#if BITSET_IS_X64
    int hw = hardware_support();
    if (hw & BITSET_AVX512) { words_or_avx512(a, b, dst, n); return; }
    if (hw & BITSET_AVX2)   { words_or_avx2(a, b, dst, n); return; }
#endif
    words_or_scalar(a, b, dst, n);
}

inline void words_and_simd(const uint64_t* a, const uint64_t* b, uint64_t* dst, size_t n)
{
#if BITSET_IS_X64
    int hw = hardware_support();
    if (hw & BITSET_AVX512) { words_and_avx512(a, b, dst, n); return; }
    if (hw & BITSET_AVX2)   { words_and_avx2(a, b, dst, n); return; }
#endif
    words_and_scalar(a, b, dst, n);
}

inline void words_xor_simd(const uint64_t* a, const uint64_t* b, uint64_t* dst, size_t n)
{
#if BITSET_IS_X64
    int hw = hardware_support();
    if (hw & BITSET_AVX512) { words_xor_avx512(a, b, dst, n); return; }
    if (hw & BITSET_AVX2)   { words_xor_avx2(a, b, dst, n); return; }
#endif
    words_xor_scalar(a, b, dst, n);
}

inline void words_andnot_simd(const uint64_t* a, const uint64_t* b, uint64_t* dst, size_t n)
{
#if BITSET_IS_X64
    int hw = hardware_support();
    if (hw & BITSET_AVX512) { words_andnot_avx512(a, b, dst, n); return; }
    if (hw & BITSET_AVX2)   { words_andnot_avx2(a, b, dst, n); return; }
#endif
    words_andnot_scalar(a, b, dst, n);
}

inline uint64_t words_popcount_simd(const uint64_t* data, size_t n)
{
#if BITSET_IS_X64
    int hw = hardware_support();
    if (hw & BITSET_AVX512_VPOPCNTDQ) { return words_popcount_avx512(data, n); }
    if (hw & BITSET_AVX2)   { return words_popcount_avx2(data, n); }
#endif
    return words_popcount_scalar(data, n);
}

} // namespace simd
} // namespace bitset

#endif // BITSET_SIMD_HPP
