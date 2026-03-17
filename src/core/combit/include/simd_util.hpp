#ifndef COMBIT_SIMD_UTIL_HPP
#define COMBIT_SIMD_UTIL_HPP

/// SIMD acceleration utilities for ComBit.
///
/// Runtime dispatch: AVX-512 → AVX2 → scalar, following the same pattern as
/// CRoaring.  All public functions work on arrays of uint16_t words that
/// form the raw uncompressed bitmap.

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

// -----------------------------------------------------------------------
// 1. Compile-time feature detection
// -----------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define COMBIT_IS_X64 1
#else
#define COMBIT_IS_X64 0
#endif

#if COMBIT_IS_X64
#include <immintrin.h>
#include <cpuid.h>
#endif

// -----------------------------------------------------------------------
// 2. Runtime ISA detection (cached, thread-safe via static local)
// -----------------------------------------------------------------------
namespace combit {
namespace simd {

enum SupportFlags : int {
    COMBIT_NONE   = 0,
    COMBIT_AVX2   = 1,
    COMBIT_AVX512 = 2,  // requires F + BW + VPOPCNTDQ
};

inline int detect_hardware() {
#if !COMBIT_IS_X64
    return COMBIT_NONE;
#else
    int flags = COMBIT_NONE;
    unsigned int eax, ebx, ecx, edx;

    // Check max CPUID level
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 7) return flags;

    // Leaf 1: check OSXSAVE (bit 27 of ECX)
    __cpuid(1, eax, ebx, ecx, edx);
    bool osxsave = (ecx >> 27) & 1;
    if (!osxsave) return flags;

    // Check XCR0 for AVX state
    unsigned int xcr0_lo, xcr0_hi;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    bool avx_os = (xcr0_lo & 0x06) == 0x06;  // SSE + AVX state
    if (!avx_os) return flags;

    // Leaf 7, subleaf 0
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    bool has_avx2 = (ebx >> 5) & 1;
    if (has_avx2) flags |= COMBIT_AVX2;

    // AVX-512 needs XCR0 bits 5-7 (opmask, ZMM hi256, Hi16 ZMM)
    bool avx512_os = (xcr0_lo & 0xE0) == 0xE0;
    bool has_avx512f  = (ebx >> 16) & 1;
    bool has_avx512bw = (ebx >> 30) & 1;
    bool has_avx512vpopcntdq = (ecx >> 14) & 1;
    if (avx512_os && has_avx512f && has_avx512bw && has_avx512vpopcntdq)
        flags |= COMBIT_AVX512;

    return flags;
#endif
}

inline int hardware_support() {
    static int cached = detect_hardware();
    return cached;
}

// -----------------------------------------------------------------------
// 3. Word-level bitwise operations: OR / AND / XOR / ANDNOT
//    All operate on uint16_t arrays; `n` is the number of uint16_t words.
//    dst may alias src_a or src_b.
// -----------------------------------------------------------------------

// ---------- AVX-512 paths ----------
#if COMBIT_IS_X64

__attribute__((target("avx512f,avx512bw")))
inline void words_or_avx512(const uint16_t* __restrict__ a,
                            const uint16_t* __restrict__ b,
                            uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    // 32 x uint16_t = 512 bits
    for (; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_or_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] | b[i];
}

__attribute__((target("avx512f,avx512bw")))
inline void words_and_avx512(const uint16_t* __restrict__ a,
                             const uint16_t* __restrict__ b,
                             uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_and_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] & b[i];
}

__attribute__((target("avx512f,avx512bw")))
inline void words_xor_avx512(const uint16_t* __restrict__ a,
                             const uint16_t* __restrict__ b,
                             uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_xor_si512(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] ^ b[i];
}

__attribute__((target("avx512f,avx512bw")))
inline void words_andnot_avx512(const uint16_t* __restrict__ a,
                                const uint16_t* __restrict__ b,
                                uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a + i));
        __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b + i));
        // andnot: a & ~b  →  _mm512_andnot(b, a)  (note: intrinsic is ~first & second)
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), _mm512_andnot_si512(vb, va));
    }
    for (; i < n; ++i) dst[i] = a[i] & ~b[i];
}

// AVX-512 popcount: use VPOPCNTDQ on 64-bit lanes, then reduce
__attribute__((target("avx512f,avx512bw,avx512vpopcntdq")))
inline uint64_t words_popcount_avx512(const uint16_t* data, size_t n)
{
    // Reinterpret as bytes; n uint16_t = 2*n bytes = n/4 uint64_t
    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(data);
    size_t n64 = (n * sizeof(uint16_t)) / sizeof(uint64_t);

    __m512i total = _mm512_setzero_si512();
    size_t i = 0;
    for (; i + 8 <= n64; i += 8) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p64 + i));
        total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
    }
    // Horizontal reduce 8 x uint64
    uint64_t buf[8];
    _mm512_storeu_si512(buf, total);
    uint64_t sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    // Scalar tail
    const uint8_t* tail = reinterpret_cast<const uint8_t*>(p64 + i);
    size_t remaining_bytes = (n * 2) - (i * 8);
    for (size_t j = 0; j < remaining_bytes; ++j) {
        sum += __builtin_popcount(tail[j]);
    }
    return sum;
}

// ---------- AVX2 paths ----------

__attribute__((target("avx2")))
inline void words_or_avx2(const uint16_t* __restrict__ a,
                          const uint16_t* __restrict__ b,
                          uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    // 16 x uint16_t = 256 bits
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_or_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] | b[i];
}

__attribute__((target("avx2")))
inline void words_and_avx2(const uint16_t* __restrict__ a,
                           const uint16_t* __restrict__ b,
                           uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_and_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] & b[i];
}

__attribute__((target("avx2")))
inline void words_xor_avx2(const uint16_t* __restrict__ a,
                           const uint16_t* __restrict__ b,
                           uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(va, vb));
    }
    for (; i < n; ++i) dst[i] = a[i] ^ b[i];
}

__attribute__((target("avx2")))
inline void words_andnot_avx2(const uint16_t* __restrict__ a,
                              const uint16_t* __restrict__ b,
                              uint16_t* __restrict__ dst, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_andnot_si256(vb, va));
    }
    for (; i < n; ++i) dst[i] = a[i] & ~b[i];
}

// AVX2 popcount via Harley-Seal / lookup table method
__attribute__((target("avx2")))
inline uint64_t words_popcount_avx2(const uint16_t* data, size_t n)
{
    // Use the classic SSSE3/AVX2 nibble-lookup popcount from Wojciech Mula
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    __m256i acc = _mm256_setzero_si256();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    size_t total_bytes = n * sizeof(uint16_t);
    size_t i = 0;

    // Process 32 bytes at a time, sub-accumulate in 8-bit counters.
    // Flush every 31 iterations: each byte lane adds up to 8 per iter,
    // 31*8 = 248 < 256, preventing uint8 overflow.
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
        // Widen 8→64 and accumulate
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    // Horizontal reduce
    uint64_t buf[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), acc);
    sum += buf[0] + buf[1] + buf[2] + buf[3];

    // Scalar tail
    for (; i < total_bytes; ++i) {
        sum += __builtin_popcount(p[i]);
    }
    return sum;
}

#endif  // COMBIT_IS_X64

// ---------- Scalar paths (always available) ----------

inline void words_or_scalar(const uint16_t* __restrict__ a,
                            const uint16_t* __restrict__ b,
                            uint16_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] | b[i];
}

inline void words_and_scalar(const uint16_t* __restrict__ a,
                             const uint16_t* __restrict__ b,
                             uint16_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] & b[i];
}

inline void words_xor_scalar(const uint16_t* __restrict__ a,
                             const uint16_t* __restrict__ b,
                             uint16_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] ^ b[i];
}

inline void words_andnot_scalar(const uint16_t* __restrict__ a,
                                const uint16_t* __restrict__ b,
                                uint16_t* __restrict__ dst, size_t n)
{
    for (size_t i = 0; i < n; ++i) dst[i] = a[i] & ~b[i];
}

inline uint64_t words_popcount_scalar(const uint16_t* data, size_t n)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += __builtin_popcount(static_cast<unsigned>(data[i]));
    return sum;
}

// -----------------------------------------------------------------------
// 4. Dispatch functions (auto-select best ISA at runtime)
// -----------------------------------------------------------------------

inline void words_or(const uint16_t* a, const uint16_t* b,
                     uint16_t* dst, size_t n)
{
#if COMBIT_IS_X64
    int hw = hardware_support();
    if (hw & COMBIT_AVX512) { words_or_avx512(a, b, dst, n); return; }
    if (hw & COMBIT_AVX2)   { words_or_avx2(a, b, dst, n); return; }
#endif
    words_or_scalar(a, b, dst, n);
}

inline void words_and(const uint16_t* a, const uint16_t* b,
                      uint16_t* dst, size_t n)
{
#if COMBIT_IS_X64
    int hw = hardware_support();
    if (hw & COMBIT_AVX512) { words_and_avx512(a, b, dst, n); return; }
    if (hw & COMBIT_AVX2)   { words_and_avx2(a, b, dst, n); return; }
#endif
    words_and_scalar(a, b, dst, n);
}

inline void words_xor(const uint16_t* a, const uint16_t* b,
                      uint16_t* dst, size_t n)
{
#if COMBIT_IS_X64
    int hw = hardware_support();
    if (hw & COMBIT_AVX512) { words_xor_avx512(a, b, dst, n); return; }
    if (hw & COMBIT_AVX2)   { words_xor_avx2(a, b, dst, n); return; }
#endif
    words_xor_scalar(a, b, dst, n);
}

inline void words_andnot(const uint16_t* a, const uint16_t* b,
                         uint16_t* dst, size_t n)
{
#if COMBIT_IS_X64
    int hw = hardware_support();
    if (hw & COMBIT_AVX512) { words_andnot_avx512(a, b, dst, n); return; }
    if (hw & COMBIT_AVX2)   { words_andnot_avx2(a, b, dst, n); return; }
#endif
    words_andnot_scalar(a, b, dst, n);
}

inline uint64_t words_popcount(const uint16_t* data, size_t n)
{
#if COMBIT_IS_X64
    int hw = hardware_support();
    if (hw & COMBIT_AVX512) { return words_popcount_avx512(data, n); }
    if (hw & COMBIT_AVX2)   { return words_popcount_avx2(data, n); }
#endif
    return words_popcount_scalar(data, n);
}

// -----------------------------------------------------------------------
// 5. Decode: extract set-bit positions into a uint32_t array
// -----------------------------------------------------------------------

#if COMBIT_IS_X64
// AVX-512 VPOPCNTDQ + compress approach for extracting set-bit positions
// is complex; we use a fast scalar path that's still very efficient for
// sparse bitmaps.  The main speedup comes from the bitwise operations above.
#endif

inline void words_decode_positions(const uint16_t* data, size_t n,
                                   uint32_t base,
                                   std::vector<uint32_t>& out)
{
    for (size_t i = 0; i < n; ++i) {
        uint16_t w = data[i];
        uint32_t word_base = base + static_cast<uint32_t>(i * 16);
        while (w) {
            // Find highest set bit (MSB-first layout: bit 0 = position 15)
            int leading = __builtin_clz(static_cast<unsigned>(w)) - 16; // clz counts from bit 31
            uint32_t bit_pos = word_base + static_cast<uint32_t>(leading);
            out.push_back(bit_pos);
            w &= ~(uint16_t(1) << (15 - leading));
        }
    }
}

}  // namespace simd
}  // namespace combit

#endif  // COMBIT_SIMD_UTIL_HPP
