// rowset.cpp — RowSet shared helpers: popcount (cardinality) + tail masking.
#include "rowset.hpp"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace wm {

size_t RowSet::count() const {
    const uint64_t* p = words_.data();
    const size_t n = words_.size();
    size_t i = 0, total = 0;
#if defined(__AVX512VPOPCNTDQ__)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 8 <= n; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(p + i));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    total += (size_t)_mm512_reduce_add_epi64(acc);
#endif
    for (; i < n; ++i) total += (size_t)__builtin_popcountll(p[i]);
    return total;
}

void RowSet::mask_tail() {
    const size_t rem = nbits_ & 63;
    if (rem && !words_.empty())
        words_.back() &= ((1ull << rem) - 1);
}

} // namespace wm
