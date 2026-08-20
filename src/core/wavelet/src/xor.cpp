// xor.cpp — RowSet logical XOR (this = this ^ o). AVX-512 with scalar fallback.
// XOR of two equal-size sets with clean tails keeps the tail clean (0^0=0).
#include "rowset.hpp"
#include <algorithm>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace wm {

void RowSet::op_xor(const RowSet& o) {
    uint64_t* a = words_.data();
    const uint64_t* b = o.words_.data();
    const size_t n = std::min(words_.size(), o.words_.size()); // overlap-safe
    size_t i = 0;
#if defined(__AVX512F__)
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_si512((void*)(a + i),
            _mm512_xor_si512(_mm512_loadu_si512((const void*)(a + i)),
                             _mm512_loadu_si512((const void*)(b + i))));
#endif
    for (; i < n; ++i) a[i] ^= b[i];
}

RowSet RowSet::XOR(const RowSet& a, const RowSet& b) { RowSet r = a; r.op_xor(b); return r; }

} // namespace wm
