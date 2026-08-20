// andnot.cpp — RowSet AND-NOT (this = this & ~o). AVX-512 with scalar fallback.
#include "rowset.hpp"
#include <algorithm>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace wm {

void RowSet::op_andnot(const RowSet& o) {
    uint64_t* a = words_.data();
    const uint64_t* b = o.words_.data();
    const size_t n = std::min(words_.size(), o.words_.size()); // overlap-safe
    size_t i = 0;
#if defined(__AVX512F__)
    // _mm512_andnot_si512(x, y) = (~x) & y ; we want a & ~b -> andnot(b, a)
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_si512((void*)(a + i),
            _mm512_andnot_si512(_mm512_loadu_si512((const void*)(b + i)),
                                _mm512_loadu_si512((const void*)(a + i))));
#endif
    for (; i < n; ++i) a[i] &= ~b[i];
}

RowSet RowSet::ANDNOT(const RowSet& a, const RowSet& b) { RowSet r = a; r.op_andnot(b); return r; }

} // namespace wm
