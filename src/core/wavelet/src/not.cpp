// not.cpp — RowSet logical NOT (this = ~this), tail bits beyond nbits_ cleared.
#include "rowset.hpp"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace wm {

void RowSet::op_not() {
    uint64_t* a = words_.data();
    const size_t n = words_.size();
    size_t i = 0;
#if defined(__AVX512F__)
    const __m512i ones = _mm512_set1_epi64(-1);
    for (; i + 8 <= n; i += 8)
        _mm512_storeu_si512((void*)(a + i),
            _mm512_xor_si512(_mm512_loadu_si512((const void*)(a + i)), ones));
#endif
    for (; i < n; ++i) a[i] = ~a[i];
    mask_tail();
}

RowSet RowSet::NOT(const RowSet& a) { RowSet r = a; r.op_not(); return r; }

} // namespace wm
