// rowset.hpp — dense row-set (one bit per row); declaration only.
//
// A predicate over a column produces a RowSet (a bitvector of n rows). To
// combine predicates across columns we AND / OR / XOR / NOT these RowSets.
// On dense row-sets this is a pure word-parallel operation that maps onto
// AVX-512 with a scalar fallback. Each operation lives in its OWN translation
// unit (src/and.cpp, or.cpp, xor.cpp, not.cpp, andnot.cpp); shared helpers
// (count, mask_tail) live in src/rowset.cpp.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace wm {

class RowSet {
public:
    RowSet() : nbits_(0) {}
    explicit RowSet(size_t nbits)
        : nbits_(nbits), words_((nbits + 63) / 64, 0ull) {}

    size_t size() const { return nbits_; }
    size_t num_words() const { return words_.size(); }
    const uint64_t* data() const { return words_.data(); }
    uint64_t* data() { return words_.data(); }

    void set(size_t i)   { words_[i >> 6] |=  (1ull << (i & 63)); }
    void clear(size_t i) { words_[i >> 6] &= ~(1ull << (i & 63)); }
    bool get(size_t i) const { return (words_[i >> 6] >> (i & 63)) & 1ull; }

    size_t mem_bytes() const { return words_.size() * sizeof(uint64_t); }

    size_t count() const;                 // popcount / cardinality   [rowset.cpp]

    // in-place set operations. Operands are expected to be the same size; if
    // they differ, the op runs over the overlap (no out-of-bounds) — see each .cpp.
    void op_and(const RowSet& o);         // this = this AND o        [and.cpp]
    void op_or(const RowSet& o);          // this = this OR  o        [or.cpp]
    void op_xor(const RowSet& o);         // this = this XOR o        [xor.cpp]
    void op_not();                        // this = NOT this (masked)  [not.cpp]
    void op_andnot(const RowSet& o);      // this = this AND (NOT o)  [andnot.cpp]

    // out-of-place convenience (defined alongside their in-place op).
    static RowSet AND(const RowSet& a, const RowSet& b);     // [and.cpp]
    static RowSet OR (const RowSet& a, const RowSet& b);     // [or.cpp]
    static RowSet XOR(const RowSet& a, const RowSet& b);     // [xor.cpp]
    static RowSet NOT(const RowSet& a);                      // [not.cpp]
    static RowSet ANDNOT(const RowSet& a, const RowSet& b);  // [andnot.cpp]

private:
    void mask_tail();                     // clear bits beyond nbits_ [rowset.cpp]

    size_t nbits_;
    std::vector<uint64_t> words_;
};

} // namespace wm
