// bitvector.hpp — succinct bit vector with AVX-512 rank/select.
//
// rank1(i) = number of 1s in [0, i). This is the primitive that drives the
// whole wavelet matrix; it reduces to popcount of a prefix, so the inner loop
// is _mm512_popcnt_epi64 (VPOPCNTDQ) over 512-bit superblocks, with a
// cumulative sample so each rank touches O(1) words.
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace wm {

class BitVector {
public:
    // sb_(1,0) keeps the same invariant build() establishes for an empty
    // vector (sb_.size() >= 1), so rank1/select1 are safe on a default object.
    BitVector() : nbits_(0), ones_(0), sb_(1, 0) {}
    BitVector(std::vector<uint64_t> words, size_t nbits)
        : nbits_(nbits), words_(std::move(words)) { build(); }

    size_t size() const { return nbits_; }
    size_t ones() const { return ones_; }
    bool get(size_t i) const { return (words_[i >> 6] >> (i & 63)) & 1ull; }

    size_t rank1(size_t i) const;                 // # of 1s in [0, i)
    size_t rank0(size_t i) const { return i - rank1(i); }
    // position of the (k+1)-th set bit; returns nbits_ (a sentinel) if k >= ones().
    size_t select1(size_t k) const;
    // position of the (k+1)-th 0-bit; returns nbits_ if k >= (size()-ones()).
    size_t select0(size_t k) const;

    size_t mem_bytes() const {
        return (words_.size() + sb_.size()) * sizeof(uint64_t);
    }

    // raw storage accessors (for pooling node bitvectors into a contiguous arena)
    const uint64_t* words_data() const { return words_.data(); }
    size_t words_size() const { return words_.size(); }
    const uint64_t* sb_data() const { return sb_.data(); }
    size_t sb_size() const { return sb_.size(); }

private:
    static constexpr size_t WORDS_PER_SB = 8;     // 512-bit superblock
    void build();
    static size_t pc_words(const uint64_t* p, size_t nwords);

    size_t nbits_;
    size_t ones_;
    std::vector<uint64_t> words_;
    std::vector<uint64_t> sb_;   // sb_[b] = popcount of words_[0 .. b*8)
};

inline size_t BitVector::pc_words(const uint64_t* p, size_t nwords) {
    size_t i = 0, t = 0;
#if defined(__AVX512VPOPCNTDQ__)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 8 <= nwords; i += 8) {
        __m512i v = _mm512_loadu_si512((const void*)(p + i));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    t += (size_t)_mm512_reduce_add_epi64(acc);
#endif
    for (; i < nwords; ++i) t += (size_t)__builtin_popcountll(p[i]);
    return t;
}

inline void BitVector::build() {
    const size_t nw = words_.size();
    const size_t num_sb = (nw + WORDS_PER_SB - 1) / WORDS_PER_SB;
    sb_.assign(num_sb + 1, 0);
    for (size_t b = 0; b < num_sb; ++b) {
        const size_t start = b * WORDS_PER_SB;
        const size_t len = std::min(WORDS_PER_SB, nw - start);
        sb_[b + 1] = sb_[b] + pc_words(words_.data() + start, len);
    }
    ones_ = sb_.back();
}

inline size_t BitVector::rank1(size_t i) const {
    const size_t w = i >> 6;                 // word holding bit i
    const size_t b = w / WORDS_PER_SB;       // superblock of that word
    size_t r = sb_[b];
    r += pc_words(words_.data() + b * WORDS_PER_SB, w - b * WORDS_PER_SB);
    const size_t rem = i & 63;
    if (rem) r += (size_t)__builtin_popcountll(words_[w] & ((1ull << rem) - 1));
    return r;
}

inline size_t BitVector::select1(size_t k) const {
    if (k >= ones_) return nbits_;           // total function: no OOB for bad k
    // largest superblock b with sb_[b] <= k  (sb_.size() >= 2 here since ones_>0)
    size_t lo = 0, hi = sb_.size() - 1;
    while (lo + 1 < hi) {
        const size_t mid = (lo + hi) >> 1;
        if (sb_[mid] <= k) lo = mid; else hi = mid;
    }
    size_t cnt = sb_[lo];
    size_t w = lo * WORDS_PER_SB;
    const size_t nw = words_.size();
    while (w < nw) {
        const size_t pc = (size_t)__builtin_popcountll(words_[w]);
        if (cnt + pc > k) break;
        cnt += pc; ++w;
    }
    if (w >= nw) return nbits_;               // defensive (unreachable when k<ones_)
    uint64_t x = words_[w];
    for (size_t t = 0, need = k - cnt; t < need; ++t) x &= x - 1; // drop lowest 1s
    return w * 64 + (size_t)__builtin_ctzll(x);
}

inline size_t BitVector::select0(size_t k) const {
    if (k >= nbits_ - ones_) return nbits_;
    // smallest pos with rank0(pos+1) == k+1  (rank0(j) = j - rank1(j))
    size_t lo = 0, hi = nbits_;
    while (lo < hi) {
        const size_t mid = (lo + hi) >> 1;
        if ((mid + 1) - rank1(mid + 1) >= k + 1) hi = mid; else lo = mid + 1;
    }
    return lo;
}

} // namespace wm
