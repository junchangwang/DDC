#ifndef SEGMENTED_BITSET_HPP
#define SEGMENTED_BITSET_HPP

#include "bitset_vector.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace bitset {

class SegmentedBitset {
public:

    static constexpr size_t DEFAULT_BITS_PER_SEG = 65536;

    SegmentedBitset() : bits_per_seg_(DEFAULT_BITS_PER_SEG), total_bits_(0) {}
    explicit SegmentedBitset(size_t bits_per_seg)
        : bits_per_seg_(bits_per_seg), total_bits_(0) {}

    void build_from(const BitsetVector& flat, size_t bits_per_seg);
    void build_from(const BitsetVector& flat) { build_from(flat, bits_per_seg_); }

    BitsetVector flatten() const;

    size_t num_segments() const { return segments_.size(); }

    size_t bits_per_seg() const { return bits_per_seg_; }

    uint64_t total_bits() const { return total_bits_; }

    const BitsetVector& segment(size_t i) const { return segments_[i]; }
    BitsetVector& segment_mut(size_t i) { return segments_[i]; }

    uint64_t popcount(bool use_simd) const;

    static SegmentedBitset seg_or(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    static SegmentedBitset seg_and(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    static void seg_or_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    static void seg_and_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

private:
    size_t bits_per_seg_;
    uint64_t total_bits_;
    std::vector<BitsetVector> segments_;
};

}

#endif
