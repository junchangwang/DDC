#ifndef SEGMENTED_BITSET_HPP
#define SEGMENTED_BITSET_HPP

#include "bitset_vector.hpp"
#include <vector>
#include <cstdint>
#include <cstddef>

namespace bitset {

/**
 * SegmentedBitset
 *
 * Splits a long bitmap into fixed-size segments, each backed by a
 * BitsetVector.  This mirrors CRoaring's container-per-chunk design:
 * each segment covers `bits_per_seg` bits (default 65536 = 8 KB).
 *
 * Segment-level operations (OR, AND, …) work on matching pairs of
 * segments independently, enabling cache-friendly processing and
 * potential parallelism.
 *
 * Segment sizes can differ between two SegmentedBitsets — the
 * operations handle this by processing at the granularity of the
 * smaller segment size.
 */
class SegmentedBitset {
public:
    /// Default segment size: 65536 bits = 1024 words = 8 KB (matches CRoaring)
    static constexpr size_t DEFAULT_BITS_PER_SEG = 65536;

    SegmentedBitset() : bits_per_seg_(DEFAULT_BITS_PER_SEG), total_bits_(0) {}
    explicit SegmentedBitset(size_t bits_per_seg)
        : bits_per_seg_(bits_per_seg), total_bits_(0) {}

    /// Build from a flat BitsetVector: split into segments.
    void build_from(const BitsetVector& flat, size_t bits_per_seg);
    void build_from(const BitsetVector& flat) { build_from(flat, bits_per_seg_); }

    /// Flatten back to a single BitsetVector.
    BitsetVector flatten() const;

    /// Number of segments.
    size_t num_segments() const { return segments_.size(); }

    /// Segment size in bits.
    size_t bits_per_seg() const { return bits_per_seg_; }

    /// Total bits across all segments.
    uint64_t total_bits() const { return total_bits_; }

    /// Access individual segments.
    const BitsetVector& segment(size_t i) const { return segments_[i]; }
    BitsetVector& segment_mut(size_t i) { return segments_[i]; }

    /// Total popcount across all segments.
    uint64_t popcount(bool use_simd) const;

    // -----------------------------------------------------------------
    //  Segment-level bitwise operations
    // -----------------------------------------------------------------

    /// OR: result[i] = a.seg[i] | b.seg[i]
    static SegmentedBitset seg_or(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    /// AND: result[i] = a.seg[i] & b.seg[i]
    static SegmentedBitset seg_and(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    /// In-place OR: a.seg[i] |= b.seg[i]
    static void seg_or_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

    /// In-place AND: a.seg[i] &= b.seg[i]
    static void seg_and_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd);

private:
    size_t bits_per_seg_;
    uint64_t total_bits_;
    std::vector<BitsetVector> segments_;
};

} // namespace bitset

#endif // SEGMENTED_BITSET_HPP
