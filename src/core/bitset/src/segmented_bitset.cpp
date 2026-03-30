#include <segmented_bitset.hpp>
#include <bitset_simd.hpp>
#include <algorithm>
#include <cstring>

namespace bitset {

// ---------------------------------------------------------------
// Build from flat BitsetVector
// ---------------------------------------------------------------
void SegmentedBitset::build_from(const BitsetVector& flat, size_t bits_per_seg) {
    bits_per_seg_ = bits_per_seg;
    total_bits_ = flat.num_bits();
    segments_.clear();

    size_t words_per_seg = bits_per_seg / 64;
    size_t total_words = flat.words_cnt();
    size_t num_segs = (total_words + words_per_seg - 1) / words_per_seg;

    segments_.resize(num_segs);
    const uint64_t* src = flat.words();

    for (size_t i = 0; i < num_segs; ++i) {
        size_t offset = i * words_per_seg;
        size_t count = std::min(words_per_seg, total_words - offset);
        segments_[i].allocate(count);
        std::memcpy(segments_[i].words_mut(), src + offset, count * sizeof(uint64_t));

        // Set num_bits for last segment
        if (i == num_segs - 1) {
            size_t remaining_bits = total_bits_ - i * bits_per_seg;
            segments_[i].set_num_bits(remaining_bits);
        } else {
            segments_[i].set_num_bits(bits_per_seg);
        }
    }
}

// ---------------------------------------------------------------
// Flatten back to single BitsetVector
// ---------------------------------------------------------------
BitsetVector SegmentedBitset::flatten() const {
    BitsetVector result;
    if (segments_.empty()) return result;

    size_t total_words = 0;
    for (auto& seg : segments_) total_words += seg.words_cnt();
    result.allocate_nozero(total_words);

    size_t offset = 0;
    for (auto& seg : segments_) {
        std::memcpy(result.words_mut() + offset, seg.words(), seg.words_cnt() * sizeof(uint64_t));
        offset += seg.words_cnt();
    }
    result.set_num_bits(total_bits_);
    return result;
}

// ---------------------------------------------------------------
// Popcount
// ---------------------------------------------------------------
uint64_t SegmentedBitset::popcount(bool use_simd) const {
    uint64_t total = 0;
    for (auto& seg : segments_)
        total += seg.popcount(use_simd);
    return total;
}

// ---------------------------------------------------------------
// Segment-level OR
// ---------------------------------------------------------------
SegmentedBitset SegmentedBitset::seg_or(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd) {
    SegmentedBitset result(a.bits_per_seg_);
    size_t n = std::max(a.segments_.size(), b.segments_.size());
    size_t m = std::min(a.segments_.size(), b.segments_.size());
    result.segments_.resize(n);
    result.total_bits_ = std::max(a.total_bits_, b.total_bits_);

    for (size_t i = 0; i < m; ++i)
        result.segments_[i] = BitsetVector::word_or(a.segments_[i], b.segments_[i], use_simd);
    // Remaining segments from the longer one
    const auto& longer = (a.segments_.size() > b.segments_.size()) ? a : b;
    for (size_t i = m; i < n; ++i)
        result.segments_[i] = longer.segments_[i];  // copy

    return result;
}

// ---------------------------------------------------------------
// Segment-level AND
// ---------------------------------------------------------------
SegmentedBitset SegmentedBitset::seg_and(const SegmentedBitset& a, const SegmentedBitset& b, bool use_simd) {
    SegmentedBitset result(a.bits_per_seg_);
    size_t m = std::min(a.segments_.size(), b.segments_.size());
    result.segments_.resize(m);
    result.total_bits_ = std::max(a.total_bits_, b.total_bits_);

    for (size_t i = 0; i < m; ++i)
        result.segments_[i] = BitsetVector::word_and(a.segments_[i], b.segments_[i], use_simd);

    return result;
}

// ---------------------------------------------------------------
// In-place OR
// ---------------------------------------------------------------
void SegmentedBitset::seg_or_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd) {
    if (b.segments_.size() > a.segments_.size()) {
        a.segments_.resize(b.segments_.size());
    }
    size_t m = std::min(a.segments_.size(), b.segments_.size());
    for (size_t i = 0; i < m; ++i) {
        if (a.segments_[i].words_cnt() == 0) {
            a.segments_[i] = b.segments_[i];
        } else {
            // Use word_or and move-assign (word_or_inplace not implemented)
            a.segments_[i] = BitsetVector::word_or(a.segments_[i], b.segments_[i], use_simd);
        }
    }
    a.total_bits_ = std::max(a.total_bits_, b.total_bits_);
}

// ---------------------------------------------------------------
// In-place AND
// ---------------------------------------------------------------
void SegmentedBitset::seg_and_inplace(SegmentedBitset& a, const SegmentedBitset& b, bool use_simd) {
    size_t m = std::min(a.segments_.size(), b.segments_.size());
    for (size_t i = 0; i < m; ++i)
        BitsetVector::word_and_inplace(a.segments_[i], b.segments_[i], use_simd);
    // Truncate segments beyond b's range (AND with missing = 0)
    for (size_t i = m; i < a.segments_.size(); ++i)
        a.segments_[i] = BitsetVector();
}

} // namespace bitset
