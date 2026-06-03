#ifndef BITMAP_VECTOR_HPP
#define BITMAP_VECTOR_HPP

#include <vector>
#include <cstdint>
#include <types.hpp>
#include <ddc_encoder.hpp>
#include <ddc_decoder.hpp>

namespace ddc {

/**
 * BitmapVector
 *
 * Represents a single compressed bitmap vector for one distinct key value.
 * Bits can be appended (set) at arbitrary positions; the vector grows as
 * needed.  The compressed representation is maintained lazily — it is
 * recomputed when requested.
 *
 * Word-level bitwise operations (OR, AND, XOR, ANDNOT) are accelerated
 * with AVX-512 / AVX2 intrinsics when available at runtime.
 */
class BitmapVector {
public:
    BitmapVector() = default;

    /// Set the bit at the given position (0-based). Positions may be set in
    /// any order; duplicates are idempotent.
    void set_bit(uint64_t position);

    /// Get the value of the bit at the given position.
    bool get_bit(uint64_t position) const;

    /// Return the total number of bits tracked (i.e., one past the highest
    /// position that has been set, rounded up to word boundary).
    uint64_t size() const;

    /// Return the number of set bits (SIMD-accelerated).
    uint64_t popcount() const;

    /// Compress the current uncompressed state and return the encoding.
    DDCEncoding encode() const;

    /// Replace the contents of this vector with the decompressed form of
    /// the given encoding.
    void load(const DDCEncoding& enc);

    /// Access the raw uncompressed words (read-only).
    const std::vector<Word>& raw_words() const { return words_; }

    /// Access the raw uncompressed words (mutable).
    std::vector<Word>& raw_words_mut() { return words_; }

    // -----------------------------------------------------------------
    //  Word-level bitwise operations (SIMD-accelerated)
    // -----------------------------------------------------------------

    /// dst = this | other
    static BitmapVector word_or(const BitmapVector& a, const BitmapVector& b);

    /// dst = this & other
    static BitmapVector word_and(const BitmapVector& a, const BitmapVector& b);

    /// dst = this ^ other
    static BitmapVector word_xor(const BitmapVector& a, const BitmapVector& b);

    /// dst = this & ~other
    static BitmapVector word_andnot(const BitmapVector& a, const BitmapVector& b);

    /// Extract all set-bit positions into a sorted vector.
    std::vector<uint32_t> decode_positions() const;

private:
    /// Ensure the internal word vector is large enough to hold position `pos`.
    void ensure_capacity(uint64_t pos);

    std::vector<Word> words_;
};

} // namespace ddc

#endif // BITMAP_VECTOR_HPP