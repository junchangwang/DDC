#ifndef BITSET_VECTOR_HPP
#define BITSET_VECTOR_HPP

#include <vector>
#include <cstdint>
#include <string>

namespace bitset {

/**
 * BitsetVector
 *
 * Represents a single uncompressed bitmap stored as packed uint64_t words.
 * Provides both plain scalar and SIMD-accelerated (AVX-512 / AVX2) bitwise
 * operations for OR, AND, XOR, ANDNOT, and popcount.
 *
 * This class is the core data structure; thin IBitmapBackend wrappers live
 * in src/benchmark/backends/.
 */
class BitsetVector {
public:
    BitsetVector() = default;

    /// Set the bit at the given position (0-based).
    void set_bit(uint64_t position);

    /// Get the value of the bit at the given position.
    bool get_bit(uint64_t position) const;

    /// Return the logical number of bits (MUST be set explicitly via
    /// set_num_bits or inferred from the word array size * 64).
    uint64_t num_bits() const { return num_bits_; }
    void set_num_bits(uint64_t n) { num_bits_ = n; }

    /// Return the number of set bits.
    /// If `use_simd` is true, uses best available SIMD (AVX-512 → AVX2).
    /// If false, uses plain scalar loop.
    uint64_t popcount(bool use_simd) const;

    /// Access the raw uint64_t words (read-only).
    const std::vector<uint64_t>& words() const { return words_; }

    /// Access the raw uint64_t words (mutable).
    std::vector<uint64_t>& words_mut() { return words_; }

    // -----------------------------------------------------------------
    //  Word-level bitwise operations
    //  use_simd = true  → AVX-512 / AVX2 runtime dispatch
    //  use_simd = false → plain scalar C++ loop
    // -----------------------------------------------------------------

    static BitsetVector word_or(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_and(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_xor(const BitsetVector& a, const BitsetVector& b, bool use_simd);
    static BitsetVector word_andnot(const BitsetVector& a, const BitsetVector& b, bool use_simd);

    /// Extract all set-bit positions into a sorted vector.
    std::vector<uint32_t> decode_positions() const;

    /// Serialize: write raw packed bits to file.
    void serialize(const std::string& path) const;

    /// Load: read raw packed bits from file. Returns true on success.
    bool load(const std::string& path);

private:
    void ensure_capacity(uint64_t pos);

    std::vector<uint64_t> words_;
    uint64_t num_bits_ = 0;
};

} // namespace bitset

#endif // BITSET_VECTOR_HPP
