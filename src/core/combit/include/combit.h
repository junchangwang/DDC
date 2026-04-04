#ifndef COMBIT_H
#define COMBIT_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <chrono>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

///
/// ComBitBtv: A fixed-length bitvector segment compressed with
/// separated leading bits and literal words.  Fill words are virtual and
/// never stored.
///
/// Word size is fixed at 8 bits.
/// fill_ones is a runtime parameter controlling the fill value.
///
/// Compressed representation:
///   - leading bitstring: one bit per word; 0 = fill, 1 = literal
///   - literal data array: literal word values, stored sequentially
///
class ComBitBtv {
public:
    static constexpr unsigned word_size = 8;
    static constexpr size_t word_byte_size = 1;
    static constexpr size_t words_per_reg = 64;              // 512 / 8
    static constexpr size_t default_segment_bits = 1 << 16;  // 65536

    struct SizeBreakdown {
        size_t leading_bits_count;
        size_t literal_bits;
        size_t total_bits;
    };

    explicit ComBitBtv(bool fill_ones = false);

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBitBtv compress(const std::vector<bool>& bits, bool fill_ones = false);
    std::vector<bool> decompress() const;

    // ----------------------------------------------------------------
    // Convenience constructors
    // ----------------------------------------------------------------

    static ComBitBtv from_string(const std::string& bitstring, bool fill_ones = false);
    std::string to_string() const;

    // ----------------------------------------------------------------
    // Bitwise operations
    // ----------------------------------------------------------------

    ComBitBtv operator&(const ComBitBtv& other) const;
    ComBitBtv operator|(const ComBitBtv& other) const;
    ComBitBtv operator^(const ComBitBtv& other) const;
    ComBitBtv operator~() const;

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;

    // ----------------------------------------------------------------
    // Size / statistics
    // ----------------------------------------------------------------

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    bool                          fill_ones()          const { return fill_ones_; }
    const std::vector<uint64_t>&  leading_bits()       const { return leading_bits_; }
    size_t                        leading_bits_count() const { return leading_bits_count_; }
    bool                          is_fill(size_t i)    const { return is_fill_bit(i); }
    size_t                        bit_count()          const { return bit_count_; }
    size_t num_leading_bits_entries() const { return leading_bits_count_; }
    size_t num_fills() const;
    size_t num_literals()  const { return literal_count_; }
    const uint8_t* literal_data() const { return literal_data_.data(); }
    uint64_t get_literal(size_t idx) const;

    // ----------------------------------------------------------------
    // Serialization
    // ----------------------------------------------------------------

    void serialize(std::ostream& os) const;
    static ComBitBtv deserialize(std::istream& is);

    // ----------------------------------------------------------------
    // Debug printing
    // ----------------------------------------------------------------

    void print(std::ostream& os = std::cout) const;

private:
    bool                    fill_ones_;
    std::vector<uint64_t>   leading_bits_;         // packed bits: 0=fill, 1=literal
    size_t                  leading_bits_count_;   // number of words (leading bits)
    std::vector<uint8_t>    literal_data_;         // packed literal word bytes
    size_t                  literal_count_;        // number of literal words
    size_t                  bit_count_;            // original bitvector length

    bool is_fill_bit(size_t i) const {
        return !((leading_bits_[i / 64] >> (i % 64)) & 1);
    }
    void set_literal_bit(size_t i) {
        leading_bits_[i / 64] |= uint64_t(1) << (i % 64);
    }

    void push_literal(uint64_t val);
    void set_literal(size_t idx, uint64_t val);

    static uint64_t read_word_from_bits(const std::vector<bool>& bits,
                                        size_t word_idx);
    static void append_word_to_bits(std::vector<bool>& bits, uint64_t word);

    friend class ComBit;
};

// ====================================================================
// ComBit: Segmented bitvector composed of ComBitBtv segments
// ====================================================================

///
/// ComBit: A segmented compressed bitvector.
///
/// The bitvector is partitioned into fixed-length segments (default 2^16
/// bits each), where each segment is independently compressed as a
/// ComBitBtv (8-bit word size).
///
class ComBit {
public:
    static constexpr size_t default_segment_bits = size_t(1) << 16;

    struct SizeBreakdown {
        size_t leading_bits_count;
        size_t literal_bits;
        size_t total_bits;
    };

    ComBit() = default;

    // ----------------------------------------------------------------
    // Compression / Decompression
    // ----------------------------------------------------------------

    static ComBit compress(const std::vector<bool>& bits,
                           bool fill_ones = false,
                           size_t segment_bits = default_segment_bits);

    std::vector<bool> decompress() const;

    // ----------------------------------------------------------------
    // Convenience constructors
    // ----------------------------------------------------------------

    static ComBit from_string(const std::string& bitstring,
                              bool fill_ones = false,
                              size_t segment_bits = default_segment_bits);

    std::string to_string() const;

    // ----------------------------------------------------------------
    // Bitwise operations (segment-wise)
    // ----------------------------------------------------------------

    ComBit operator&(const ComBit& other) const;
    ComBit operator|(const ComBit& other) const;
    ComBit operator^(const ComBit& other) const;
    ComBit operator~() const;

    // ----------------------------------------------------------------
    // Queries
    // ----------------------------------------------------------------

    size_t popcount() const;
    std::vector<size_t> set_bit_positions() const;

    // ----------------------------------------------------------------
    // Size / statistics
    // ----------------------------------------------------------------

    SizeBreakdown size_breakdown() const;
    size_t compressed_size_bits()  const { return size_breakdown().total_bits; }
    size_t compressed_size_bytes() const { return (compressed_size_bits() + 7) / 8; }
    size_t original_size_bits()    const { return bit_count_; }
    double compression_ratio() const;

    // ----------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------

    size_t bit_count()     const { return bit_count_; }
    size_t num_segments()  const { return segments_.size(); }
    size_t segment_bits()  const { return segment_bits_; }

    const std::vector<ComBitBtv>& segments() const { return segments_; }
    const ComBitBtv& segment(size_t i) const { return segments_[i]; }

    // ----------------------------------------------------------------
    // Serialization
    // ----------------------------------------------------------------

    void serialize(std::ostream& os) const;
    static ComBit deserialize(std::istream& is);

    // ----------------------------------------------------------------
    // Debug printing
    // ----------------------------------------------------------------

    void print(std::ostream& os = std::cout) const;

private:
    std::vector<ComBitBtv> segments_;
    size_t bit_count_ = 0;
    size_t segment_bits_ = default_segment_bits;
};

#endif // COMBIT_H
