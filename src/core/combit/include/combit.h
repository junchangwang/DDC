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
/// ComBitBtv: A fixed-length bitvector segment compressed with a
/// three-level structure:
///
///   L1 – literal data: 8-bit words of the original bitvector (only
///         non-fill words are stored).
///   L2 – leading bitstring for L1: one bit per 8-bit word of the
///         original bitvector (0 = fill, 1 = literal).  Stored as a
///         packed byte array; only non-zero bytes are kept when the L3
///         layer is active.
///   L3 – leading bitstring for L2: one bit per 8-bit chunk of L2
///         (i.e. per group of 64 original words = 512 bits).
///         0 = the L2 byte is all-zero (entire 64-word region is fills),
///         1 = the L2 byte is a literal and stored in l2_literals_.
///
/// When the L2 density is high (≥ 1/64), the L3 layer is bypassed and
/// L2 is stored flat (use_l3_ == false) for zero overhead.
///
/// Word size is fixed at 8 bits.
/// fill_ones is a runtime parameter controlling the fill value.
///
class ComBitBtv {
public:
    static constexpr unsigned word_size = 8;
    static constexpr size_t word_byte_size = 1;
    static constexpr size_t words_per_reg = 64;              // 512 / 8
    static constexpr size_t l2_bits_per_l3_bit = 8;          // 8 L2 bits per L3 bit
    static constexpr size_t words_per_l3_bit = 64;           // 8 * 8
    static constexpr size_t default_segment_bits = 1 << 16;  // 65536

    struct SizeBreakdown {
        size_t l3_bits;            // L3 leading bits
        size_t l2_literal_bits;    // L2 stored literal bytes * 8
        size_t l1_literal_bits;    // L1 stored literal bytes * 8
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
    bool                          use_l3()             const { return use_l3_; }
    size_t                        l2_count()           const { return l2_count_; }
    size_t                        l3_count()           const { return l3_count_; }
    size_t                        bit_count()          const { return bit_count_; }
    size_t num_fills() const;
    size_t num_literals()  const { return l1_literal_count_; }

    // Raw data access (used by bitwise operators)
    const uint8_t* l3_data()         const { return l3_bits_.data(); }
    const uint8_t* l2_flat_data()    const { return l2_flat_.data(); }
    const uint8_t* l2_literal_data() const { return l2_literals_.data(); }
    const uint8_t* l1_literal_data() const { return l1_literals_.data(); }
    size_t         l2_literal_count() const { return l2_literal_count_; }

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
    size_t                  bit_count_;            // original bitvector length

    // --- L2: leading bitstring for L1 (1 bit per 8-bit word) ---
    // When use_l3_ == false, l2_flat_ stores the packed L2 bytes directly.
    // When use_l3_ == true,  L2 is compressed via l3/l2_literals_.
    bool                    use_l3_;
    size_t                  l2_count_;             // total L2 bits (= num 8-bit words)
    std::vector<uint8_t>    l2_flat_;              // flat L2 bytes (used when !use_l3_)

    // --- L3: leading bitstring for L2 (1 bit per 8-bit chunk of L2) ---
    size_t                  l3_count_;             // total L3 bits
    std::vector<uint8_t>    l3_bits_;              // packed L3 bytes
    std::vector<uint8_t>    l2_literals_;          // L2 literal bytes (non-zero L2 chunks)
    size_t                  l2_literal_count_;

    // --- L1: literal data ---
    std::vector<uint8_t>    l1_literals_;          // L1 literal word bytes
    size_t                  l1_literal_count_;

    // Rebuild flat L2 from L3 + L2 literals (for decompression / scalar paths)
    std::vector<uint8_t> expand_l2() const;

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
        size_t l3_bits;
        size_t l2_literal_bits;
        size_t l1_literal_bits;
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
