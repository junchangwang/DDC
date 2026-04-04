#include "combit.h"

#include <stdexcept>

// ====================================================================
// ComBitBtv member function definitions (3-level: L3/L2/L1)
// ====================================================================

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

ComBitBtv::ComBitBtv(bool fill_ones)
    : fill_ones_(fill_ones), bit_count_(0), use_l3_(false),
      l2_count_(0), l3_count_(0), l2_literal_count_(0),
      l1_literal_count_(0) {}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

uint64_t ComBitBtv::get_literal(size_t idx) const {
    return l1_literals_[idx];
}

uint64_t ComBitBtv::read_word_from_bits(const std::vector<bool>& bits,
                                        size_t word_idx) {
    uint64_t word = 0;
    size_t start = word_idx * 8;
    for (unsigned i = 0; i < 8 && start + i < bits.size(); i++) {
        if (bits[start + i])
            word |= uint64_t(1) << (7 - i);
    }
    return word;
}

void ComBitBtv::append_word_to_bits(std::vector<bool>& bits, uint64_t word) {
    for (int i = 7; i >= 0; i--)
        bits.push_back((word >> i) & 1);
}

/// Rebuild the flat L2 byte array from L3 + l2_literals_.
/// If !use_l3_, just returns l2_flat_ directly.
std::vector<uint8_t>
ComBitBtv::expand_l2() const {
    if (!use_l3_)
        return l2_flat_;

    size_t l2_byte_count = (l2_count_ + 7) / 8;
    std::vector<uint8_t> l2(l2_byte_count, 0);
    size_t lit_idx = 0;
    for (size_t i = 0; i < l3_count_; i++) {
        uint8_t l3_byte = l3_bits_[i / 8];
        bool is_literal = (l3_byte >> (i % 8)) & 1;
        if (is_literal) {
            if (i < l2_byte_count)
                l2[i] = l2_literals_[lit_idx++];
        }
    }
    return l2;
}

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::compress(const std::vector<bool>& bits, bool fill_ones) {
    ComBitBtv result(fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.l2_count_ = num_words;

    // Step 1: Build flat L2 (one bit per word, packed into bytes)
    // and collect L1 literals.
    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    const uint64_t fill_val = fill_ones ? uint64_t(0xFF) : uint64_t(0);
    for (size_t i = 0; i < num_words; i++) {
        uint64_t word = read_word_from_bits(bits, i);
        if (word != fill_val) {
            l2_flat[i / 8] |= uint8_t(1) << (i % 8);
            result.l1_literals_.push_back(static_cast<uint8_t>(word));
        }
    }
    result.l1_literal_count_ = result.l1_literals_.size();

    // Step 2: Decide whether to use L3 compression on L2.
    // Count non-zero L2 bytes (each byte covers 8 words = 64 bits).
    size_t l2_nonzero = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != 0) l2_nonzero++;
    }

    // Use L3 when L2 density < 1/64 (i.e., ≤ ~1.5% of L2 bytes are non-zero)
    // or more practically, when L3 + L2 literals < flat L2 size.
    size_t flat_cost = l2_byte_count;
    size_t l3_byte_count = (l2_byte_count + 7) / 8;
    size_t l3_cost = l3_byte_count + l2_nonzero;

    if (l3_cost < flat_cost) {
        result.use_l3_ = true;
        result.l3_count_ = l2_byte_count;  // one L3 bit per L2 byte

        result.l3_bits_.assign(l3_byte_count, 0);
        for (size_t i = 0; i < l2_byte_count; i++) {
            if (l2_flat[i] != 0) {
                result.l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
                result.l2_literals_.push_back(l2_flat[i]);
            }
        }
        result.l2_literal_count_ = result.l2_literals_.size();
        // l2_flat_ left empty when L3 is active
    } else {
        result.use_l3_ = false;
        result.l3_count_ = 0;
        result.l2_flat_ = std::move(l2_flat);
        result.l2_literal_count_ = 0;
    }

    return result;
}

// ----------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------

std::vector<bool>
ComBitBtv::decompress() const {
    std::vector<bool> result;
    result.reserve(bit_count_);

    // Reconstruct flat L2
    auto l2 = expand_l2();

    size_t lit_idx = 0;
    for (size_t i = 0; i < l2_count_; i++) {
        uint8_t l2_byte = l2[i / 8];
        bool is_literal = (l2_byte >> (i % 8)) & 1;
        if (!is_literal) {
            for (unsigned b = 0; b < 8; b++)
                result.push_back(fill_ones_);
        } else {
            append_word_to_bits(result, l1_literals_[lit_idx++]);
        }
    }

    result.resize(bit_count_);
    return result;
}

// ----------------------------------------------------------------
// Convenience constructors
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::from_string(const std::string& bitstring, bool fill_ones) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, fill_ones);
}

std::string
ComBitBtv::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size() + bits.size() / 8);
    for (size_t i = 0; i < bits.size(); i++) {
        if (i > 0 && i % 8 == 0) s += ' ';
        s += bits[i] ? '1' : '0';
    }
    return s;
}

// ----------------------------------------------------------------
// operator~
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator~() const {
    auto bits = decompress();
    for (size_t i = 0; i < bits.size(); i++)
        bits[i] = !bits[i];
    return compress(bits, fill_ones_);
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

size_t
ComBitBtv::popcount() const {
    // Reconstruct flat L2 for iteration
    auto l2 = expand_l2();

    size_t count = 0;
    size_t bits_seen = 0;
    size_t lit_idx = 0;

    for (size_t i = 0; i < l2_count_; i++) {
        uint8_t l2_byte = l2[i / 8];
        bool is_literal = (l2_byte >> (i % 8)) & 1;

        if (!is_literal) {
            if (fill_ones_) {
                size_t remaining = (bits_seen < bit_count_)
                    ? bit_count_ - bits_seen : 0;
                count += std::min(size_t(8), remaining);
            }
            bits_seen += 8;
        } else {
            uint64_t w = l1_literals_[lit_idx++];
            size_t remaining = (bits_seen < bit_count_)
                ? bit_count_ - bits_seen : 0;
            if (remaining >= 8) {
                count += __builtin_popcountll(w);
            } else {
                for (size_t b = 0; b < remaining; b++) {
                    if (w & (uint64_t(1) << (7 - b)))
                        count++;
                }
            }
            bits_seen += 8;
        }
    }
    return count;
}

std::vector<size_t>
ComBitBtv::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) pos.push_back(i);
    }
    return pos;
}

// ----------------------------------------------------------------
// Size / statistics
// ----------------------------------------------------------------

ComBitBtv::SizeBreakdown
ComBitBtv::size_breakdown() const {
    SizeBreakdown sb;
    if (use_l3_) {
        sb.l3_bits = l3_count_;
        sb.l2_literal_bits = l2_literal_count_ * 8;
    } else {
        sb.l3_bits = 0;
        sb.l2_literal_bits = l2_flat_.size() * 8;
    }
    sb.l1_literal_bits = l1_literal_count_ * 8;
    sb.total_bits = sb.l3_bits + sb.l2_literal_bits + sb.l1_literal_bits;
    return sb;
}

double
ComBitBtv::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

// ----------------------------------------------------------------
// num_fills
// ----------------------------------------------------------------

size_t
ComBitBtv::num_fills() const {
    auto l2 = expand_l2();
    size_t lit = 0;
    for (size_t i = 0; i < (l2_count_ + 7) / 8; i++)
        lit += __builtin_popcount(l2[i]);
    return l2_count_ - lit;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

void
ComBitBtv::print(std::ostream& os) const {
    os << "ComBitBtv compressed bitvector (3-level):\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  L2 count: " << l2_count_ << " (words)\n";
    os << "  Use L3: " << (use_l3_ ? "yes" : "no") << "\n";

    if (use_l3_) {
        os << "  L3 count: " << l3_count_ << " bits\n";
        os << "  L2 literals: " << l2_literal_count_ << " bytes\n";
    } else {
        os << "  L2 flat: " << l2_flat_.size() << " bytes\n";
    }
    os << "  L1 literals: " << l1_literal_count_ << " bytes\n";

    auto sb = size_breakdown();
    os << "  Size breakdown: L3=" << sb.l3_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

// ====================================================================
// ComBit (segmented) member function definitions
// ====================================================================

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

ComBit
ComBit::compress(const std::vector<bool>& bits, bool fill_ones,
                 size_t segment_bits) {
    ComBit result;
    result.bit_count_ = bits.size();
    result.segment_bits_ = segment_bits;

    size_t offset = 0;
    while (offset < bits.size()) {
        size_t seg_len = std::min(segment_bits, bits.size() - offset);
        std::vector<bool> seg_bits(bits.begin() + static_cast<ptrdiff_t>(offset),
                                   bits.begin() + static_cast<ptrdiff_t>(offset + seg_len));
        result.segments_.emplace_back(
            ComBitBtv::compress(seg_bits, fill_ones));
        offset += seg_len;
    }

    return result;
}

// ----------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------

std::vector<bool>
ComBit::decompress() const {
    std::vector<bool> result;
    result.reserve(bit_count_);

    for (const auto& seg : segments_) {
        auto seg_bits = seg.decompress();
        result.insert(result.end(), seg_bits.begin(), seg_bits.end());
    }

    return result;
}

// ----------------------------------------------------------------
// Convenience constructors
// ----------------------------------------------------------------

ComBit
ComBit::from_string(const std::string& bitstring, bool fill_ones,
                    size_t segment_bits) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, fill_ones, segment_bits);
}

std::string
ComBit::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); i++)
        s += bits[i] ? '1' : '0';
    return s;
}

// ----------------------------------------------------------------
// operator~
// ----------------------------------------------------------------

ComBit
ComBit::operator~() const {
    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (const auto& seg : segments_)
        result.segments_.push_back(~seg);

    return result;
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

size_t
ComBit::popcount() const {
    size_t count = 0;
    for (const auto& seg : segments_)
        count += seg.popcount();
    return count;
}

std::vector<size_t>
ComBit::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i]) pos.push_back(i);
    return pos;
}

// ----------------------------------------------------------------
// Size / statistics
// ----------------------------------------------------------------

ComBit::SizeBreakdown
ComBit::size_breakdown() const {
    SizeBreakdown sb{0, 0, 0, 0};
    for (const auto& seg : segments_) {
        auto ssb = seg.size_breakdown();
        sb.l3_bits += ssb.l3_bits;
        sb.l2_literal_bits += ssb.l2_literal_bits;
        sb.l1_literal_bits += ssb.l1_literal_bits;
        sb.total_bits += ssb.total_bits;
    }
    return sb;
}

double
ComBit::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

void
ComBit::print(std::ostream& os) const {
    os << "ComBit segmented bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Segment size:  " << segment_bits_ << " bits\n";
    os << "  Num segments:  " << segments_.size() << "\n";

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& s = segments_[i];
        os << "  Segment " << i << ": "
           << "ComBitBtv"
           << " fill_ones=" << s.fill_ones()
           << " bits=" << s.bit_count()
           << " use_l3=" << s.use_l3()
           << " fills=" << s.num_fills()
           << " literals=" << s.num_literals()
           << "\n";
    }

    auto sb = size_breakdown();
    os << "  Size breakdown: L3=" << sb.l3_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

// ====================================================================
// Serialization / Deserialization
// ====================================================================

// Helper: write POD value to stream
template<typename T>
static void write_val(std::ostream& os, T val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

// Helper: read POD value from stream
template<typename T>
static T read_val(std::istream& is) {
    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

// ----------------------------------------------------------------
// ComBitBtv::serialize / deserialize
// ----------------------------------------------------------------

void ComBitBtv::serialize(std::ostream& os) const {
    write_val<uint8_t>(os, static_cast<uint8_t>(8));  // word size tag
    write_val<uint8_t>(os, fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, use_l3_ ? 1 : 0);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, l2_count_);

    if (use_l3_) {
        write_val<uint64_t>(os, l3_count_);
        uint64_t l3_bytes = l3_bits_.size();
        write_val<uint64_t>(os, l3_bytes);
        if (l3_bytes > 0)
            os.write(reinterpret_cast<const char*>(l3_bits_.data()), l3_bytes);

        write_val<uint64_t>(os, l2_literal_count_);
        if (l2_literal_count_ > 0)
            os.write(reinterpret_cast<const char*>(l2_literals_.data()), l2_literal_count_);
    } else {
        uint64_t l2f_size = l2_flat_.size();
        write_val<uint64_t>(os, l2f_size);
        if (l2f_size > 0)
            os.write(reinterpret_cast<const char*>(l2_flat_.data()), l2f_size);
    }

    write_val<uint64_t>(os, l1_literal_count_);
    if (l1_literal_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_literals_.data()), l1_literal_count_);
}

ComBitBtv ComBitBtv::deserialize(std::istream& is) {
    uint8_t ws = read_val<uint8_t>(is);
    if (ws != 8)
        throw std::runtime_error("ComBitBtv::deserialize: expected word size 8, got " +
                                 std::to_string(ws));
    uint8_t fo = read_val<uint8_t>(is);
    uint8_t ul3 = read_val<uint8_t>(is);

    ComBitBtv btv(fo != 0);
    btv.use_l3_ = (ul3 != 0);
    btv.bit_count_ = read_val<uint64_t>(is);
    btv.l2_count_ = read_val<uint64_t>(is);

    if (btv.use_l3_) {
        btv.l3_count_ = read_val<uint64_t>(is);
        uint64_t l3_bytes = read_val<uint64_t>(is);
        btv.l3_bits_.resize(l3_bytes);
        if (l3_bytes > 0)
            is.read(reinterpret_cast<char*>(btv.l3_bits_.data()), l3_bytes);

        btv.l2_literal_count_ = read_val<uint64_t>(is);
        btv.l2_literals_.resize(btv.l2_literal_count_);
        if (btv.l2_literal_count_ > 0)
            is.read(reinterpret_cast<char*>(btv.l2_literals_.data()), btv.l2_literal_count_);
    } else {
        uint64_t l2f_size = read_val<uint64_t>(is);
        btv.l2_flat_.resize(l2f_size);
        if (l2f_size > 0)
            is.read(reinterpret_cast<char*>(btv.l2_flat_.data()), l2f_size);
    }

    btv.l1_literal_count_ = read_val<uint64_t>(is);
    btv.l1_literals_.resize(btv.l1_literal_count_);
    if (btv.l1_literal_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_literals_.data()), btv.l1_literal_count_);

    return btv;
}

// ----------------------------------------------------------------
// ComBit::serialize / deserialize
// ----------------------------------------------------------------

void ComBit::serialize(std::ostream& os) const {
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());

    for (const auto& seg : segments_)
        seg.serialize(os);
}

ComBit ComBit::deserialize(std::istream& is) {
    ComBit cb;
    cb.bit_count_ = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);

    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(ComBitBtv::deserialize(is));

    return cb;
}
