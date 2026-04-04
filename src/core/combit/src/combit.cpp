#include "combit.h"

#include <stdexcept>

// ====================================================================
// ComBitBtv member function definitions
// ====================================================================

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

ComBitBtv::ComBitBtv(bool fill_ones)
    : fill_ones_(fill_ones),
      leading_bits_count_(0), literal_count_(0), bit_count_(0) {}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

void ComBitBtv::push_literal(uint64_t val) {
    literal_data_.push_back(static_cast<uint8_t>(val));
    literal_count_++;
}

void ComBitBtv::set_literal(size_t idx, uint64_t val) {
    literal_data_[idx] = static_cast<uint8_t>(val);
}

uint64_t ComBitBtv::get_literal(size_t idx) const {
    return literal_data_[idx];
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

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::compress(const std::vector<bool>& bits, bool fill_ones) {
    ComBitBtv result(fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.leading_bits_count_ = num_words;
    result.leading_bits_.assign((num_words + 63) / 64, 0);
    const uint64_t fill_val = fill_ones ? uint64_t(0xFF) : uint64_t(0);
    for (size_t i = 0; i < num_words; i++) {
        uint64_t word = read_word_from_bits(bits, i);
        if (word != fill_val) {
            result.set_literal_bit(i);
            result.push_literal(word);
        }
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

    size_t lit_idx = 0;
    for (size_t i = 0; i < leading_bits_count_; i++) {
        if (is_fill_bit(i)) {
            for (unsigned b = 0; b < 8; b++)
                result.push_back(fill_ones_);
        } else {
            append_word_to_bits(result, get_literal(lit_idx++));
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
    size_t count = 0;
    size_t bits_seen = 0;
    size_t lit_idx = 0;

    for (size_t i = 0; i < leading_bits_count_; i++) {
        if (is_fill_bit(i)) {
            if (fill_ones_) {
                size_t remaining = (bits_seen < bit_count_)
                    ? bit_count_ - bits_seen : 0;
                count += std::min(size_t(8), remaining);
            }
            bits_seen += 8;
        } else {
            uint64_t w = get_literal(lit_idx++);
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
    sb.leading_bits_count = leading_bits_count_;
    sb.literal_bits       = literal_count_ * 8;
    sb.total_bits         = sb.leading_bits_count + sb.literal_bits;
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
    size_t n = 0;
    for (size_t w = 0; w < leading_bits_.size(); w++)
        n += __builtin_popcountll(leading_bits_[w]);
    return leading_bits_count_ - n;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

void
ComBitBtv::print(std::ostream& os) const {
    os << "ComBitBtv compressed bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";

    os << "  Leading bits: ";
    for (size_t i = 0; i < leading_bits_count_; i++)
        os << (is_fill_bit(i) ? '0' : '1');
    os << " (" << leading_bits_count_ << " entries)\n";

    os << "  Literal words: [";
    for (size_t i = 0; i < literal_count_; i++) {
        if (i > 0) os << ", ";
        uint64_t val = get_literal(i);
        for (int b = 7; b >= 0; b--)
            os << ((val >> b) & 1);
    }
    os << "]\n";

    auto sb = size_breakdown();
    os << "  Size breakdown: meta=" << sb.leading_bits_count
       << "  literal=" << sb.literal_bits
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
    SizeBreakdown sb{0, 0, 0};
    for (const auto& seg : segments_) {
        auto ssb = seg.size_breakdown();
        sb.leading_bits_count += ssb.leading_bits_count;
        sb.literal_bits += ssb.literal_bits;
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
           << " fills=" << s.num_fills()
           << " literals=" << s.num_literals()
           << "\n";
    }

    auto sb = size_breakdown();
    os << "  Size breakdown: meta=" << sb.leading_bits_count
       << "  literal=" << sb.literal_bits
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
    write_val<uint8_t>(os, static_cast<uint8_t>(8));
    write_val<uint8_t>(os, fill_ones_ ? 1 : 0);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, leading_bits_count_);

    uint64_t lb_vec_size = leading_bits_.size();
    write_val<uint64_t>(os, lb_vec_size);
    if (lb_vec_size > 0)
        os.write(reinterpret_cast<const char*>(leading_bits_.data()),
                 lb_vec_size * sizeof(uint64_t));

    write_val<uint64_t>(os, literal_count_);
    uint64_t lit_data_size = literal_data_.size();
    write_val<uint64_t>(os, lit_data_size);
    if (lit_data_size > 0)
        os.write(reinterpret_cast<const char*>(literal_data_.data()), lit_data_size);
}

ComBitBtv ComBitBtv::deserialize(std::istream& is) {
    uint8_t ws = read_val<uint8_t>(is);
    if (ws != 8)
        throw std::runtime_error("ComBitBtv::deserialize: expected word size 8, got " +
                                 std::to_string(ws));
    uint8_t fo = read_val<uint8_t>(is);

    ComBitBtv btv(fo != 0);
    btv.bit_count_ = read_val<uint64_t>(is);
    btv.leading_bits_count_ = read_val<uint64_t>(is);

    uint64_t lb_vec_size = read_val<uint64_t>(is);
    btv.leading_bits_.resize(lb_vec_size);
    if (lb_vec_size > 0)
        is.read(reinterpret_cast<char*>(btv.leading_bits_.data()),
                lb_vec_size * sizeof(uint64_t));

    btv.literal_count_ = read_val<uint64_t>(is);
    uint64_t lit_data_size = read_val<uint64_t>(is);
    btv.literal_data_.resize(lit_data_size);
    if (lit_data_size > 0)
        is.read(reinterpret_cast<char*>(btv.literal_data_.data()), lit_data_size);

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
