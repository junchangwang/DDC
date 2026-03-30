#include "combit.h"

#include <stdexcept>

// ====================================================================
// ComBitBtv<WordSize> member function definitions
// ====================================================================

// ----------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------

template<unsigned WordSize>
ComBitBtv<WordSize>::ComBitBtv(bool fill_ones)
    : fill_ones_(fill_ones),
      leading_bits_count_(0), literal_count_(0), bit_count_(0) {}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

template<unsigned WordSize>
void ComBitBtv<WordSize>::push_literal(uint64_t val) {
    const size_t old_size = literal_data_.size();
    literal_data_.resize(old_size + word_byte_size);
    std::memcpy(literal_data_.data() + old_size, &val, word_byte_size);
    literal_count_++;
}

template<unsigned WordSize>
void ComBitBtv<WordSize>::set_literal(size_t idx, uint64_t val) {
    std::memcpy(literal_data_.data() + idx * word_byte_size, &val, word_byte_size);
}

template<unsigned WordSize>
uint64_t ComBitBtv<WordSize>::get_literal(size_t idx) const {
    uint64_t val = 0;
    std::memcpy(&val, literal_data_.data() + idx * word_byte_size, word_byte_size);
    return val;
}

template<unsigned WordSize>
uint64_t ComBitBtv<WordSize>::read_word_from_bits(const std::vector<bool>& bits,
                                                size_t word_idx) {
    uint64_t word = 0;
    size_t start = word_idx * WordSize;
    for (unsigned i = 0; i < WordSize && start + i < bits.size(); i++) {
        if (bits[start + i])
            word |= uint64_t(1) << (WordSize - 1 - i);
    }
    return word;
}

template<unsigned WordSize>
void ComBitBtv<WordSize>::append_word_to_bits(std::vector<bool>& bits,
                                            uint64_t word) {
    for (int i = static_cast<int>(WordSize) - 1; i >= 0; i--)
        bits.push_back((word >> i) & 1);
}

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

template<unsigned WordSize>
ComBitBtv<WordSize>
ComBitBtv<WordSize>::compress(const std::vector<bool>& bits, bool fill_ones) {
    ComBitBtv result(fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + WordSize - 1) / WordSize;
    if (num_words == 0) return result;

    result.leading_bits_count_ = num_words;
    result.leading_bits_.assign((num_words + 63) / 64, 0);
    for (size_t i = 0; i < num_words; i++) {
        uint64_t word = read_word_from_bits(bits, i);
        if (word == 0) {
            result.set_fill_bit(i);
        } else {
            result.push_literal(word);
        }
    }

    return result;
}

// ----------------------------------------------------------------
// Decompression
// ----------------------------------------------------------------

template<unsigned WordSize>
std::vector<bool>
ComBitBtv<WordSize>::decompress() const {
    std::vector<bool> result;
    result.reserve(bit_count_);

    size_t lit_idx = 0;
    for (size_t i = 0; i < leading_bits_count_; i++) {
        if (is_fill_bit(i)) {
            for (unsigned b = 0; b < WordSize; b++)
                result.push_back(false);
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

template<unsigned WordSize>
ComBitBtv<WordSize>
ComBitBtv<WordSize>::from_string(const std::string& bitstring, bool fill_ones) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, fill_ones);
}

template<unsigned WordSize>
std::string
ComBitBtv<WordSize>::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size() + bits.size() / WordSize);
    for (size_t i = 0; i < bits.size(); i++) {
        if (i > 0 && i % WordSize == 0) s += ' ';
        s += bits[i] ? '1' : '0';
    }
    return s;
}

// ----------------------------------------------------------------
// operator~
// ----------------------------------------------------------------

template<unsigned WordSize>
ComBitBtv<WordSize>
ComBitBtv<WordSize>::operator~() const {
    auto bits = decompress();
    for (size_t i = 0; i < bits.size(); i++)
        bits[i] = !bits[i];
    return compress(bits, fill_ones_);
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

template<unsigned WordSize>
size_t
ComBitBtv<WordSize>::popcount() const {
    size_t count = 0;
    size_t bits_seen = 0;
    size_t lit_idx = 0;

    for (size_t i = 0; i < leading_bits_count_; i++) {
        if (is_fill_bit(i)) {
            bits_seen += WordSize;
        } else {
            uint64_t w = get_literal(lit_idx++);
            size_t remaining = (bits_seen < bit_count_)
                ? bit_count_ - bits_seen : 0;
            if (remaining >= WordSize) {
                count += __builtin_popcountll(w);
            } else {
                for (size_t b = 0; b < remaining; b++) {
                    if (w & (uint64_t(1) << (WordSize - 1 - b)))
                        count++;
                }
            }
            bits_seen += WordSize;
        }
    }
    return count;
}

template<unsigned WordSize>
std::vector<size_t>
ComBitBtv<WordSize>::set_bit_positions() const {
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

template<unsigned WordSize>
typename ComBitBtv<WordSize>::SizeBreakdown
ComBitBtv<WordSize>::size_breakdown() const {
    SizeBreakdown sb;
    sb.leading_bits_count = leading_bits_count_;
    sb.literal_bits       = literal_count_ * WordSize;
    sb.total_bits         = sb.leading_bits_count + sb.literal_bits;
    return sb;
}

template<unsigned WordSize>
double
ComBitBtv<WordSize>::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

// ----------------------------------------------------------------
// num_fills
// ----------------------------------------------------------------

template<unsigned WordSize>
size_t
ComBitBtv<WordSize>::num_fills() const {
    size_t n = 0;
    for (size_t w = 0; w < leading_bits_.size(); w++)
        n += __builtin_popcountll(leading_bits_[w]);
    return n;
}

// ----------------------------------------------------------------
// Debug printing
// ----------------------------------------------------------------

template<unsigned WordSize>
void
ComBitBtv<WordSize>::print(std::ostream& os) const {
    os << "ComBitBtv<" << WordSize << "> compressed bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";

    os << "  Leading bits: ";
    for (size_t i = 0; i < leading_bits_count_; i++)
        os << (is_fill_bit(i) ? '1' : '0');
    os << " (" << leading_bits_count_ << " entries)\n";

    os << "  Literal words: [";
    for (size_t i = 0; i < literal_count_; i++) {
        if (i > 0) os << ", ";
        uint64_t val = get_literal(i);
        for (int b = static_cast<int>(WordSize) - 1; b >= 0; b--)
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
// Explicit template instantiations for ComBitBtv
// ====================================================================

template class ComBitBtv<8>;
template class ComBitBtv<16>;
template class ComBitBtv<32>;
template class ComBitBtv<64>;

// ====================================================================
// ComBit (segmented) member function definitions
// ====================================================================

// ----------------------------------------------------------------
// Compression
// ----------------------------------------------------------------

template<unsigned WordSize>
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
            ComBitBtv<WordSize>::compress(seg_bits, fill_ones));
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
        auto seg_bits = std::visit(
            [](const auto& s) { return s.decompress(); }, seg);
        result.insert(result.end(), seg_bits.begin(), seg_bits.end());
    }

    return result;
}

// ----------------------------------------------------------------
// Convenience constructors
// ----------------------------------------------------------------

template<unsigned WordSize>
ComBit
ComBit::from_string(const std::string& bitstring, bool fill_ones,
                    size_t segment_bits) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress<WordSize>(bits, fill_ones, segment_bits);
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

    for (const auto& seg : segments_) {
        auto rseg = std::visit(
            [](const auto& s) -> ComBitBtvSegment { return ~s; }, seg);
        result.segments_.push_back(std::move(rseg));
    }

    return result;
}

// ----------------------------------------------------------------
// Queries
// ----------------------------------------------------------------

size_t
ComBit::popcount() const {
    size_t count = 0;
    for (const auto& seg : segments_)
        count += std::visit(
            [](const auto& s) { return s.popcount(); }, seg);
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
        std::visit([&sb](const auto& s) {
            auto ssb = s.size_breakdown();
            sb.leading_bits_count += ssb.leading_bits_count;
            sb.literal_bits += ssb.literal_bits;
            sb.total_bits += ssb.total_bits;
        }, seg);
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
        os << "  Segment " << i << ": ";
        std::visit([&os](const auto& s) {
            os << "ComBitBtv<" << s.word_size << ">"
               << " fill_ones=" << s.fill_ones()
               << " bits=" << s.bit_count()
               << " fills=" << s.num_fills()
               << " literals=" << s.num_literals();
        }, segments_[i]);
        os << "\n";
    }

    auto sb = size_breakdown();
    os << "  Size breakdown: meta=" << sb.leading_bits_count
       << "  literal=" << sb.literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

// ====================================================================
// Explicit template instantiations for ComBit::compress / from_string
// ====================================================================

template ComBit ComBit::compress<8>(const std::vector<bool>&, bool, size_t);
template ComBit ComBit::compress<16>(const std::vector<bool>&, bool, size_t);
template ComBit ComBit::compress<32>(const std::vector<bool>&, bool, size_t);
template ComBit ComBit::compress<64>(const std::vector<bool>&, bool, size_t);

template ComBit ComBit::from_string<8>(const std::string&, bool, size_t);
template ComBit ComBit::from_string<16>(const std::string&, bool, size_t);
template ComBit ComBit::from_string<32>(const std::string&, bool, size_t);
template ComBit ComBit::from_string<64>(const std::string&, bool, size_t);

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
// ComBitBtv<WS>::serialize / deserialize
// ----------------------------------------------------------------

template<unsigned WS>
void ComBitBtv<WS>::serialize(std::ostream& os) const {
    write_val<uint8_t>(os, static_cast<uint8_t>(WS));
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

template<unsigned WS>
ComBitBtv<WS> ComBitBtv<WS>::deserialize(std::istream& is) {
    // Note: the word_size tag byte must already be consumed by the caller
    uint8_t fo = read_val<uint8_t>(is);

    ComBitBtv<WS> btv(fo != 0);
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

// Explicit instantiations for serialize/deserialize
template void ComBitBtv<8>::serialize(std::ostream&) const;
template void ComBitBtv<16>::serialize(std::ostream&) const;
template void ComBitBtv<32>::serialize(std::ostream&) const;
template void ComBitBtv<64>::serialize(std::ostream&) const;

template ComBitBtv<8>  ComBitBtv<8>::deserialize(std::istream&);
template ComBitBtv<16> ComBitBtv<16>::deserialize(std::istream&);
template ComBitBtv<32> ComBitBtv<32>::deserialize(std::istream&);
template ComBitBtv<64> ComBitBtv<64>::deserialize(std::istream&);

// ----------------------------------------------------------------
// ComBit::serialize / deserialize
// ----------------------------------------------------------------

void ComBit::serialize(std::ostream& os) const {
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());

    for (const auto& seg : segments_) {
        std::visit([&os](const auto& btv) { btv.serialize(os); }, seg);
    }
}

ComBit ComBit::deserialize(std::istream& is) {
    ComBit cb;
    cb.bit_count_ = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);

    for (uint64_t i = 0; i < num_segs; i++) {
        // Read word_size tag byte, then dispatch to correct deserializer
        uint8_t ws = read_val<uint8_t>(is);

        switch (ws) {
            case 8:  cb.segments_.push_back(ComBitBtv<8>::deserialize(is));  break;
            case 16: cb.segments_.push_back(ComBitBtv<16>::deserialize(is)); break;
            case 32: cb.segments_.push_back(ComBitBtv<32>::deserialize(is)); break;
            case 64: cb.segments_.push_back(ComBitBtv<64>::deserialize(is)); break;
            default:
                throw std::runtime_error("ComBit::deserialize: invalid word size " +
                                         std::to_string(ws));
        }
    }
    return cb;
}
