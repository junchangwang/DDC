#include <bitset_vector.hpp>
#include <bitset_simd.hpp>
#include <fstream>
#include <algorithm>

namespace bitset {

void BitsetVector::ensure_capacity(uint64_t pos) {
    size_t word_index = pos / 64;
    if (word_index >= words_.size())
        words_.resize(word_index + 1, 0);
}

void BitsetVector::set_bit(uint64_t position) {
    ensure_capacity(position);
    size_t word_index = position / 64;
    size_t bit_offset = position % 64;
    words_[word_index] |= (uint64_t(1) << bit_offset);
    if (position >= num_bits_)
        num_bits_ = position + 1;
}

bool BitsetVector::get_bit(uint64_t position) const {
    size_t word_index = position / 64;
    if (word_index >= words_.size()) return false;
    size_t bit_offset = position % 64;
    return (words_[word_index] >> bit_offset) & 1;
}

uint64_t BitsetVector::popcount(bool use_simd) const {
    if (words_.empty()) return 0;
    if (use_simd)
        return simd::words_popcount_simd(words_.data(), words_.size());
    else
        return simd::words_popcount_scalar(words_.data(), words_.size());
}

// -----------------------------------------------------------------
//  Static word-level bitwise operations
// -----------------------------------------------------------------

BitsetVector BitsetVector::word_or(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_.size(), nb = b.words_.size();
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.words_.resize(max_n, 0);

    if (min_n > 0) {
        if (use_simd)
            simd::words_or_simd(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
        else
            simd::words_or_scalar(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
    }
    const auto& longer = (na > nb) ? a : b;
    for (size_t i = min_n; i < max_n; ++i)
        result.words_[i] = longer.words_[i];
    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

BitsetVector BitsetVector::word_and(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t min_n = std::min(a.words_.size(), b.words_.size());
    if (min_n == 0) return result;
    result.words_.resize(min_n);

    if (use_simd)
        simd::words_and_simd(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
    else
        simd::words_and_scalar(a.words_.data(), b.words_.data(), result.words_.data(), min_n);

    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

BitsetVector BitsetVector::word_xor(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_.size(), nb = b.words_.size();
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.words_.resize(max_n, 0);

    if (min_n > 0) {
        if (use_simd)
            simd::words_xor_simd(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
        else
            simd::words_xor_scalar(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
    }
    const auto& longer = (na > nb) ? a : b;
    for (size_t i = min_n; i < max_n; ++i)
        result.words_[i] = longer.words_[i];
    result.num_bits_ = std::max(a.num_bits_, b.num_bits_);
    return result;
}

BitsetVector BitsetVector::word_andnot(const BitsetVector& a, const BitsetVector& b, bool use_simd) {
    BitsetVector result;
    size_t na = a.words_.size(), nb = b.words_.size();
    if (na == 0) return result;

    result.words_.resize(na, 0);
    size_t min_n = std::min(na, nb);
    if (min_n > 0) {
        if (use_simd)
            simd::words_andnot_simd(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
        else
            simd::words_andnot_scalar(a.words_.data(), b.words_.data(), result.words_.data(), min_n);
    }
    if (na > nb) {
        for (size_t i = min_n; i < na; ++i)
            result.words_[i] = a.words_[i];
    }
    result.num_bits_ = a.num_bits_;
    return result;
}

std::vector<uint32_t> BitsetVector::decode_positions() const {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < words_.size(); ++i) {
        uint64_t w = words_[i];
        uint32_t base = static_cast<uint32_t>(i * 64);
        while (w) {
            int bit = __builtin_ctzll(w);
            uint32_t pos = base + bit;
            if (pos >= num_bits_) break;
            out.push_back(pos);
            w &= w - 1;
        }
    }
    return out;
}

void BitsetVector::serialize(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    size_t total_bytes = (num_bits_ + 7) / 8;
    out.write(reinterpret_cast<const char*>(words_.data()), total_bytes);
}

bool BitsetVector::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;

    std::streamsize file_size = in.tellg();
    in.seekg(0, std::ios::beg);

    num_bits_ = file_size * 8;
    size_t num_words = (file_size + 7) / 8;
    words_.resize(num_words, 0);
    in.read(reinterpret_cast<char*>(words_.data()), file_size);
    return true;
}

} // namespace bitset
