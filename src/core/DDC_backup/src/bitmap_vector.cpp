#include <algorithm>
#include <bitmap_vector.hpp>
#include <simd_util.hpp>

namespace ddc {

void BitmapVector::ensure_capacity(uint64_t pos) {
    size_t word_index = static_cast<size_t>(pos / WORD_SIZE);
    if (word_index >= words_.size()) {
        words_.resize(word_index + 1, LITERAL_ALL_ZERO);
    }
}

void BitmapVector::set_bit(uint64_t position) {
    ensure_capacity(position);
    size_t word_index = static_cast<size_t>(position / WORD_SIZE);
    size_t bit_offset = static_cast<size_t>(position % WORD_SIZE);
    // MSB-first: bit 0 of the word is the leftmost (highest) bit
    words_[word_index] |= Word(1) << (WORD_LEFTMOST - bit_offset);
}

bool BitmapVector::get_bit(uint64_t position) const {
    size_t word_index = static_cast<size_t>(position / WORD_SIZE);
    if (word_index >= words_.size()) {
        return false;
    }
    size_t bit_offset = static_cast<size_t>(position % WORD_SIZE);
    return (words_[word_index] >> (WORD_LEFTMOST - bit_offset)) & 1;
}

uint64_t BitmapVector::size() const {
    return static_cast<uint64_t>(words_.size()) * WORD_SIZE;
}

uint64_t BitmapVector::popcount() const {
    if (words_.empty()) return 0;
    return simd::words_popcount(words_.data(), words_.size());
}

DDCEncoding BitmapVector::encode() const {
    return DDCEncoder::encode_words(words_);
}

void BitmapVector::load(const DDCEncoding& enc) {
    words_ = DDCDecoder::decode_words(enc);
}

// -----------------------------------------------------------------
//  Word-level bitwise operations (SIMD-accelerated)
// -----------------------------------------------------------------

BitmapVector BitmapVector::word_or(const BitmapVector& a, const BitmapVector& b) {
    BitmapVector result;
    size_t na = a.words_.size();
    size_t nb = b.words_.size();
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.words_.resize(max_n, LITERAL_ALL_ZERO);

    if (min_n > 0) {
        simd::words_or(a.words_.data(), b.words_.data(),
                       result.words_.data(), min_n);
    }
    // Copy the tail from the longer vector
    if (na > nb) {
        std::copy(a.words_.begin() + min_n, a.words_.end(),
                  result.words_.begin() + min_n);
    } else if (nb > na) {
        std::copy(b.words_.begin() + min_n, b.words_.end(),
                  result.words_.begin() + min_n);
    }
    return result;
}

BitmapVector BitmapVector::word_and(const BitmapVector& a, const BitmapVector& b) {
    BitmapVector result;
    size_t min_n = std::min(a.words_.size(), b.words_.size());
    if (min_n == 0) return result;
    result.words_.resize(min_n);
    simd::words_and(a.words_.data(), b.words_.data(),
                    result.words_.data(), min_n);
    return result;
}

BitmapVector BitmapVector::word_xor(const BitmapVector& a, const BitmapVector& b) {
    BitmapVector result;
    size_t na = a.words_.size();
    size_t nb = b.words_.size();
    size_t max_n = std::max(na, nb);
    size_t min_n = std::min(na, nb);
    result.words_.resize(max_n, LITERAL_ALL_ZERO);

    if (min_n > 0) {
        simd::words_xor(a.words_.data(), b.words_.data(),
                        result.words_.data(), min_n);
    }
    // XOR with 0 = identity, copy the longer tail
    if (na > nb) {
        std::copy(a.words_.begin() + min_n, a.words_.end(),
                  result.words_.begin() + min_n);
    } else if (nb > na) {
        std::copy(b.words_.begin() + min_n, b.words_.end(),
                  result.words_.begin() + min_n);
    }
    return result;
}

BitmapVector BitmapVector::word_andnot(const BitmapVector& a, const BitmapVector& b) {
    BitmapVector result;
    size_t na = a.words_.size();
    size_t nb = b.words_.size();
    if (na == 0) return result;

    result.words_.resize(na, LITERAL_ALL_ZERO);
    size_t min_n = std::min(na, nb);
    if (min_n > 0) {
        simd::words_andnot(a.words_.data(), b.words_.data(),
                           result.words_.data(), min_n);
    }
    // a & ~0 = a  for tail positions where b has no words
    if (na > nb) {
        std::copy(a.words_.begin() + min_n, a.words_.end(),
                  result.words_.begin() + min_n);
    }
    return result;
}

std::vector<uint32_t> BitmapVector::decode_positions() const {
    std::vector<uint32_t> out;
    if (!words_.empty()) {
        simd::words_decode_positions(words_.data(), words_.size(), 0, out);
    }
    return out;
}

} // namespace ddc