#include <algorithm>
#include <combit/bitmap_vector.hpp>

namespace combit {

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
    uint64_t count = 0;
    for (Word w : words_) {
        // Use GCC/Clang built-in; falls back to manual for portability
        count += __builtin_popcount(static_cast<unsigned>(w));
    }
    return count;
}

ComBitEncoding BitmapVector::encode() const {
    return ComBitEncoder::encode_words(words_);
}

void BitmapVector::load(const ComBitEncoding& enc) {
    words_ = ComBitDecoder::decode_words(enc);
}

} // namespace combit