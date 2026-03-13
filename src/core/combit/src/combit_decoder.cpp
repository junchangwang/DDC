#include <cassert>
#include <combit_decoder.hpp>

namespace combit {

std::vector<Word> ComBitDecoder::decode_words(const ComBitEncoding& enc) {
    assert(enc.header.size() == enc.content.size());

    std::vector<Word> result;

    for (size_t i = 0; i < enc.content.size(); ++i) {
        if (enc.header[i]) {
            // Compressed (fill) word
            uint8_t bit = fill_bit(enc.content[i]);
            Word    len = fill_length(enc.content[i]);
            Word fill_word = (bit == 1) ? LITERAL_ALL_ONE : LITERAL_ALL_ZERO;
            for (Word j = 0; j < len; ++j) {
                result.push_back(fill_word);
            }
        } else {
            // Literal word
            result.push_back(enc.content[i]);
        }
    }

    return result;
}

std::vector<bool> ComBitDecoder::decode(const ComBitEncoding& enc) {
    std::vector<Word> words = decode_words(enc);
    std::vector<bool> bits;
    bits.reserve(words.size() * WORD_SIZE);

    for (Word w : words) {
        for (size_t bi = 0; bi < WORD_SIZE; ++bi) {
            bits.push_back((w >> (WORD_LEFTMOST - bi)) & 1);
        }
    }

    return bits;
}

} // namespace combit