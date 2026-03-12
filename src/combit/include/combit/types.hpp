#ifndef COMBIT_TYPES_HPP
#define COMBIT_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace combit {

// Configurable word size (in bits). Default matches the reference: 16 bits.
using Word = uint16_t;

constexpr size_t WORD_SIZE = sizeof(Word) * 8;
constexpr size_t WORD_LEFTMOST = WORD_SIZE - 1;

// All zeros / all ones literal words
constexpr Word LITERAL_ALL_ZERO = 0;
constexpr Word LITERAL_ALL_ONE = static_cast<Word>(~Word(0));

// Mask to extract the fill length (everything except the MSB)
constexpr Word FILL_MASK = static_cast<Word>(~(Word(1) << WORD_LEFTMOST));

// Maximum run-length that can be stored in a single fill word
constexpr Word MAX_FILL_LENGTH = static_cast<Word>((Word(1) << WORD_LEFTMOST) - 1);

/// Construct a fill (compressed) word.
/// @param bit 0 or 1 — which bit value is being run-length encoded
/// @param length number of *words* that are compressed
inline constexpr Word make_fill_word(uint8_t bit, Word length) {
    return (static_cast<Word>(bit) << WORD_LEFTMOST) | length;
}

/// Extract the fill length (number of words compressed) from a compressed word.
inline constexpr Word fill_length(Word w) {
    return w & FILL_MASK;
}

/// Extract the fill bit (0 or 1) from a compressed word.
inline constexpr uint8_t fill_bit(Word w) {
    return static_cast<uint8_t>(w >> WORD_LEFTMOST);
}

/// The result of ComBit encoding: a header section and a content section.
struct ComBitEncoding {
    std::vector<bool> header;    // one bit per content word: 1=compressed, 0=literal
    std::vector<Word> content;   // the content words

    /// Total number of uncompressed bits represented by this encoding.
    size_t decompressed_bit_count() const;

    /// Pretty-print for debugging.
    std::string to_string() const;

    /// Equality comparison.
    bool operator==(const ComBitEncoding& other) const {
        return header == other.header && content == other.content;
    }
    bool operator!=(const ComBitEncoding& other) const {
        return !(*this == other);
    }
};

} // namespace combit

#endif // COMBIT_TYPES_HPP