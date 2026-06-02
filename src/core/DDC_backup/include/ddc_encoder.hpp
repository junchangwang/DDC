#ifndef DDC_ENCODER_HPP
#define DDC_ENCODER_HPP

#include <vector>
#include <types.hpp>

namespace ddc {

/**
 * DDC Encoder
 *
 * Encodes an uncompressed bitmap vector into DDC (Compressed Bitvector) format.
 *
 * The encoding works by scanning the input bits word-by-word. Consecutive
 * words that are all-zero or all-one are merged into a single compressed
 * (fill) word. Mixed words are stored verbatim as literal (uncompressed)
 * words.
 *
 * The result is an DDCEncoding containing:
 *   - header:  one bit per content word (1 = compressed, 0 = literal)
 *   - content: the sequence of compressed and literal words
 */
class DDCEncoder {
public:
    /**
     * Encode an uncompressed bitmap vector.
     *
     * @param bits   The raw uncompressed bits (vector of bool, MSB-first
     *               within each conceptual word). The total length need
     *               not be a multiple of WORD_SIZE; trailing bits in the
     *               last word are treated as zero-padded.
     * @return       The DDC-compressed encoding.
     */
    static DDCEncoding encode(const std::vector<bool>& bits);

    /**
     * Encode from an already word-aligned vector of Words.
     *
     * @param words  The raw uncompressed words (each word is WORD_SIZE bits).
     * @return       The DDC-compressed encoding.
     */
    static DDCEncoding encode_words(const std::vector<Word>& words);

private:
    /// Flush a pending run of identical words into the encoding.
    /// Handles runs longer than MAX_FILL_LENGTH by emitting multiple
    /// fill words.
    static void flush_run(DDCEncoding& enc, uint8_t fill_bit_val, size_t run_length);
};

} // namespace ddc

#endif // DDC_ENCODER_HPP