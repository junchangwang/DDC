#ifndef COMBIT_DECODER_HPP
#define COMBIT_DECODER_HPP

#include <vector>
#include <types.hpp>

namespace combit {

/**
 * ComBit Decoder
 *
 * Decompresses an ComBitEncoding back into an uncompressed bitmap vector.
 */
class ComBitDecoder {
public:
    /**
     * Decode an ComBit encoding back to an uncompressed word vector.
     *
     * @param enc  The ComBit-compressed encoding.
     * @return     A vector of uncompressed Words.
     */
    static std::vector<Word> decode_words(const ComBitEncoding& enc);

    /**
     * Decode an ComBit encoding back to a flat bit vector.
     *
     * @param enc  The ComBit-compressed encoding.
     * @return     A vector of bools representing each bit (MSB-first per word).
     */
    static std::vector<bool> decode(const ComBitEncoding& enc);
};

} // namespace combit

#endif // COMBIT_DECODER_HPP