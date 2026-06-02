#ifndef DDC_DECODER_HPP
#define DDC_DECODER_HPP

#include <vector>
#include <types.hpp>

namespace ddc {

/**
 * DDC Decoder
 *
 * Decompresses an DDCEncoding back into an uncompressed bitmap vector.
 */
class DDCDecoder {
public:
    /**
     * Decode an DDC encoding back to an uncompressed word vector.
     *
     * @param enc  The DDC-compressed encoding.
     * @return     A vector of uncompressed Words.
     */
    static std::vector<Word> decode_words(const DDCEncoding& enc);

    /**
     * Decode an DDC encoding back to a flat bit vector.
     *
     * @param enc  The DDC-compressed encoding.
     * @return     A vector of bools representing each bit (MSB-first per word).
     */
    static std::vector<bool> decode(const DDCEncoding& enc);
};

} // namespace ddc

#endif // DDC_DECODER_HPP