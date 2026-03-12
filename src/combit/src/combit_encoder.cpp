#include <cassert>
#include <combit/combit_encoder.hpp>

namespace combit {

/*
 * Flush a pending run of all-zero or all-one words.
 *
 * A single fill word can encode at most MAX_FILL_LENGTH words.
 * If the run is longer we emit multiple fill words.
 *
 * A run of length 1 that is all-zero or all-one is still stored as a
 * compressed fill word (length = 1) to keep the encoding canonical.
 * This matches the PostgreSQL reference behaviour.
 */
void ComBitEncoder::flush_run(ComBitEncoding& enc, uint8_t fill_bit_val, size_t run_length) {
    while (run_length > 0) {
        Word len = static_cast<Word>(
            run_length > MAX_FILL_LENGTH ? MAX_FILL_LENGTH : run_length);
        enc.header.push_back(true);   // 1 → compressed word
        enc.content.push_back(make_fill_word(fill_bit_val, len));
        run_length -= len;
    }
}

ComBitEncoding ComBitEncoder::encode_words(const std::vector<Word>& words) {
    ComBitEncoding enc;

    if (words.empty()) {
        return enc;
    }

    // State for the current run being accumulated
    enum class RunKind { NONE, ZEROS, ONES };
    RunKind  cur_kind   = RunKind::NONE;
    size_t   cur_length = 0;

    for (size_t i = 0; i < words.size(); ++i) {
        Word w = words[i];

        if (w == LITERAL_ALL_ZERO) {
            if (cur_kind == RunKind::ZEROS) {
                ++cur_length;
            } else {
                // Flush previous run if any
                if (cur_kind == RunKind::ONES) {
                    flush_run(enc, 1, cur_length);
                }
                cur_kind   = RunKind::ZEROS;
                cur_length = 1;
            }
        } else if (w == LITERAL_ALL_ONE) {
            if (cur_kind == RunKind::ONES) {
                ++cur_length;
            } else {
                if (cur_kind == RunKind::ZEROS) {
                    flush_run(enc, 0, cur_length);
                }
                cur_kind   = RunKind::ONES;
                cur_length = 1;
            }
        } else {
            // Mixed (literal) word — flush any pending run first
            if (cur_kind == RunKind::ZEROS) {
                flush_run(enc, 0, cur_length);
            } else if (cur_kind == RunKind::ONES) {
                flush_run(enc, 1, cur_length);
            }
            cur_kind   = RunKind::NONE;
            cur_length = 0;

            // Emit the literal word
            enc.header.push_back(false);  // 0 → literal (uncompressed)
            enc.content.push_back(w);
        }
    }

    // Flush the last pending run
    if (cur_kind == RunKind::ZEROS) {
        flush_run(enc, 0, cur_length);
    } else if (cur_kind == RunKind::ONES) {
        flush_run(enc, 1, cur_length);
    }

    assert(enc.header.size() == enc.content.size());
    return enc;
}

ComBitEncoding ComBitEncoder::encode(const std::vector<bool>& bits) {
    // Convert the bit vector into words (MSB-first within each word,
    // zero-padding the last word if necessary).
    std::vector<Word> words;
    size_t num_words = (bits.size() + WORD_SIZE - 1) / WORD_SIZE;
    words.reserve(num_words);

    for (size_t wi = 0; wi < num_words; ++wi) {
        Word w = 0;
        for (size_t bi = 0; bi < WORD_SIZE; ++bi) {
            size_t idx = wi * WORD_SIZE + bi;
            if (idx < bits.size() && bits[idx]) {
                w |= Word(1) << (WORD_LEFTMOST - bi);
            }
        }
        words.push_back(w);
    }

    return encode_words(words);
}

} // namespace combit