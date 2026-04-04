#include "combit.h"

// ----------------------------------------------------------------
// Bitwise AND operations
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv AND operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator&(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return ComBitBtv(false);

    const size_t total_words = leading_bits_count_;
    const size_t total_512regions = bit_count_ / 512;

    ComBitBtv result(false);
    result.bit_count_ = bit_count_;
    result.leading_bits_count_ = total_words;
    result.leading_bits_.assign((total_words + 63) / 64, ~uint64_t(0));
    result.literal_data_.resize(total_words);
    result.literal_count_ = total_words;

    const uint8_t* a_lit = literal_data_.data();
    const uint8_t* b_lit = other.literal_data_.data();
    uint8_t* r_ptr = result.literal_data_.data();

    size_t word_pos = 0;
    size_t a_lit_off = 0, b_lit_off = 0;
    size_t r_byte_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    for (size_t region = 0; region < total_512regions; region++) {
        uint64_t raw_a = (leading_bits_[word_pos / 64] >> (word_pos % 64));
        uint64_t raw_b = (other.leading_bits_[word_pos / 64] >> (word_pos % 64));

        const __m512i fill_a = fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();
        const __m512i fill_b = other.fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();

        auto ma = static_cast<__mmask64>(raw_a);
        auto mb = static_cast<__mmask64>(raw_b);

        __m512i va = _mm512_mask_expandloadu_epi8(fill_a, ma, a_lit + a_lit_off);
        __m512i vb = _mm512_mask_expandloadu_epi8(fill_b, mb, b_lit + b_lit_off);

        a_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));
        b_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(r_ptr + r_byte_off),
                            _mm512_and_si512(va, vb));

        word_pos += words_per_reg;
        r_byte_off += 64;
    }
#endif

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // Scalar tail.
    {
        alignas(64) uint8_t buf_a[64], buf_b[64];
        std::memset(buf_a, fill_ones_ ? 0xFF : 0x00, 64);
        std::memset(buf_b, other.fill_ones_ ? 0xFF : 0x00, 64);

        const size_t remaining = total_words - word_pos;

        for (size_t i = 0; i < remaining; i++) {
            if (!is_fill_bit(word_pos + i)) {
                buf_a[i] = a_lit[a_lit_off];
                a_lit_off++;
            }
        }
        for (size_t i = 0; i < remaining; i++) {
            if (!other.is_fill_bit(word_pos + i)) {
                buf_b[i] = b_lit[b_lit_off];
                b_lit_off++;
            }
        }

        for (size_t i = 0; i < remaining; i++)
            buf_a[i] &= buf_b[i];
        std::memcpy(r_ptr + r_byte_off, buf_a, remaining);
    }

#ifdef COMBIT_DEBUG
    auto t2 = clock::now();
    auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::cout << "  [AND] "
              << "expand_and: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    return result;
}

// ----------------------------------------------------------------
// ComBit (segmented) AND operator
// ----------------------------------------------------------------

ComBit
ComBit::operator&(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++)
        result.segments_.push_back(segments_[i] & other.segments_[i]);

    return result;
}
