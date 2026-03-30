#include "combit.h"

// ----------------------------------------------------------------
// Bitwise XOR operations
// ----------------------------------------------------------------

/// AVX-512 expand-load dispatch by word size (compile-time).
#ifdef __AVX512VBMI2__
template<unsigned WS>
static inline void avx512_expand_load(__m512i fill,
                                       const uint8_t* data, size_t lit_off,
                                       uint64_t raw_mask,
                                       __m512i& out, size_t& consumed) {
    constexpr size_t wbs = WS / 8;
    if constexpr (WS == 8) {
        auto m = static_cast<__mmask64>(raw_mask);
        out = _mm512_mask_expandloadu_epi8(fill, m, data + lit_off * wbs);
        consumed = __builtin_popcountll(static_cast<uint64_t>(m));
    } else if constexpr (WS == 16) {
        auto m = static_cast<__mmask32>(raw_mask);
        out = _mm512_mask_expandloadu_epi16(fill, m, data + lit_off * wbs);
        consumed = __builtin_popcountll(static_cast<uint64_t>(m));
    } else if constexpr (WS == 32) {
        auto m = static_cast<__mmask16>(raw_mask);
        out = _mm512_mask_expandloadu_epi32(fill, m, data + lit_off * wbs);
        consumed = __builtin_popcountll(static_cast<uint64_t>(m));
    } else {
        static_assert(WS == 64);
        auto m = static_cast<__mmask8>(raw_mask);
        out = _mm512_mask_expandloadu_epi64(fill, m, data + lit_off * wbs);
        consumed = __builtin_popcountll(static_cast<uint64_t>(m));
    }
}
#endif

// ----------------------------------------------------------------
// Same-word-size XOR operator (ComBitBtv)
// ----------------------------------------------------------------

/// Same-word-size XOR operator.  Returns ComBitBtv<64>.
template<unsigned WordSize>
ComBitBtv<64>
ComBitBtv<WordSize>::operator^(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return ComBitBtv<64>(false);

    constexpr size_t wbs = word_byte_size;
    const size_t total_words = leading_bits_count_;
    const size_t total_words_r = (bit_count_ + 63) / 64;
    const size_t total_512regions = bit_count_ / 512;

    ComBitBtv<64> result(false);
    result.bit_count_ = bit_count_;
    result.leading_bits_count_ = total_words_r;
    result.leading_bits_.assign((total_words_r + 63) / 64, 0);
    result.literal_data_.resize(total_words_r * 8);
    result.literal_count_ = total_words_r;

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
        uint64_t raw_a = ~(leading_bits_[word_pos / 64] >> (word_pos % 64));
        uint64_t raw_b = ~(other.leading_bits_[word_pos / 64] >> (word_pos % 64));

        const __m512i fill_a = fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();
        const __m512i fill_b = other.fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();

        __m512i va, vb;
        size_t a_consumed, b_consumed;
        avx512_expand_load<WordSize>(fill_a, a_lit, a_lit_off, raw_a, va, a_consumed);
        avx512_expand_load<WordSize>(fill_b, b_lit, b_lit_off, raw_b, vb, b_consumed);

        a_lit_off += a_consumed;
        b_lit_off += b_consumed;

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(r_ptr + r_byte_off),
                            _mm512_xor_si512(va, vb));

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
                std::memcpy(buf_a + i * wbs, a_lit + a_lit_off * wbs, wbs);
                a_lit_off++;
            }
        }
        for (size_t i = 0; i < remaining; i++) {
            if (!other.is_fill_bit(word_pos + i)) {
                std::memcpy(buf_b + i * wbs, b_lit + b_lit_off * wbs, wbs);
                b_lit_off++;
            }
        }

        const size_t tail_bytes = remaining * wbs;
        for (size_t i = 0; i < tail_bytes; i++)
            buf_a[i] ^= buf_b[i];
        std::memcpy(r_ptr + r_byte_off, buf_a, tail_bytes);
    }

#ifdef COMBIT_DEBUG
    auto t2 = clock::now();
    auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::cout << "  [XOR<" << WordSize << ">] "
              << "expand_xor: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    return result;
}

// ----------------------------------------------------------------
// Cross-word-size XOR (ComBitBtv)
// ----------------------------------------------------------------

template<unsigned A, unsigned B>
ComBitBtv<64> cross_xor(const ComBitBtv<A>& a, const ComBitBtv<B>& b) {
    assert(a.bit_count() == b.bit_count());
    const size_t bit_count = a.bit_count();
    if (bit_count == 0) return ComBitBtv<64>(false);

    if constexpr (A == B) {
        return a ^ b;
    }

    // Different word sizes: cross-WS expand-load logic.
    const size_t total_words_r = (bit_count + 63) / 64;

    ComBitBtv<64> result(false);
    result.bit_count_ = bit_count;
    result.leading_bits_count_ = total_words_r;
    result.leading_bits_.assign((total_words_r + 63) / 64, 0);
    result.literal_data_.resize(total_words_r * 8);
    result.literal_count_ = total_words_r;

    constexpr size_t wbsa = A / 8;
    constexpr size_t wbsb = B / 8;
    constexpr size_t wbsr = 8;
    constexpr size_t words_per_reg_a = 512 / A;
    constexpr size_t words_per_reg_b = 512 / B;
    constexpr size_t words_per_reg_r = 8;

    const size_t total_words_a = (bit_count + A - 1) / A;
    const size_t total_words_b = (bit_count + B - 1) / B;
    const size_t total_512regions = bit_count / 512;

    const uint8_t* a_lit = a.literal_data();
    const uint8_t* b_lit = b.literal_data();
    uint8_t* r_ptr = result.literal_data_.data();

    size_t a_word_pos = 0, b_word_pos = 0, r_word_pos = 0;
    size_t a_lit_off = 0, b_lit_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    for (size_t region = 0; region < total_512regions; region++) {
        uint64_t raw_a = ~(a.leading_bits()[a_word_pos / 64] >> (a_word_pos % 64));
        uint64_t raw_b = ~(b.leading_bits()[b_word_pos / 64] >> (b_word_pos % 64));

        const __m512i fill_a = a.fill_ones()
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();
        const __m512i fill_b = b.fill_ones()
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();

        __m512i va, vb;
        size_t a_consumed, b_consumed;
        avx512_expand_load<A>(fill_a, a_lit, a_lit_off, raw_a, va, a_consumed);
        avx512_expand_load<B>(fill_b, b_lit, b_lit_off, raw_b, vb, b_consumed);

        a_lit_off += a_consumed;
        b_lit_off += b_consumed;

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(r_ptr + r_word_pos * wbsr),
                            _mm512_xor_si512(va, vb));

        a_word_pos += words_per_reg_a;
        b_word_pos += words_per_reg_b;
        r_word_pos += words_per_reg_r;
    }
#endif

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // Scalar tail.
    {
        alignas(64) uint8_t buf_a[64], buf_b[64];
        std::memset(buf_a, a.fill_ones() ? 0xFF : 0x00, 64);
        std::memset(buf_b, b.fill_ones() ? 0xFF : 0x00, 64);

        const size_t remaining_a = total_words_a - a_word_pos;
        const size_t remaining_b = total_words_b - b_word_pos;
        const size_t remaining_r = total_words_r - r_word_pos;

        for (size_t i = 0; i < remaining_a; i++) {
            if (!a.is_fill(a_word_pos + i)) {
                std::memcpy(buf_a + i * wbsa, a_lit + a_lit_off * wbsa, wbsa);
                a_lit_off++;
            }
        }
        for (size_t i = 0; i < remaining_b; i++) {
            if (!b.is_fill(b_word_pos + i)) {
                std::memcpy(buf_b + i * wbsb, b_lit + b_lit_off * wbsb, wbsb);
                b_lit_off++;
            }
        }

        const size_t result_tail_bytes = remaining_r * wbsr;
        for (size_t i = 0; i < result_tail_bytes; i++)
            buf_a[i] ^= buf_b[i];
        std::memcpy(r_ptr + r_word_pos * wbsr, buf_a, result_tail_bytes);
    }

#ifdef COMBIT_DEBUG
    auto t2 = clock::now();
    auto us = [](auto aa, auto bb) {
        return std::chrono::duration<double, std::micro>(bb - aa).count();
    };
    std::cout << "  [cross_XOR<" << A << "," << B << ">->64] "
              << "expand_xor: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    return result;
}

// ----------------------------------------------------------------
// ComBit (segmented) XOR operator
// ----------------------------------------------------------------

ComBit
ComBit::operator^(const ComBit& other) const {
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    ComBit result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    for (size_t i = 0; i < segments_.size(); i++) {
        auto rseg = std::visit(
            [](const auto& a, const auto& b) -> ComBitBtvSegment {
                return cross_xor(a, b);
            }, segments_[i], other.segments_[i]);
        result.segments_.push_back(std::move(rseg));
    }

    return result;
}

// ====================================================================
// Explicit template instantiations
// ====================================================================

// cross_xor explicit instantiations for all WordSize combinations:
#define INSTANTIATE_CROSS_XOR(A, B) \
    template ComBitBtv<64> cross_xor<A, B>(const ComBitBtv<A>&, const ComBitBtv<B>&);

INSTANTIATE_CROSS_XOR(8, 8)
INSTANTIATE_CROSS_XOR(8, 16)
INSTANTIATE_CROSS_XOR(8, 32)
INSTANTIATE_CROSS_XOR(8, 64)
INSTANTIATE_CROSS_XOR(16, 8)
INSTANTIATE_CROSS_XOR(16, 16)
INSTANTIATE_CROSS_XOR(16, 32)
INSTANTIATE_CROSS_XOR(16, 64)
INSTANTIATE_CROSS_XOR(32, 8)
INSTANTIATE_CROSS_XOR(32, 16)
INSTANTIATE_CROSS_XOR(32, 32)
INSTANTIATE_CROSS_XOR(32, 64)
INSTANTIATE_CROSS_XOR(64, 8)
INSTANTIATE_CROSS_XOR(64, 16)
INSTANTIATE_CROSS_XOR(64, 32)
INSTANTIATE_CROSS_XOR(64, 64)

#undef INSTANTIATE_CROSS_XOR
