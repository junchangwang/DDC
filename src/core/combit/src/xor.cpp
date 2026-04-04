#include "combit.h"

// ----------------------------------------------------------------
// Bitwise XOR operations (3-level: L3/L2/L1)
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv XOR operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator^(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return ComBitBtv(false);

    const size_t total_words = l2_count_;
    const size_t total_regions = total_words / words_per_reg;

    auto l2_a = expand_l2();
    auto l2_b = other.expand_l2();

    ComBitBtv result(false);
    result.bit_count_ = bit_count_;
    result.use_l3_ = false;
    result.l2_count_ = total_words;
    size_t l2_byte_count = (total_words + 7) / 8;
    result.l2_flat_.assign(l2_byte_count, 0xFF);
    if (total_words % 8 != 0)
        result.l2_flat_.back() = uint8_t((1u << (total_words % 8)) - 1);
    result.l1_literals_.resize(total_words);
    result.l1_literal_count_ = total_words;

    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    uint8_t* r_l1 = result.l1_literals_.data();

    size_t word_pos = 0;
    size_t a_l1_off = 0, b_l1_off = 0;
    size_t r_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    for (size_t region = 0; region < total_regions; region++) {
        size_t l2_byte_idx = word_pos / 8;

        uint64_t mask_a, mask_b;
        std::memcpy(&mask_a, l2_a.data() + l2_byte_idx, 8);
        std::memcpy(&mask_b, l2_b.data() + l2_byte_idx, 8);

        const __m512i fill_a = fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();
        const __m512i fill_b = other.fill_ones_
            ? _mm512_set1_epi8(static_cast<char>(-1))
            : _mm512_setzero_si512();

        auto ma = static_cast<__mmask64>(mask_a);
        auto mb = static_cast<__mmask64>(mask_b);

        __m512i va = _mm512_mask_expandloadu_epi8(fill_a, ma, a_l1 + a_l1_off);
        __m512i vb = _mm512_mask_expandloadu_epi8(fill_b, mb, b_l1 + b_l1_off);

        a_l1_off += __builtin_popcountll(mask_a);
        b_l1_off += __builtin_popcountll(mask_b);

        _mm512_storeu_si512(reinterpret_cast<__m512i*>(r_l1 + r_off),
                            _mm512_xor_si512(va, vb));

        word_pos += words_per_reg;
        r_off += 64;
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
            size_t wi = word_pos + i;
            uint8_t l2a_byte = l2_a[wi / 8];
            if ((l2a_byte >> (wi % 8)) & 1) {
                buf_a[i] = a_l1[a_l1_off++];
            }
        }
        for (size_t i = 0; i < remaining; i++) {
            size_t wi = word_pos + i;
            uint8_t l2b_byte = l2_b[wi / 8];
            if ((l2b_byte >> (wi % 8)) & 1) {
                buf_b[i] = b_l1[b_l1_off++];
            }
        }

        for (size_t i = 0; i < remaining; i++)
            buf_a[i] ^= buf_b[i];
        std::memcpy(r_l1 + r_off, buf_a, remaining);
    }

#ifdef COMBIT_DEBUG
    auto t2 = clock::now();
    auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::cout << "  [XOR] "
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

    for (size_t i = 0; i < segments_.size(); i++)
        result.segments_.push_back(segments_[i] ^ other.segments_[i]);

    return result;
}
