#include "combit.h"

// ----------------------------------------------------------------
// Bitwise AND operations (3-level: L3/L2/L1)
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// ComBitBtv AND operator
// ----------------------------------------------------------------

ComBitBtv
ComBitBtv::operator&(const ComBitBtv& other) const {
    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return ComBitBtv(false);

    const size_t total_words = l2_count_;

    // Result: all words become literals (we compute every word).
    ComBitBtv result(false);
    result.bit_count_ = bit_count_;
    result.use_l3_ = false;
    result.l2_count_ = total_words;
    size_t l2_byte_count = (total_words + 7) / 8;
    result.l2_flat_.assign(l2_byte_count, 0xFF);
    // Mask off trailing bits in last L2 byte.
    if (total_words % 8 != 0)
        result.l2_flat_.back() = uint8_t((1u << (total_words % 8)) - 1);
    result.l1_literals_.resize(total_words);
    result.l1_literal_count_ = total_words;

    const uint8_t* a_l1 = l1_literals_.data();
    const uint8_t* b_l1 = other.l1_literals_.data();
    uint8_t* r_l1 = result.l1_literals_.data();

    size_t a_l1_off = 0, b_l1_off = 0;
    size_t r_off = 0;

#ifdef COMBIT_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    // === AVX-512 main loop ===
    // Each region: 64 words = 512 bits = 8 L2 bytes = 1 L3 byte (8 L3 bits).
    // Read L3 byte → expand-load L2 literals → 64-bit mask → expand-load L1.
    const size_t avx_regions = total_words / words_per_reg;
    size_t a_l2_off = 0, b_l2_off = 0;

    const __m512i fill_a_vec = fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();
    const __m512i fill_b_vec = other.fill_ones_
        ? _mm512_set1_epi8(static_cast<char>(-1))
        : _mm512_setzero_si512();

    if (use_l3_) {
        for (size_t region = 0; region < avx_regions; region++) {
            uint8_t l3a = l3_bits_[region];
            __m512i l2a_v = _mm512_maskz_expandloadu_epi8(
                static_cast<__mmask64>(l3a), l2_literals_.data() + a_l2_off);
            a_l2_off += __builtin_popcount(l3a);
            __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

            uint8_t l3b = other.l3_bits_[region];
            __m512i l2b_v = _mm512_maskz_expandloadu_epi8(
                static_cast<__mmask64>(l3b), other.l2_literals_.data() + b_l2_off);
            b_l2_off += __builtin_popcount(l3b);
            __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

            __m512i va = _mm512_mask_expandloadu_epi8(fill_a_vec, ma, a_l1 + a_l1_off);
            __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb, b_l1 + b_l1_off);
            a_l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
            b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
            _mm512_storeu_si512(r_l1 + r_off, _mm512_and_si512(va, vb));
            r_off += 64;
        }
    } else {
        const uint64_t* l2a = reinterpret_cast<const uint64_t*>(l2_flat_.data());
        const uint64_t* l2b = reinterpret_cast<const uint64_t*>(other.l2_flat_.data());
        for (size_t region = 0; region < avx_regions; region++) {
            __mmask64 ma = static_cast<__mmask64>(l2a[region]);
            __mmask64 mb = static_cast<__mmask64>(l2b[region]);

            __m512i va = _mm512_mask_expandloadu_epi8(fill_a_vec, ma, a_l1 + a_l1_off);
            __m512i vb = _mm512_mask_expandloadu_epi8(fill_b_vec, mb, b_l1 + b_l1_off);
            a_l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
            b_l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
            _mm512_storeu_si512(r_l1 + r_off, _mm512_and_si512(va, vb));
            r_off += 64;
        }
    }

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // === AVX-512 scalar tail (< 64 words) — temporarily disabled ===
    // TODO: re-enable tail processing
    /*
    {
        const size_t remaining = total_words - avx_regions * words_per_reg;
        if (remaining > 0) {
            size_t tail_l2_bytes = (remaining + 7) / 8;

            // Reconstruct L2 bytes for tail from L3 + L2 literals.
            uint8_t l2a_tail[8] = {}, l2b_tail[8] = {};
            if (use_l3_ && avx_regions < l3_bits_.size()) {
                uint8_t l3a = l3_bits_[avx_regions];
                for (size_t b = 0; b < tail_l2_bytes; b++) {
                    if ((l3a >> b) & 1)
                        l2a_tail[b] = l2_literals_[a_l2_off++];
                }
            } else if (!use_l3_) {
                size_t base = avx_regions * 8;
                for (size_t j = 0; j < tail_l2_bytes; j++)
                    l2a_tail[j] = l2_flat_[base + j];
            }

            if (other.use_l3_ && avx_regions < other.l3_bits_.size()) {
                uint8_t l3b = other.l3_bits_[avx_regions];
                for (size_t b = 0; b < tail_l2_bytes; b++) {
                    if ((l3b >> b) & 1)
                        l2b_tail[b] = other.l2_literals_[b_l2_off++];
                }
            } else if (!other.use_l3_) {
                size_t base = avx_regions * 8;
                for (size_t j = 0; j < tail_l2_bytes; j++)
                    l2b_tail[j] = other.l2_flat_[base + j];
            }

            alignas(64) uint8_t buf_a[64], buf_b[64];
            std::memset(buf_a, fill_ones_ ? 0xFF : 0x00, 64);
            std::memset(buf_b, other.fill_ones_ ? 0xFF : 0x00, 64);

            for (size_t i = 0; i < remaining; i++) {
                if ((l2a_tail[i / 8] >> (i % 8)) & 1)
                    buf_a[i] = a_l1[a_l1_off++];
            }
            for (size_t i = 0; i < remaining; i++) {
                if ((l2b_tail[i / 8] >> (i % 8)) & 1)
                    buf_b[i] = b_l1[b_l1_off++];
            }

            for (size_t i = 0; i < remaining; i++)
                buf_a[i] &= buf_b[i];
            std::memcpy(r_l1 + r_off, buf_a, remaining);
        }
    }
    */

#else  // !__AVX512VBMI2__

#ifdef COMBIT_DEBUG
    auto t1 = clock::now();
#endif

    // === Scalar fallback (no AVX-512) ===
    {
        auto l2_a = expand_l2();
        auto l2_b = other.expand_l2();

        alignas(64) uint8_t buf_a[64], buf_b[64];
        size_t pos = 0;
        while (pos < total_words) {
            size_t chunk = std::min(size_t(64), total_words - pos);
            std::memset(buf_a, fill_ones_ ? 0xFF : 0x00, 64);
            std::memset(buf_b, other.fill_ones_ ? 0xFF : 0x00, 64);

            for (size_t i = 0; i < chunk; i++) {
                size_t wi = pos + i;
                if ((l2_a[wi / 8] >> (wi % 8)) & 1)
                    buf_a[i] = a_l1[a_l1_off++];
            }
            for (size_t i = 0; i < chunk; i++) {
                size_t wi = pos + i;
                if ((l2_b[wi / 8] >> (wi % 8)) & 1)
                    buf_b[i] = b_l1[b_l1_off++];
            }

            for (size_t i = 0; i < chunk; i++)
                buf_a[i] &= buf_b[i];
            std::memcpy(r_l1 + r_off, buf_a, chunk);
            r_off += chunk;
            pos += chunk;
        }
    }

#endif  // __AVX512VBMI2__

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
