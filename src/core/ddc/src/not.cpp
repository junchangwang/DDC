#include "ddc.h"

#include <cassert>
#include <algorithm>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

DDCBtv&
DDCBtv::negate_inplace() {
    if (has_l2v_) { *this = ~*this; return *this; }
    if (bit_count_ == 0) return *this;
    assert(state_ != State::Uncompressed);

    l1_fill_ones_ = !l1_fill_ones_;

    uint8_t* data = l1_lits_.data();
    size_t n = l1_lits_.size();

#ifdef __AVX512F__

    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
    size_t i = 0;

    for (; i + 256 <= n; i += 256) {
        __m512i v0 = _mm512_loadu_si512(data + i);
        __m512i v1 = _mm512_loadu_si512(data + i +  64);
        __m512i v2 = _mm512_loadu_si512(data + i + 128);
        __m512i v3 = _mm512_loadu_si512(data + i + 192);
        _mm512_storeu_si512(data + i,       _mm512_xor_si512(v0, ones));
        _mm512_storeu_si512(data + i +  64, _mm512_xor_si512(v1, ones));
        _mm512_storeu_si512(data + i + 128, _mm512_xor_si512(v2, ones));
        _mm512_storeu_si512(data + i + 192, _mm512_xor_si512(v3, ones));
    }

    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(data + i);
        _mm512_storeu_si512(data + i, _mm512_xor_si512(v, ones));
    }

    if (i < n) {
        size_t tail = n - i;
        __mmask64 m = (tail >= 64) ? __mmask64(-1)
                                   : __mmask64((uint64_t(1) << tail) - 1);
        __m512i v = _mm512_maskz_loadu_epi8(m, data + i);
        _mm512_mask_storeu_epi8(data + i, m, _mm512_xor_si512(v, ones));
    }
#else
    for (size_t k = 0; k < n; k++) data[k] ^= 0xFF;
#endif

    // Tail masking
    if (bit_count_ % 8 != 0 && l1_lit_count_ > 0 && is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;
        size_t byte_off = l1_lit_count_ - 1;
        l1_lits_[byte_off] &= static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return *this;
}

DDCBtv
DDCBtv::operator~() const {
    if (has_l2v_) {
        if (bit_count_ % 8 == 0) {
            DDCBtv d = *this;
            for (auto& b : d.l1_lits_) b = (uint8_t)~b;
            for (auto& b : d.l2v_bits_) b = (uint8_t)~b;
            if (l1_lit_count_ != 0) {
                std::vector<uint32_t> w(l1_lit_count_);
                std::vector<uint8_t>  v(l1_lit_count_);
                const size_t nl = collect_literals(w.data(), v.data(), l1_lit_count_);
                for (size_t k = 0; k < nl; k++)
                    d.l2v_bits_[w[k] / 8] &= (uint8_t)~(uint8_t(1) << (w[k] % 8));
            }
            if (l2_count_ % 8)
                d.l2v_bits_.back() &= uint8_t((1u << (l2_count_ % 8)) - 1);
            return d;
        }
        DDCBtv d = to_decompressed();
        for (auto& b : d.l1_lits_) b = (uint8_t)~b;
        d.mask_tail_byte();
        return d;
    }

    if (bit_count_ == 0) return DDCBtv();
    assert(state_ != State::Uncompressed);

    if (l1_lit_count_ == 0) {
        return make_all_fill(bit_count_, l2_count_, !l1_fill_ones_);
    }

    DDCBtv result( !l1_fill_ones_,
                      l2_fill_ones_,
                     state_);
    result.bit_count_       = bit_count_;
    result.l2_count_        = l2_count_;
    result.l3_count_        = l3_count_;
    result.l4_count_        = l4_count_;
    result.l3_fill_ones_    = l3_fill_ones_;
    result.l2_lit_count_    = l2_lit_count_;
    result.l3_lit_count_    = l3_lit_count_;
    result.l1_lit_count_    = l1_lit_count_;

    result.l2_lits_         = l2_lits_;
    result.l3_lits_         = l3_lits_;
    result.l4_bits_         = l4_bits_;

    result.l1_lits_.resize(l1_lit_count_);
    const uint8_t* src = l1_lits_.data();
    uint8_t* dst = result.l1_lits_.data();
    size_t n = l1_lit_count_;
    size_t i = 0;

#ifdef __AVX512F__
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));

    for (; i + 256 <= n; i += 256) {
        __m512i v0 = _mm512_loadu_si512(src + i);
        __m512i v1 = _mm512_loadu_si512(src + i +  64);
        __m512i v2 = _mm512_loadu_si512(src + i + 128);
        __m512i v3 = _mm512_loadu_si512(src + i + 192);
        _mm512_storeu_si512(dst + i,       _mm512_xor_si512(v0, ones));
        _mm512_storeu_si512(dst + i +  64, _mm512_xor_si512(v1, ones));
        _mm512_storeu_si512(dst + i + 128, _mm512_xor_si512(v2, ones));
        _mm512_storeu_si512(dst + i + 192, _mm512_xor_si512(v3, ones));
    }
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(src + i);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(v, ones));
    }
    if (i < n) {
        size_t tail = n - i;
        __mmask64 m = (tail >= 64) ? __mmask64(-1)
                                   : __mmask64((uint64_t(1) << tail) - 1);
        __m512i v = _mm512_maskz_loadu_epi8(m, src + i);
        _mm512_mask_storeu_epi8(dst + i, m, _mm512_xor_si512(v, ones));
    }
#else
    for (; i < n; i++) dst[i] = src[i] ^ 0xFF;
#endif

    if (bit_count_ % 8 != 0 && result.l1_lit_count_ > 0
        && result.is_last_word_literal()) {
        size_t valid_bits = bit_count_ % 8;
        size_t byte_off = result.l1_lit_count_ - 1;
        result.l1_lits_[byte_off] &=
            static_cast<uint8_t>(0xFF << (8 - valid_bits));
    }

    return result;
}

DDC&
DDC::negate_inplace() {
    ensure_flat();
    for (auto& s : segments_) s.negate_inplace();
    invalidate_masks();
    return *this;
}

// Run complement
DDC
DDC::operator~() const {
    DDC result;
    result.bit_count_    = bit_count_;
    result.segment_bits_ = segment_bits_;
    result.sparse_form_  = true;
    const size_t n = total_segments();

    auto push_run = [&](uint32_t id, uint32_t len) {
        if (!result.ones_runs_.empty() &&
            result.ones_runs_.back().first + result.ones_runs_.back().second == id)
            result.ones_runs_.back().second += len;
        else
            result.ones_runs_.emplace_back(id, len);
    };

    if (sparse_form_) {
        result.segments_.reserve(segments_.size());
        result.seg_ids_.reserve(segments_.size());
        size_t k = 0, m = 0, c = 0;
        while (c < n) {
            const size_t next_id  = (k < seg_ids_.size())   ? seg_ids_[k]           : n;
            const size_t next_run = (m < ones_runs_.size()) ? ones_runs_[m].first   : n;
            const size_t nxt = std::min(next_id, next_run);
            if (c < nxt) push_run((uint32_t)c, (uint32_t)(std::min(nxt, n) - c));
            if (nxt >= n) break;
            if (next_id <= next_run) {
                const DDCBtv& s = segments_[k++];
                if (s.is_all_zero())      push_run((uint32_t)nxt, 1);
                else if (!s.is_all_ones()) {
                    result.segments_.push_back(~s);
                    result.seg_ids_.push_back((uint32_t)nxt);
                }
                c = nxt + 1;
            } else {
                c = next_run + ones_runs_[m].second;
                m++;
            }
        }
        return result;
    }

    ensure_masks();
    {
        size_t mixed_total = 0;
        for (size_t wi = 0; wi * 64 < n; wi++) {
            const size_t left = n - wi * 64;
            const uint64_t valid = (left >= 64) ? ~0ull : ((1ull << left) - 1);
            mixed_total += (size_t)__builtin_popcountll(
                valid & ~(zmask_[wi] | omask_[wi]));
        }
        result.segments_.reserve(mixed_total);
        result.seg_ids_.reserve(mixed_total);
    }
    for (size_t wi = 0; wi * 64 < n; wi++) {
        const size_t base = wi * 64;
        const size_t left = n - base;
        const uint64_t valid = (left >= 64) ? ~0ull : ((1ull << left) - 1);
        const uint64_t zw = zmask_[wi] & valid;
        uint64_t mixed = valid & ~(zmask_[wi] | omask_[wi]);
        uint64_t b = zw;
        while (b) {
            const int s = __builtin_ctzll(b);
            const uint64_t rest = b >> s;
            const int len = (~rest) ? __builtin_ctzll(~rest) : (64 - s);
            push_run((uint32_t)(base + s), (uint32_t)len);
            if (s + len >= 64) break;
            b &= ~(((len < 64 ? (1ull << len) - 1 : ~0ull)) << s);
        }
        while (mixed) {
            const int i = __builtin_ctzll(mixed);
            mixed &= mixed - 1;
            result.segments_.push_back(~segments_[base + i]);
            result.seg_ids_.push_back((uint32_t)(base + i));
        }
    }
    return result;
}
