#include "ddc.h"
#include "zero_run_bypass.h"
#include <algorithm>

#ifdef __AVX512VBMI2__
__attribute__((noinline)) static size_t
or_batch_fast(DDCBtv::SideCtx& A, DDCBtv::SideCtx& B,
              const uint8_t* l3a_buf, const uint8_t* l3b_buf,
              size_t batch_size, uint8_t* r_l1, size_t r_off) {
    static constexpr size_t PF_DIST = 128;
    const __m512i vec_ff = _mm512_set1_epi8(static_cast<char>(0xFF));
    const __m512i vec_00 = _mm512_setzero_si512();
    for (size_t r = 0; r < batch_size; r++) {
        const uint8_t l3a = l3a_buf[r];
        const uint8_t l3b = l3b_buf[r];

        if (l3a == 0 || l3b == 0) {
            const bool a_uf = (l3a == 0) && !A.l2_fill_ones;
            const bool b_uf = (l3b == 0) && !B.l2_fill_ones;
            const bool a_al = (l3a == 0) &&  A.l2_fill_ones;
            const bool b_al = (l3b == 0) &&  B.l2_fill_ones;
            const bool a_ones = a_uf && A.l1_fill_ones;
            const bool b_ones = b_uf && B.l1_fill_ones;
            if (a_uf && b_uf) {
                _mm512_storeu_si512(r_l1 + r_off,
                                    (a_ones || b_ones) ? vec_ff : vec_00);
                r_off += 64;
                continue;
            }
            if (a_al && b_uf) {
                if (b_ones)
                    _mm512_storeu_si512(r_l1 + r_off, vec_ff);
                else
                    std::memcpy(r_l1 + r_off, A.l1_lits + A.l1_lit_off, 64);
                A.l1_lit_off += 64;
                r_off += 64;
                continue;
            }
            if (b_al && a_uf) {
                if (a_ones)
                    _mm512_storeu_si512(r_l1 + r_off, vec_ff);
                else
                    std::memcpy(r_l1 + r_off, B.l1_lits + B.l1_lit_off, 64);
                B.l1_lit_off += 64;
                r_off += 64;
                continue;
            }
            if (a_ones || b_ones) {
                DDCBtv::SideCtx& M = a_ones ? B : A;
                const uint8_t l3m = a_ones ? l3b : l3a;
                __m512i l2m_v = _mm512_mask_expandloadu_epi8(M.l2_fill_vec,
                    static_cast<__mmask64>(l3m), M.l2_lits + M.l2_lit_off);
                M.l2_lit_off += __builtin_popcount(l3m);
                __mmask64 mm = static_cast<__mmask64>(
                    _mm_cvtsi128_si64(_mm512_castsi512_si128(l2m_v)));
                M.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mm));
                _mm512_storeu_si512(r_l1 + r_off, vec_ff);
                r_off += 64;
                continue;
            }
        }

        _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(r_l1 + r_off + PF_DIST), _MM_HINT_T0);

        __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
            static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
        A.l2_lit_off += __builtin_popcount(l3a);
        __mmask64 ma = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
        B.l2_lit_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma, A.l1_lits + A.l1_lit_off);
        A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));

        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb, B.l1_lits + B.l1_lit_off);
        B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i vr = _mm512_or_si512(va, vb);
        _mm512_storeu_si512(r_l1 + r_off, vr);
        r_off += 64;
    }
    return r_off;
}
#endif

// OR two compressed bitvectors
DDCBtv
DDCBtv::operator|(const DDCBtv& other) const {
    if (has_l2v_ || other.has_l2v_)
        return dense_binop(other, '|');

    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Compressed);
    assert(other.state_ == State::Compressed);

    if (bit_count_ == 0) return DDCBtv();

    const size_t total_words = l2_count_;

    const bool compress = ddc_compress_results;
    DDCBtv result = compress ? DDCBtv(false, false, State::Compressed)
                                : DDCBtv(false, true, State::Decompressed);
    result.bit_count_ = bit_count_;
    result.l2_count_ = total_words;
    size_t l2_byte_count = (total_words + 7) / 8;

    if (compress) {
        result.l2_flat_.assign(l2_byte_count, 0x00);
    } else {

        result.l3_count_ = l2_byte_count;
        result.l2_lit_count_ = 0;
    }

    result.l1_lits_.resize(total_words);
    result.l1_lit_count_ = total_words;

    const uint8_t* a_l1 = l1_lits_.data();
    const uint8_t* b_l1 = other.l1_lits_.data();
    uint8_t* r_l1 = result.l1_lits_.data();

    size_t r_off = 0;

#ifdef DDC_DEBUG
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
#endif

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    const bool try_fast = !compress && ddc_zrb::run_hint();

    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const bool a_struct_zero = a_zero_when_l3_zero && !A.l3_fill_ones;
    const bool b_struct_zero = b_zero_when_l3_zero && !B.l3_fill_ones;

    // batches
    const size_t batch_count = (avx_regions + 63) / 64;
    if (try_fast) {
    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t batch_start = batch * 64;
        const size_t batch_end   = std::min(batch_start + 64, avx_regions);
        const size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, A.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, B.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }
        if ((a_struct_zero && a_l4_mask == 0) &&
            (b_struct_zero && b_l4_mask == 0)) {
            std::memset(r_l1 + r_off, 0, batch_size * 64);
            r_off += batch_size * 64;
            continue;
        }
        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(A.l3_fill_vec,
            static_cast<__mmask64>(a_l4_mask), A.l3_lits + A.l3_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(B.l3_fill_vec,
            static_cast<__mmask64>(b_l4_mask), B.l3_lits + B.l3_lit_off);
        A.l3_lit_off += __builtin_popcountll(a_l4_mask);
        B.l3_lit_off += __builtin_popcountll(b_l4_mask);
        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);
        r_off = or_batch_fast(A, B, l3a_buf, l3b_buf, batch_size, r_l1, r_off);
    }
    } else {
    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t batch_start = batch * 64;
        const size_t batch_end   = std::min(batch_start + 64, avx_regions);
        const size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, A.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, B.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }

        const bool a_batch_zero = a_struct_zero && a_l4_mask == 0;
        const bool b_batch_zero = b_struct_zero && b_l4_mask == 0;
        // bypass empty batch
        if (a_batch_zero && b_batch_zero) {
            if (!compress) {
                std::memset(r_l1 + r_off, 0, batch_size * 64);
                r_off += batch_size * 64;
            }

            continue;
        }

        // expand L3
        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(A.l3_fill_vec,
            static_cast<__mmask64>(a_l4_mask), A.l3_lits + A.l3_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(B.l3_fill_vec,
            static_cast<__mmask64>(b_l4_mask), B.l3_lits + B.l3_lit_off);
        A.l3_lit_off += __builtin_popcountll(a_l4_mask);
        B.l3_lit_off += __builtin_popcountll(b_l4_mask);

        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);

        for (size_t r = 0; r < batch_size; r++) {
            const size_t region = batch_start + r;
            const uint8_t l3a = l3a_buf[r];
            const uint8_t l3b = l3b_buf[r];

            _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_lit_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_lit_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<char*>(r_l1 + r_off + PF_DIST), _MM_HINT_T0);

            // expand L2/L1 then OR
            __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
            A.l2_lit_off += __builtin_popcount(l3a);
            __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));

            __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
            B.l2_lit_off += __builtin_popcount(l3b);
            __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

            __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma, A.l1_lits + A.l1_lit_off);
            A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));

            __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb, B.l1_lits + B.l1_lit_off);
            B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

            __m512i vr = _mm512_or_si512(va, vb);
            if (compress) {  // emit compressed
                __mmask64 lit_mask = _mm512_test_epi8_mask(vr, vr);
                uint64_t mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, vr);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, vr);
                r_off += 64;
            }
        }
    }
    }

    // scalar tail
    if (avx_regions * words_per_reg < total_words) {
        const uint8_t a_l3_fill = A.l3_fill_ones ? 0xFF : 0x00;
        const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;
        const uint8_t l1_fill_a = A.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l1_fill_b = B.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_a = A.l2_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_b = B.l2_fill_ones ? 0xFF : 0x00;
        bool a_l4_lit = (A.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        bool b_l4_lit = (B.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3a = a_l4_lit ? A.l3_lits[A.l3_lit_off++] : a_l3_fill;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2a = ((l3a >> l2i) & 1) ? A.l2_lits[A.l2_lit_off++] : l2_fill_a;
            uint8_t l2b = ((l3b >> l2i) & 1) ? B.l2_lits[B.l2_lit_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wa = ((l2a >> bit) & 1) ? A.l1_lits[A.l1_lit_off++] : l1_fill_a;
                uint8_t wb = ((l2b >> bit) & 1) ? B.l1_lits[B.l1_lit_off++] : l1_fill_b;
                uint8_t vr = wa | wb;
                if (compress) {
                    if (vr != 0x00) {
                        result.l2_flat_[pos / 8] |= uint8_t(1) << (pos % 8);
                        r_l1[r_off++] = vr;
                    }
                } else {
                    r_l1[r_off++] = vr;
                }
            }
        }
    }

#ifdef DDC_DEBUG
    auto t1 = clock::now();
#endif

#else

#ifdef DDC_DEBUG
    auto t1 = clock::now();
#endif

    // scalar fallback
    {
        size_t a_l1_off = 0, b_l1_off = 0;
        auto l2_a = expand_l2();
        auto l2_b = other.expand_l2();

        alignas(64) uint8_t buf_a[64], buf_b[64];
        size_t pos = 0;
        while (pos < total_words) {
            size_t chunk = std::min(size_t(64), total_words - pos);
            std::memset(buf_a, l1_fill_ones_ ? 0xFF : 0x00, 64);
            std::memset(buf_b, other.l1_fill_ones_ ? 0xFF : 0x00, 64);

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
                buf_a[i] |= buf_b[i];
            if (compress) {
                for (size_t i = 0; i < chunk; i++) {
                    if (buf_a[i] != 0x00) {
                        size_t wi = pos + i;
                        result.l2_flat_[wi / 8] |= uint8_t(1) << (wi % 8);
                        r_l1[r_off++] = buf_a[i];
                    }
                }
            } else {
                std::memcpy(r_l1 + r_off, buf_a, chunk);
                r_off += chunk;
            }
            pos += chunk;
        }
    }

#endif

#ifdef DDC_DEBUG
    auto t2 = clock::now();
    auto us = [](auto a, auto b) {
        return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::cout << "  [OR] "
              << "expand_or: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    if (compress) result.compact_l2_l3(r_off);
    result.mask_tail_byte();
    return result;
}

// per-segment OR
DDC
DDC::operator|(const DDC& other) const {
    assert(bit_count_ == other.bit_count_);

    DDC result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    const size_t n = total_segments();

    // bypass
    if (ddc_zrb::enabled() && n > 0) {
        assert(segment_bits_ == other.segment_bits_);
        const id_vec_t* ia = sparse_form_ ? &seg_ids_ : nullptr;
        const id_vec_t* ib = other.sparse_form_ ? &other.seg_ids_ : nullptr;
        ensure_masks();                       // v3.1: cached per-bitmap masks
        other.ensure_masks();
        const uint64_t* za = zmask_.data();       const uint64_t* oa = omask_.data();
        const uint64_t* zb = other.zmask_.data(); const uint64_t* ob = other.omask_.data();
        // v3 absorption masks: g = both-zero (gap), o = either-ones (x|1=1)
        const size_t nw = zmask_.size();
        uint64_t gsb[64], osb[64];          // 64 words = 4096 segments
        std::vector<uint64_t> gh, oh;       // heap fallback for huge grids
        uint64_t *g, *o;
        if (nw <= 64) { g = gsb; o = osb; }
        else { gh.resize(nw); oh.resize(nw); g = gh.data(); o = oh.data(); }
        size_t gap_bits = 0, ones_bits = 0;
        for (size_t w = 0; w < nw; w++) {
            g[w] = za[w] & zb[w];
            o[w] = oa[w] | ob[w];
            gap_bits  += (size_t)__builtin_popcountll(g[w]);
            ones_bits += (size_t)__builtin_popcountll(o[w]);
        }
        ddc_zrb::SegView<seg_vec_t, id_vec_t> va{&segments_, ia};
        ddc_zrb::SegView<seg_vec_t, id_vec_t> vb{&other.segments_, ib};
        ddc_zrb::JumpLog jl;
        id_vec_t rid;
        run_vec_t rones;
        const size_t present = n - gap_bits - ones_bits;   // exact stored count
        result.segments_.reserve(present);
        rid.reserve(present);
        rones.reserve(8);

        const bool collapse_uniform = (gap_bits + ones_bits) * 2 >= n;
        ddc_zrb::set_run_hint(collapse_uniform);
        auto emit_mixed = [&](DDCBtv&& r_seg, size_t idx) {
            if (collapse_uniform) {
                const int u = r_seg.uniform_class();
                if (u == 0) return;
                if (u == 1) {
                    if (!rones.empty() && rones.back().first + rones.back().second == idx)
                        rones.back().second++;
                    else rones.emplace_back((uint32_t)idx, 1u);
                    return;
                }
            }
            result.segments_.push_back(std::move(r_seg));
            rid.push_back((uint32_t)idx);
        };

        size_t i = 0;
        while (i < n) {
            if (ddc_zrb::test_bit(g, i)) {        // both zero → implicit gap
                const size_t L = ddc_zrb::run_len_set(g, i, n);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(o, i)) {        // either ones → virtual ones run
                const size_t L = ddc_zrb::run_len_set(o, i, n);
                if (!rones.empty() && rones.back().first + rones.back().second == i)
                    rones.back().second += (uint32_t)L;
                else
                    rones.emplace_back((uint32_t)i, (uint32_t)L);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(zb, i)) {       // b zero → result = a range (a mixed)
                const size_t L = std::min({ddc_zrb::run_len_set(zb, i, n),
                                           ddc_zrb::run_len_clear(za, i, n),
                                           ddc_zrb::run_len_clear(o, i, n)});
                va.emit_range(i, i + L, result.segments_, rid);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(za, i)) {       // a zero → result = b range (b mixed)
                const size_t L = std::min({ddc_zrb::run_len_set(za, i, n),
                                           ddc_zrb::run_len_clear(zb, i, n),
                                           ddc_zrb::run_len_clear(o, i, n)});
                vb.emit_range(i, i + L, result.segments_, rid);
                jl.jump(L);
                i += L;
                continue;
            }

            const DDCBtv& sa = *va.at(i);         // both present & mixed here
            const DDCBtv& sb = *vb.at(i);
            if (&sa == &sb) {
                // idempotent
                DDCBtv tmp = sa;
                if (collapse_uniform) emit_mixed(std::move(tmp), i);
                else { result.segments_.push_back(std::move(tmp)); rid.push_back((uint32_t)i); }
                jl.jump(1); i++; continue;
            }
            if (sa.has_l2v() && sb.has_l2v()) {
                const int u = ddc_dual_probe('|', sa, sb);
                if (u == 0) { jl.jump(1); i++; continue; }
                if (u == 1) { if (!rones.empty() && rones.back().first + rones.back().second == i)
                    rones.back().second++;
                else rones.emplace_back((uint32_t)i, 1u);
                jl.jump(1); i++; continue; }
            }
            if (sa.state() != DDCBtv::State::Compressed) {
                DDCBtv tmp = sa;
                tmp |= sb;
                if (collapse_uniform) emit_mixed(std::move(tmp), i);
                else { result.segments_.push_back(std::move(tmp)); rid.push_back((uint32_t)i); }
                i++;
                continue;
            }
            if (sb.state() != DDCBtv::State::Compressed) {
                DDCBtv tmp = sb;
                tmp |= sa;
                if (collapse_uniform) emit_mixed(std::move(tmp), i);
                else { result.segments_.push_back(std::move(tmp)); rid.push_back((uint32_t)i); }
                i++;
                continue;
            }
            if (collapse_uniform) emit_mixed(sa | sb, i);
            else { result.segments_.push_back(sa | sb); rid.push_back((uint32_t)i); }
            i++;
        }

        ddc_zrb::set_run_hint(false);
        if (rid.size() == n) {
            result.sparse_form_ = false;          // fully materialised
        } else {
            result.sparse_form_ = true;
            result.seg_ids_ = std::move(rid);
            result.ones_runs_ = std::move(rones);
        }
        if (ddc_zrb::debug_on())
            ddc_zrb::report("OR", segment_bits_, n, zmask_, omask_, other.zmask_, other.omask_, jl);
        return result;
    }

    ensure_flat();
    other.ensure_flat();
    // legacy path (DDC_ZERO_RUN_BYPASS=0) — byte-for-byte the original loop
    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // fill shortcuts
        if (sb.is_all_zero()) { result.segments_.push_back(sa); continue; }
        if (sa.is_all_zero()) { result.segments_.push_back(sb); continue; }
        if (sa.is_all_ones()) { result.segments_.push_back(sa); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(sb); continue; }

        if (sa.state() != DDCBtv::State::Compressed) {
            DDCBtv tmp = sa;
            tmp |= sb;
            result.segments_.push_back(std::move(tmp));
            continue;
        }
        if (sb.state() != DDCBtv::State::Compressed) {

            DDCBtv tmp = sb;
            tmp |= sa;
            result.segments_.push_back(std::move(tmp));
            continue;
        }

        result.segments_.push_back(sa | sb);
    }

    return result;
}

// OR into decompressed lhs
DDCBtv&
DDCBtv::operator|=(const DDCBtv& other) {
    if (has_l2v_ || other.has_l2v_) {
        *this = dense_binop(other, '|');
        return *this;
    }

    assert(bit_count_ == other.bit_count_);
    assert(state_ == State::Decompressed);
    assert(other.state_ != State::Uncompressed);

    if (bit_count_ == 0) return *this;

    const size_t total_words = l2_count_;
    uint8_t* r_l1 = l1_lits_.data();
    const uint8_t* b_l1 = other.l1_lits_.data();

#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    // both decompressed: dense SIMD OR
    if (other.state_ == State::Decompressed) {
        for (size_t region = 0; region < avx_regions; region++) {
            __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
            __m512i vb = _mm512_loadu_si512(b_l1 + region * 64);
            _mm512_storeu_si512(r_l1 + region * 64,
                _mm512_or_si512(va, vb));
        }
        for (size_t pos = avx_regions * words_per_reg; pos < total_words; pos++)
            r_l1[pos] |= b_l1[pos];
        mask_tail_byte();
        return *this;
    }

    SideCtx B = other.make_side(b_l1);
    static constexpr size_t PF_DIST = 128;

    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const bool b_l4_skipable = b_zero_when_l3_zero && !B.l3_fill_ones;
    const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;

    // compressed rhs: expand-and-OR
    for (size_t region = 0; region < avx_regions; region++) {
        // branchless L4 skip
        if (b_l4_skipable && (region & 63) == 0
            && region + 64 <= avx_regions) {
            uint64_t l4_chunk;
            std::memcpy(&l4_chunk, B.l4_bits + region / 8, 8);
            if (l4_chunk == 0) {
                region += 63;
                continue;
            }
        }

        bool b_l4_lit = (B.l4_bits[region / 8] >> (region % 8)) & 1;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;

        if (b_zero_when_l3_zero && l3b == 0) continue;

        _mm_prefetch(reinterpret_cast<const char*>(
            B.l1_lits + B.l1_lit_off + PF_DIST), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<char*>(
            r_l1 + region * 64 + PF_DIST), _MM_HINT_T0);

        __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
            static_cast<__mmask64>(l3b),
            B.l2_lits + B.l2_lit_off);
        B.l2_lit_off += __builtin_popcount(l3b);
        __mmask64 mb = static_cast<__mmask64>(
            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));

        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb,
            B.l1_lits + B.l1_lit_off);
        B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));

        __m512i va = _mm512_loadu_si512(r_l1 + region * 64);
        _mm512_storeu_si512(r_l1 + region * 64,
            _mm512_or_si512(va, vb));
    }

    if (avx_regions * words_per_reg < total_words) {
        const uint8_t l1_fill_b = B.l1_fill_ones ? 0xFF : 0x00;
        const uint8_t l2_fill_b = B.l2_fill_ones ? 0xFF : 0x00;
        bool b_l4_lit = (B.l4_bits[avx_regions / 8] >> (avx_regions % 8)) & 1;
        uint8_t l3b = b_l4_lit ? B.l3_lits[B.l3_lit_off++] : b_l3_fill;
        size_t pos = avx_regions * words_per_reg;
        for (int l2i = 0; pos < total_words; l2i++) {
            uint8_t l2b = ((l3b >> l2i) & 1) ? B.l2_lits[B.l2_lit_off++] : l2_fill_b;
            for (int bit = 0; bit < 8 && pos < total_words; bit++, pos++) {
                uint8_t wb = ((l2b >> bit) & 1) ? B.l1_lits[B.l1_lit_off++] : l1_fill_b;
                r_l1[pos] |= wb;
            }
        }
    }
#else
    {
        size_t b_l1_off = 0;
        auto l2_b = other.expand_l2();
        for (size_t w = 0; w < total_words; w++) {
            uint8_t wb = other.l1_fill_ones_ ? 0xFF : 0x00;
            if ((l2_b[w / 8] >> (w % 8)) & 1)
                wb = b_l1[b_l1_off++];
            r_l1[w] |= wb;
        }
    }
#endif
    mask_tail_byte();
    return *this;
}

// per-segment OR in place
DDC&
DDC::operator|=(const DDC& other) {
    invalidate_masks();
    ensure_flat();
    other.ensure_flat();
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    const size_t num_segs = segments_.size();

    for (size_t i = 0; i < num_segs; i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_zero()) continue;
        if (seg.is_all_ones()) {
            segments_[i] = DDCBtv::make_all_fill(
                seg.bit_count(), seg.l2_count(), true);
            continue;
        }

        if (segments_[i].is_all_ones()) continue;
        if (segments_[i].is_all_zero()) {
            segments_[i] = seg;
            continue;
        }
        if (segments_[i].state() != DDCBtv::State::Decompressed) {
            segments_[i] = segments_[i] | seg;
            continue;
        }

        if (i + 2 < num_segs) {
            __builtin_prefetch(
                other.segments_[i + 2].l1_lit_data(), 0, 3);
            __builtin_prefetch(
                other.segments_[i + 2].l4_data(), 0, 3);
        }

        segments_[i] |= other.segments_[i];
    }

    return *this;
}

// fast union of many bitvectors
DDC
DDC::OR_many(size_t number, const DDC** Btvs) {
    for (size_t z = 0; z < number; z++) Btvs[z]->ensure_flat();
    assert(number > 0);

    if (number == 1) return *Btvs[0];

    const size_t num_segs = Btvs[0]->num_segments();

    size_t total_nz = 0;
    size_t total_slots = 0;
    for (size_t i = 0; i < number; i++) {
        const size_t nsegs = Btvs[i]->num_segments();
        for (size_t s = 0; s < nsegs; s++) {
            const DDCBtv& seg = Btvs[i]->segment(s);
            total_nz += seg.num_lits();
            total_slots += seg.l2_count();
        }
    }
    const bool use_scatter = (total_nz * 20 < total_slots);  // sparsity heuristic

    DDC result;
    result.bit_count_ = Btvs[0]->bit_count_;
    result.segment_bits_ = Btvs[0]->segment_bits_;
    result.segments_.reserve(num_segs);

    if (use_scatter) {

        for (size_t s = 0; s < num_segs; s++) {
            const DDCBtv& src = Btvs[0]->segment(s);
            DDCBtv seg( false,
                           true,
                          DDCBtv::State::Decompressed);
            seg.bit_count_ = src.bit_count_;
            seg.l2_count_ = src.l2_count_;
            const size_t l2_byte_count = (src.l2_count_ + 7) / 8;
            seg.l3_count_ = l2_byte_count;

            seg.l2_lit_count_ = 0;
            seg.l1_lits_.assign(src.l2_count_, 0x00);
            seg.l1_lit_count_ = src.l2_count_;
            result.segments_.push_back(std::move(seg));
        }

        // scatter literals into dense accumulators
        for (size_t i = 0; i < number; i++) {
            size_t cur_seg   = 0;
            size_t seg_start = 0;
            size_t seg_end   = result.segments_[0].l2_count_;
            uint8_t* dst     = result.segments_[0].l1_lits_.data();

            Btvs[i]->for_each_literal(
                [&cur_seg, &seg_start, &seg_end, &dst, &result](uint32_t word_pos, uint8_t val) {
                    while (word_pos >= seg_end) {
                        cur_seg++;
                        seg_start = seg_end;
                        seg_end  += result.segments_[cur_seg].l2_count_;
                        dst       = result.segments_[cur_seg]
                                          .l1_lits_.data();
                    }
                    dst[word_pos - seg_start] |= val;
                });
        }
    } else {
        // pairwise OR-reduce
        for (size_t s = 0; s < num_segs; s++) {
            const DDCBtv& s0 = Btvs[0]->segment(s);
            const DDCBtv& s1 = Btvs[1]->segment(s);
            // Degenerate operands (hollow make_all_fill segments from
            // ensure_flat carry no L2-L4 arrays; chained results may be
            // Decompressed) must not reach the compressed-only kernel.  The
            // canonical dense-benchmark path below stays byte-identical.
            const bool degenerate =
                s0.is_all_zero() || s1.is_all_zero() ||
                s0.is_all_ones() || s1.is_all_ones() ||
                s0.state() != DDCBtv::State::Compressed ||
                s1.state() != DDCBtv::State::Compressed;
            if (!degenerate) {
                result.segments_.push_back(s0 | s1);
            } else {
                DDCBtv acc = DDCBtv::make_decompressed_zero(
                    s0.bit_count(), s0.l2_count());
                for (const DDCBtv* x : {&s0, &s1}) {
                    if (x->is_all_zero()) continue;
                    if (x->is_all_ones()) {
                        std::memset(acc.l1_lits_.data(), 0xFF,
                                    acc.l1_lits_.size());
                        acc.mask_tail_byte();
                        continue;
                    }
                    acc |= *x;
                }
                result.segments_.push_back(std::move(acc));
            }
            bool saturated = result.segments_[s].is_all_ones();
            for (size_t i = 2; i < number && !saturated; i++) {
                const auto& seg = Btvs[i]->segment(s);

                if (seg.is_all_zero()) continue;
                if (seg.is_all_ones()) {           // x|1=1: absorb and stop
                    result.segments_[s] = DDCBtv::make_all_fill(
                        seg.bit_count(), seg.l2_count(), true);
                    saturated = true;
                    continue;
                }
                if (result.segments_[s].state() != DDCBtv::State::Decompressed) {
                    // accumulator came from a shortcut/compressed pairwise
                    // result: densify once so |= (Decompressed lhs) is legal
                    DDCBtv acc = DDCBtv::make_decompressed_zero(
                        result.segments_[s].bit_count(),
                        result.segments_[s].l2_count());
                    if (!result.segments_[s].is_all_zero())
                        acc |= result.segments_[s];
                    result.segments_[s] = std::move(acc);
                }

                if (i + 2 < number) {
                    const auto& next = Btvs[i + 2]->segment(s);
                    __builtin_prefetch(next.l1_lit_data(), 0, 3);
                    __builtin_prefetch(next.l4_data(), 0, 3);
                }

                result.segments_[s] |= seg;
            }
        }
    }

    return result;
}
