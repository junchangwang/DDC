#include "ddc.h"
#include "zero_run_bypass.h"
#include <algorithm>

// XOR kernel
DDCBtv
DDCBtv::operator^(const DDCBtv& other) const {
    if (has_l2v_ || other.has_l2v_)
        return dense_binop(other, '^');

    assert(bit_count_ == other.bit_count_);
    if (bit_count_ == 0) return DDCBtv();
    assert(state_ != State::Uncompressed);
    assert(other.state_ != State::Uncompressed);

    // dense
    if (state_ == State::Decompressed || other.state_ == State::Decompressed) {
        auto densify = [](const DDCBtv& s) {
            DDCBtv t = DDCBtv::make_decompressed_zero(s.bit_count_, s.l2_count_);
            t |= s;
            return t;
        };
        DDCBtv r  = (state_ == State::Decompressed) ? *this : densify(*this);
        DDCBtv db = (other.state_ == State::Decompressed) ? other : densify(other);
        uint8_t* rp = r.l1_lits_.data();
        const uint8_t* bp = db.l1_lits_.data();
        const size_t nw = r.l1_lits_.size();
        size_t i = 0;
#ifdef __AVX512F__
        for (; i + 64 <= nw; i += 64) {
            __m512i x = _mm512_loadu_si512(rp + i);
            __m512i y = _mm512_loadu_si512(bp + i);
            _mm512_storeu_si512(rp + i, _mm512_xor_si512(x, y));
        }
#endif
        for (; i < nw; i++) rp[i] ^= bp[i];
        r.mask_tail_byte();
        return r;
    }

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

// SIMD path
#ifdef __AVX512VBMI2__
    const size_t avx_regions = total_words / words_per_reg;

    SideCtx A = this->make_side(a_l1);
    SideCtx B = other.make_side(b_l1);

    static constexpr size_t PF_DIST = 128;

    uint8_t* result_l2 = result.l2_flat_.data();

    const bool a_zero_when_l3_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const bool b_zero_when_l3_zero = !B.l1_fill_ones && !B.l2_fill_ones;

    const uint8_t a_l3_fill = A.l3_fill_ones ? 0xFF : 0x00;
    const uint8_t b_l3_fill = B.l3_fill_ones ? 0xFF : 0x00;

    const bool a_struct_zero = a_zero_when_l3_zero && !A.l3_fill_ones;
    const bool b_struct_zero = b_zero_when_l3_zero && !B.l3_fill_ones;

    // batches
    const size_t batch_count = (avx_regions + 63) / 64;
    for (size_t batch = 0; batch < batch_count; batch++) {
        const size_t batch_start = batch * 64;
        const size_t batch_end   = std::min(batch_start + 64, avx_regions);
        const size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, A.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, B.l4_bits + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            const uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }
        const bool a_batch_zero = a_struct_zero && a_l4_mask == 0;
        const bool b_batch_zero = b_struct_zero && b_l4_mask == 0;
        if (a_batch_zero && b_batch_zero) {
            if (!compress) {
                std::memset(r_l1 + r_off, 0, batch_size * 64);
                r_off += batch_size * 64;
            }
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

        for (size_t r = 0; r < batch_size; r++) {
        const size_t region = batch_start + r;
        const uint8_t l3a = l3a_buf[r];
        const uint8_t l3b = l3b_buf[r];

        const bool a_is_zero = a_zero_when_l3_zero && l3a == 0;
        const bool b_is_zero = b_zero_when_l3_zero && l3b == 0;

        if (a_is_zero && b_is_zero) {  // both empty

            if (!compress) {
                _mm512_storeu_si512(r_l1 + r_off, _mm512_setzero_si512());
                r_off += 64;
            }

            continue;
        }
        if (a_is_zero) {

            __m512i l2b_v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                static_cast<__mmask64>(l3b), B.l2_lits + B.l2_lit_off);
            B.l2_lit_off += __builtin_popcount(l3b);
            __mmask64 mb = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b_v)));
            __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec, mb,
                B.l1_lits + B.l1_lit_off);
            B.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(mb));
            if (compress) {
                __mmask64 lit_mask = _mm512_test_epi8_mask(vb, vb);
                uint64_t mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, vb);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, vb);
                r_off += 64;
            }
            continue;
        }
        if (b_is_zero) {

            __m512i l2a_v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                static_cast<__mmask64>(l3a), A.l2_lits + A.l2_lit_off);
            A.l2_lit_off += __builtin_popcount(l3a);
            __mmask64 ma = static_cast<__mmask64>(
                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a_v)));
            __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec, ma,
                A.l1_lits + A.l1_lit_off);
            A.l1_lit_off += __builtin_popcountll(static_cast<uint64_t>(ma));
            if (compress) {
                __mmask64 lit_mask = _mm512_test_epi8_mask(va, va);
                uint64_t mask_val = static_cast<uint64_t>(lit_mask);
                std::memcpy(result_l2 + region * 8, &mask_val, 8);
                _mm512_mask_compressstoreu_epi8(r_l1 + r_off, lit_mask, va);
                r_off += __builtin_popcountll(mask_val);
            } else {
                _mm512_storeu_si512(r_l1 + r_off, va);
                r_off += 64;
            }
            continue;
        }

        // prefetch
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

        __m512i vr = _mm512_xor_si512(va, vb);  // XOR + emit
        if (compress) {
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

    // scalar tail
    if (avx_regions * words_per_reg < total_words) {
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
                uint8_t vr = wa ^ wb;
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

    {
        // scalar fallback
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
                buf_a[i] ^= buf_b[i];
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
    std::cout << "  [XOR] "
              << "expand_xor: " << std::fixed << std::setprecision(1) << us(t0, t1)
              << " us | scalar_tail: " << us(t1, t2)
              << " us | total: " << us(t0, t2) << " us"
              << " | count(1): " << result.popcount()
              << "\n";
#endif

    if (compress) result.compact_l2_l3(r_off);
    result.mask_tail_byte();
    return result;
}

// per-segment XOR
DDC
DDC::operator^(const DDC& other) const {
    assert(bit_count_ == other.bit_count_);

    DDC result;
    result.bit_count_ = bit_count_;
    result.segment_bits_ = segment_bits_;

    const size_t n = total_segments();

    // Dynamic zero-run bypass v2: XOR with a zero run is identity — both-zero
    // runs become implicit gaps; one-side-zero runs copy the other side's
    // range in one jump (gap-form aware for chained operands).
    if (ddc_zrb::enabled() && n > 0) {
        assert(segment_bits_ == other.segment_bits_);
        const id_vec_t* ia = sparse_form_ ? &seg_ids_ : nullptr;
        const id_vec_t* ib = other.sparse_form_ ? &other.seg_ids_ : nullptr;
        ensure_masks();                       // v3.1: cached per-bitmap masks
        other.ensure_masks();
        const uint64_t* za = zmask_.data();       const uint64_t* oa = omask_.data();
        const uint64_t* zb = other.zmask_.data(); const uint64_t* ob = other.omask_.data();
        // v3 absorption masks: g = 0^0 or 1^1 (gap), o = 1^0 or 0^1 (ones)
        const size_t nw = zmask_.size();
        uint64_t gsb[64], osb[64];          // 64 words = 4096 segments
        std::vector<uint64_t> gh, oh;       // heap fallback for huge grids
        uint64_t *g, *o;
        if (nw <= 64) { g = gsb; o = osb; }
        else { gh.resize(nw); oh.resize(nw); g = gh.data(); o = oh.data(); }
        size_t gap_bits = 0, ones_bits = 0;
        for (size_t w = 0; w < nw; w++) {
            g[w] = (za[w] & zb[w]) | (oa[w] & ob[w]);
            o[w] = (oa[w] & zb[w]) | (za[w] & ob[w]);
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
            if (ddc_zrb::test_bit(g, i)) {        // 0^0 / 1^1 → implicit gap
                const size_t L = ddc_zrb::run_len_set(g, i, n);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(o, i)) {        // 1^0 / 0^1 → virtual ones run
                const size_t L = ddc_zrb::run_len_set(o, i, n);
                if (!rones.empty() && rones.back().first + rones.back().second == i)
                    rones.back().second += (uint32_t)L;
                else
                    rones.emplace_back((uint32_t)i, (uint32_t)L);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(za, i)) {       // a zero → result = b range (b mixed)
                const size_t L = std::min({ddc_zrb::run_len_set(za, i, n),
                                           ddc_zrb::run_len_clear(zb, i, n),
                                           ddc_zrb::run_len_clear(ob, i, n)});
                vb.emit_range(i, i + L, result.segments_, rid);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(zb, i)) {       // b zero → result = a range (a mixed)
                const size_t L = std::min({ddc_zrb::run_len_set(zb, i, n),
                                           ddc_zrb::run_len_clear(za, i, n),
                                           ddc_zrb::run_len_clear(oa, i, n)});
                va.emit_range(i, i + L, result.segments_, rid);
                jl.jump(L);
                i += L;
                continue;
            }
            if (ddc_zrb::test_bit(oa, i)) {       // a ones, b mixed → ~b
                const DDCBtv& sb = *vb.at(i);
                if (collapse_uniform) emit_mixed(~sb, i);
            else { result.segments_.push_back(~sb); rid.push_back((uint32_t)i); }
                i++;
                continue;
            }
            if (ddc_zrb::test_bit(ob, i)) {       // b ones, a mixed → ~a
                const DDCBtv& sa = *va.at(i);
                if (collapse_uniform) emit_mixed(~sa, i);
            else { result.segments_.push_back(~sa); rid.push_back((uint32_t)i); }
                i++;
                continue;
            }
            const DDCBtv& sa = *va.at(i);         // both present & mixed here
            const DDCBtv& sb = *vb.at(i);
            if (&sa == &sb) { jl.jump(1); i++; continue; }
            if (sa.has_l2v() && sb.has_l2v()) {
                const int u = ddc_dual_probe('^', sa, sb);
                if (u == 0) { jl.jump(1); i++; continue; }
                if (u == 1) { if (!rones.empty() && rones.back().first + rones.back().second == i)
                    rones.back().second++;
                else rones.emplace_back((uint32_t)i, 1u);
                jl.jump(1); i++; continue; }
            }
            if (collapse_uniform) emit_mixed(sa ^ sb, i);
            else { result.segments_.push_back(sa ^ sb); rid.push_back((uint32_t)i); }
            i++;
        }

        ddc_zrb::set_run_hint(false);
        if (rid.size() == n) {
            result.sparse_form_ = false;
        } else {
            result.sparse_form_ = true;
            result.seg_ids_ = std::move(rid);
            result.ones_runs_ = std::move(rones);
        }
        if (ddc_zrb::debug_on())
            ddc_zrb::report("XOR", segment_bits_, n, zmask_, omask_, other.zmask_, other.omask_, jl);
        return result;
    }

    ensure_flat();
    other.ensure_flat();
    // legacy path (DDC_ZERO_RUN_BYPASS=0) — byte-for-byte the original loop
    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& sa = segments_[i];
        const auto& sb = other.segments_[i];

        // fast-path fills
        if (sa.is_all_zero()) { result.segments_.push_back(sb); continue; }
        if (sb.is_all_zero()) { result.segments_.push_back(sa); continue; }
        if (sa.is_all_ones()) { result.segments_.push_back(~sb); continue; }
        if (sb.is_all_ones()) { result.segments_.push_back(~sa); continue; }

        result.segments_.push_back(sa ^ sb);
    }

    return result;
}

DDCBtv&
DDCBtv::operator^=(const DDCBtv& other) {
    if (has_l2v_ || other.has_l2v_) {
        *this = dense_binop(other, '^');
        return *this;
    }

    *this = *this ^ other;
    return *this;
}

DDC&
DDC::operator^=(const DDC& other) {
    invalidate_masks();
    ensure_flat();
    other.ensure_flat();
    assert(bit_count_ == other.bit_count_);
    assert(segments_.size() == other.segments_.size());

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& seg = other.segments_[i];
        if (seg.is_all_zero()) continue;
        if (segments_[i].is_all_zero()) {
            segments_[i] = seg;
            continue;
        }
        if (seg.is_all_ones()) {
            segments_[i] = ~segments_[i];
            continue;
        }
        if (segments_[i].is_all_ones()) {
            segments_[i] = ~seg;
            continue;
        }
        segments_[i] = segments_[i] ^ seg;
    }
    return *this;
}
