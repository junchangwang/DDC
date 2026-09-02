#include "ddc.h"
#include "zero_run_bypass.h"

#include <atomic>
#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>

bool ddc_compress_results = false;

namespace {

struct DDCLoadStats {
    std::atomic<size_t> n     {0};
    std::atomic<size_t> l1    {0};
    std::atomic<size_t> l2    {0};
    std::atomic<size_t> l3    {0};
    std::atomic<size_t> l4    {0};
    std::atomic<size_t> total {0};

    void add(const DDC& cb) {
        auto sb = cb.size_breakdown();
        size_t b1 = (sb.l1_literal_bits + 7) / 8;
        size_t b2 = (sb.l2_literal_bits + 7) / 8;
        size_t b3 = (sb.l3_literal_bits + 7) / 8;
        size_t b4 = (sb.l4_bits         + 7) / 8;
        l1    += b1;
        l2    += b2;
        l3    += b3;
        l4    += b4;
        total += b1 + b2 + b3 + b4;
        n     += 1;
    }
};

DDCLoadStats g_load_stats;

__attribute__((destructor))
void ddc_print_load_breakdown() {
    if (g_load_stats.n.load() == 0) return;
    const char* off = std::getenv("DDC_NO_BREAKDOWN");
    if (off && *off && off[0] != '0') return;

    const double inv_mib = 1.0 / (1024.0 * 1024.0);

    std::fprintf(stdout,
        "  [Breakdown] DDC   L1=%.2f MiB  L2=%.2f MiB  L3=%.2f MiB"
        "  L4=%.2f MiB  total=%.2f MiB\n",
        g_load_stats.l1.load()    * inv_mib,
        g_load_stats.l2.load()    * inv_mib,
        g_load_stats.l3.load()    * inv_mib,
        g_load_stats.l4.load()    * inv_mib,
        g_load_stats.total.load() * inv_mib);
    std::fflush(stdout);
}

}

DDCBtv::DDCBtv(bool l1_fill_ones, bool l2_fill_ones, State state)
    : state_(state),
      l1_fill_ones_(l1_fill_ones), l2_fill_ones_(l2_fill_ones),
      bit_count_(0),
      l2_count_(0), l3_count_(0), l2_lit_count_(0),
      l3_fill_ones_(false),
      l4_count_(0), l3_lit_count_(0),
      l1_lit_count_(0) {}

DDCBtv
DDCBtv::make_all_fill(size_t bit_count, size_t l2_count, bool l1_fill_ones) {
    DDCBtv s(l1_fill_ones);
    s.bit_count_ = bit_count;
    s.l2_count_  = l2_count;
    s.l3_count_  = (l2_count + 7) / 8;
    return s;
}

bool ddc_dual_enabled() {
    static const bool v = [] {
        const char* e = std::getenv("DDC_DUAL_FILL");
        return !(e && *e == '0');
    }();
    return v;
}

size_t DDCBtv::collect_literals(uint32_t* words, uint8_t* vals, size_t cap) const {
    size_t n = 0, l3i = 0, l2i = 0, l1i = 0;
    for (size_t B = 0; B < l4_bits_.size(); B++) {
        uint8_t l4b = l4_bits_[B];
        while (l4b) {
            const int t = __builtin_ctz(l4b); l4b &= (uint8_t)(l4b - 1);
            uint8_t l3b = l3_lits_[l3i++];
            while (l3b) {
                const int u = __builtin_ctz(l3b); l3b &= (uint8_t)(l3b - 1);
                uint8_t l2b = l2_lits_[l2i++];
                while (l2b) {
                    const int s = __builtin_ctz(l2b); l2b &= (uint8_t)(l2b - 1);
                    if (n < cap) {
                        words[n] = (uint32_t)((((size_t)B * 8 + t) * 8 + u) * 8 + s);
                        vals[n] = l1_lits_[l1i];
                    }
                    n++; l1i++;
                }
            }
        }
    }
    return n;
}

DDCBtv DDCBtv::compress_dual(const std::vector<bool>& bits) {
    const size_t num_words = (bits.size() + 7) / 8;
    size_t zf = 0, of = 0;
    std::vector<uint8_t> wordbuf(num_words, 0);
    for (size_t i = 0; i < num_words; i++) {
        uint8_t word = 0;
        const size_t start = i * 8;
        for (size_t b = 0; b < 8 && start + b < bits.size(); b++)
            if (bits[start + b]) word |= uint8_t(1) << (7 - b);
        wordbuf[i] = word;
        const bool full = (start + 8 <= bits.size());
        if (word == 0x00) zf++;
        else if (word == 0xFF && full) of++;
    }
    if (zf == 0 || of == 0)
        return compress(bits, of > 0);

    DDCBtv result(false);
    result.bit_count_ = bits.size();
    result.l2_count_ = num_words;
    const size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);
    result.l2v_bits_.assign(l2_byte_count, 0);
    for (size_t i = 0; i < num_words; i++) {
        const uint8_t word = wordbuf[i];
        const bool full = ((i + 1) * 8 <= bits.size());
        if (word == 0x00) continue;
        if (word == 0xFF && full) {
            result.l2v_bits_[i / 8] |= uint8_t(1) << (i % 8);
            continue;
        }
        l2_flat[i / 8] |= uint8_t(1) << (i % 8);
        result.l1_lits_.push_back(word);
    }
    result.l1_lit_count_ = result.l1_lits_.size();
    result.has_l2v_ = true;

    result.l2_fill_ones_ = false;
    result.l3_count_ = l2_byte_count;
    const size_t l3_byte_count = (l2_byte_count + 7) / 8;
    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != 0x00) {
            result.l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            result.l2_lits_.push_back(l2_flat[i]);
        }
    }
    result.l2_lit_count_ = result.l2_lits_.size();
    result.compress_l3_to_l4();
    if (result.l3_fill_ones_)
        return compress(bits, false);
    result.l1_lits_.shrink_to_fit();
    result.l2_lits_.shrink_to_fit();
    return result;
}

DDCBtv DDCBtv::to_decompressed() const {
    if (state_ == State::Decompressed) return *this;
    DDCBtv d(false, true, State::Decompressed);
    d.bit_count_ = bit_count_;
    d.l2_count_  = l2_count_;
    d.l3_count_  = (l2_count_ + 7) / 8;
    d.l1_lits_.assign(l2_count_, 0x00);
    d.l1_lit_count_ = l2_count_;
    if (has_l2v_) {
        uint8_t* p = d.l1_lits_.data();
        size_t i = 0;
#ifdef __AVX512BW__
        for (; i + 64 <= l2_count_; i += 64) {
            uint64_t m;
            std::memcpy(&m, l2v_bits_.data() + i / 8, 8);
            _mm512_storeu_si512(p + i, _mm512_movm_epi8((__mmask64)m));
        }
#endif
        for (; i < l2_count_; i++)
            if ((l2v_bits_[i / 8] >> (i % 8)) & 1) p[i] = 0xFF;
        std::vector<uint32_t> w(l1_lit_count_);
        std::vector<uint8_t> v(l1_lit_count_);
        const size_t n = collect_literals(w.data(), v.data(), l1_lit_count_);
        for (size_t k = 0; k < n && k < l1_lit_count_; k++) p[w[k]] = v[k];
    } else {
        DDCBtv z = make_decompressed_zero(bit_count_, l2_count_);
        z |= *this;
        return z;
    }
    d.mask_tail_byte();
    return d;
}

DDCBtv DDCBtv::dense_binop(const DDCBtv& other, char op) const {
    DDCBtv da = to_decompressed();
    DDCBtv db = other.to_decompressed();
    uint8_t* pa = da.l1_lits_.data();
    const uint8_t* pb = db.l1_lits_.data();
    const size_t n = da.l1_lits_.size();
    for (size_t i = 0; i < n; i++) {
        switch (op) {
            case '|': pa[i] |= pb[i]; break;
            case '&': pa[i] &= pb[i]; break;
            default:  pa[i] ^= pb[i]; break;
        }
    }
    da.mask_tail_byte();
    return da;
}

int ddc_dual_probe(char op, const DDCBtv& a, const DDCBtv& b) {
    if (!a.has_l2v() || !b.has_l2v()) return -1;
    constexpr size_t CAP = 64;
    uint32_t wa[CAP], wb[CAP];
    uint8_t va[CAP], vb[CAP];
    const size_t na = a.collect_literals(wa, va, CAP);
    const size_t nb = b.collect_literals(wb, vb, CAP);
    if (na > CAP || nb > CAP) return -1;
    const size_t num_words = a.l2_count();
    const size_t nq = (num_words + 63) / 64;
    uint64_t litq_stack[128];
    std::vector<uint64_t> litq_heap;
    uint64_t* litq = litq_stack;
    if (nq > 128) { litq_heap.assign(nq, 0); litq = litq_heap.data(); }
    else std::memset(litq_stack, 0, nq * 8);
    for (size_t k = 0; k < na; k++) litq[wa[k] / 64] |= 1ull << (wa[k] % 64);
    for (size_t k = 0; k < nb; k++) litq[wb[k] / 64] |= 1ull << (wb[k] % 64);
    if (num_words % 64)
        litq[nq - 1] |= ~((1ull << (num_words % 64)) - 1);

    const uint8_t* pa = a.l2v_data();
    const uint8_t* pb = b.l2v_data();
    // Uniform probe
    uint64_t acc_or = 0, acc_not = 0;
    const size_t full_bytes = (num_words + 7) / 8;
    const uint8_t* litb = reinterpret_cast<const uint8_t*>(litq);
    size_t bi = 0;
#ifdef __AVX512F__
    {
        __m512i vor = _mm512_setzero_si512();
        __m512i vnot = _mm512_setzero_si512();
        const __m512i vones = _mm512_set1_epi8(static_cast<char>(0xFF));
        for (; bi + 64 <= full_bytes; bi += 64) {
            const __m512i qa = _mm512_loadu_si512(pa + bi);
            const __m512i qb = _mm512_loadu_si512(pb + bi);
            __m512i r;
            if (op == '|')      r = _mm512_or_si512(qa, qb);
            else if (op == '&') r = _mm512_and_si512(qa, qb);
            else                r = _mm512_xor_si512(qa, qb);
            const __m512i f = _mm512_loadu_si512(litb + bi);
            vor  = _mm512_or_si512(vor,  _mm512_andnot_si512(f, r));
            vnot = _mm512_or_si512(vnot, _mm512_andnot_si512(f,
                        _mm512_xor_si512(r, vones)));
        }
        acc_or  |= _mm512_reduce_or_epi64(vor);
        acc_not |= _mm512_reduce_or_epi64(vnot);
    }
#endif
    for (size_t q = bi / 8; q < nq; q++) {
        uint64_t qa = 0, qb = 0;
        const size_t bytes = std::min<size_t>(8, full_bytes - q * 8);
        std::memcpy(&qa, pa + q * 8, bytes);
        std::memcpy(&qb, pb + q * 8, bytes);
        uint64_t r;
        if (op == '|')      r = qa | qb;
        else if (op == '&') r = qa & qb;
        else                r = qa ^ qb;
        const uint64_t f = ~litq[q];
        acc_or  |= r & f;
        acc_not |= ~r & f;
    }
    bool all_zero = (acc_or == 0), all_ones = (acc_not == 0);
    if (!all_zero && !all_ones) return -1;
    const size_t rem = a.bit_count() % 8;
    auto val_at = [&](const DDCBtv& s, const uint32_t* w, const uint8_t* v,
                      size_t n, uint32_t word) -> uint8_t {
        for (size_t k = 0; k < n; k++) if (w[k] == word) return v[k];
        return ((s.l2v_data()[word / 8] >> (word % 8)) & 1) ? 0xFF : 0x00;
    };
    uint64_t seen[2 * CAP];
    size_t ns = 0;
    auto check_word = [&](uint32_t word) {
        for (size_t k = 0; k < ns; k++) if (seen[k] == word) return;
        seen[ns++] = word;
        const uint8_t xa = val_at(a, wa, va, na, word);
        const uint8_t xb = val_at(b, wb, vb, nb, word);
        uint8_t r;
        switch (op) {
            case '|': r = xa | xb; break;
            case '&': r = xa & xb; break;
            default:  r = xa ^ xb; break;
        }
        const bool last = (word + 1 == num_words);
        const uint8_t full = (last && rem) ? (uint8_t)(0xFF << (8 - rem)) : 0xFF;
        if (r != 0)    all_zero = false;
        if (r != full) all_ones = false;
    };
    for (size_t k = 0; k < na; k++) check_word(wa[k]);
    for (size_t k = 0; k < nb; k++) check_word(wb[k]);
    if (all_zero) return 0;
    if (all_ones) return 1;
    return -1;
}

int DDCBtv::uniform_class() const {
    if (has_l2v_) return -1;
    if (state_ == State::Compressed) {
        if (l1_lit_count_ == 0) return l1_fill_ones_ ? 1 : 0;
        return -1;
    }
    const uint8_t* p = l1_lits_.data();
    const size_t n = l1_lit_count_;
    if (n == 0) return l1_fill_ones_ ? 1 : 0;
    const size_t rem = bit_count_ % 8;
    const uint8_t tail_full = rem ? static_cast<uint8_t>(0xFF << (8 - rem))
                                  : 0xFF;
    if (n == 1)
        return p[0] == 0x00 ? 0 : (p[0] == tail_full ? 1 : -1);
    if (p[0] == 0x00) {
        size_t i = 0;
#ifdef __AVX512BW__
        for (; i + 64 <= n; i += 64) {
            const __m512i v = _mm512_loadu_si512(p + i);
            if (_mm512_test_epi8_mask(v, v)) return -1;
        }
#endif
        for (; i < n; i++)
            if (p[i] != 0) return -1;
        return 0;
    }
    if (p[0] == 0xFF) {
        size_t i = 0;
#ifdef __AVX512BW__
        const size_t n_avx = (n > 64) ? ((n - 1) / 64) * 64 : 0;
        const __m512i ff = _mm512_set1_epi8(static_cast<char>(0xFF));
        for (; i < n_avx; i += 64) {
            const __m512i v = _mm512_loadu_si512(p + i);
            if (_mm512_cmpneq_epi8_mask(v, ff)) return -1;
        }
#endif
        for (; i < n; i++) {
            const uint8_t full = (i + 1 == n) ? tail_full : 0xFF;
            if (p[i] != full) return -1;
        }
        return 1;
    }
    return -1;
}

DDCBtv
DDCBtv::make_decompressed_zero(size_t bit_count, size_t l2_count) {

    DDCBtv s( false,
                 true,
                State::Decompressed);
    s.bit_count_ = bit_count;
    s.l2_count_  = l2_count;
    s.l3_count_  = (l2_count + 7) / 8;
    s.l2_lit_count_ = 0;
    s.l1_lits_.assign(l2_count, 0x00);
    s.l1_lit_count_ = l2_count;
    return s;
}

uint64_t DDCBtv::get_literal(size_t idx) const {

    return l1_lits_[idx];
}

std::vector<uint8_t>
DDCBtv::expand_l3() const {
    if (state_ != State::Compressed) {
        size_t expected_bytes = (l3_count_ + 7) / 8;
        if (l3_bits_.size() == expected_bytes)
            return l3_bits_;
        return std::vector<uint8_t>(expected_bytes, 0x00);
    }

    size_t l3_byte_count = (l3_count_ + 7) / 8;
    uint8_t l3_fill_val = l3_fill_ones_ ? 0xFF : 0x00;
    std::vector<uint8_t> l3(l3_byte_count, l3_fill_val);
    size_t lit_idx = 0;
    for (size_t i = 0; i < l4_count_; i++) {
        uint8_t l4_byte = l4_bits_[i / 8];
        bool is_literal = (l4_byte >> (i % 8)) & 1;
        if (is_literal) {
            if (i < l3_byte_count)
                l3[i] = l3_lits_[lit_idx++];
        }
    }
    return l3;
}

std::vector<uint8_t>
DDCBtv::expand_l2() const {
    size_t l2_byte_count = (l2_count_ + 7) / 8;
    uint8_t l2_fill_val = l2_fill_ones_ ? 0xFF : 0x00;
    std::vector<uint8_t> l2(l2_byte_count, l2_fill_val);
    auto l3 = expand_l3();
    size_t lit_idx = 0;
    for (size_t i = 0; i < l3_count_; i++) {
        uint8_t l3_byte = l3[i / 8];
        bool is_literal = (l3_byte >> (i % 8)) & 1;
        if (is_literal) {
            if (i < l2_byte_count)
                l2[i] = l2_lits_[lit_idx++];
        }
    }
    return l2;
}

bool
DDCBtv::is_last_word_literal() const {
    size_t last_word = l2_count_ - 1;
    size_t l2_byte_idx = last_word / 8;
    size_t l3_byte_pos = l2_byte_idx / 8;
    size_t l3_bit_pos  = l2_byte_idx % 8;
    auto l3 = expand_l3();
    bool l3_lit = (l3[l3_byte_pos] >> l3_bit_pos) & 1;
    uint8_t l2_byte;
    if (l3_lit) {
        size_t l2_lit_idx = 0;
        for (size_t b = 0; b < l3_byte_pos; b++)
            l2_lit_idx += __builtin_popcount(l3[b]);
        if (l3_bit_pos > 0)
            l2_lit_idx += __builtin_popcount(
                l3[l3_byte_pos] & ((1u << l3_bit_pos) - 1));
        l2_byte = l2_lits_[l2_lit_idx];
    } else {
        l2_byte = l2_fill_ones_ ? 0xFF : 0x00;
    }
    return (l2_byte >> (last_word % 8)) & 1;
}

void
DDCBtv::compact_l2_l3(size_t actual_l1_count) {
    l1_lits_.resize(actual_l1_count);
    l1_lit_count_ = actual_l1_count;

    size_t l2_byte_count = (l2_count_ + 7) / 8;
    size_t l2_nonzero = 0;
    size_t l2_non_ff = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat_[i] != 0x00) l2_nonzero++;
        if (l2_flat_[i] != 0xFF) l2_non_ff++;
    }

    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    l2_fill_ones_ = best_l2_fill_lit;
    l3_count_ = l2_byte_count;
    l3_bits_.assign(l3_byte_count, 0);
    l2_lits_.clear();
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat_[i] != l2_fill_val) {
            l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            l2_lits_.push_back(l2_flat_[i]);
        }
    }
    l2_lit_count_ = l2_lits_.size();
    l2_flat_.clear();

    compress_l3_to_l4();

    state_ = State::Compressed;
}

void
DDCBtv::compress_l3_to_l4() {
    size_t l3_byte_count = l3_bits_.size();
    size_t l3_nonzero = 0;
    size_t l3_non_ff  = 0;
    for (size_t i = 0; i < l3_byte_count; i++) {
        if (l3_bits_[i] != 0x00) l3_nonzero++;
        if (l3_bits_[i] != 0xFF) l3_non_ff++;
    }

    bool best_l3_fill_lit = (l3_non_ff < l3_nonzero);
    uint8_t l3_fill_val = best_l3_fill_lit ? 0xFF : 0x00;

    size_t l4_byte_count = (l3_byte_count + 7) / 8;

    l3_fill_ones_ = best_l3_fill_lit;
    l4_count_ = l3_byte_count;
    l4_bits_.assign(l4_byte_count, 0);
    l3_lits_.clear();
    for (size_t i = 0; i < l3_byte_count; i++) {
        if (l3_bits_[i] != l3_fill_val) {
            l4_bits_[i / 8] |= uint8_t(1) << (i % 8);
            l3_lits_.push_back(l3_bits_[i]);
        }
    }
    l3_lit_count_ = l3_lits_.size();

    l3_bits_.clear();
    l3_bits_.shrink_to_fit();
    l3_lits_.shrink_to_fit();
}

// Sparse compression
DDCBtv
DDCBtv::compress_sparse_segment(const std::vector<uint16_t>& sorted_positions,
                                   size_t seg_bits, bool l1_fill_ones) {
    if (l1_fill_ones) {

        std::vector<bool> bits(seg_bits, true);
        for (uint16_t p : sorted_positions) bits[p] = false;
        return compress(bits, true);
    }

    DDCBtv result( false);
    result.bit_count_ = seg_bits;
    size_t num_words = (seg_bits + 7) / 8;
    if (num_words == 0) return result;
    result.l2_count_ = num_words;

    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    size_t i = 0;
    while (i < sorted_positions.size()) {
        size_t word_idx = sorted_positions[i] / 8;
        uint8_t word = 0;
        while (i < sorted_positions.size() &&
               (size_t(sorted_positions[i] / 8)) == word_idx) {
            uint16_t bit_in_byte = sorted_positions[i] % 8;
            word |= uint8_t(1) << (7 - bit_in_byte);
            i++;
        }

        l2_flat[word_idx / 8] |= uint8_t(1) << (word_idx % 8);
        result.l1_lits_.push_back(word);
    }
    result.l1_lit_count_ = result.l1_lits_.size();

    size_t l2_nonzero = 0, l2_non_ff = 0;
    for (size_t k = 0; k < l2_byte_count; k++) {
        if (l2_flat[k] != 0x00) l2_nonzero++;
        if (l2_flat[k] != 0xFF) l2_non_ff++;
    }
    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;
    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_     = l2_byte_count;
    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t k = 0; k < l2_byte_count; k++) {
        if (l2_flat[k] != l2_fill_val) {
            result.l3_bits_[k / 8] |= uint8_t(1) << (k % 8);
            result.l2_lits_.push_back(l2_flat[k]);
        }
    }
    result.l2_lit_count_ = result.l2_lits_.size();

    result.compress_l3_to_l4();

    result.l1_lits_.shrink_to_fit();
    result.l2_lits_.shrink_to_fit();
    return result;
}

DDCBtv
DDCBtv::compress(const std::vector<bool>& bits, bool l1_fill_ones) {
    DDCBtv result(l1_fill_ones);
    result.bit_count_ = bits.size();

    size_t num_words = (bits.size() + 7) / 8;
    if (num_words == 0) return result;

    result.l2_count_ = num_words;

    size_t l2_byte_count = (num_words + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);

    const uint8_t fill_byte = l1_fill_ones ? 0xFF : 0x00;

    for (size_t i = 0; i < num_words; i++) {

        uint8_t word = 0;
        size_t start = i * 8;
        for (size_t b = 0; b < 8 && start + b < bits.size(); b++) {
            if (bits[start + b])
                word |= uint8_t(1) << (7 - b);
        }
        if (word != fill_byte) {
            l2_flat[i / 8] |= uint8_t(1) << (i % 8);
            result.l1_lits_.push_back(word);
        }
    }
    result.l1_lit_count_ = result.l1_lits_.size();

    size_t l2_nonzero = 0;
    size_t l2_non_ff  = 0;
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != 0x00) l2_nonzero++;
        if (l2_flat[i] != 0xFF) l2_non_ff++;
    }

    bool best_l2_fill_lit = (l2_non_ff < l2_nonzero);
    uint8_t l2_fill_val = best_l2_fill_lit ? 0xFF : 0x00;

    size_t l3_byte_count = (l2_byte_count + 7) / 8;

    result.l2_fill_ones_ = best_l2_fill_lit;
    result.l3_count_ = l2_byte_count;

    result.l3_bits_.assign(l3_byte_count, 0);
    for (size_t i = 0; i < l2_byte_count; i++) {
        if (l2_flat[i] != l2_fill_val) {
            result.l3_bits_[i / 8] |= uint8_t(1) << (i % 8);
            result.l2_lits_.push_back(l2_flat[i]);
        }
    }
    result.l2_lit_count_ = result.l2_lits_.size();

    result.compress_l3_to_l4();

    result.l1_lits_.shrink_to_fit();
    result.l2_lits_.shrink_to_fit();

    return result;
}

std::vector<bool>
DDCBtv::decompress() const {
    assert(state_ != State::Uncompressed);
    std::vector<bool> result;
    result.reserve(bit_count_);

    auto l2 = expand_l2();

    size_t lit_idx = 0;
    for (size_t i = 0; i < l2_count_; i++) {
        uint8_t l2_byte = l2[i / 8];
        bool is_literal = (l2_byte >> (i % 8)) & 1;
        if (!is_literal) {
            const bool fv = has_l2v_
                ? ((l2v_bits_[i / 8] >> (i % 8)) & 1)
                : l1_fill_ones_;
            for (unsigned b = 0; b < 8; b++)
                result.push_back(fv);
        } else {
            uint8_t word = l1_lits_[lit_idx++];
            for (unsigned b = 0; b < 8; b++)
                result.push_back((word >> (7 - b)) & 1);
        }
    }

    result.resize(bit_count_);
    return result;
}

DDCBtv
DDCBtv::from_string(const std::string& bitstring, bool l1_fill_ones) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, l1_fill_ones);
}

std::string
DDCBtv::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size() + bits.size() / 8);
    for (size_t i = 0; i < bits.size(); i++) {
        if (i > 0 && i % 8 == 0) s += ' ';
        s += bits[i] ? '1' : '0';
    }
    return s;
}

size_t
DDCBtv::popcount() const {
    if (l2_count_ == 0) return 0;
    assert(state_ != State::Uncompressed);

    size_t fill_count = l2_count_ - l1_lit_count_;
    size_t count;
    if (has_l2v_) {
        size_t ones_fills = 0;
        for (uint8_t byte : l2v_bits_)
            ones_fills += (size_t)__builtin_popcount(byte);
        count = ones_fills * 8;
    } else {
        count = l1_fill_ones_ ? fill_count * 8 : 0;
    }

    const uint8_t* data = l1_lits_.data();
    size_t n = l1_lits_.size();
    size_t i = 0;

#ifdef __AVX512VPOPCNTDQ__
    __m512i acc = _mm512_setzero_si512();
    for (; i + 64 <= n; i += 64) {
        __m512i chunk = _mm512_loadu_si512(data + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(chunk));
    }
    count += _mm512_reduce_add_epi64(acc);
#endif

    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, data + i, 8);
        count += __builtin_popcountll(w);
    }
    for (; i < n; i++)
        count += __builtin_popcount(data[i]);

    if (bit_count_ % 8 != 0) {
        const size_t rem = bit_count_ % 8;
        const bool last_lit =
            state_ == State::Decompressed ? true : is_last_word_literal();
        if (!last_lit) {
            if (l1_fill_ones_) count -= 8 - rem;
        } else if (l1_lit_count_ > 0) {
            // Tail masking
            const uint8_t pad_mask =
                static_cast<uint8_t>(~static_cast<uint8_t>(0xFF << (8 - rem)));
            count -= __builtin_popcount(
                static_cast<uint8_t>(l1_lits_[l1_lit_count_ - 1] & pad_mask));
        }
    }

    return count;
}

std::vector<size_t>
DDCBtv::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) pos.push_back(i);
    }
    return pos;
}

DDCBtv::SizeBreakdown
DDCBtv::size_breakdown() const {
    SizeBreakdown sb{};
    sb.l3_bits         = l3_count_;
    sb.l4_bits         = l4_count_;
    sb.l3_literal_bits = l3_lit_count_ * 8;
    sb.l2_literal_bits = l2_lit_count_ * 8;
    sb.l1_literal_bits = l1_lit_count_ * 8;
    sb.total_bits      = sb.l4_bits + sb.l3_literal_bits
                       + sb.l2_literal_bits + sb.l1_literal_bits;
    return sb;
}

double
DDCBtv::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

size_t
DDCBtv::num_fills() const {
    return l2_count_ - l1_lit_count_;
}

void
DDCBtv::print(std::ostream& os) const {
    os << "DDCBtv compressed bitvector (4-level):\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Word size:     8 bits\n";
    os << "  L2 count: " << l2_count_ << " (words)\n";
    os << "  L3 count: " << l3_count_ << " bits\n";
    os << "  L4 count: " << l4_count_ << " bits\n";
    os << "  L3 literals: " << l3_lit_count_ << " bytes\n";
    os << "  L2 literals: " << l2_lit_count_ << " bytes\n";
    os << "  L1 literals: " << l1_lit_count_ << " words ("
       << l1_lit_count_ << " bytes)\n";

    auto sb = size_breakdown();
    os << "  Size breakdown: L4=" << sb.l4_bits
       << "  L3_lit=" << sb.l3_literal_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

DDC
DDC::compress(const std::vector<bool>& bits, bool l1_fill_ones,
                 size_t segment_bits, bool adaptive_l1_polarity) {
    DDC result;
    result.bit_count_ = bits.size();
    result.segment_bits_ = segment_bits;

    size_t offset = 0;
    while (offset < bits.size()) {
        size_t seg_len = std::min(segment_bits, bits.size() - offset);
        std::vector<bool> seg_bits(bits.begin() + static_cast<ptrdiff_t>(offset),
                                   bits.begin() + static_cast<ptrdiff_t>(offset + seg_len));
        if (adaptive_l1_polarity && ddc_dual_enabled()) {
            // Dual fills
            result.segments_.emplace_back(DDCBtv::compress_dual(seg_bits));
            offset += seg_len;
            continue;
        }
        bool polarity = l1_fill_ones;
        if (adaptive_l1_polarity) {
            // Segment polarity
            size_t ones = 0;
            for (bool b : seg_bits) ones += b;
            polarity = ones * 2 > seg_len;
        }
        result.segments_.emplace_back(
            DDCBtv::compress(seg_bits, polarity));
        offset += seg_len;
    }

    return result;
}

DDC
DDC::from_sparse_positions(const std::vector<uint32_t>& positions,
                              size_t num_rows,
                              size_t segment_bits) {
    DDC result;
    result.bit_count_ = num_rows;
    result.segment_bits_ = segment_bits;
    const size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;

    std::vector<std::vector<uint32_t>> seg_positions(num_segs);
    for (uint32_t p : positions) {
        seg_positions[p / segment_bits].push_back(p % segment_bits);
    }

    result.segments_.reserve(num_segs);
    for (size_t s = 0; s < num_segs; s++) {
        size_t seg_len = std::min(segment_bits, num_rows - s * segment_bits);
        if (seg_positions[s].empty()) {

            const size_t l2_count = (seg_len + 7) / 8;
            result.segments_.push_back(
                DDCBtv::make_decompressed_zero(seg_len, l2_count));
        } else {

            std::vector<bool> seg_bits(seg_len, false);
            for (uint32_t p : seg_positions[s]) seg_bits[p] = true;
            result.segments_.emplace_back(
                DDCBtv::compress(seg_bits,  false));
        }
    }

    return result;
}

std::vector<bool>
DDC::decompress() const {
    if (sparse_form_) {
        std::vector<bool> result(bit_count_, false);
        for (size_t k = 0; k < segments_.size(); k++) {
            const size_t off = static_cast<size_t>(seg_ids_[k]) * segment_bits_;
            auto seg_bits = segments_[k].decompress();
            for (size_t j = 0; j < seg_bits.size(); j++)
                if (seg_bits[j]) result[off + j] = true;
        }
        for (const auto& run : ones_runs_) {
            for (uint32_t id = run.first; id < run.first + run.second; id++) {
                const size_t off  = static_cast<size_t>(id) * segment_bits_;
                const size_t bits = std::min(segment_bits_, bit_count_ - off);
                for (size_t j = 0; j < bits; j++) result[off + j] = true;
            }
        }
        return result;
    }
    std::vector<bool> result;
    result.reserve(bit_count_);

    for (const auto& seg : segments_) {
        auto seg_bits = seg.decompress();
        result.insert(result.end(), seg_bits.begin(), seg_bits.end());
    }

    return result;
}

// Cached masks
void
DDC::ensure_masks() const {
    if (masks_valid_) return;
    const size_t n = total_segments();
    ddc_zrb::build_masks_any(segments_,
                             sparse_form_ ? &seg_ids_ : nullptr,
                             sparse_form_ ? &ones_runs_ : nullptr,
                             zmask_, omask_, n);
    masks_valid_ = true;
}

// Canonicalize
void
DDC::ensure_flat() const {
    if (!sparse_form_) return;
    const size_t n = total_segments();
    seg_vec_t flat;
    flat.reserve(n);
    size_t k = 0, m = 0;
    for (size_t i = 0; i < n; i++) {
        if (k < seg_ids_.size() && seg_ids_[k] == i) {
            flat.push_back(std::move(segments_[k]));
            k++;
        } else if (m < ones_runs_.size() && i >= ones_runs_[m].first
                   && i < (size_t)ones_runs_[m].first + ones_runs_[m].second) {
            const size_t bits = std::min(segment_bits_, bit_count_ - i * segment_bits_);
            flat.push_back(DDCBtv::make_all_fill(bits, (bits + 7) / 8, true));
            if (i + 1 == (size_t)ones_runs_[m].first + ones_runs_[m].second) m++;
        } else {
            const size_t bits = std::min(segment_bits_, bit_count_ - i * segment_bits_);
            flat.push_back(DDCBtv::make_all_fill(bits, (bits + 7) / 8, false));
        }
    }
    segments_ = std::move(flat);
    seg_ids_.clear();
    seg_ids_.shrink_to_fit();
    ones_runs_.clear();
    ones_runs_.shrink_to_fit();
    sparse_form_ = false;
}

DDC
DDC::from_string(const std::string& bitstring, bool l1_fill_ones,
                    size_t segment_bits) {
    std::vector<bool> bits;
    bits.reserve(bitstring.size());
    for (char c : bitstring) {
        if (c == '0')      bits.push_back(false);
        else if (c == '1') bits.push_back(true);
    }
    return compress(bits, l1_fill_ones, segment_bits);
}

std::string
DDC::to_string() const {
    auto bits = decompress();
    std::string s;
    s.reserve(bits.size());
    for (size_t i = 0; i < bits.size(); i++)
        s += bits[i] ? '1' : '0';
    return s;
}

size_t
DDC::popcount() const {
    size_t count = 0;
    for (const auto& seg : segments_)
        count += seg.popcount();
    for (const auto& run : ones_runs_)
        for (uint32_t id = run.first; id < run.first + run.second; id++)
            count += std::min(segment_bits_,
                              bit_count_ - static_cast<size_t>(id) * segment_bits_);
    return count;
}

std::vector<size_t>
DDC::set_bit_positions() const {
    auto bits = decompress();
    std::vector<size_t> pos;
    for (size_t i = 0; i < bits.size(); i++)
        if (bits[i]) pos.push_back(i);
    return pos;
}

DDC::SizeBreakdown
DDC::size_breakdown() const {
    ensure_flat();
    SizeBreakdown sb{0, 0, 0, 0, 0, 0};
    for (const auto& seg : segments_) {
        auto ssb = seg.size_breakdown();
        sb.l3_bits         += ssb.l3_bits;
        sb.l2_literal_bits += ssb.l2_literal_bits;
        sb.l1_literal_bits += ssb.l1_literal_bits;
        sb.total_bits      += ssb.total_bits;
        sb.l4_bits         += ssb.l4_bits;
        sb.l3_literal_bits += ssb.l3_literal_bits;
    }
    return sb;
}

double
DDC::compression_ratio() const {
    size_t cb = compressed_size_bits();
    return cb > 0 ? static_cast<double>(bit_count_) / cb : 0.0;
}

void
DDC::print(std::ostream& os) const {
    ensure_flat();
    os << "DDC segmented bitvector:\n";
    os << "  Original size: " << bit_count_ << " bits\n";
    os << "  Segment size:  " << segment_bits_ << " bits\n";
    os << "  Num segments:  " << segments_.size() << "\n";

    for (size_t i = 0; i < segments_.size(); i++) {
        const auto& s = segments_[i];
        os << "  Segment " << i << ": "
           << "DDCBtv"
           << " l1_fill_ones=" << s.l1_fill_ones()
           << " bits=" << s.bit_count()
           << " fills=" << s.num_fills()
           << " literals=" << s.num_lits()
           << "\n";
    }

    auto sb = size_breakdown();
    os << "  Size breakdown: L4=" << sb.l4_bits
       << "  L3_lit=" << sb.l3_literal_bits
       << "  L2_lit=" << sb.l2_literal_bits
       << "  L1_lit=" << sb.l1_literal_bits
       << "  total=" << sb.total_bits << " bits\n";
    os << "  Compression ratio: " << std::fixed << std::setprecision(2)
       << compression_ratio() << "x\n";
}

template<typename T>
static void write_val(std::ostream& os, T val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

template<typename T>
static T read_val(std::istream& is) {
    T val;
    is.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

static constexpr uint8_t DDC_FMT_V1 = 8;
static constexpr uint8_t DDC_FMT_V2 = 9;

// V2 format
void DDCBtv::serialize(std::ostream& os) const {
    assert(!has_l2v_ && "dual segments serialize via v5 type-3 only");
    assert(state_ == State::Compressed);
    write_val<uint8_t>(os, DDC_FMT_V2);
    write_val<uint8_t>(os, l1_fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, l2_fill_ones_ ? 1 : 0);
    write_val<uint8_t>(os, l3_fill_ones_ ? 1 : 0);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, l2_count_);
    write_val<uint64_t>(os, l3_count_);

    write_val<uint64_t>(os, l4_count_);
    uint64_t l4_bytes = l4_bits_.size();
    write_val<uint64_t>(os, l4_bytes);
    if (l4_bytes > 0)
        os.write(reinterpret_cast<const char*>(l4_bits_.data()), l4_bytes);

    write_val<uint64_t>(os, l3_lit_count_);
    if (l3_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_lits_.data()), l3_lit_count_);

    write_val<uint64_t>(os, l2_lit_count_);
    if (l2_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_lits_.data()), l2_lit_count_);

    write_val<uint64_t>(os, l1_lit_count_);
    if (l1_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_lits_.data()), l1_lit_count_);
}

DDCBtv DDCBtv::deserialize(std::istream& is) {
    uint8_t fmt = read_val<uint8_t>(is);
    if (fmt != DDC_FMT_V1 && fmt != DDC_FMT_V2)
        throw std::runtime_error("DDCBtv::deserialize: unknown format tag " +
                                 std::to_string(fmt));
    uint8_t fo = read_val<uint8_t>(is);
    uint8_t fl = read_val<uint8_t>(is);

    DDCBtv btv(fo != 0, fl != 0);

    if (fmt == DDC_FMT_V2) {

        uint8_t fl3 = read_val<uint8_t>(is);
        btv.l3_fill_ones_ = (fl3 != 0);
        btv.bit_count_ = read_val<uint64_t>(is);
        btv.l2_count_  = read_val<uint64_t>(is);
        btv.l3_count_  = read_val<uint64_t>(is);

        btv.l4_count_ = read_val<uint64_t>(is);
        uint64_t l4_bytes = read_val<uint64_t>(is);
        btv.l4_bits_.resize(l4_bytes);
        if (l4_bytes > 0)
            is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);

        btv.l3_lit_count_ = read_val<uint64_t>(is);
        btv.l3_lits_.resize(btv.l3_lit_count_);
        if (btv.l3_lit_count_ > 0)
            is.read(reinterpret_cast<char*>(btv.l3_lits_.data()), btv.l3_lit_count_);
    } else {

        btv.bit_count_ = read_val<uint64_t>(is);
        btv.l2_count_  = read_val<uint64_t>(is);
        btv.l3_count_  = read_val<uint64_t>(is);
        uint64_t l3_bytes = read_val<uint64_t>(is);
        btv.l3_bits_.resize(l3_bytes);
        if (l3_bytes > 0)
            is.read(reinterpret_cast<char*>(btv.l3_bits_.data()), l3_bytes);
    }

    btv.l2_lit_count_ = read_val<uint64_t>(is);
    btv.l2_lits_.resize(btv.l2_lit_count_);
    if (btv.l2_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_lits_.data()), btv.l2_lit_count_);

    btv.l1_lit_count_ = read_val<uint64_t>(is);
    btv.l1_lits_.resize(btv.l1_lit_count_);
    if (btv.l1_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_lits_.data()), btv.l1_lit_count_);

    if (fmt == DDC_FMT_V1)
        btv.compress_l3_to_l4();

    return btv;
}

void DDC::serialize(std::ostream& os) const {
    ensure_flat();
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());

    for (const auto& seg : segments_)
        seg.serialize(os);
}

DDC DDC::deserialize(std::istream& is) {
    DDC cb;
    cb.bit_count_ = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);

    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize(is));

    g_load_stats.add(cb);
    return cb;
}

static constexpr uint8_t DDC_FMT_V3_MAGIC = 0xCB;

void DDCBtv::serialize_packed(std::ostream& os, bool is_last_seg) const {
    assert(!has_l2v_ && "dual segments serialize via v5 type-3 only");
    assert(state_ == State::Compressed);

    uint8_t flags = 0;
    if (l1_fill_ones_) flags |= 0x01;
    if (l2_fill_ones_) flags |= 0x02;
    if (l3_fill_ones_) flags |= 0x04;
    if (is_last_seg)   flags |= 0x08;
    write_val<uint8_t>(os, flags);

    if (is_last_seg)
        write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));

    uint32_t l4_bytes = static_cast<uint32_t>(l4_bits_.size());
    write_val<uint32_t>(os, l4_bytes);
    write_val<uint32_t>(os, static_cast<uint32_t>(l3_lit_count_));
    write_val<uint32_t>(os, static_cast<uint32_t>(l2_lit_count_));
    write_val<uint32_t>(os, static_cast<uint32_t>(l1_lit_count_));

    if (l4_bytes > 0)
        os.write(reinterpret_cast<const char*>(l4_bits_.data()), l4_bytes);
    if (l3_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_lits_.data()), l3_lit_count_);
    if (l2_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_lits_.data()), l2_lit_count_);
    if (l1_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_lits_.data()), l1_lit_count_);
}

DDCBtv DDCBtv::deserialize_packed(std::istream& is, size_t segment_bits) {
    uint8_t flags = read_val<uint8_t>(is);
    bool l1_fo  = (flags & 0x01) != 0;
    bool l2_fo  = (flags & 0x02) != 0;
    bool l3_fo  = (flags & 0x04) != 0;
    bool islast = (flags & 0x08) != 0;

    DDCBtv btv(l1_fo, l2_fo);
    btv.l3_fill_ones_ = l3_fo;

    btv.bit_count_ = islast ? read_val<uint32_t>(is) : segment_bits;
    btv.l2_count_  = (btv.bit_count_ + 7) / 8;
    btv.l3_count_  = (btv.l2_count_  + 7) / 8;
    btv.l4_count_  = (btv.l3_count_  + 7) / 8;

    uint32_t l4_bytes   = read_val<uint32_t>(is);
    btv.l3_lit_count_   = read_val<uint32_t>(is);
    btv.l2_lit_count_   = read_val<uint32_t>(is);
    btv.l1_lit_count_   = read_val<uint32_t>(is);

    btv.l4_bits_.resize(l4_bytes);
    if (l4_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);
    btv.l3_lits_.resize(btv.l3_lit_count_);
    if (btv.l3_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l3_lits_.data()), btv.l3_lit_count_);
    btv.l2_lits_.resize(btv.l2_lit_count_);
    if (btv.l2_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_lits_.data()), btv.l2_lit_count_);
    btv.l1_lits_.resize(btv.l1_lit_count_);
    if (btv.l1_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_lits_.data()), btv.l1_lit_count_);
    return btv;
}

void DDC::serialize_packed(std::ostream& os) const {
    ensure_flat();
    write_val<uint8_t>(os, DDC_FMT_V3_MAGIC);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());
    for (size_t i = 0; i < segments_.size(); i++)
        segments_[i].serialize_packed(os,  (i + 1 == segments_.size()));
}

DDC DDC::deserialize_packed(std::istream& is) {
    uint8_t magic = read_val<uint8_t>(is);
    if (magic != DDC_FMT_V3_MAGIC)
        throw std::runtime_error("DDC::deserialize_packed: bad magic "
                                 + std::to_string(magic));
    DDC cb;
    cb.bit_count_    = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);
    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize_packed(is, cb.segment_bits_));
    return cb;
}

static constexpr uint8_t DDC_FMT_V4_MAGIC = 0xC4;

// V4 format
void DDCBtv::serialize_v4(std::ostream& os, bool is_last_seg) const {
    assert(!has_l2v_ && "dual segments serialize via v5 type-3 only");
    assert(state_ == State::Compressed);

    uint8_t flags = 0;
    if (l1_fill_ones_)            flags |= 0x01;
    if (l2_fill_ones_)            flags |= 0x02;
    if (l3_fill_ones_)            flags |= 0x04;
    if (is_last_seg)              flags |= 0x08;
    if (l1_lit_count_ > 0)        flags |= 0x10;
    if (l2_lit_count_ > 0)        flags |= 0x20;
    if (l3_lit_count_ > 0)        flags |= 0x40;
    write_val<uint8_t>(os, flags);

    if (is_last_seg)
        write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));

    if (l3_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l3_lit_count_));
    if (l2_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l2_lit_count_));
    if (l1_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l1_lit_count_));

    // Hollow segments
    {
        const size_t l4_cnt    = (l3_count_ + 7) / 8;
        const size_t l4_expect = (l4_cnt + 7) / 8;
        const size_t have      = std::min(l4_bits_.size(), l4_expect);
        if (have > 0)
            os.write(reinterpret_cast<const char*>(l4_bits_.data()), have);
        for (size_t i = have; i < l4_expect; i++)
            os.put('\0');
    }

    if (l3_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l3_lits_.data()), l3_lit_count_);
    if (l2_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l2_lits_.data()), l2_lit_count_);
    if (l1_lit_count_ > 0)
        os.write(reinterpret_cast<const char*>(l1_lits_.data()), l1_lit_count_);
}

DDCBtv DDCBtv::deserialize_v4(std::istream& is, size_t segment_bits) {
    uint8_t flags = read_val<uint8_t>(is);
    bool l1_fo       = (flags & 0x01) != 0;
    bool l2_fo       = (flags & 0x02) != 0;
    bool l3_fo       = (flags & 0x04) != 0;
    bool islast      = (flags & 0x08) != 0;
    bool has_l1_lit  = (flags & 0x10) != 0;
    bool has_l2_lit  = (flags & 0x20) != 0;
    bool has_l3_lit  = (flags & 0x40) != 0;

    DDCBtv btv(l1_fo, l2_fo);
    btv.l3_fill_ones_ = l3_fo;

    btv.bit_count_ = islast ? read_val<uint32_t>(is) : segment_bits;
    btv.l2_count_  = (btv.bit_count_ + 7) / 8;
    btv.l3_count_  = (btv.l2_count_  + 7) / 8;
    btv.l4_count_  = (btv.l3_count_  + 7) / 8;
    size_t l4_bytes = (btv.l4_count_ + 7) / 8;

    btv.l3_lit_count_ = has_l3_lit ? read_val<uint32_t>(is) : 0;
    btv.l2_lit_count_ = has_l2_lit ? read_val<uint32_t>(is) : 0;
    btv.l1_lit_count_ = has_l1_lit ? read_val<uint32_t>(is) : 0;

    btv.l4_bits_.resize(l4_bytes);
    if (l4_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);
    btv.l3_lits_.resize(btv.l3_lit_count_);
    if (btv.l3_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l3_lits_.data()), btv.l3_lit_count_);
    btv.l2_lits_.resize(btv.l2_lit_count_);
    if (btv.l2_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l2_lits_.data()), btv.l2_lit_count_);
    btv.l1_lits_.resize(btv.l1_lit_count_);
    if (btv.l1_lit_count_ > 0)
        is.read(reinterpret_cast<char*>(btv.l1_lits_.data()), btv.l1_lit_count_);
    return btv;
}

static constexpr uint8_t DDC_FMT_V5_MAGIC = 0xC5;

static void rle_encode_bytes(const uint8_t* p, size_t n, std::ostream& os) {
    uint64_t enc_len = 0;
    std::ostringstream body;
    size_t i = 0;
    while (i < n) {
        size_t run = 1;
        while (i + run < n && p[i + run] == p[i] && run < 0x7FFF) run++;
        if (run >= 4) {
            uint16_t hdr = static_cast<uint16_t>(0x8000u | run);
            body.write(reinterpret_cast<const char*>(&hdr), 2);
            body.put(static_cast<char>(p[i]));
            enc_len += 3;
            i += run;
        } else {
            size_t start = i;
            size_t lit = 0;
            while (i < n && lit < 0x7FFF) {
                size_t r = 1;
                while (i + r < n && p[i + r] == p[i] && r < 0x7FFF) r++;
                if (r >= 4) break;
                i += r; lit += r;
            }
            uint16_t hdr = static_cast<uint16_t>(lit);
            body.write(reinterpret_cast<const char*>(&hdr), 2);
            body.write(reinterpret_cast<const char*>(p + start), lit);
            enc_len += 2 + lit;
        }
    }
    write_val<uint32_t>(os, static_cast<uint32_t>(enc_len));
    const std::string s = body.str();
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

static void rle_decode_bytes(std::istream& is, uint8_t* out, size_t n) {
    uint32_t enc_len = read_val<uint32_t>(is);
    size_t produced = 0;
    uint32_t consumed = 0;
    while (consumed < enc_len) {
        uint16_t hdr = read_val<uint16_t>(is);
        consumed += 2;
        if (hdr & 0x8000u) {
            size_t run = hdr & 0x7FFFu;
            uint8_t v = read_val<uint8_t>(is);
            consumed += 1;
            if (produced + run > n)
                throw std::runtime_error("DDC v5: l1 RLE overflow");
            std::memset(out + produced, v, run);
            produced += run;
        } else {
            size_t lit = hdr;
            if (produced + lit > n)
                throw std::runtime_error("DDC v5: l1 RLE overflow");
            is.read(reinterpret_cast<char*>(out + produced),
                    static_cast<std::streamsize>(lit));
            consumed += static_cast<uint32_t>(lit);
            produced += lit;
        }
    }
    if (produced != n)
        throw std::runtime_error("DDC v5: l1 RLE length mismatch");
}

void DDCBtv::serialize_v5(std::ostream& os, bool is_last_seg) const {
    if (state_ == State::Decompressed) {
        uint8_t flags = 0x80;
        if (l1_fill_ones_) flags |= 0x01;
        if (l2_fill_ones_) flags |= 0x02;
        if (is_last_seg)   flags |= 0x08;
        write_val<uint8_t>(os, flags);
        if (is_last_seg)
            write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));
        rle_encode_bytes(l1_lits_.data(), l1_lit_count_, os);
        return;
    }
    assert(state_ == State::Compressed);

    uint8_t flags = 0;
    if (l1_fill_ones_)            flags |= 0x01;
    if (l2_fill_ones_)            flags |= 0x02;
    if (l3_fill_ones_)            flags |= 0x04;
    if (is_last_seg)              flags |= 0x08;
    if (l1_lit_count_ > 0)        flags |= 0x10;
    if (l2_lit_count_ > 0)        flags |= 0x20;
    if (l3_lit_count_ > 0)        flags |= 0x40;
    write_val<uint8_t>(os, flags);

    if (is_last_seg)
        write_val<uint32_t>(os, static_cast<uint32_t>(bit_count_));

    if (l3_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l3_lit_count_));
    if (l2_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l2_lit_count_));
    if (l1_lit_count_ > 0)
        write_val<uint32_t>(os, static_cast<uint32_t>(l1_lit_count_));

    {
        const size_t l4_cnt    = (l3_count_ + 7) / 8;
        const size_t l4_expect = (l4_cnt + 7) / 8;
        const size_t have      = std::min(l4_bits_.size(), l4_expect);
        if (have > 0)
            os.write(reinterpret_cast<const char*>(l4_bits_.data()), have);
        for (size_t i = have; i < l4_expect; i++)
            os.put('\0');
    }

    if (l3_lit_count_ > 0)
        rle_encode_bytes(l3_lits_.data(), l3_lit_count_, os);
    if (l2_lit_count_ > 0)
        rle_encode_bytes(l2_lits_.data(), l2_lit_count_, os);
    if (l1_lit_count_ > 0)
        rle_encode_bytes(l1_lits_.data(), l1_lit_count_, os);
}

DDCBtv DDCBtv::deserialize_v5(std::istream& is, size_t segment_bits) {
    uint8_t flags = read_val<uint8_t>(is);
    if (flags & 0x80) {
        DDCBtv btv((flags & 0x01) != 0, (flags & 0x02) != 0,
                   State::Decompressed);
        btv.bit_count_ = (flags & 0x08) ? read_val<uint32_t>(is)
                                        : segment_bits;
        btv.l2_count_  = (btv.bit_count_ + 7) / 8;
        btv.l3_count_  = (btv.l2_count_  + 7) / 8;
        btv.l1_lits_.resize(btv.l2_count_);
        btv.l1_lit_count_ = btv.l2_count_;
        rle_decode_bytes(is, btv.l1_lits_.data(), btv.l1_lit_count_);
        return btv;
    }
    bool l1_fo       = (flags & 0x01) != 0;
    bool l2_fo       = (flags & 0x02) != 0;
    bool l3_fo       = (flags & 0x04) != 0;
    bool islast      = (flags & 0x08) != 0;
    bool has_l1_lit  = (flags & 0x10) != 0;
    bool has_l2_lit  = (flags & 0x20) != 0;
    bool has_l3_lit  = (flags & 0x40) != 0;

    DDCBtv btv(l1_fo, l2_fo);
    btv.l3_fill_ones_ = l3_fo;

    btv.bit_count_ = islast ? read_val<uint32_t>(is) : segment_bits;
    btv.l2_count_  = (btv.bit_count_ + 7) / 8;
    btv.l3_count_  = (btv.l2_count_  + 7) / 8;
    btv.l4_count_  = (btv.l3_count_  + 7) / 8;
    size_t l4_bytes = (btv.l4_count_ + 7) / 8;

    btv.l3_lit_count_ = has_l3_lit ? read_val<uint32_t>(is) : 0;
    btv.l2_lit_count_ = has_l2_lit ? read_val<uint32_t>(is) : 0;
    btv.l1_lit_count_ = has_l1_lit ? read_val<uint32_t>(is) : 0;

    btv.l4_bits_.resize(l4_bytes);
    if (l4_bytes > 0)
        is.read(reinterpret_cast<char*>(btv.l4_bits_.data()), l4_bytes);
    btv.l3_lits_.resize(btv.l3_lit_count_);
    if (btv.l3_lit_count_ > 0)
        rle_decode_bytes(is, btv.l3_lits_.data(), btv.l3_lit_count_);
    btv.l2_lits_.resize(btv.l2_lit_count_);
    if (btv.l2_lit_count_ > 0)
        rle_decode_bytes(is, btv.l2_lits_.data(), btv.l2_lit_count_);
    btv.l1_lits_.resize(btv.l1_lit_count_);
    if (btv.l1_lit_count_ > 0)
        rle_decode_bytes(is, btv.l1_lits_.data(), btv.l1_lit_count_);
    return btv;
}

void DDC::serialize_v5(std::ostream& os) const {
    ensure_flat();
    write_val<uint8_t>(os, DDC_FMT_V5_MAGIC);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());
    const size_t n = segments_.size();
    size_t i = 0;
    while (i < n) {
        const DDCBtv& seg = segments_[i];
        const bool pure = (seg.num_lits() == 0) && !seg.has_l2v();
        if (pure) {
            const bool pol = seg.is_all_ones();
            size_t j = i;
            while (j < n && segments_[j].num_lits() == 0
                   && !segments_[j].has_l2v()
                   && segments_[j].is_all_ones() == pol)
                j++;
            write_val<uint8_t>(os, pol ? 1 : 0);
            write_val<uint32_t>(os, static_cast<uint32_t>(j - i));
            i = j;
        } else {
            write_val<uint8_t>(os, seg.has_l2v() ? 3 : 2);
            seg.serialize_v5(os, i + 1 == n);
            if (seg.has_l2v())
                rle_encode_bytes(seg.l2v_data(), (seg.l2_count() + 7) / 8, os);
            i++;
        }
    }
}

DDC DDC::deserialize_v5(std::istream& is) {
    uint8_t magic = read_val<uint8_t>(is);
    if (magic != DDC_FMT_V5_MAGIC)
        throw std::runtime_error("DDC::deserialize_v5: bad magic "
                                 + std::to_string(magic));
    DDC cb;
    cb.bit_count_    = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);
    while (cb.segments_.size() < num_segs) {
        uint8_t type = read_val<uint8_t>(is);
        if (type == 2 || type == 3) {
            DDCBtv seg = DDCBtv::deserialize_v5(is, cb.segment_bits_);
            if (type == 3) {
                seg.l2v_bits_.assign((seg.l2_count_ + 7) / 8, 0);
                rle_decode_bytes(is, seg.l2v_bits_.data(),
                                 seg.l2v_bits_.size());
                seg.has_l2v_ = true;
            }
            cb.segments_.push_back(std::move(seg));
        } else if (type == 0 || type == 1) {
            uint32_t run = read_val<uint32_t>(is);
            if (cb.segments_.size() + run > num_segs)
                throw std::runtime_error("DDC v5: fill run overflows directory");
            for (uint32_t k = 0; k < run; k++) {
                const size_t idx = cb.segments_.size();
                const size_t off = idx * cb.segment_bits_;
                const size_t len = std::min(cb.segment_bits_,
                                            cb.bit_count_ - off);
                cb.segments_.push_back(DDCBtv::make_all_fill(
                    len, (len + 7) / 8, type == 1));
            }
        } else {
            throw std::runtime_error("DDC v5: unknown record type "
                                     + std::to_string(type));
        }
    }
    return cb;
}

DDC DDC::load_any(std::istream& is) {
    const int first = is.peek();
    if (first == DDC_FMT_V5_MAGIC) return deserialize_v5(is);
    if (first == DDC_FMT_V4_MAGIC) return deserialize_v4(is);
    if (first == DDC_FMT_V3_MAGIC) return deserialize_packed(is);
    return deserialize(is);
}

void DDC::serialize_v4(std::ostream& os) const {
    ensure_flat();
    write_val<uint8_t>(os, DDC_FMT_V4_MAGIC);
    write_val<uint64_t>(os, bit_count_);
    write_val<uint64_t>(os, segment_bits_);
    write_val<uint64_t>(os, segments_.size());
    for (size_t i = 0; i < segments_.size(); i++)
        segments_[i].serialize_v4(os,  (i + 1 == segments_.size()));
}

DDC DDC::deserialize_v4(std::istream& is) {
    uint8_t magic = read_val<uint8_t>(is);
    if (magic != DDC_FMT_V4_MAGIC)
        throw std::runtime_error("DDC::deserialize_v4: bad magic "
                                 + std::to_string(magic));
    DDC cb;
    cb.bit_count_    = read_val<uint64_t>(is);
    cb.segment_bits_ = read_val<uint64_t>(is);
    uint64_t num_segs = read_val<uint64_t>(is);
    cb.segments_.reserve(num_segs);
    for (uint64_t i = 0; i < num_segs; i++)
        cb.segments_.push_back(DDCBtv::deserialize_v4(is, cb.segment_bits_));
    return cb;
}

SparseDDC
SparseDDC::from_positions(const std::vector<uint32_t>& positions,
                             size_t num_rows,
                             size_t segment_bits) {
    SparseDDC s;
    s.bit_count_    = num_rows;
    s.segment_bits_ = segment_bits;
    s.num_set_bits_ = positions.size();

    if (positions.empty()) return s;

    std::map<uint32_t, std::vector<uint32_t>> by_seg;
    for (uint32_t p : positions) {
        by_seg[static_cast<uint32_t>(p / segment_bits)]
            .push_back(static_cast<uint32_t>(p % segment_bits));
    }

    s.seg_indices_.reserve(by_seg.size());
    s.seg_data_.reserve(by_seg.size());
    std::vector<uint16_t> sorted_pos;
    for (auto& [seg_idx, local_pos] : by_seg) {
        size_t seg_off = static_cast<size_t>(seg_idx) * segment_bits;
        size_t seg_len = std::min(segment_bits, num_rows - seg_off);

        sorted_pos.clear();
        sorted_pos.reserve(local_pos.size());
        for (uint32_t p : local_pos) sorted_pos.push_back(static_cast<uint16_t>(p));
        std::sort(sorted_pos.begin(), sorted_pos.end());
        s.seg_indices_.push_back(seg_idx);
        s.seg_data_.emplace_back(
            DDCBtv::compress_sparse_segment(sorted_pos, seg_len,
                                                false));
    }
    return s;
}

void
SparseDDC::apply_or_to(DDC& dst) const {
    assert(dst.bit_count() == bit_count_);
    assert(dst.segment_bits() == segment_bits_);
    auto& dst_segs = dst.segments();
    for (size_t i = 0; i < seg_indices_.size(); i++) {
        uint32_t idx = seg_indices_[i];
        const DDCBtv& src = seg_data_[i];
        if (src.is_all_zero()) continue;
        DDCBtv& d = dst_segs[idx];
        if (d.is_all_zero()) {

            if (d.state() == DDCBtv::State::Decompressed) {
                d |= src;
            } else {
                d = src;
            }
            continue;
        }
        if (d.is_all_ones()) continue;
        if (src.is_all_ones()) {
            d = DDCBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
            continue;
        }

        if (d.state() == DDCBtv::State::Compressed &&
            src.state() == DDCBtv::State::Compressed) {
            d = d | src;
        } else {

            if (d.state() != DDCBtv::State::Decompressed) {
                DDCBtv tmp = d;
                tmp |= src;
                d = std::move(tmp);
            } else {
                d |= src;
            }
        }
    }
}

size_t
SparseDDC::storage_bytes() const {
    size_t total = sizeof(*this);
    total += seg_indices_.capacity() * sizeof(uint32_t);
    total += seg_data_.capacity()    * sizeof(DDCBtv);
    for (const auto& s : seg_data_) {
        total += s.size_breakdown().total_bits / 8;
    }
    return total;
}

DDC
SparseDDC::or_many(size_t count, const SparseDDC** sparses,
                      size_t num_rows, size_t segment_bits) {
    DDC dst = DDC::from_sparse_positions({}, num_rows, segment_bits);
    if (count == 0) return dst;

    size_t num_segs = (num_rows + segment_bits - 1) / segment_bits;
    std::vector<std::vector<const DDCBtv*>> by_seg(num_segs);
    size_t total_segs = 0;
    for (size_t s = 0; s < count; s++) total_segs += sparses[s]->seg_indices_.size();
    if (total_segs == 0) return dst;
    if (num_segs > 0) {
        size_t avg = total_segs / num_segs + 4;
        for (auto& v : by_seg) v.reserve(avg);
    }
    for (size_t s = 0; s < count; s++) {
        const auto& sp = *sparses[s];
        const auto& idxs = sp.seg_indices();
        const auto& data = sp.seg_data();
        for (size_t i = 0; i < idxs.size(); i++)
            by_seg[idxs[i]].push_back(&data[i]);
    }

    auto& dst_segs = dst.segments();
    for (uint32_t seg = 0; seg < num_segs; seg++) {
        auto& srcs = by_seg[seg];
        if (srcs.empty()) continue;
        DDCBtv& d = dst_segs[seg];
        for (const DDCBtv* src_ptr : srcs) {
            const DDCBtv& src = *src_ptr;
            if (src.is_all_zero()) continue;
            if (d.is_all_zero())   { d = src; continue; }
            if (d.is_all_ones())   break;
            if (src.is_all_ones()) {
                d = DDCBtv::make_all_fill(src.bit_count(), src.l2_count(), true);
                break;
            }
            if (d.state() == DDCBtv::State::Compressed
                && src.state() == DDCBtv::State::Compressed) {
                d = d | src;
            } else {
                if (d.state() != DDCBtv::State::Decompressed) {
                    DDCBtv tmp = d;
                    tmp |= src;
                    d = std::move(tmp);
                } else {
                    d |= src;
                }
            }
        }
    }
    return dst;
}
