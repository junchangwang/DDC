

#include "ddc_n.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <ostream>
#include <istream>
#include <stdexcept>
#include <string>

#ifdef __AVX512VBMI2__
#include <immintrin.h>
#endif

static thread_local DDCNPathCounts g_ddc_n_path_counts{};
const DDCNPathCounts& ddc_n_path_counts() { return g_ddc_n_path_counts; }
void ddc_n_reset_path_counts()                { g_ddc_n_path_counts = {}; }

static inline bool get_bit_lsb(const uint8_t* buf, size_t bit_off) {
    return (buf[bit_off >> 3] >> (bit_off & 7)) & 1;
}
static inline void set_bit_lsb(uint8_t* buf, size_t bit_off) {
    buf[bit_off >> 3] |= uint8_t(1) << (bit_off & 7);
}

static inline bool get_bit_msb(const uint8_t* buf, size_t bit_off) {
    return (buf[bit_off >> 3] >> (7 - (bit_off & 7))) & 1;
}
static inline void set_bit_msb(uint8_t* buf, size_t bit_off) {
    buf[bit_off >> 3] |= uint8_t(0x80) >> (bit_off & 7);
}

static inline uint8_t bits_to_l1_byte(const std::vector<bool>& bits,
                                      size_t seg_start_bit,
                                      size_t l1_byte_idx,
                                      size_t total_bits) {
    uint8_t v = 0;
    size_t base = seg_start_bit + l1_byte_idx * 8;
    for (int b = 0; b < 8; b++) {
        size_t pos = base + b;
        if (pos >= total_bits) break;
        if (bits[pos]) v |= uint8_t(0x80) >> b;
    }
    return v;
}

// pick fill 0/FF
static uint8_t pick_fill(const std::vector<uint8_t>& bytes, bool& fill_ones) {
    size_t nonzero = 0, non_ff = 0;
#ifdef __AVX512VBMI2__
    const __m512i zero_vec = _mm512_setzero_si512();
    const __m512i ff_vec   = _mm512_set1_epi8(static_cast<char>(0xFF));
    size_t i = 0;
    size_t n = bytes.size();
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(bytes.data() + i);
        nonzero += __builtin_popcountll(_mm512_cmpneq_epi8_mask(v, zero_vec));
        non_ff  += __builtin_popcountll(_mm512_cmpneq_epi8_mask(v, ff_vec));
    }
    for (; i < n; i++) {
        if (bytes[i] != 0x00) nonzero++;
        if (bytes[i] != 0xFF) non_ff++;
    }
#else
    for (uint8_t b : bytes) {
        if (b != 0x00) nonzero++;
        if (b != 0xFF) non_ff++;
    }
#endif
    fill_ones = (non_ff < nonzero);
    return fill_ones ? 0xFF : 0x00;
}

// compress one layer
static size_t compress_layer(const uint8_t* src, size_t src_count,
                             uint8_t fill_val,
                             uint8_t* lits_out,
                             uint8_t* marker_bits) {
    size_t lits_off = 0;
#ifdef __AVX512VBMI2__
    // SIMD compressstore
    const __m512i fill_vec = _mm512_set1_epi8(static_cast<char>(fill_val));
    size_t i = 0;
    for (; i + 64 <= src_count; i += 64) {
        __m512i v = _mm512_loadu_si512(src + i);
        __mmask64 nonfill = _mm512_cmpneq_epi8_mask(v, fill_vec);

        std::memcpy(marker_bits + i / 8, &nonfill, 8);
        _mm512_mask_compressstoreu_epi8(lits_out + lits_off, nonfill, v);
        lits_off += __builtin_popcountll(nonfill);
    }
    for (; i < src_count; i++) {
        if (src[i] != fill_val) {
            marker_bits[i >> 3] |= uint8_t(1) << (i & 7);
            lits_out[lits_off++] = src[i];
        }
    }
#else
    for (size_t i = 0; i < src_count; i++) {
        if (src[i] != fill_val) {
            marker_bits[i >> 3] |= uint8_t(1) << (i & 7);
            lits_out[lits_off++] = src[i];
        }
    }
#endif
    return lits_off;
}

// build segment, fold L1..L5
static DDCNSeg
ddc_n_compress_seg_from_l1(const uint8_t* l1_full, size_t l2_count,
                              size_t seg_bit_count, int depth,
                              bool l1_fill_ones) {
    DDCNSeg s{};
    s.depth = depth;
    s.bit_count = seg_bit_count;
    s.l1_fill_ones = l1_fill_ones;
    s.l2_count = l2_count;
    s.l3_count = (s.l2_count + 7) / 8;
    s.l4_count = (s.l3_count + 7) / 8;
    s.l5_count = (s.l4_count + 7) / 8;

    const uint8_t l1_fill_val = l1_fill_ones ? 0xFF : 0x00;
    size_t l2_byte_count = (s.l2_count + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);
    s.l1_lits.assign(s.l2_count, 0);
    size_t lit_n = compress_layer(l1_full, s.l2_count, l1_fill_val,
                                  s.l1_lits.data(), l2_flat.data());
    s.l1_lits.resize(lit_n);

    if (depth == 2) {
        s.top_raw = std::move(l2_flat);
        return s;
    }

    uint8_t l2_fill_val = pick_fill(l2_flat, s.l2_fill_ones);
    size_t l3_byte_count = (s.l3_count + 7) / 8;
    std::vector<uint8_t> l3_flat(l3_byte_count, 0);
    s.l2_lits.assign(l2_byte_count, 0);
    lit_n = compress_layer(l2_flat.data(), l2_byte_count, l2_fill_val,
                           s.l2_lits.data(), l3_flat.data());
    s.l2_lits.resize(lit_n);

    if (depth == 3) {
        s.top_raw = std::move(l3_flat);
        return s;
    }

    uint8_t l3_fill_val = pick_fill(l3_flat, s.l3_fill_ones);
    size_t l4_byte_count = (s.l4_count + 7) / 8;
    std::vector<uint8_t> l4_flat(l4_byte_count, 0);
    s.l3_lits.assign(l3_byte_count, 0);
    lit_n = compress_layer(l3_flat.data(), l3_byte_count, l3_fill_val,
                           s.l3_lits.data(), l4_flat.data());
    s.l3_lits.resize(lit_n);

    if (depth == 4) {
        s.top_raw = std::move(l4_flat);
        return s;
    }

    uint8_t l4_fill_val = pick_fill(l4_flat, s.l4_fill_ones);
    size_t l5_byte_count = (s.l5_count + 7) / 8;
    std::vector<uint8_t> l5_flat(l5_byte_count, 0);
    s.l4_lits.assign(l4_byte_count, 0);
    lit_n = compress_layer(l4_flat.data(), l4_byte_count, l4_fill_val,
                           s.l4_lits.data(), l5_flat.data());
    s.l4_lits.resize(lit_n);

    s.top_raw = std::move(l5_flat);
    return s;
}

static DDCNSeg
ddc_n_compress_seg(const std::vector<bool>& bits,
                      size_t seg_start, size_t seg_bit_count,
                      int depth, bool l1_fill_ones) {
    size_t l2_count = (seg_bit_count + 7) / 8;
    std::vector<uint8_t> l1_full(l2_count);
    size_t total_bits = bits.size();
    for (size_t i = 0; i < l2_count; i++)
        l1_full[i] = bits_to_l1_byte(bits, seg_start, i, total_bits);
    return ddc_n_compress_seg_from_l1(l1_full.data(), l2_count,
                                         seg_bit_count, depth, l1_fill_ones);
}

// compress: per-segment
DDCN ddc_n_compress(const std::vector<bool>& bits, int depth,
                          size_t segment_bits, bool l1_fill_ones) {
    assert(depth >= 2 && depth <= 5);
    DDCN cb{};
    cb.depth = depth;
    cb.bit_count = bits.size();
    cb.segment_bits = segment_bits;
    size_t n_segs = (cb.bit_count + segment_bits - 1) / segment_bits;
    cb.segments.reserve(n_segs);
    for (size_t s = 0; s < n_segs; s++) {
        size_t start = s * segment_bits;
        size_t cnt = std::min(segment_bits, cb.bit_count - start);
        cb.segments.push_back(
            ddc_n_compress_seg(bits, start, cnt, depth, l1_fill_ones));
    }
    return cb;
}

// expand L5->L1
static std::vector<uint8_t> expand_l1_stream(const DDCNSeg& s) {

    std::vector<uint8_t> l4_flat;
    if (s.depth >= 4) {
        size_t l4_byte_count = (s.l4_count + 7) / 8;
        l4_flat.assign(l4_byte_count, 0);
        if (s.depth == 4) {
            l4_flat = s.top_raw;
        } else {
            size_t lit_cursor = 0;
            uint8_t fill = s.l4_fill_ones ? 0xFF : 0x00;
            for (size_t i = 0; i < l4_byte_count; i++) {
                if (get_bit_lsb(s.top_raw.data(), i)) {
                    l4_flat[i] = s.l4_lits[lit_cursor++];
                } else {
                    l4_flat[i] = fill;
                }
            }
        }
    }

    std::vector<uint8_t> l3_flat;
    if (s.depth >= 3) {
        size_t l3_byte_count = (s.l3_count + 7) / 8;
        l3_flat.assign(l3_byte_count, 0);
        if (s.depth == 3) {
            l3_flat = s.top_raw;
        } else {
            size_t lit_cursor = 0;
            uint8_t fill = s.l3_fill_ones ? 0xFF : 0x00;
            for (size_t i = 0; i < l3_byte_count; i++) {

                bool is_lit = get_bit_lsb(l4_flat.data(), i);
                if (is_lit) l3_flat[i] = s.l3_lits[lit_cursor++];
                else        l3_flat[i] = fill;
            }
        }
    }

    size_t l2_byte_count = (s.l2_count + 7) / 8;
    std::vector<uint8_t> l2_flat(l2_byte_count, 0);
    if (s.depth == 2) {
        l2_flat = s.top_raw;
    } else {
        size_t lit_cursor = 0;
        uint8_t fill = s.l2_fill_ones ? 0xFF : 0x00;
        for (size_t i = 0; i < l2_byte_count; i++) {
            bool is_lit = get_bit_lsb(l3_flat.data(), i);
            if (is_lit) l2_flat[i] = s.l2_lits[lit_cursor++];
            else        l2_flat[i] = fill;
        }
    }

    std::vector<uint8_t> l1_flat(s.l2_count, 0);
    uint8_t l1_fill = s.l1_fill_ones ? 0xFF : 0x00;
    size_t lit_cursor = 0;
    for (size_t i = 0; i < s.l2_count; i++) {
        bool is_lit = get_bit_lsb(l2_flat.data(), i);
        if (is_lit) l1_flat[i] = s.l1_lits[lit_cursor++];
        else        l1_flat[i] = l1_fill;
    }
    return l1_flat;
}

std::vector<bool> ddc_n_decompress(const DDCN& cb) {
    std::vector<bool> out(cb.bit_count, false);
    size_t base = 0;
    for (size_t s = 0; s < cb.segments.size(); s++) {
        const auto& seg = cb.segments[s];
        std::vector<uint8_t> l1 = expand_l1_stream(seg);
        for (size_t i = 0; i < seg.l2_count; i++) {
            uint8_t v = l1[i];
            for (int b = 0; b < 8; b++) {
                size_t pos = base + i * 8 + b;
                if (pos >= base + seg.bit_count) break;
                if (pos >= cb.bit_count) break;
                out[pos] = (v >> (7 - b)) & 1;
            }
        }
        base += seg.bit_count;
    }
    return out;
}

size_t ddc_n_popcount(const DDCN& cb) {
    size_t total = 0;
    for (size_t s = 0; s < cb.segments.size(); s++) {
        const auto& seg = cb.segments[s];
        std::vector<uint8_t> l1 = expand_l1_stream(seg);
        size_t valid = seg.l2_count;

        if (valid > 0 && seg.bit_count % 8 != 0) {
            size_t valid_bits = seg.bit_count % 8;
            l1[valid - 1] &= uint8_t(0xFF) << (8 - valid_bits);
        }
        for (size_t i = 0; i < valid; i++) total += __builtin_popcount(l1[i]);
    }
    return total;
}

enum class Op { AND, OR, XOR };

// scalar AND/OR/XOR
static std::vector<uint8_t> apply_op_dec(const DDCN& a, const DDCN& b, Op op) {
    assert(a.depth == b.depth);
    assert(a.bit_count == b.bit_count);
    assert(a.segments.size() == b.segments.size());

    size_t total = 0;
    for (const auto& sg : a.segments) total += sg.l2_count;
    std::vector<uint8_t> out(total);

    size_t off = 0;
    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        const auto& sb = b.segments[s];
        std::vector<uint8_t> la = expand_l1_stream(sa);
        std::vector<uint8_t> lb = expand_l1_stream(sb);
        switch (op) {
            case Op::AND:
                for (size_t i = 0; i < la.size(); i++) out[off + i] = la[i] & lb[i];
                break;
            case Op::OR:
                for (size_t i = 0; i < la.size(); i++) out[off + i] = la[i] | lb[i];
                break;
            case Op::XOR:
                for (size_t i = 0; i < la.size(); i++) out[off + i] = la[i] ^ lb[i];
                break;
        }
        off += sa.l2_count;
    }
    return out;
}

std::vector<uint8_t> ddc_n_and_dec(const DDCN& a, const DDCN& b) {
    return apply_op_dec(a, b, Op::AND);
}
std::vector<uint8_t> ddc_n_or_dec(const DDCN& a, const DDCN& b) {
    return apply_op_dec(a, b, Op::OR);
}
std::vector<uint8_t> ddc_n_xor_dec(const DDCN& a, const DDCN& b) {
    return apply_op_dec(a, b, Op::XOR);
}

std::vector<bool> bytes_to_bits(const std::vector<uint8_t>& bytes, size_t bit_count) {
    std::vector<bool> out(bit_count, false);
    for (size_t i = 0; i < bytes.size(); i++) {
        uint8_t v = bytes[i];
        for (int b = 0; b < 8; b++) {
            size_t pos = i * 8 + b;
            if (pos >= bit_count) break;
            out[pos] = (v >> (7 - b)) & 1;
        }
    }
    return out;
}

size_t ddc_n_seg_bytes(const DDCNSeg& seg) {
    return seg.l1_lits.size() + seg.l2_lits.size() + seg.l3_lits.size()
         + seg.l4_lits.size() + seg.top_raw.size();
}

size_t ddc_n_total_bytes(const DDCN& cb) {
    size_t t = 0;
    for (const auto& seg : cb.segments) t += ddc_n_seg_bytes(seg);
    return t;
}

static constexpr uint8_t DDC_N_FMT_V1 = 0xCE;

template<typename T>
static inline void wval(std::ostream& os, T v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
template<typename T>
static inline T rval(std::istream& is) {
    T v;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}
static inline void wbuf(std::ostream& os, const std::vector<uint8_t>& v) {
    wval<uint64_t>(os, v.size());
    if (!v.empty())
        os.write(reinterpret_cast<const char*>(v.data()), v.size());
}
static inline std::vector<uint8_t> rbuf(std::istream& is) {
    uint64_t n = rval<uint64_t>(is);
    std::vector<uint8_t> v(n);
    if (n > 0) is.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

static void serialize_seg(const DDCNSeg& s, std::ostream& os) {
    wval<uint8_t> (os, uint8_t(s.depth));
    wval<uint64_t>(os, s.bit_count);
    wval<uint64_t>(os, s.l2_count);
    wval<uint64_t>(os, s.l3_count);
    wval<uint64_t>(os, s.l4_count);
    wval<uint64_t>(os, s.l5_count);
    wval<uint8_t> (os, s.l1_fill_ones ? 1 : 0);
    wval<uint8_t> (os, s.l2_fill_ones ? 1 : 0);
    wval<uint8_t> (os, s.l3_fill_ones ? 1 : 0);
    wval<uint8_t> (os, s.l4_fill_ones ? 1 : 0);
    wbuf(os, s.l1_lits);
    wbuf(os, s.l2_lits);
    wbuf(os, s.l3_lits);
    wbuf(os, s.l4_lits);
    wbuf(os, s.top_raw);
}

static DDCNSeg deserialize_seg(std::istream& is) {
    DDCNSeg s{};
    s.depth        = int(rval<uint8_t>(is));
    s.bit_count    = rval<uint64_t>(is);
    s.l2_count     = rval<uint64_t>(is);
    s.l3_count     = rval<uint64_t>(is);
    s.l4_count     = rval<uint64_t>(is);
    s.l5_count     = rval<uint64_t>(is);
    s.l1_fill_ones = rval<uint8_t>(is) != 0;
    s.l2_fill_ones = rval<uint8_t>(is) != 0;
    s.l3_fill_ones = rval<uint8_t>(is) != 0;
    s.l4_fill_ones = rval<uint8_t>(is) != 0;
    s.l1_lits = rbuf(is);
    s.l2_lits = rbuf(is);
    s.l3_lits = rbuf(is);
    s.l4_lits = rbuf(is);
    s.top_raw = rbuf(is);
    return s;
}

// serialize
void ddc_n_serialize(const DDCN& cb, std::ostream& os) {
    wval<uint8_t> (os, DDC_N_FMT_V1);
    wval<uint8_t> (os, uint8_t(cb.depth));
    wval<uint64_t>(os, cb.bit_count);
    wval<uint64_t>(os, cb.segment_bits);
    wval<uint64_t>(os, cb.segments.size());
    for (const auto& seg : cb.segments) serialize_seg(seg, os);
}

DDCN ddc_n_deserialize(std::istream& is) {
    uint8_t fmt = rval<uint8_t>(is);
    if (fmt != DDC_N_FMT_V1)
        throw std::runtime_error("ddc_n_deserialize: bad fmt tag " +
                                 std::to_string(int(fmt)));
    DDCN cb;
    cb.depth        = int(rval<uint8_t>(is));
    cb.bit_count    = rval<uint64_t>(is);
    cb.segment_bits = rval<uint64_t>(is);
    uint64_t n      = rval<uint64_t>(is);
    cb.segments.reserve(n);
    for (uint64_t i = 0; i < n; i++) cb.segments.push_back(deserialize_seg(is));
    return cb;
}

#ifdef __AVX512VBMI2__
#include <immintrin.h>

enum class OpKind { AND, OR, XOR };

static constexpr size_t PF_DIST = 256;

// 64-byte region op (L3-driven expand)
template <OpKind OP>
static inline void region_op_avx(
    uint8_t l3a, uint8_t l3b,
    const uint8_t* l2_lits_a, const uint8_t* l2_lits_b,
    const uint8_t* l1_lits_a, const uint8_t* l1_lits_b,
    size_t& l2a_off, size_t& l2b_off,
    size_t& l1a_off, size_t& l1b_off,
    __m512i l2_fill_a, __m512i l2_fill_b,
    __m512i l1_fill_a, __m512i l1_fill_b,
    uint8_t* dst)
{
    __m512i l2a = _mm512_mask_expandloadu_epi8(l2_fill_a,
        static_cast<__mmask64>(l3a), l2_lits_a + l2a_off);
    l2a_off += __builtin_popcount(l3a);
    __mmask64 ma = static_cast<__mmask64>(
        _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
    __m512i va = _mm512_mask_expandloadu_epi8(l1_fill_a, ma,
        l1_lits_a + l1a_off);
    l1a_off += __builtin_popcountll(static_cast<uint64_t>(ma));

    __m512i l2b = _mm512_mask_expandloadu_epi8(l2_fill_b,
        static_cast<__mmask64>(l3b), l2_lits_b + l2b_off);
    l2b_off += __builtin_popcount(l3b);
    __mmask64 mb = static_cast<__mmask64>(
        _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b)));
    __m512i vb = _mm512_mask_expandloadu_epi8(l1_fill_b, mb,
        l1_lits_b + l1b_off);
    l1b_off += __builtin_popcountll(static_cast<uint64_t>(mb));

    __m512i vr;
    if      constexpr (OP == OpKind::AND) vr = _mm512_and_si512(va, vb);
    else if constexpr (OP == OpKind::OR ) vr = _mm512_or_si512 (va, vb);
    else                                  vr = _mm512_xor_si512(va, vb);

    _mm512_storeu_si512(dst, vr);
}

template <OpKind OP>
static inline void region_op_l2_avx(
    __mmask64 ma, __mmask64 mb,
    const uint8_t* l1_lits_a, const uint8_t* l1_lits_b,
    size_t& l1a_off, size_t& l1b_off,
    __m512i l1_fill_a, __m512i l1_fill_b,
    uint8_t* dst)
{
    __m512i va = _mm512_mask_expandloadu_epi8(l1_fill_a, ma,
        l1_lits_a + l1a_off);
    l1a_off += __builtin_popcountll(static_cast<uint64_t>(ma));
    __m512i vb = _mm512_mask_expandloadu_epi8(l1_fill_b, mb,
        l1_lits_b + l1b_off);
    l1b_off += __builtin_popcountll(static_cast<uint64_t>(mb));

    __m512i vr;
    if      constexpr (OP == OpKind::AND) vr = _mm512_and_si512(va, vb);
    else if constexpr (OP == OpKind::OR ) vr = _mm512_or_si512 (va, vb);
    else                                  vr = _mm512_xor_si512(va, vb);

    _mm512_storeu_si512(dst, vr);
}

struct SideAvx {
    const uint8_t* l1_lits;
    const uint8_t* l2_lits;
    const uint8_t* l3_lits;
    const uint8_t* l4_lits;
    const uint8_t* top_raw;
    bool   l1_fill_ones, l2_fill_ones, l3_fill_ones, l4_fill_ones;
    __m512i l1_fill_vec, l2_fill_vec;
    size_t l1_off, l2_off;
};

static inline SideAvx make_side_avx(const DDCNSeg& s) {
    SideAvx a{};
    a.l1_lits = s.l1_lits.data();
    a.l2_lits = s.l2_lits.data();
    a.l3_lits = s.l3_lits.data();
    a.l4_lits = s.l4_lits.data();
    a.top_raw = s.top_raw.data();
    a.l1_fill_ones = s.l1_fill_ones;
    a.l2_fill_ones = s.l2_fill_ones;
    a.l3_fill_ones = s.l3_fill_ones;
    a.l4_fill_ones = s.l4_fill_ones;
    a.l1_fill_vec  = s.l1_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();
    a.l2_fill_vec  = s.l2_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();
    a.l1_off = 0; a.l2_off = 0;
    return a;
}

// depth-2 segment op
template <OpKind OP>
static void seg_op_l2(const DDCNSeg& sa, const DDCNSeg& sb, uint8_t* dst) {
    SideAvx A = make_side_avx(sa), B = make_side_avx(sb);
    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    constexpr bool has_region_bypass = (OP != OpKind::AND);

    const bool fills_zero = !A.l1_fill_ones && !B.l1_fill_ones;

    const size_t batch_count = (avx_regions + 7) / 8;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t r_start = batch * 8;
        size_t r_end   = std::min(r_start + 8, avx_regions);
        size_t batch_size = r_end - r_start;

        if (batch_size == 8) {

            const bool a_can_be_zero = !A.l1_fill_ones;
            const bool b_can_be_zero = !B.l1_fill_ones;
            __m512i va_chunk = _mm512_loadu_si512(A.top_raw + r_start * 8);
            __m512i vb_chunk = _mm512_loadu_si512(B.top_raw + r_start * 8);
            const bool a_batch_zero = a_can_be_zero
                && _mm512_test_epi8_mask(va_chunk, va_chunk) == 0;
            const bool b_batch_zero = b_can_be_zero
                && _mm512_test_epi8_mask(vb_chunk, vb_chunk) == 0;
            if constexpr (OP == OpKind::AND) {
                if (a_batch_zero || b_batch_zero) {
                    // batch bypass: advance offsets, emit zeros
                    if (!a_batch_zero) {
                        for (size_t r = r_start; r < r_end; r++) {
                            uint64_t ma; std::memcpy(&ma, A.top_raw + r * 8, 8);
                            A.l1_off += __builtin_popcountll(ma);
                        }
                    }
                    if (!b_batch_zero) {
                        for (size_t r = r_start; r < r_end; r++) {
                            uint64_t mb; std::memcpy(&mb, B.top_raw + r * 8, 8);
                            B.l1_off += __builtin_popcountll(mb);
                        }
                    }
                    std::memset(dst + r_start * 64, 0, batch_size * 64);
                    continue;
                }
            } else {
                if (a_batch_zero && b_batch_zero) {
                    if constexpr (OP == OpKind::OR)
                        g_ddc_n_path_counts.zz += batch_size;
                    std::memset(dst + r_start * 64, 0, batch_size * 64);
                    continue;
                }
            }
        }

        for (size_t r = r_start; r < r_end; r++) {
            uint64_t ma = 0, mb = 0;
            std::memcpy(&ma, A.top_raw + r * 8, 8);
            std::memcpy(&mb, B.top_raw + r * 8, 8);
            if constexpr (has_region_bypass) {
                // per-region bypass
                if (fills_zero && ma == 0) {
                    if (mb == 0) {
                        if constexpr (OP == OpKind::OR) g_ddc_n_path_counts.zz++;
                        _mm512_storeu_si512(dst + r * 64, _mm512_setzero_si512());
                    } else {
                        if constexpr (OP == OpKind::OR) g_ddc_n_path_counts.az++;
                        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec,
                            static_cast<__mmask64>(mb), B.l1_lits + B.l1_off);
                        B.l1_off += __builtin_popcountll(mb);
                        _mm512_storeu_si512(dst + r * 64, vb);
                    }
                    continue;
                }
                if (fills_zero && mb == 0) {
                    if constexpr (OP == OpKind::OR) g_ddc_n_path_counts.bz++;
                    __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                        static_cast<__mmask64>(ma), A.l1_lits + A.l1_off);
                    A.l1_off += __builtin_popcountll(ma);
                    _mm512_storeu_si512(dst + r * 64, va);
                    continue;
                }
            }
            if constexpr (OP == OpKind::OR) g_ddc_n_path_counts.full++;

            _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<char*>(dst + r * 64 + PF_DIST), _MM_HINT_T0);
            region_op_l2_avx<OP>(static_cast<__mmask64>(ma), static_cast<__mmask64>(mb),
                A.l1_lits, B.l1_lits, A.l1_off, B.l1_off,
                A.l1_fill_vec, B.l1_fill_vec, dst + r * 64);
        }
    }

    // scalar tail
    if (avx_regions * 64 < total_words) {
        size_t tail_words = total_words - avx_regions * 64;
        uint8_t l1_fill_a = sa.l1_fill_ones ? 0xFF : 0x00;
        uint8_t l1_fill_b = sb.l1_fill_ones ? 0xFF : 0x00;
        for (size_t i = 0; i < tail_words; i++) {
            size_t l2_byte_idx = avx_regions * 8 + i / 8;
            size_t bit_in_byte = i % 8;
            bool a_lit = (A.top_raw[l2_byte_idx] >> bit_in_byte) & 1;
            bool b_lit = (B.top_raw[l2_byte_idx] >> bit_in_byte) & 1;
            uint8_t wa = a_lit ? A.l1_lits[A.l1_off++] : l1_fill_a;
            uint8_t wb = b_lit ? B.l1_lits[B.l1_off++] : l1_fill_b;
            uint8_t vr = (OP == OpKind::AND) ? (wa & wb)
                       : (OP == OpKind::OR ) ? (wa | wb)
                                             : (wa ^ wb);
            dst[avx_regions * 64 + i] = vr;
        }
    }
}

// depth-3 segment op
template <OpKind OP>
static void seg_op_l3(const DDCNSeg& sa, const DDCNSeg& sb, uint8_t* dst) {
    SideAvx A = make_side_avx(sa), B = make_side_avx(sb);
    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    constexpr bool has_region_bypass = (OP != OpKind::AND);

    const bool fills_zero = !A.l1_fill_ones && !B.l1_fill_ones
                         && !A.l2_fill_ones && !B.l2_fill_ones;

    const size_t batch_count = (avx_regions + 63) / 64;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t r_start = batch * 64;
        size_t r_end   = std::min(r_start + 64, avx_regions);
        size_t batch_size = r_end - r_start;

        if (batch_size == 64) {

            const bool a_can_be_zero = !A.l1_fill_ones && !A.l2_fill_ones;
            const bool b_can_be_zero = !B.l1_fill_ones && !B.l2_fill_ones;
            __m512i va_chunk = _mm512_loadu_si512(A.top_raw + r_start);
            __m512i vb_chunk = _mm512_loadu_si512(B.top_raw + r_start);
            const bool a_batch_zero = a_can_be_zero
                && _mm512_test_epi8_mask(va_chunk, va_chunk) == 0;
            const bool b_batch_zero = b_can_be_zero
                && _mm512_test_epi8_mask(vb_chunk, vb_chunk) == 0;
            if constexpr (OP == OpKind::AND) {
                if (a_batch_zero || b_batch_zero) {

                    if (!a_batch_zero) {
                        for (size_t r = r_start; r < r_end; r++) {
                            uint8_t l3a = A.top_raw[r];
                            __m512i l2v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                                static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                            A.l2_off += __builtin_popcount(l3a);
                            __mmask64 m = static_cast<__mmask64>(
                                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                            A.l1_off += __builtin_popcountll(
                                static_cast<uint64_t>(m));
                        }
                    }
                    if (!b_batch_zero) {
                        for (size_t r = r_start; r < r_end; r++) {
                            uint8_t l3b = B.top_raw[r];
                            __m512i l2v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                                static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                            B.l2_off += __builtin_popcount(l3b);
                            __mmask64 m = static_cast<__mmask64>(
                                _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                            B.l1_off += __builtin_popcountll(
                                static_cast<uint64_t>(m));
                        }
                    }
                    std::memset(dst + r_start * 64, 0, batch_size * 64);
                    continue;
                }
            } else {
                if (a_batch_zero && b_batch_zero) {
                    std::memset(dst + r_start * 64, 0, batch_size * 64);
                    continue;
                }
            }
        }

        for (size_t r = r_start; r < r_end; r++) {
            uint8_t l3a = A.top_raw[r];
            uint8_t l3b = B.top_raw[r];
            if constexpr (has_region_bypass) {

                if (fills_zero && l3a == 0) {
                    if (l3b == 0) {
                        _mm512_storeu_si512(dst + r * 64, _mm512_setzero_si512());
                    } else {

                        __m512i l2b = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 mb = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b)));
                        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec,
                            mb, B.l1_lits + B.l1_off);
                        B.l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
                        _mm512_storeu_si512(dst + r * 64, vb);
                    }
                    continue;
                }
                if (fills_zero && l3b == 0) {
                    __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                        static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                    A.l2_off += __builtin_popcount(l3a);
                    __mmask64 ma = static_cast<__mmask64>(
                        _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                    __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                        ma, A.l1_lits + A.l1_off);
                    A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                    _mm512_storeu_si512(dst + r * 64, va);
                    continue;
                }
            }

            _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<char*>(dst + r * 64 + PF_DIST), _MM_HINT_T0);
            region_op_avx<OP>(l3a, l3b,
                A.l2_lits, B.l2_lits,
                A.l1_lits, B.l1_lits,
                A.l2_off, B.l2_off, A.l1_off, B.l1_off,
                A.l2_fill_vec, B.l2_fill_vec,
                A.l1_fill_vec, B.l1_fill_vec,
                dst + r * 64);
        }
    }

    if (avx_regions * 64 < total_words) {
        size_t tail_words = total_words - avx_regions * 64;
        uint8_t l1_fill_a = sa.l1_fill_ones ? 0xFF : 0x00;
        uint8_t l1_fill_b = sb.l1_fill_ones ? 0xFF : 0x00;
        uint8_t l2_fill_a = sa.l2_fill_ones ? 0xFF : 0x00;
        uint8_t l2_fill_b = sb.l2_fill_ones ? 0xFF : 0x00;
        uint8_t l3a = A.top_raw[avx_regions];
        uint8_t l3b = B.top_raw[avx_regions];
        size_t tail_l2_bytes = (tail_words + 7) / 8;
        for (size_t lb = 0; lb < tail_l2_bytes; lb++) {
            bool a_l2_lit = (l3a >> lb) & 1;
            bool b_l2_lit = (l3b >> lb) & 1;
            uint8_t l2a = a_l2_lit ? A.l2_lits[A.l2_off++] : l2_fill_a;
            uint8_t l2b = b_l2_lit ? B.l2_lits[B.l2_off++] : l2_fill_b;
            for (size_t bit = 0; bit < 8; bit++) {
                size_t i = lb * 8 + bit;
                if (i >= tail_words) break;
                bool a_lit = (l2a >> bit) & 1;
                bool b_lit = (l2b >> bit) & 1;
                uint8_t wa = a_lit ? A.l1_lits[A.l1_off++] : l1_fill_a;
                uint8_t wb = b_lit ? B.l1_lits[B.l1_off++] : l1_fill_b;
                uint8_t vr = (OP == OpKind::AND) ? (wa & wb)
                           : (OP == OpKind::OR ) ? (wa | wb)
                                                 : (wa ^ wb);
                dst[avx_regions * 64 + i] = vr;
            }
        }
    }
}

// depth-4 segment op (configurable bypass)
template <OpKind OP, bool HAS_BATCH_BYPASS = true,
          bool HAS_REGION_BYPASS = (OP != OpKind::AND)>
static void seg_op_l4(const DDCNSeg& sa, const DDCNSeg& sb, uint8_t* dst) {
    SideAvx A = make_side_avx(sa), B = make_side_avx(sb);

    const bool fills_zero = !sa.l1_fill_ones && !sb.l1_fill_ones
                         && !sa.l2_fill_ones && !sb.l2_fill_ones
                         && !sa.l3_fill_ones && !sb.l3_fill_ones;
    __m512i l3_fill_a_vec = sa.l3_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();
    __m512i l3_fill_b_vec = sb.l3_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();

    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    size_t l3a_lit_off = 0, l3b_lit_off = 0;
    size_t batch_count = (avx_regions + 63) / 64;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t batch_start = batch * 64;
        size_t batch_end   = std::min(batch_start + 64, avx_regions);
        size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, sa.top_raw.data() + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, sb.top_raw.data() + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }

        if constexpr (HAS_BATCH_BYPASS) {
        const bool a_can_be_zero = !sa.l1_fill_ones && !sa.l2_fill_ones
                                && !sa.l3_fill_ones;
        const bool b_can_be_zero = !sb.l1_fill_ones && !sb.l2_fill_ones
                                && !sb.l3_fill_ones;
        const bool a_batch_zero = a_can_be_zero && a_l4_mask == 0;
        const bool b_batch_zero = b_can_be_zero && b_l4_mask == 0;
        if constexpr (OP == OpKind::AND) {
            if (a_batch_zero || b_batch_zero) {
                if (!a_batch_zero) {
                    __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(l3_fill_a_vec,
                        static_cast<__mmask64>(a_l4_mask),
                        sa.l3_lits.data() + l3a_lit_off);
                    l3a_lit_off += __builtin_popcountll(a_l4_mask);
                    alignas(64) uint8_t l3a_buf[64];
                    _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
                    for (size_t r = 0; r < batch_size; r++) {
                        uint8_t l3a = l3a_buf[r];
                        __m512i l2v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                            static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                        A.l2_off += __builtin_popcount(l3a);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        A.l1_off += __builtin_popcountll(static_cast<uint64_t>(m));
                    }
                }
                if (!b_batch_zero) {
                    __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(l3_fill_b_vec,
                        static_cast<__mmask64>(b_l4_mask),
                        sb.l3_lits.data() + l3b_lit_off);
                    l3b_lit_off += __builtin_popcountll(b_l4_mask);
                    alignas(64) uint8_t l3b_buf[64];
                    _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);
                    for (size_t r = 0; r < batch_size; r++) {
                        uint8_t l3b = l3b_buf[r];
                        __m512i l2v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        B.l1_off += __builtin_popcountll(static_cast<uint64_t>(m));
                    }
                }
                std::memset(dst + batch_start * 64, 0, batch_size * 64);
                continue;
            }
        } else {
            if (a_batch_zero && b_batch_zero) {
                std::memset(dst + batch_start * 64, 0, batch_size * 64);
                continue;
            }
        }
        }

        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(l3_fill_a_vec,
            static_cast<__mmask64>(a_l4_mask), sa.l3_lits.data() + l3a_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(l3_fill_b_vec,
            static_cast<__mmask64>(b_l4_mask), sb.l3_lits.data() + l3b_lit_off);
        l3a_lit_off += __builtin_popcountll(a_l4_mask);
        l3b_lit_off += __builtin_popcountll(b_l4_mask);

        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);

        for (size_t r = 0; r < batch_size; r++) {
            uint8_t l3a = l3a_buf[r];
            uint8_t l3b = l3b_buf[r];

            if constexpr (HAS_REGION_BYPASS) {
              if constexpr (OP == OpKind::AND) {

                const bool a_is_zero = fills_zero && l3a == 0;
                const bool b_is_zero = fills_zero && l3b == 0;
                if (a_is_zero || b_is_zero) {
                    if (!a_is_zero) {
                        __m512i l2v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                            static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                        A.l2_off += __builtin_popcount(l3a);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        A.l1_off += __builtin_popcountll(static_cast<uint64_t>(m));
                    }
                    if (!b_is_zero) {
                        __m512i l2v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        B.l1_off += __builtin_popcountll(static_cast<uint64_t>(m));
                    }
                    _mm512_storeu_si512(dst + (batch_start + r) * 64,
                                        _mm512_setzero_si512());
                    continue;
                }
              } else {
                if (fills_zero && l3a == 0) {
                    if (l3b == 0) {
                        _mm512_storeu_si512(dst + (batch_start + r) * 64,
                                            _mm512_setzero_si512());
                    } else {
                        __m512i l2b = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 mb = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b)));
                        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec,
                            mb, B.l1_lits + B.l1_off);
                        B.l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
                        _mm512_storeu_si512(dst + (batch_start + r) * 64, vb);
                    }
                    continue;
                }
                if (fills_zero && l3b == 0) {
                    __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                        static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                    A.l2_off += __builtin_popcount(l3a);
                    __mmask64 ma = static_cast<__mmask64>(
                        _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                    __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                        ma, A.l1_lits + A.l1_off);
                    A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                    _mm512_storeu_si512(dst + (batch_start + r) * 64, va);
                    continue;
                }
              }
            }

            _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<char*>(dst + (batch_start + r) * 64 + PF_DIST), _MM_HINT_T0);
            region_op_avx<OP>(l3a, l3b,
                A.l2_lits, B.l2_lits,
                A.l1_lits, B.l1_lits,
                A.l2_off, B.l2_off, A.l1_off, B.l1_off,
                A.l2_fill_vec, B.l2_fill_vec,
                A.l1_fill_vec, B.l1_fill_vec,
                dst + (batch_start + r) * 64);
        }
    }

    if (avx_regions * 64 < total_words) {
        auto la_full = expand_l1_stream(sa);
        auto lb_full = expand_l1_stream(sb);
        for (size_t i = avx_regions * 64; i < total_words; i++) {
            uint8_t va = la_full[i], vb = lb_full[i];
            dst[i] = (OP == OpKind::AND) ? (va & vb)
                   : (OP == OpKind::OR ) ? (va | vb)
                                         : (va ^ vb);
        }
    }
}

// depth-5 segment op
template <OpKind OP>
static void seg_op_l5(const DDCNSeg& sa, const DDCNSeg& sb, uint8_t* dst) {
    SideAvx A = make_side_avx(sa), B = make_side_avx(sb);
    constexpr bool has_region_bypass = (OP != OpKind::AND);

    // expand L4 into buf
    constexpr size_t MAX_L4 = 256;
    alignas(64) uint8_t l4a_buf[MAX_L4] = {0}, l4b_buf[MAX_L4] = {0};
    size_t l4_byte_count = (sa.l4_count + 7) / 8;
    assert(l4_byte_count <= MAX_L4);
    {
        uint8_t fill = sa.l4_fill_ones ? 0xFF : 0x00;
        size_t lit = 0;
        for (size_t i = 0; i < l4_byte_count; i++) {
            bool is_lit = (sa.top_raw[i / 8] >> (i % 8)) & 1;
            l4a_buf[i] = is_lit ? sa.l4_lits[lit++] : fill;
        }
    }
    {
        uint8_t fill = sb.l4_fill_ones ? 0xFF : 0x00;
        size_t lit = 0;
        for (size_t i = 0; i < l4_byte_count; i++) {
            bool is_lit = (sb.top_raw[i / 8] >> (i % 8)) & 1;
            l4b_buf[i] = is_lit ? sb.l4_lits[lit++] : fill;
        }
    }

    const bool fills_zero = !sa.l1_fill_ones && !sb.l1_fill_ones
                         && !sa.l2_fill_ones && !sb.l2_fill_ones
                         && !sa.l3_fill_ones && !sb.l3_fill_ones;
    __m512i l3_fill_a_vec = sa.l3_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();
    __m512i l3_fill_b_vec = sb.l3_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();

    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    size_t l3a_lit_off = 0, l3b_lit_off = 0;
    size_t batch_count = (avx_regions + 63) / 64;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t batch_start = batch * 64;
        size_t batch_end   = std::min(batch_start + 64, avx_regions);
        size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0, b_l4_mask = 0;
        std::memcpy(&a_l4_mask, l4a_buf + batch_start / 8, (batch_size + 7) / 8);
        std::memcpy(&b_l4_mask, l4b_buf + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
            b_l4_mask &= valid;
        }

        const bool a_can_be_zero = !sa.l1_fill_ones && !sa.l2_fill_ones
                                && !sa.l3_fill_ones;
        const bool b_can_be_zero = !sb.l1_fill_ones && !sb.l2_fill_ones
                                && !sb.l3_fill_ones;
        const bool a_batch_zero = a_can_be_zero && a_l4_mask == 0;
        const bool b_batch_zero = b_can_be_zero && b_l4_mask == 0;
        if constexpr (OP == OpKind::AND) {
            if (a_batch_zero || b_batch_zero) {

                if (!a_batch_zero) {
                    __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(l3_fill_a_vec,
                        static_cast<__mmask64>(a_l4_mask),
                        sa.l3_lits.data() + l3a_lit_off);
                    l3a_lit_off += __builtin_popcountll(a_l4_mask);
                    alignas(64) uint8_t l3a_buf[64];
                    _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
                    for (size_t r = 0; r < batch_size; r++) {
                        uint8_t l3a = l3a_buf[r];
                        __m512i l2v = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                            static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                        A.l2_off += __builtin_popcount(l3a);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        A.l1_off += __builtin_popcountll(
                            static_cast<uint64_t>(m));
                    }
                }
                if (!b_batch_zero) {
                    __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(l3_fill_b_vec,
                        static_cast<__mmask64>(b_l4_mask),
                        sb.l3_lits.data() + l3b_lit_off);
                    l3b_lit_off += __builtin_popcountll(b_l4_mask);
                    alignas(64) uint8_t l3b_buf[64];
                    _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);
                    for (size_t r = 0; r < batch_size; r++) {
                        uint8_t l3b = l3b_buf[r];
                        __m512i l2v = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 m = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2v)));
                        B.l1_off += __builtin_popcountll(
                            static_cast<uint64_t>(m));
                    }
                }
                std::memset(dst + batch_start * 64, 0, batch_size * 64);
                continue;
            }
        } else {
            if (a_batch_zero && b_batch_zero) {
                std::memset(dst + batch_start * 64, 0, batch_size * 64);
                continue;
            }
        }

        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(l3_fill_a_vec,
            static_cast<__mmask64>(a_l4_mask), sa.l3_lits.data() + l3a_lit_off);
        __m512i l3b_chunk = _mm512_mask_expandloadu_epi8(l3_fill_b_vec,
            static_cast<__mmask64>(b_l4_mask), sb.l3_lits.data() + l3b_lit_off);
        l3a_lit_off += __builtin_popcountll(a_l4_mask);
        l3b_lit_off += __builtin_popcountll(b_l4_mask);

        alignas(64) uint8_t l3a_buf[64], l3b_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3b_buf), l3b_chunk);

        for (size_t r = 0; r < batch_size; r++) {
            uint8_t l3a = l3a_buf[r];
            uint8_t l3b = l3b_buf[r];

            if constexpr (has_region_bypass) {

                if (fills_zero && l3a == 0) {
                    if (l3b == 0) {
                        _mm512_storeu_si512(dst + (batch_start + r) * 64,
                                            _mm512_setzero_si512());
                    } else {
                        __m512i l2b = _mm512_mask_expandloadu_epi8(B.l2_fill_vec,
                            static_cast<__mmask64>(l3b), B.l2_lits + B.l2_off);
                        B.l2_off += __builtin_popcount(l3b);
                        __mmask64 mb = static_cast<__mmask64>(
                            _mm_cvtsi128_si64(_mm512_castsi512_si128(l2b)));
                        __m512i vb = _mm512_mask_expandloadu_epi8(B.l1_fill_vec,
                            mb, B.l1_lits + B.l1_off);
                        B.l1_off += __builtin_popcountll(static_cast<uint64_t>(mb));
                        _mm512_storeu_si512(dst + (batch_start + r) * 64, vb);
                    }
                    continue;
                }
                if (fills_zero && l3b == 0) {
                    __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                        static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                    A.l2_off += __builtin_popcount(l3a);
                    __mmask64 ma = static_cast<__mmask64>(
                        _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                    __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                        ma, A.l1_lits + A.l1_off);
                    A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                    _mm512_storeu_si512(dst + (batch_start + r) * 64, va);
                    continue;
                }
            }

            _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B.l1_lits + B.l1_off + PF_DIST), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<char*>(dst + (batch_start + r) * 64 + PF_DIST), _MM_HINT_T0);
            region_op_avx<OP>(l3a, l3b,
                A.l2_lits, B.l2_lits,
                A.l1_lits, B.l1_lits,
                A.l2_off, B.l2_off, A.l1_off, B.l1_off,
                A.l2_fill_vec, B.l2_fill_vec,
                A.l1_fill_vec, B.l1_fill_vec,
                dst + (batch_start + r) * 64);
        }
    }

    if (avx_regions * 64 < total_words) {
        auto la_full = expand_l1_stream(sa);
        auto lb_full = expand_l1_stream(sb);
        for (size_t i = avx_regions * 64; i < total_words; i++) {
            uint8_t va = la_full[i], vb = lb_full[i];
            dst[i] = (OP == OpKind::AND) ? (va & vb)
                   : (OP == OpKind::OR ) ? (va | vb)
                                         : (va ^ vb);
        }
    }
}

// self-AND = decompress, depth 2
static void seg_self_and_l2(const DDCNSeg& sa, uint8_t* dst) {
    SideAvx A = make_side_avx(sa);
    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    const bool fills_zero = !A.l1_fill_ones;
    const size_t batch_count = (avx_regions + 7) / 8;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t r_start = batch * 8;
        size_t r_end   = std::min(r_start + 8, avx_regions);
        size_t batch_size = r_end - r_start;

        if (fills_zero) {

            alignas(64) uint8_t buf[64] = {0};
            std::memcpy(buf, A.top_raw + r_start * 8, batch_size * 8);
            __m512i chunk = _mm512_load_si512(reinterpret_cast<__m512i*>(buf));
            uint8_t nz = static_cast<uint8_t>(_mm512_test_epi64_mask(chunk, chunk));

            if (batch_size < 8) nz &= static_cast<uint8_t>((1u << batch_size) - 1);
            if (nz == 0) {
                std::memset(dst + r_start * 64, 0, batch_size * 64);
                continue;
            }

            std::memset(dst + r_start * 64, 0, batch_size * 64);
            uint32_t work = nz;
            while (work) {
                int r_off = __builtin_ctz(work);
                work &= work - 1;
                size_t r = r_start + r_off;

                _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<char*>(dst + r * 64 + PF_DIST), _MM_HINT_T0);
                uint64_t ma;
                std::memcpy(&ma, A.top_raw + r * 8, 8);
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    static_cast<__mmask64>(ma), A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(ma);
                _mm512_storeu_si512(dst + r * 64, va);
            }
        } else {

            for (size_t r = r_start; r < r_end; r++) {
                uint64_t ma = 0;
                std::memcpy(&ma, A.top_raw + r * 8, 8);
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    static_cast<__mmask64>(ma), A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(ma);
                _mm512_storeu_si512(dst + r * 64, va);
            }
        }
    }

    if (avx_regions * 64 < total_words) {
        size_t tail_words = total_words - avx_regions * 64;
        uint8_t l1_fill_a = sa.l1_fill_ones ? 0xFF : 0x00;
        for (size_t i = 0; i < tail_words; i++) {
            size_t l2_byte_idx = avx_regions * 8 + i / 8;
            size_t bit_in_byte = i % 8;
            bool a_lit = (A.top_raw[l2_byte_idx] >> bit_in_byte) & 1;
            dst[avx_regions * 64 + i] = a_lit ? A.l1_lits[A.l1_off++] : l1_fill_a;
        }
    }
}

static void seg_self_and_l3(const DDCNSeg& sa, uint8_t* dst) {
    SideAvx A = make_side_avx(sa);
    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    const bool fills_zero = !A.l1_fill_ones && !A.l2_fill_ones;
    const size_t batch_count = (avx_regions + 63) / 64;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t r_start = batch * 64;
        size_t r_end   = std::min(r_start + 64, avx_regions);
        size_t batch_size = r_end - r_start;

        alignas(64) uint8_t l3_buf[64] = {0};
        std::memcpy(l3_buf, A.top_raw + r_start, batch_size);

        if (fills_zero) {
            __m512i chunk = _mm512_load_si512(reinterpret_cast<__m512i*>(l3_buf));
            uint64_t nz = static_cast<uint64_t>(_mm512_test_epi8_mask(chunk, chunk));
            if (nz == 0) {
                std::memset(dst + r_start * 64, 0, batch_size * 64);
                continue;
            }

            std::memset(dst + r_start * 64, 0, batch_size * 64);
            while (nz) {
                size_t r = __builtin_ctzll(nz);
                nz &= nz - 1;
                uint8_t l3a = l3_buf[r];

                _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<char*>(dst + (r_start + r) * 64 + PF_DIST), _MM_HINT_T0);

                __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                    static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                A.l2_off += __builtin_popcount(l3a);
                __mmask64 ma = static_cast<__mmask64>(
                    _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    ma, A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                _mm512_storeu_si512(dst + (r_start + r) * 64, va);
            }
        } else {

            for (size_t r = r_start; r < r_end; r++) {
                uint8_t l3a = A.top_raw[r];
                __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                    static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                A.l2_off += __builtin_popcount(l3a);
                __mmask64 ma = static_cast<__mmask64>(
                    _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    ma, A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                _mm512_storeu_si512(dst + r * 64, va);
            }
        }
    }

    if (avx_regions * 64 < total_words) {
        size_t tail_words = total_words - avx_regions * 64;
        uint8_t l1_fill_a = sa.l1_fill_ones ? 0xFF : 0x00;
        uint8_t l2_fill_a = sa.l2_fill_ones ? 0xFF : 0x00;
        uint8_t l3a = A.top_raw[avx_regions];
        size_t tail_l2_bytes = (tail_words + 7) / 8;
        for (size_t lb = 0; lb < tail_l2_bytes; lb++) {
            bool a_l2_lit = (l3a >> lb) & 1;
            uint8_t l2a = a_l2_lit ? A.l2_lits[A.l2_off++] : l2_fill_a;
            for (size_t bit = 0; bit < 8; bit++) {
                size_t i = lb * 8 + bit;
                if (i >= tail_words) break;
                bool a_lit = (l2a >> bit) & 1;
                dst[avx_regions * 64 + i] = a_lit ? A.l1_lits[A.l1_off++] : l1_fill_a;
            }
        }
    }
}

static void seg_self_and_l5(const DDCNSeg& sa, uint8_t* dst) {
    SideAvx A = make_side_avx(sa);

    constexpr size_t MAX_L4 = 256;
    alignas(64) uint8_t l4a_buf[MAX_L4] = {0};
    size_t l4_byte_count = (sa.l4_count + 7) / 8;
    assert(l4_byte_count <= MAX_L4);
    {
        uint8_t fill = sa.l4_fill_ones ? 0xFF : 0x00;
        size_t lit = 0;
        for (size_t i = 0; i < l4_byte_count; i++) {
            bool is_lit = (sa.top_raw[i / 8] >> (i % 8)) & 1;
            l4a_buf[i] = is_lit ? sa.l4_lits[lit++] : fill;
        }
    }

    const bool fills_zero = !sa.l1_fill_ones && !sa.l2_fill_ones
                         && !sa.l3_fill_ones;
    __m512i l3_fill_a_vec = sa.l3_fill_ones
        ? _mm512_set1_epi8(static_cast<char>(-1)) : _mm512_setzero_si512();

    size_t total_words = sa.l2_count;
    size_t avx_regions = total_words / 64;
    size_t l3a_lit_off = 0;
    size_t batch_count = (avx_regions + 63) / 64;

    for (size_t batch = 0; batch < batch_count; batch++) {
        size_t batch_start = batch * 64;
        size_t batch_end   = std::min(batch_start + 64, avx_regions);
        size_t batch_size  = batch_end - batch_start;

        uint64_t a_l4_mask = 0;
        std::memcpy(&a_l4_mask, l4a_buf + batch_start / 8, (batch_size + 7) / 8);
        if (batch_size < 64) {
            uint64_t valid = (uint64_t(1) << batch_size) - 1;
            a_l4_mask &= valid;
        }

        if (fills_zero && a_l4_mask == 0) {
            std::memset(dst + batch_start * 64, 0, batch_size * 64);
            continue;
        }

        __m512i l3a_chunk = _mm512_mask_expandloadu_epi8(l3_fill_a_vec,
            static_cast<__mmask64>(a_l4_mask), sa.l3_lits.data() + l3a_lit_off);
        l3a_lit_off += __builtin_popcountll(a_l4_mask);

        alignas(64) uint8_t l3a_buf[64];
        _mm512_store_si512(reinterpret_cast<__m512i*>(l3a_buf), l3a_chunk);

        if (fills_zero) {
            uint64_t nz = static_cast<uint64_t>(_mm512_test_epi8_mask(l3a_chunk, l3a_chunk));
            uint64_t valid = (batch_size < 64) ? ((uint64_t(1) << batch_size) - 1)
                                               : ~uint64_t(0);
            nz &= valid;
            if (nz == 0) {
                std::memset(dst + batch_start * 64, 0, batch_size * 64);
                continue;
            }
            std::memset(dst + batch_start * 64, 0, batch_size * 64);
            while (nz) {
                size_t r = __builtin_ctzll(nz);
                nz &= nz - 1;
                uint8_t l3a = l3a_buf[r];

                _mm_prefetch(reinterpret_cast<const char*>(A.l1_lits + A.l1_off + PF_DIST), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<char*>(dst + (batch_start + r) * 64 + PF_DIST), _MM_HINT_T0);
                __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                    static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                A.l2_off += __builtin_popcount(l3a);
                __mmask64 ma = static_cast<__mmask64>(
                    _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    ma, A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                _mm512_storeu_si512(dst + (batch_start + r) * 64, va);
            }
        } else {

            for (size_t r = 0; r < batch_size; r++) {
                uint8_t l3a = l3a_buf[r];
                __m512i l2a = _mm512_mask_expandloadu_epi8(A.l2_fill_vec,
                    static_cast<__mmask64>(l3a), A.l2_lits + A.l2_off);
                A.l2_off += __builtin_popcount(l3a);
                __mmask64 ma = static_cast<__mmask64>(
                    _mm_cvtsi128_si64(_mm512_castsi512_si128(l2a)));
                __m512i va = _mm512_mask_expandloadu_epi8(A.l1_fill_vec,
                    ma, A.l1_lits + A.l1_off);
                A.l1_off += __builtin_popcountll(static_cast<uint64_t>(ma));
                _mm512_storeu_si512(dst + (batch_start + r) * 64, va);
            }
        }
    }

    if (avx_regions * 64 < total_words) {
        auto la_full = expand_l1_stream(sa);
        for (size_t i = avx_regions * 64; i < total_words; i++)
            dst[i] = la_full[i];
    }
}

// AVX dispatch by depth -> decompressed bytes
template <OpKind OP>
static std::vector<uint8_t> apply_op_dec_avx(const DDCN& a, const DDCN& b) {
    assert(a.depth == b.depth);
    assert(a.bit_count == b.bit_count);
    size_t total = 0;
    for (const auto& sg : a.segments) total += sg.l2_count;
    std::vector<uint8_t> out(total);

    const bool is_self_and = (OP == OpKind::AND) && (&a == &b);

    size_t off = 0;
    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        const auto& sb = b.segments[s];
        if (is_self_and) {
            if      (a.depth == 2) seg_self_and_l2(sa, out.data() + off);
            else if (a.depth == 3) seg_self_and_l3(sa, out.data() + off);
            else if (a.depth == 5) seg_self_and_l5(sa, out.data() + off);
        } else {
            if      (a.depth == 2) seg_op_l2<OP>(sa, sb, out.data() + off);
            else if (a.depth == 3) seg_op_l3<OP>(sa, sb, out.data() + off);
            else if (a.depth == 4) seg_op_l4<OP>(sa, sb, out.data() + off);
            else if (a.depth == 5) seg_op_l5<OP>(sa, sb, out.data() + off);
        }
        off += sa.l2_count;
    }
    return out;
}

std::vector<uint8_t> ddc_n_and_dec_avx(const DDCN& a, const DDCN& b) {
    return apply_op_dec_avx<OpKind::AND>(a, b);
}
std::vector<uint8_t> ddc_n_or_dec_avx(const DDCN& a, const DDCN& b) {
    return apply_op_dec_avx<OpKind::OR>(a, b);
}
std::vector<uint8_t> ddc_n_xor_dec_avx(const DDCN& a, const DDCN& b) {
    return apply_op_dec_avx<OpKind::XOR>(a, b);
}

// op then re-compress result
template <OpKind OP>
static DDCN apply_op_compressed_avx(const DDCN& a, const DDCN& b) {
    assert(a.depth == b.depth);
    assert(a.bit_count == b.bit_count);
    assert(a.segments.size() == b.segments.size());

    DDCN result;
    result.depth = a.depth;
    result.bit_count = a.bit_count;
    result.segment_bits = a.segment_bits;
    result.segments.reserve(a.segments.size());

    alignas(64) static thread_local uint8_t scratch[65536 / 8 + 64];

    auto seg_struct_zero = [](const DDCNSeg& s) {
        if (s.l1_fill_ones) return false;
        if (s.depth >= 3 && s.l2_fill_ones) return false;
        if (s.depth >= 4 && s.l3_fill_ones) return false;
        if (s.depth >= 5 && s.l4_fill_ones) return false;

        const uint8_t* p = s.top_raw.data();
        size_t n = s.top_raw.size();
        size_t i = 0;
        for (; i + 64 <= n; i += 64) {
            __m512i v = _mm512_loadu_si512(p + i);
            if (_mm512_test_epi8_mask(v, v)) return false;
        }
        for (; i < n; i++) if (p[i]) return false;
        return true;
    };

    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        const auto& sb = b.segments[s];

        bool fill_ones;
        if constexpr (OP == OpKind::AND) fill_ones = sa.l1_fill_ones && sb.l1_fill_ones;
        else                             fill_ones = sa.l1_fill_ones || sb.l1_fill_ones;

        // whole-segment bypass
        const bool a_zero = seg_struct_zero(sa);
        const bool b_zero = seg_struct_zero(sb);
        if (a_zero && b_zero) {

            result.segments.push_back(sa);
            continue;
        }
        if constexpr (OP == OpKind::AND) {
            if (a_zero) { result.segments.push_back(sa); continue; }
            if (b_zero) { result.segments.push_back(sb); continue; }
        } else {
            if (a_zero) { result.segments.push_back(sb); continue; }
            if (b_zero) { result.segments.push_back(sa); continue; }
        }

        if      (a.depth == 2) seg_op_l2<OP>(sa, sb, scratch);
        else if (a.depth == 3) seg_op_l3<OP>(sa, sb, scratch);
        else if (a.depth == 4) seg_op_l4<OP>(sa, sb, scratch);
        else if (a.depth == 5) seg_op_l5<OP>(sa, sb, scratch);

        result.segments.push_back(
            ddc_n_compress_seg_from_l1(scratch, sa.l2_count, sa.bit_count,
                                          a.depth, fill_ones));
    }
    return result;
}

DDCN ddc_n_or (const DDCN& a, const DDCN& b) {
    return apply_op_compressed_avx<OpKind::OR>(a, b);
}
DDCN ddc_n_and(const DDCN& a, const DDCN& b) {
    return apply_op_compressed_avx<OpKind::AND>(a, b);
}

template <OpKind OP, bool B_BP, bool R_BP>
static std::vector<uint8_t> dec_l4_cfg(const DDCN& a, const DDCN& b) {
    assert(a.depth == 4 && b.depth == 4);
    size_t total = 0;
    for (const auto& sg : a.segments) total += sg.l2_count;
    std::vector<uint8_t> out(total);
    size_t off = 0;
    for (size_t s = 0; s < a.segments.size(); s++) {
        seg_op_l4<OP, B_BP, R_BP>(a.segments[s], b.segments[s], out.data() + off);
        off += a.segments[s].l2_count;
    }
    return out;
}

std::vector<uint8_t> ddc_n_or_dec_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg c) {
    switch (c) {
        case BypassCfg::BP_ALL:   return dec_l4_cfg<OpKind::OR, true,  true >(a, b);
        case BypassCfg::BP_NO_L4: return dec_l4_cfg<OpKind::OR, false, true >(a, b);
        case BypassCfg::BP_NO_L3: return dec_l4_cfg<OpKind::OR, true,  false>(a, b);
        case BypassCfg::BP_NONE:  return dec_l4_cfg<OpKind::OR, false, false>(a, b);
    }
    return {};
}
std::vector<uint8_t> ddc_n_and_dec_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg c) {
    switch (c) {
        case BypassCfg::BP_ALL:   return dec_l4_cfg<OpKind::AND, true,  true >(a, b);
        case BypassCfg::BP_NO_L4: return dec_l4_cfg<OpKind::AND, false, true >(a, b);
        case BypassCfg::BP_NO_L3: return dec_l4_cfg<OpKind::AND, true,  false>(a, b);
        case BypassCfg::BP_NONE:  return dec_l4_cfg<OpKind::AND, false, false>(a, b);
    }
    return {};
}

template <OpKind OP, bool B_BP, bool R_BP>
static DDCN compressed_l4_cfg(const DDCN& a, const DDCN& b) {
    assert(a.depth == 4 && b.depth == 4);
    DDCN result;
    result.depth = 4;
    result.bit_count = a.bit_count;
    result.segment_bits = a.segment_bits;
    result.segments.reserve(a.segments.size());
    alignas(64) static thread_local uint8_t scratch[65536 / 8 + 64];
    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        const auto& sb = b.segments[s];
        bool fill_ones;
        if constexpr (OP == OpKind::AND) fill_ones = sa.l1_fill_ones && sb.l1_fill_ones;
        else                             fill_ones = sa.l1_fill_ones || sb.l1_fill_ones;
        seg_op_l4<OP, B_BP, R_BP>(sa, sb, scratch);
        result.segments.push_back(
            ddc_n_compress_seg_from_l1(scratch, sa.l2_count, sa.bit_count, 4, fill_ones));
    }
    return result;
}

DDCN ddc_n_or_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg c) {
    switch (c) {
        case BypassCfg::BP_ALL:   return compressed_l4_cfg<OpKind::OR, true,  true >(a, b);
        case BypassCfg::BP_NO_L4: return compressed_l4_cfg<OpKind::OR, false, true >(a, b);
        case BypassCfg::BP_NO_L3: return compressed_l4_cfg<OpKind::OR, true,  false>(a, b);
        case BypassCfg::BP_NONE:  return compressed_l4_cfg<OpKind::OR, false, false>(a, b);
    }
    return {};
}
DDCN ddc_n_and_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg c) {
    switch (c) {
        case BypassCfg::BP_ALL:   return compressed_l4_cfg<OpKind::AND, true,  true >(a, b);
        case BypassCfg::BP_NO_L4: return compressed_l4_cfg<OpKind::AND, false, true >(a, b);
        case BypassCfg::BP_NO_L3: return compressed_l4_cfg<OpKind::AND, true,  false>(a, b);
        case BypassCfg::BP_NONE:  return compressed_l4_cfg<OpKind::AND, false, false>(a, b);
    }
    return {};
}

// NOT: flip fill + XOR lits
void ddc_n_not_inplace(DDCN& a) {
    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));

    for (auto& s : a.segments) {
        s.l1_fill_ones = !s.l1_fill_ones;

        uint8_t* data = s.l1_lits.data();
        size_t n = s.l1_lits.size();
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
    }

    if (a.bit_count % 8 != 0 && !a.segments.empty()) {
        DDCNSeg& last = a.segments.back();
        if (!last.l1_lits.empty()) {
            size_t valid_bits = a.bit_count % 8;
            last.l1_lits.back() &= static_cast<uint8_t>(0xFF << (8 - valid_bits));
        }
    }
}

#else
// scalar fallback (no AVX512)
std::vector<uint8_t> ddc_n_and_dec_avx(const DDCN& a, const DDCN& b) { return ddc_n_and_dec(a, b); }
std::vector<uint8_t> ddc_n_or_dec_avx (const DDCN& a, const DDCN& b) { return ddc_n_or_dec (a, b); }
std::vector<uint8_t> ddc_n_xor_dec_avx(const DDCN& a, const DDCN& b) { return ddc_n_xor_dec(a, b); }

DDCN ddc_n_or(const DDCN& a, const DDCN& b) {
    auto bytes = ddc_n_or_dec(a, b);
    DDCN r; r.depth = a.depth; r.bit_count = a.bit_count; r.segment_bits = a.segment_bits;
    r.segments.reserve(a.segments.size());
    size_t off = 0;
    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        bool fill_ones = sa.l1_fill_ones || b.segments[s].l1_fill_ones;
        r.segments.push_back(ddc_n_compress_seg_from_l1(bytes.data() + off,
            sa.l2_count, sa.bit_count, a.depth, fill_ones));
        off += sa.l2_count;
    }
    return r;
}
DDCN ddc_n_and(const DDCN& a, const DDCN& b) {
    auto bytes = ddc_n_and_dec(a, b);
    DDCN r; r.depth = a.depth; r.bit_count = a.bit_count; r.segment_bits = a.segment_bits;
    r.segments.reserve(a.segments.size());
    size_t off = 0;
    for (size_t s = 0; s < a.segments.size(); s++) {
        const auto& sa = a.segments[s];
        bool fill_ones = sa.l1_fill_ones && b.segments[s].l1_fill_ones;
        r.segments.push_back(ddc_n_compress_seg_from_l1(bytes.data() + off,
            sa.l2_count, sa.bit_count, a.depth, fill_ones));
        off += sa.l2_count;
    }
    return r;
}
void ddc_n_not_inplace(DDCN& a) {
    for (auto& s : a.segments) {
        s.l1_fill_ones = !s.l1_fill_ones;
        for (auto& b : s.l1_lits) b ^= 0xFF;
    }
    if (a.bit_count % 8 != 0 && !a.segments.empty()) {
        auto& last = a.segments.back();
        if (!last.l1_lits.empty()) {
            size_t valid_bits = a.bit_count % 8;
            last.l1_lits.back() &= static_cast<uint8_t>(0xFF << (8 - valid_bits));
        }
    }
}

std::vector<uint8_t> ddc_n_or_dec_l4_cfg (const DDCN& a, const DDCN& b, BypassCfg) { return ddc_n_or_dec(a, b); }
std::vector<uint8_t> ddc_n_and_dec_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg) { return ddc_n_and_dec(a, b); }
DDCN ddc_n_or_l4_cfg (const DDCN& a, const DDCN& b, BypassCfg) { return ddc_n_or(a, b); }
DDCN ddc_n_and_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg) { return ddc_n_and(a, b); }
#endif
