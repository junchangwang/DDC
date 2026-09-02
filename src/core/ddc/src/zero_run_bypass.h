#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ddc_zrb {

inline bool& enabled_ref() {
    static bool cached = [] {
        const char* e = std::getenv("DDC_ZERO_RUN_BYPASS");
        return !(e && *e == '0');
    }();
    return cached;
}
inline bool enabled() { return enabled_ref(); }
inline void set_enabled(bool v) { enabled_ref() = v; }

inline bool& run_hint_ref() {
    static thread_local bool v = false;
    return v;
}
inline bool run_hint() { return run_hint_ref(); }
inline void set_run_hint(bool v) { run_hint_ref() = v; }

inline bool& debug_ref() {
    static bool cached = [] {
        const char* e = std::getenv("DDC_BYPASS_DEBUG");
        return e && *e == '1';
    }();
    return cached;
}
inline void set_debug(bool v) { debug_ref() = v; }
inline bool debug_on() { return debug_ref(); }

template <class Segs, class Pred>
inline void build_mask(const Segs& segs, std::vector<uint64_t>& m, Pred pred) {
    const size_t n = segs.size();
    m.assign((n + 63) / 64, 0);
    for (size_t i = 0; i < n; i++)
        if (pred(segs[i])) m[i >> 6] |= 1ull << (i & 63);
}

inline bool test_bit(const uint64_t* m, size_t i) {
    return (m[i >> 6] >> (i & 63)) & 1;
}
inline bool test_bit(const std::vector<uint64_t>& m, size_t i) {
    return test_bit(m.data(), i);
}

inline size_t run_len_set(const uint64_t* m, size_t i, size_t n) {
    const size_t words = (n + 63) / 64;
    size_t w = i >> 6, off = i & 63, len;
    uint64_t inv = ~(m[w] >> off);
    if (inv == 0) {
        len = 64 - off;
        for (w++; w < words; w++) {
            if (~m[w] == 0) { len += 64; continue; }
            len += (size_t)__builtin_ctzll(~m[w]);
            break;
        }
    } else {
        len = (size_t)__builtin_ctzll(inv);
    }
    const size_t cap = n - i;
    return len < cap ? len : cap;
}
inline size_t run_len_set(const std::vector<uint64_t>& m, size_t i, size_t n) {
    return run_len_set(m.data(), i, n);
}

inline size_t run_len_clear(const uint64_t* m, size_t i, size_t n) {
    const size_t words = (n + 63) / 64;
    size_t w = i >> 6, off = i & 63, len;
    uint64_t bits = m[w] >> off;
    if (bits == 0) {
        len = 64 - off;
        for (w++; w < words; w++) {
            if (m[w] == 0) { len += 64; continue; }
            len += (size_t)__builtin_ctzll(m[w]);
            break;
        }
    } else {
        len = (size_t)__builtin_ctzll(bits);
    }
    const size_t cap = n - i;
    return len < cap ? len : cap;
}
inline size_t run_len_clear(const std::vector<uint64_t>& m, size_t i, size_t n) {
    return run_len_clear(m.data(), i, n);
}

template <class SegVec, class IdVec>
struct SegView {
    const SegVec* segs;
    const IdVec* ids;
    size_t k = 0;
    inline const typename SegVec::value_type* at(size_t i) {
        if (!ids) return &(*segs)[i];
        while (k < ids->size() && (*ids)[k] < i) k++;
        return (k < ids->size() && (*ids)[k] == i) ? &(*segs)[k] : nullptr;
    }
    template <class Out, class OutIds>
    inline void emit_range(size_t from, size_t to, Out& out,
                           OutIds& out_ids) {
        if (!ids) {
            out.insert(out.end(), segs->begin() + from, segs->begin() + to);
            for (size_t i = from; i < to; i++)
                out_ids.push_back((uint32_t)i);
            return;
        }
        while (k < ids->size() && (*ids)[k] < from) k++;
        while (k < ids->size() && (*ids)[k] < to) {
            out.push_back((*segs)[k]);
            out_ids.push_back((*ids)[k]);
            k++;
        }
    }
};

template <class SegVec, class IdVec, class RunVec, class MaskVec>
inline void build_masks_any(const SegVec& segs,
                            const IdVec* ids,
                            const RunVec* ones_runs,
                            MaskVec& z, MaskVec& o,
                            size_t n) {
    const size_t words = (n + 63) / 64;
    o.assign(words, 0);
    if (!ids) {
        z.assign(words, 0);
        for (size_t i = 0; i < n; i++) {
            const auto& s = segs[i];
            if (s.is_all_zero())      z[i >> 6] |= 1ull << (i & 63);
            else if (s.is_all_ones()) o[i >> 6] |= 1ull << (i & 63);
        }
        return;
    }
    z.assign(words, ~0ull);
    if (n & 63) z.back() = (1ull << (n & 63)) - 1;
    for (size_t k = 0; k < ids->size(); k++) {
        const uint32_t id = (*ids)[k];
        if (!segs[k].is_all_zero()) {
            z[id >> 6] &= ~(1ull << (id & 63));
            if (segs[k].is_all_ones()) o[id >> 6] |= 1ull << (id & 63);
        }
    }
    if (ones_runs)
        for (const auto& run : *ones_runs)
            for (uint32_t id = run.first; id < run.first + run.second; id++) {
                z[id >> 6] &= ~(1ull << (id & 63));
                o[id >> 6] |= 1ull << (id & 63);
            }
}

struct JumpLog {
    size_t jumps = 0, jumped_segs = 0, max_span = 0;
    inline void jump(size_t L) {
        jumps++; jumped_segs += L;
        if (L > max_span) max_span = L;
    }
};

template <class MaskVec>
inline void report(const char* op, size_t segment_bits, size_t n,
                   const MaskVec& za, const MaskVec& oma,
                   const MaskVec& zb, const MaskVec& omb,
                   const JumpLog& jl) {
    size_t nza = 0, nzb = 0, oa = 0, ob = 0;
    for (uint64_t w : za) nza += (size_t)__builtin_popcountll(w);
    for (uint64_t w : zb) nzb += (size_t)__builtin_popcountll(w);
    for (uint64_t w : oma) oa += (size_t)__builtin_popcountll(w);
    for (uint64_t w : omb) ob += (size_t)__builtin_popcountll(w);
    std::fprintf(stderr,
        "[zrb] %s seg_bits=%zu segs=%zu | A zero=%zu ones=%zu mixed=%zu"
        " | B zero=%zu ones=%zu mixed=%zu | jumps=%zu segs_jumped=%zu max_span=%zu\n",
        op, segment_bits, n, nza, oa, n - nza - oa, nzb, ob, n - nzb - ob,
        jl.jumps, jl.jumped_segs, jl.max_span);
}

}
