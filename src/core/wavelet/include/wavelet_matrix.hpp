// wavelet_matrix.hpp — a succinct column index (Claude & Navarro's wavelet matrix).
//
// Stores a column of N values over alphabet [0, sigma) in ~N*ceil(log2 sigma)
// bits plus rank samples (near the information-theoretic floor), and answers
// the operators a compressed bitmap (Roaring) is BAD at, directly on the
// compressed form: access / rank / range_count / quantile. Navigation is all
// rank0/rank1 (AVX-512 in bitvector.hpp). Predicate row-sets are combined with
// AVX-512 AND/OR/XOR/NOT in rowset.hpp; range_predicate() bridges the two.
#pragma once
#include "bitvector.hpp"
#include "rowset.hpp"
#include <cstdint>
#include <vector>
#include <algorithm>
#include <utility>

namespace wm {

class WaveletMatrix {
public:
    WaveletMatrix() : n_(0), L_(0), sigma_(1) {}
    explicit WaveletMatrix(const std::vector<uint32_t>& data) { build(data); }

    void build(const std::vector<uint32_t>& data);

    size_t   size()   const { return n_; }
    uint64_t sigma()  const { return sigma_; }   // mx+1; can be 2^32, hence 64-bit
    int      levels() const { return L_; }

    uint32_t access(size_t i) const;
    size_t   rank(uint32_t c, size_t i) const;               // # of c in [0, i)
    size_t   count_lt(size_t l, size_t r, uint64_t t) const; // # in [l,r) with value < t
    size_t   range_count(size_t l, size_t r, uint32_t lo, uint32_t hi) const {
        if (lo >= hi) return 0;
        return count_lt(l, r, hi) - count_lt(l, r, lo);
    }
    uint32_t quantile(size_t l, size_t r, size_t k) const;   // k-th smallest (0-indexed)

    // ---- ORIGINAL fused predicate+aggregate operators (one descent, zero materialization) ----
    // These exploit value-order + position together: a single tree descent over the value
    // range yields the aggregate directly on the compressed form. Bitmaps/BSI must instead
    // materialize a row-set and scan the raw column. Cost is O(value-range visited), i.e.
    // INDEPENDENT of (r-l) — the win for large-range / whole-column OLAP aggregates.
    template <class F> void for_each_value(size_t l, size_t r, uint32_t lo, uint32_t hi, F cb) const;
    uint64_t filtered_sum(size_t l, size_t r, uint32_t lo, uint32_t hi) const;            // SUM of values in [lo,hi) over rows [l,r)
    double   filtered_avg(size_t l, size_t r, uint32_t lo, uint32_t hi) const;            // AVG (= sum/count)
    std::vector<std::pair<uint32_t, size_t>> groupby_count(size_t l, size_t r, uint32_t lo, uint32_t hi) const; // (value,count) by value
    std::vector<std::pair<uint32_t, size_t>> top_k_freq(size_t l, size_t r, uint32_t lo, uint32_t hi, size_t k) const; // k most frequent values
    // implicit zone-map: min/max of any row range in O(log sigma) (subsumes a separate min-max/zone-map index)
    uint32_t range_min(size_t l, size_t r) const { return (l < r) ? quantile(l, r, 0) : 0; }
    uint32_t range_max(size_t l, size_t r) const { return (l < r) ? quantile(l, r, (r - l) - 1) : 0; }

    // Materialize {i : lo <= value(i) < hi} via an O(N*L) access scan
    // (correct; doubles as the reference for range_predicate_fast).
    RowSet range_predicate(uint32_t lo, uint32_t hi) const;

    // Same result via wavelet range-report: descend to the value range and map
    // matching positions up through the levels with select, touching O(matches)
    // rather than every row. Sub-linear for SELECTIVE predicates; for dense ones
    // it is ~O(N) (the price of not storing per-value bitmaps — by design).
    RowSet range_predicate_fast(uint32_t lo, uint32_t hi) const;

    // Never-lose: O(log sigma) cardinality probe picks report (selective) vs scan (dense).
    RowSet range_predicate_auto(uint32_t lo, uint32_t hi) const;

    size_t mem_bytes() const;

private:
    size_t map_up(int lev, size_t q) const;   // level-`lev` position -> level-0 (original) index

    size_t n_;
    int L_;
    uint64_t sigma_;                 // mx + 1 (widened so UINT32_MAX can't wrap)
    std::vector<BitVector> level_;   // L_ bitvectors, MSB at level 0
    std::vector<size_t> z_;          // z_[l] = # of zeros at level l
};

inline void WaveletMatrix::build(const std::vector<uint32_t>& data) {
    n_ = data.size();
    uint32_t mx = 0;
    for (uint32_t v : data) mx = std::max(mx, v);
    sigma_ = (uint64_t)mx + 1;                          // widened: no wrap at UINT32_MAX
    L_ = (mx == 0) ? 1 : (32 - __builtin_clz(mx));      // bits needed to hold [0, mx]

    level_.clear();
    level_.reserve(L_);
    z_.assign(L_, 0);
    const size_t nw = (n_ + 63) / 64;
    std::vector<uint32_t> cur = data, nxt(n_);

    for (int l = 0; l < L_; ++l) {
        const int bitpos = L_ - 1 - l;
        std::vector<uint64_t> words(nw, 0ull);
        for (size_t i = 0; i < n_; ++i)
            if ((cur[i] >> bitpos) & 1u) words[i >> 6] |= (1ull << (i & 63));

        BitVector bv(std::move(words), n_);
        const size_t zeros = n_ - bv.ones();
        z_[l] = zeros;

        size_t p0 = 0, p1 = zeros;             // stable partition: 0s then 1s
        for (size_t i = 0; i < n_; ++i) {
            if ((cur[i] >> bitpos) & 1u) nxt[p1++] = cur[i];
            else                         nxt[p0++] = cur[i];
        }
        level_.push_back(std::move(bv));
        std::swap(cur, nxt);
    }
}

inline uint32_t WaveletMatrix::access(size_t i) const {
    if (i >= n_) return 0;                     // guard empty matrix / OOB
    uint32_t val = 0;
    size_t pos = i;
    for (int l = 0; l < L_; ++l) {
        const BitVector& bv = level_[l];
        const uint32_t b = bv.get(pos) ? 1u : 0u;
        val = (val << 1) | b;
        pos = b ? z_[l] + bv.rank1(pos) : bv.rank0(pos);
    }
    return val;
}

inline size_t WaveletMatrix::count_lt(size_t l, size_t r, uint64_t t) const {
    if (r > n_) r = n_;                        // clamp range end into the column
    if (l >= r || t == 0) return 0;
    if (t >= sigma_) return r - l;             // every value (<= mx) is < t
    size_t cnt = 0;
    for (int lev = 0; lev < L_; ++lev) {
        const BitVector& bv = level_[lev];
        const uint64_t bit = (t >> (L_ - 1 - lev)) & 1ull;
        const size_t l0 = bv.rank0(l), r0 = bv.rank0(r);
        if (bit) {
            cnt += (r0 - l0);                  // all 0-branch elems are < t
            l = z_[lev] + (l - l0);            // descend into 1-branch
            r = z_[lev] + (r - r0);
        } else {
            l = l0; r = r0;                    // descend into 0-branch
        }
    }
    return cnt;
}

inline size_t WaveletMatrix::rank(uint32_t c, size_t i) const {
    if (c >= sigma_) return 0;
    return count_lt(0, i, (uint64_t)c + 1) - count_lt(0, i, (uint64_t)c);
}

inline uint32_t WaveletMatrix::quantile(size_t l, size_t r, size_t k) const {
    if (r > n_) r = n_;
    if (l >= r || k >= (r - l)) return 0;      // empty range / out-of-range k
    uint32_t val = 0;
    for (int lev = 0; lev < L_; ++lev) {
        const BitVector& bv = level_[lev];
        const size_t l0 = bv.rank0(l), r0 = bv.rank0(r);
        const size_t zeros = r0 - l0;
        if (k < zeros) {
            l = l0; r = r0;
        } else {
            k -= zeros;
            val |= (1u << (L_ - 1 - lev));
            l = z_[lev] + (l - l0);
            r = z_[lev] + (r - r0);
        }
    }
    return val;
}

inline RowSet WaveletMatrix::range_predicate(uint32_t lo, uint32_t hi) const {
    RowSet rs(n_);
    for (size_t i = 0; i < n_; ++i) {
        const uint32_t v = access(i);
        if (v >= lo && v < hi) rs.set(i);
    }
    return rs;
}

inline size_t WaveletMatrix::map_up(int lev, size_t q) const {
    for (int l = lev - 1; l >= 0; --l)
        q = (q < z_[l]) ? level_[l].select0(q) : level_[l].select1(q - z_[l]);
    return q;
}

inline RowSet WaveletMatrix::range_predicate_fast(uint32_t vlo, uint32_t vhi) const {
    RowSet rs(n_);
    if (n_ == 0 || vlo >= vhi) return rs;
    struct Frame { int lev; uint64_t a, b; size_t l, r; };  // node = values [a,b), positions [l,r) at level lev
    std::vector<Frame> st;
    st.push_back({0, 0, (uint64_t)1 << L_, 0, n_});
    while (!st.empty()) {
        const Frame f = st.back(); st.pop_back();
        if (f.l >= f.r || f.b <= vlo || f.a >= vhi) continue;          // empty or disjoint
        if (f.a >= vlo && f.b <= vhi) {                                // fully contained
            for (size_t q = f.l; q < f.r; ++q) rs.set(map_up(f.lev, q));
            continue;
        }
        const BitVector& bv = level_[f.lev];                          // partial -> descend
        const uint64_t mid = (f.a + f.b) >> 1;
        const size_t l0 = bv.rank0(f.l), r0 = bv.rank0(f.r);
        st.push_back({f.lev + 1, f.a, mid, l0, r0});
        st.push_back({f.lev + 1, mid, f.b, z_[f.lev] + (f.l - l0), z_[f.lev] + (f.r - r0)});
    }
    return rs;
}

inline RowSet WaveletMatrix::range_predicate_auto(uint32_t lo, uint32_t hi) const {
    if (lo >= hi) return RowSet(n_);
    const size_t cnt = range_count(0, n_, lo, hi);          // O(log sigma) selectivity probe
    return (cnt * 8 < n_) ? range_predicate_fast(lo, hi)    // < ~12% -> report
                          : range_predicate(lo, hi);        // dense -> scan
}

template <class F>
inline void WaveletMatrix::for_each_value(size_t l, size_t r, uint32_t vlo, uint32_t vhi, F cb) const {
    if (l >= r || vlo >= vhi || n_ == 0) return;
    struct Frame { int lev; uint64_t a, b; size_t l, r; };  // node = values [a,b), positions [l,r) at level lev
    std::vector<Frame> st;
    st.push_back({0, 0, (uint64_t)1 << L_, l, r});
    while (!st.empty()) {
        const Frame f = st.back(); st.pop_back();
        if (f.l >= f.r || f.b <= vlo || f.a >= vhi) continue;          // empty or disjoint
        if (f.b - f.a == 1) { cb((uint32_t)f.a, f.r - f.l); continue; } // leaf in [vlo,vhi): value f.a, count r-l
        const BitVector& bv = level_[f.lev];
        const uint64_t mid = (f.a + f.b) >> 1;
        const size_t l0 = bv.rank0(f.l), r0 = bv.rank0(f.r);
        st.push_back({f.lev + 1, f.a, mid, l0, r0});
        st.push_back({f.lev + 1, mid, f.b, z_[f.lev] + (f.l - l0), z_[f.lev] + (f.r - r0)});
    }
}

inline uint64_t WaveletMatrix::filtered_sum(size_t l, size_t r, uint32_t lo, uint32_t hi) const {
    uint64_t s = 0;
    for_each_value(l, r, lo, hi, [&](uint32_t v, size_t c) { s += (uint64_t)v * c; });
    return s;
}

inline double WaveletMatrix::filtered_avg(size_t l, size_t r, uint32_t lo, uint32_t hi) const {
    uint64_t s = 0, cnt = 0;
    for_each_value(l, r, lo, hi, [&](uint32_t v, size_t c) { s += (uint64_t)v * c; cnt += c; });
    return cnt ? (double)s / (double)cnt : 0.0;
}

inline std::vector<std::pair<uint32_t, size_t>>
WaveletMatrix::groupby_count(size_t l, size_t r, uint32_t lo, uint32_t hi) const {
    std::vector<std::pair<uint32_t, size_t>> out;
    for_each_value(l, r, lo, hi, [&](uint32_t v, size_t c) { out.emplace_back(v, c); });
    std::sort(out.begin(), out.end());                      // by value ascending
    return out;
}

inline std::vector<std::pair<uint32_t, size_t>>
WaveletMatrix::top_k_freq(size_t l, size_t r, uint32_t lo, uint32_t hi, size_t k) const {
    auto h = groupby_count(l, r, lo, hi);
    auto by_freq = [](const std::pair<uint32_t, size_t>& a, const std::pair<uint32_t, size_t>& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    };
    if (h.size() > k) { std::partial_sort(h.begin(), h.begin() + k, h.end(), by_freq); h.resize(k); }
    else std::sort(h.begin(), h.end(), by_freq);
    return h;
}

inline size_t WaveletMatrix::mem_bytes() const {
    size_t b = z_.size() * sizeof(size_t);
    for (const auto& bv : level_) b += bv.mem_bytes();
    return b;
}

} // namespace wm
