// compressed_wavelet.hpp — entropy-shaped, VALUE-ORDERED compressed wavelet.
//
// The plain wavelet matrix spends ceil(log2 sigma) bits/elem regardless of skew.
// This variant shapes the tree by symbol frequency so each value's code length
// ~ -log2(freq/n): total stored bits = sum_v freq[v]*depth[v] -> ~ n*H0 (zeroth-
// order entropy). Crucially the tree is kept in VALUE ORDER (leaves left->right
// are values 0..sigma-1), so it is an OPTIMAL ALPHABETIC binary tree (Hu-Tucker
// class), and range_count / quantile / k-th-smallest still work — unlike a plain
// Huffman tree, whose leaves are frequency-ordered and break value-range queries.
//
// The optimal alphabetic tree is computed by an interval DP
//   dp[i][j] = W(i,j) + min_{i<=k<j} ( dp[i][k] + dp[k+1][j] )
// accelerated to O(sigma^2) by Knuth's optimization (the split point is monotone;
// W = prefix-sum of frequencies satisfies the quadrangle inequality). Bitvectors
// stay plain, so AVX-512 VPOPCNTDQ rank is reused unchanged.
#pragma once
#include "bitvector.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace wm {

class CompressedWavelet {
public:
    CompressedWavelet() : n_(0), sigma_(1), root_(-1), total_bits_(0) {}
    explicit CompressedWavelet(const std::vector<uint32_t>& data) { build(data); }
    void build(const std::vector<uint32_t>& data);

    size_t   size()  const { return n_; }
    uint64_t sigma() const { return sigma_; }

    uint32_t access(size_t i) const;
    size_t   count_lt(size_t l, size_t r, uint64_t t) const;
    size_t   range_count(size_t l, size_t r, uint32_t lo, uint32_t hi) const {
        if (lo >= hi) return 0;
        return count_lt(l, r, hi) - count_lt(l, r, lo);
    }
    uint32_t quantile(size_t l, size_t r, size_t k) const;
    size_t   rank(uint32_t c, size_t i) const {
        if (c >= sigma_) return 0;
        return count_lt(0, i, (uint64_t)c + 1) - count_lt(0, i, (uint64_t)c);
    }

    size_t mem_bytes() const;
    size_t total_code_bits() const { return total_bits_; }  // == sum freq*depth (== DP optimum)

    // storage breakdown by component (our "containers"): data bitvectors / rank index / tree structure
    struct Breakdown { size_t data_bytes = 0, rank_bytes = 0, struct_bytes = 0, total_bytes = 0; };
    Breakdown breakdown() const {
        Breakdown b;
        for (const auto& nd : nodes_) if (nd.left != -1) {
            b.data_bytes += nd.bv.words_size() * sizeof(uint64_t);
            b.rank_bytes += nd.bv.sb_size() * sizeof(uint64_t);
        }
        b.struct_bytes = nodes_.size() * (sizeof(uint32_t) * 3 + sizeof(int) * 2);
        b.total_bytes = b.data_bytes + b.rank_bytes + b.struct_bytes;
        return b;
    }

private:
    struct Node {
        uint32_t lo = 0, hi = 0, mid = 0;   // covers values [lo,hi); split at mid
        int left = -1, right = -1;          // child node indices; leaf iff left==-1
        BitVector bv;                       // present on internal nodes only
    };
    size_t n_;
    uint64_t sigma_;
    int root_;
    size_t total_bits_;
    std::vector<Node> nodes_;

    static constexpr int DP_THRESHOLD = 4096;   // optimal DP below this, weight-balanced above
    static uint32_t weight_median(uint32_t i, uint32_t j, const std::vector<size_t>& pre);
    int build_rec(uint32_t i, uint32_t j, int S, const std::vector<int>* sp,
                  const std::vector<size_t>& pre, std::vector<uint32_t>&& vals);
};

// split [i,j] (values i..j) into [i,k] and [k+1,j] at the weight median (Shannon-Fano
// alphabetic): near-optimal (within ~1 bit/symbol of H0), O(log sigma) per split, O(sigma) space.
inline uint32_t CompressedWavelet::weight_median(uint32_t i, uint32_t j,
                                                 const std::vector<size_t>& pre) {
    const size_t total = pre[j + 1] - pre[i];
    const size_t target = pre[i] + total / 2;          // cumulative weight target
    uint32_t L = i, R = j - 1;                          // largest k in [i,j-1] with pre[k+1] <= target
    while (L < R) { uint32_t m = (L + R + 1) / 2; if (pre[m + 1] <= target) L = m; else R = m - 1; }
    const uint32_t k1 = L, k2 = (L + 1 <= j - 1) ? L + 1 : L;
    auto imbalance = [&](uint32_t k) {
        const size_t left = pre[k + 1] - pre[i], right = total - left;
        return left > right ? left - right : right - left;
    };
    return imbalance(k1) <= imbalance(k2) ? k1 : k2;
}

inline void CompressedWavelet::build(const std::vector<uint32_t>& data) {
    n_ = data.size();
    nodes_.clear(); root_ = -1; total_bits_ = 0;
    uint32_t mx = 0; for (uint32_t v : data) mx = std::max(mx, v);
    sigma_ = (uint64_t)mx + 1;
    if (n_ == 0) return;
    const int S = (int)sigma_;

    std::vector<size_t> pre(S + 1, 0);            // frequency prefix sums; W(i,j)=pre[j+1]-pre[i]
    for (uint32_t v : data) pre[v + 1]++;
    for (int i = 0; i < S; ++i) pre[i + 1] += pre[i];

    nodes_.reserve(2 * (size_t)S);
    std::vector<uint32_t> vals = data;

    if (S <= DP_THRESHOLD) {
        // OPTIMAL alphabetic binary tree, Knuth-optimized O(S^2) time/space
        auto W = [&](int i, int j) { return pre[j + 1] - pre[i]; };
        const size_t INF = std::numeric_limits<size_t>::max();
        std::vector<size_t> dp((size_t)S * S, 0);
        std::vector<int>    sp((size_t)S * S, 0);
        auto DP = [&](int i, int j) -> size_t& { return dp[(size_t)i * S + j]; };
        auto SP = [&](int i, int j) -> int&    { return sp[(size_t)i * S + j]; };
        for (int i = 0; i < S; ++i) { DP(i, i) = 0; SP(i, i) = i; }
        for (int len = 2; len <= S; ++len)
            for (int i = 0; i + len - 1 < S; ++i) {
                int j = i + len - 1, klo = SP(i, j - 1), khi = SP(i + 1, j);
                if (klo < i) klo = i;
                if (khi > j - 1) khi = j - 1;
                size_t best = INF; int bk = klo;
                for (int k = klo; k <= khi; ++k) { size_t c = DP(i, k) + DP(k + 1, j); if (c < best) { best = c; bk = k; } }
                DP(i, j) = best + W(i, j); SP(i, j) = bk;
            }
        root_ = build_rec(0, S - 1, S, &sp, pre, std::move(vals));
    } else {
        // NEAR-OPTIMAL Shannon-Fano alphabetic tree: O(S log S) time, O(S) space (no S^2 table)
        root_ = build_rec(0, S - 1, S, nullptr, pre, std::move(vals));
    }
    for (const auto& nd : nodes_) if (nd.left != -1) total_bits_ += nd.bv.size();   // = sum freq*depth
}

inline int CompressedWavelet::build_rec(uint32_t i, uint32_t j, int S,
                                        const std::vector<int>* sp,
                                        const std::vector<size_t>& pre,
                                        std::vector<uint32_t>&& vals) {
    const int idx = (int)nodes_.size();
    nodes_.emplace_back();
    nodes_[idx].lo = i; nodes_[idx].hi = j + 1;
    if (i == j) { nodes_[idx].mid = i; return idx; }     // leaf

    const uint32_t k = sp ? (uint32_t)(*sp)[(size_t)i * S + j] : weight_median(i, j, pre);
    const uint32_t mid = k + 1;                           // left [i,k] (<mid), right [k+1,j]
    const size_t m = vals.size();
    std::vector<uint64_t> words((m + 63) / 64, 0);
    std::vector<uint32_t> lvals, rvals;
    lvals.reserve(m); rvals.reserve(m);
    for (size_t p = 0; p < m; ++p) {
        if (vals[p] >= mid) { words[p >> 6] |= (1ull << (p & 63)); rvals.push_back(vals[p]); }
        else lvals.push_back(vals[p]);
    }
    vals.clear(); vals.shrink_to_fit();
    nodes_[idx].mid = mid;
    nodes_[idx].bv = BitVector(std::move(words), m);
    const int L = build_rec(i, k, S, sp, pre, std::move(lvals));
    const int R = build_rec(k + 1, j, S, sp, pre, std::move(rvals));
    nodes_[idx].left = L; nodes_[idx].right = R;
    return idx;
}

inline uint32_t CompressedWavelet::access(size_t i) const {
    if (i >= n_) return 0;
    int u = root_; size_t pos = i;
    while (nodes_[u].left != -1) {
        const Node& nd = nodes_[u];
        if (nd.bv.get(pos)) { pos = nd.bv.rank1(pos); u = nd.right; }
        else                { pos = nd.bv.rank0(pos); u = nd.left;  }
    }
    return nodes_[u].lo;
}

inline size_t CompressedWavelet::count_lt(size_t l, size_t r, uint64_t t) const {
    if (n_ == 0 || l >= r || t == 0) return 0;
    if (t >= sigma_) return r - l;
    size_t cnt = 0; int u = root_;
    while (nodes_[u].left != -1) {
        const Node& nd = nodes_[u];
        const size_t l0 = nd.bv.rank0(l), r0 = nd.bv.rank0(r);
        if (t <= nd.mid) {                 // all values < t live in the left child
            l = l0; r = r0; u = nd.left;
        } else {                           // left child (< mid <= t-1) all count; recurse right
            cnt += (r0 - l0);
            l = l - l0; r = r - r0; u = nd.right;
        }
        if (l >= r) return cnt;
    }
    if (nodes_[u].lo < t) cnt += (r - l);  // leaf value < t
    return cnt;
}

inline uint32_t CompressedWavelet::quantile(size_t l, size_t r, size_t k) const {
    if (n_ == 0 || l >= r || k >= (r - l)) return 0;
    int u = root_;
    while (nodes_[u].left != -1) {
        const Node& nd = nodes_[u];
        const size_t l0 = nd.bv.rank0(l), r0 = nd.bv.rank0(r), zeros = r0 - l0;
        if (k < zeros) { l = l0; r = r0; u = nd.left; }
        else { k -= zeros; l = l - l0; r = r - r0; u = nd.right; }
    }
    return nodes_[u].lo;
}

inline size_t CompressedWavelet::mem_bytes() const {
    size_t b = nodes_.size() * (sizeof(uint32_t) * 3 + sizeof(int) * 2);  // tree overhead
    for (const auto& nd : nodes_) b += nd.bv.mem_bytes();
    return b;
}

} // namespace wm
