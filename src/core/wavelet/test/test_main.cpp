// test_main.cpp — correctness (vs brute force) + edge cases (the review's
// confirmed bugs) + micro-benchmark.
#include "wavelet_matrix.hpp"
#include "rowset.hpp"
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace wm;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
#define CHECK(cond) do { if(!(cond)){ \
    std::fprintf(stderr,"FAIL %s:%d  %s\n",__FILE__,__LINE__,#cond); std::exit(1);} } while(0)

static void bitvector_checks() {
    std::mt19937 rng(777);
    for (size_t nbits : {1u, 63u, 64u, 65u, 511u, 512u, 513u, 5000u, 100000u}) {
        std::vector<uint64_t> words((nbits + 63) / 64, 0);
        std::vector<size_t> ones_pos;
        for (size_t i = 0; i < nbits; ++i)
            if (rng() & 1) { words[i >> 6] |= (1ull << (i & 63)); ones_pos.push_back(i); }
        BitVector bv(words, nbits);
        size_t acc = 0;
        for (size_t i = 0; i <= nbits; ++i) {
            CHECK(bv.rank1(i) == acc);
            if (i < nbits && bv.get(i)) ++acc;
        }
        CHECK(bv.ones() == ones_pos.size());
        for (size_t k = 0; k < ones_pos.size(); ++k) CHECK(bv.select1(k) == ones_pos[k]);
        // out-of-range select1 must return the sentinel nbits_, NOT read OOB (review C1/C6)
        CHECK(bv.select1(ones_pos.size()) == nbits);
        CHECK(bv.select1(ones_pos.size() + 50) == nbits);
    }
    // default-constructed / all-zero / empty bitvectors must be safe (review C2)
    BitVector def;                       CHECK(def.rank1(0) == 0); CHECK(def.select1(0) == 0);
    BitVector zeros(std::vector<uint64_t>(8, 0), 512);
    CHECK(zeros.ones() == 0); CHECK(zeros.rank1(512) == 0); CHECK(zeros.select1(0) == 512);
    std::printf("[OK] BitVector rank1/select1 + out-of-range/empty/default safety\n");
}

static void correctness() {
    std::mt19937 rng(12345);
    const size_t n = 5000;
    const uint32_t sigma = 37;                 // deliberately not a power of two
    std::vector<uint32_t> data(n);
    for (auto& x : data) x = rng() % sigma;
    WaveletMatrix w(data);

    for (size_t i = 0; i < n; ++i) CHECK(w.access(i) == data[i]);

    std::uniform_int_distribution<size_t>   dpos(0, n);
    std::uniform_int_distribution<uint32_t> dval(0, sigma);
    for (int t = 0; t < 3000; ++t) {
        size_t a = dpos(rng), b = dpos(rng); if (a > b) std::swap(a, b);
        uint32_t lo = dval(rng), hi = dval(rng); if (lo > hi) std::swap(lo, hi);
        size_t ref = 0;
        for (size_t i = a; i < b; ++i) if (data[i] >= lo && data[i] < hi) ++ref;
        CHECK(w.range_count(a, b, lo, hi) == ref);
        uint32_t c = dval(rng) % sigma;
        size_t rr = 0;
        for (size_t i = 0; i < b; ++i) if (data[i] == c) ++rr;
        CHECK(w.rank(c, b) == rr);
        if (b > a) {
            std::vector<uint32_t> sub(data.begin() + a, data.begin() + b);
            std::sort(sub.begin(), sub.end());
            size_t k = rng() % (b - a);
            CHECK(w.quantile(a, b, k) == sub[k]);
        }
    }
    std::printf("[OK] wavelet access/rank/range_count/quantile vs brute force (n=%zu sigma=%u L=%d)\n",
                n, sigma, w.levels());

    for (int t = 0; t < 200; ++t) {
        uint32_t lo1 = dval(rng), hi1 = dval(rng); if (lo1 > hi1) std::swap(lo1, hi1);
        uint32_t lo2 = dval(rng), hi2 = dval(rng); if (lo2 > hi2) std::swap(lo2, hi2);
        RowSet A = w.range_predicate(lo1, hi1);
        RowSet B = w.range_predicate(lo2, hi2);
        RowSet rAND = RowSet::AND(A, B), rOR = RowSet::OR(A, B);
        RowSet rXOR = RowSet::XOR(A, B), rNOT = RowSet::NOT(A), rANB = RowSet::ANDNOT(A, B);
        size_t cAND = 0, cOR = 0, cXOR = 0;
        for (size_t i = 0; i < n; ++i) {
            bool a = (data[i] >= lo1 && data[i] < hi1);
            bool b = (data[i] >= lo2 && data[i] < hi2);
            CHECK(rAND.get(i) == (a && b));
            CHECK(rOR.get(i)  == (a || b));
            CHECK(rXOR.get(i) == (a != b));
            CHECK(rNOT.get(i) == (!a));
            CHECK(rANB.get(i) == (a && !b));
            cAND += (a && b); cOR += (a || b); cXOR += (a != b);
        }
        CHECK(rAND.count() == cAND);
        CHECK(rOR.count()  == cOR);
        CHECK(rXOR.count() == cXOR);
    }
    std::printf("[OK] RowSet AND/OR/XOR/NOT/ANDNOT + popcount vs brute force\n");
}

static void edge_cases() {
    // empty column (review C4/C5): no operator may crash
    WaveletMatrix e(std::vector<uint32_t>{});
    CHECK(e.size() == 0);
    CHECK(e.access(0) == 0);
    CHECK(e.range_count(0, 0, 0, 1) == 0);
    CHECK(e.quantile(0, 0, 0) == 0);
    CHECK(e.range_predicate(0, 5).count() == 0);

    // single distinct value
    WaveletMatrix one(std::vector<uint32_t>(100, 7u));
    CHECK(one.levels() == 3);                 // 7 needs 3 bits
    CHECK(one.access(42) == 7);
    CHECK(one.range_count(0, 100, 7, 8) == 100);
    CHECK(one.quantile(0, 100, 50) == 7);
    WaveletMatrix allzero(std::vector<uint32_t>(100, 0u));
    CHECK(allzero.levels() == 1);
    CHECK(allzero.access(10) == 0);
    CHECK(allzero.range_count(0, 100, 0, 1) == 100);

    // UINT32_MAX present — sigma_=mx+1 must not wrap (review C3/C7)
    std::vector<uint32_t> big{0u, 0xFFFFFFFFu, 5u, 0xFFFFFFFFu, 100u};
    WaveletMatrix bw(big);
    CHECK(bw.levels() == 32);
    for (size_t i = 0; i < big.size(); ++i) CHECK(bw.access(i) == big[i]);
    CHECK(bw.range_count(0, 5, 0xFFFFFFFFu, 0xFFFFFFFFu) == 0); // empty value range
    CHECK(bw.rank(0xFFFFFFFFu, 5) == 2);
    {   // count values >= 100: count_lt(0,n,sigma) - count_lt(0,n,100)
        size_t ge100 = bw.size() - bw.count_lt(0, bw.size(), 100);
        CHECK(ge100 == 3);                    // 0xFFFFFFFF,0xFFFFFFFF,100
    }

    // range end past column size must clamp, not read OOB (review C9)
    WaveletMatrix sm(std::vector<uint32_t>{1, 2, 3, 2, 1});
    CHECK(sm.count_lt(0, 1000, 3) == 4);      // r clamps to 5; values <3 = {1,2,2,1}
    CHECK(sm.range_count(0, 1000, 0, 3) == 4);
    CHECK(sm.quantile(0, 1000, 0) == 1);

    // RowSet mismatched sizes must not crash under NDEBUG (review C8): overlap-only
    RowSet big2(500); big2.set(10); big2.set(400);
    RowSet small2(64); small2.set(10);
    RowSet r = RowSet::AND(big2, small2);     // must not OOB
    CHECK(r.get(10) == true);
    std::printf("[OK] edge cases (empty / single-value / UINT32_MAX / r>n clamp / size-mismatch)\n");
}

static void bench() {
    std::mt19937 rng(999);
    const size_t n = 1u << 22;                 // ~4.19M rows
    const uint32_t sigma = 4096;               // 12-bit column
    std::vector<uint32_t> data(n);
    for (auto& x : data) x = rng() % sigma;

    auto t0 = clk::now();
    WaveletMatrix w(data);
    auto t1 = clk::now();
    std::printf("\n[bench] n=%zu sigma=%u L=%d  build=%.1f ms  index=%.2f MB (%.2f bits/elem, raw uint32=32)\n",
                n, sigma, w.levels(), ms(t0, t1), w.mem_bytes() / 1e6, w.mem_bytes() * 8.0 / n);

    const size_t Q = 200000;
    std::vector<size_t> ql(Q), qr(Q);
    std::vector<uint32_t> qlo(Q), qhi(Q);
    std::uniform_int_distribution<size_t>   dpos(0, n);
    std::uniform_int_distribution<uint32_t> dval(0, sigma);
    for (size_t i = 0; i < Q; ++i) {
        size_t a = dpos(rng), b = dpos(rng); if (a > b) std::swap(a, b);
        uint32_t lo = dval(rng), hi = dval(rng); if (lo > hi) std::swap(lo, hi);
        ql[i] = a; qr[i] = b; qlo[i] = lo; qhi[i] = hi;
    }
    auto b0 = clk::now(); size_t s1 = 0;
    for (size_t i = 0; i < Q; ++i) s1 += w.range_count(ql[i], qr[i], qlo[i], qhi[i]);
    auto b1 = clk::now();
    std::printf("[bench] range_count : %zu queries / %.1f ms = %.2f M q/s  (chk %zu)\n",
                Q, ms(b0, b1), Q / ms(b0, b1) / 1e3, s1);
    auto c0 = clk::now(); size_t s2 = 0;
    for (size_t i = 0; i < Q; ++i) { size_t a = ql[i], b = qr[i]; if (b > a) s2 += w.quantile(a, b, (b - a) / 2); }
    auto c1 = clk::now();
    std::printf("[bench] quantile/med: %zu queries / %.1f ms = %.2f M q/s  (chk %zu)\n",
                Q, ms(c0, c1), Q / ms(c0, c1) / 1e3, s2);
    RowSet A = w.range_predicate(100, 600), B = w.range_predicate(300, 3000);
    const int reps = 300;
    auto d0 = clk::now(); size_t s3 = 0;
    for (int r = 0; r < reps; ++r) { RowSet C = RowSet::AND(A, B); s3 += C.count(); }
    auto d1 = clk::now();
    const double gb = (double)(A.num_words() * sizeof(uint64_t)) * reps / 1e9;
    std::printf("[bench] RowSet AND  : %d ops on %zu-row sets / %.1f ms = %.1f GB/s  (chk %zu)\n",
                reps, A.size(), ms(d0, d1), gb / (ms(d0, d1) / 1e3), s3);
}

int main() {
    bitvector_checks();
    correctness();
    edge_cases();
    bench();
    std::printf("\nAll checks passed.\n");
    return 0;
}
