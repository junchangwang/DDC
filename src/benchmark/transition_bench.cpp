// transition_bench.cpp
//
// Construct two transition scenarios where CRoaring's container layout flips
// mid-operation, and measure who eats that conversion cost.
//
// Scenario 1 — OR (array → bitset)
//   Both inputs sit just under CRoaring's array-container threshold
//   (≈ 4096 bits per 65536-bit container).  Their OR overshoots the
//   threshold, so every result container is reconstructed as a bitset.
//   We bill that reconstruction to CRoaring; ComBit / EWAH / WAH / Concise
//   do not have container layouts, so they pay nothing extra.
//
// Scenario 2 — AND (bitset → array)
//   Both inputs are dense enough to live as bitset containers.  Their
//   AND has expected cardinality well below the threshold, so every
//   result container is reconstructed as an array.  Again CRoaring pays;
//   the RLE / 4-level backends do not.
//
// Implementation notes
// - Single 65536-bit roaring container per "segment".  We replicate the
//   pattern across N segments to get a measurable, repeatable timing.
// - Deterministic seed → identical positions across backends → directly
//   comparable popcount on the result.
// - Median of N_ITER for noise.
// - We compare ComBit against CRoaring (CR plain, no run-optimization)
//   for the headline numbers, then also report WAH/EWAH/Concise for the
//   broader picture (they have no container transition so the result is
//   "free of conversion" but their raw op cost may still differ).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "combit.h"
#include "croaring/roaring.hh"
#include "fastbit/bitvector.h"
#include "ewah/ewah.h"
#include "Concise/concise.h"

namespace {

using clk = std::chrono::high_resolution_clock;

inline double ms_since(clk::time_point t0) {
    return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count() / 1000.0;
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Build a deterministic random sorted position list: `count` set bits chosen
// uniformly from [base, base + 65536) without replacement.
std::vector<uint32_t> sample_positions(uint64_t seed, uint32_t base, uint32_t count) {
    std::mt19937 rng(seed);
    std::vector<uint32_t> ix(65536);
    std::iota(ix.begin(), ix.end(), 0);
    for (uint32_t i = 0; i < count; i++) {
        std::uniform_int_distribution<uint32_t> pick(i, 65535);
        std::swap(ix[i], ix[pick(rng)]);
    }
    std::vector<uint32_t> out;
    out.reserve(count);
    for (uint32_t i = 0; i < count; i++) out.push_back(base + ix[i]);
    std::sort(out.begin(), out.end());
    return out;
}

// Build two N_SEG-segment bitmaps where each segment has `count_a` / `count_b`
// random set bits sampled with disjoint seeds.  Returned: concatenated sorted
// position lists.
struct BitmapPair {
    std::vector<uint32_t> pos_a;
    std::vector<uint32_t> pos_b;
    size_t                num_rows;
};

BitmapPair build_pair(size_t n_seg, uint32_t count_a, uint32_t count_b, uint64_t seed) {
    BitmapPair p;
    p.num_rows = n_seg * 65536;
    p.pos_a.reserve(n_seg * count_a);
    p.pos_b.reserve(n_seg * count_b);
    for (size_t s = 0; s < n_seg; s++) {
        uint32_t base = static_cast<uint32_t>(s * 65536);
        auto a = sample_positions(seed + 2 * s + 0, base, count_a);
        auto b = sample_positions(seed + 2 * s + 1, base, count_b);
        p.pos_a.insert(p.pos_a.end(), a.begin(), a.end());
        p.pos_b.insert(p.pos_b.end(), b.begin(), b.end());
    }
    return p;
}

// Build A and B such that B is A with a small fraction (overlap_ratio of A's
// bits kept) plus a fresh perturbation of fresh_bits new bits.  Used to
// construct the "worst path" Scenario 3: both arrays just under threshold,
// totalCardinality > threshold (triggers bitset alloc), but actual |A∪B|
// drops back below threshold so CRoaring has to convert bitset → array.
BitmapPair build_overlapping_pair(size_t n_seg, uint32_t count_a,
                                  double overlap_ratio, uint32_t fresh_bits,
                                  uint64_t seed) {
    BitmapPair p;
    p.num_rows = n_seg * 65536;
    p.pos_a.reserve(n_seg * count_a);
    uint32_t carry = static_cast<uint32_t>(overlap_ratio * count_a);
    p.pos_b.reserve(n_seg * (carry + fresh_bits));

    for (size_t s = 0; s < n_seg; s++) {
        uint32_t base = static_cast<uint32_t>(s * 65536);
        auto a = sample_positions(seed + 3 * s + 0, base, count_a);
        // B inherits the first `carry` positions of A, then picks `fresh_bits`
        // brand-new positions disjoint from A.
        std::vector<uint32_t> b(a.begin(), a.begin() + carry);
        // Disjoint sampler: pick from [base, base+65536) avoiding `a`.
        std::mt19937 rng(seed + 3 * s + 1);
        std::vector<uint8_t> in_a(65536, 0);
        for (auto v : a) in_a[v - base] = 1;
        uint32_t fresh_added = 0;
        while (fresh_added < fresh_bits) {
            uint32_t pos = std::uniform_int_distribution<uint32_t>(0, 65535)(rng);
            if (!in_a[pos]) { in_a[pos] = 2; b.push_back(base + pos); fresh_added++; }
        }
        std::sort(b.begin(), b.end());

        p.pos_a.insert(p.pos_a.end(), a.begin(), a.end());
        p.pos_b.insert(p.pos_b.end(), b.begin(), b.end());
    }
    return p;
}

// ----------------------------------------------------------------
// CRoaring helpers
// ----------------------------------------------------------------
roaring::Roaring cr_build(const std::vector<uint32_t>& pos) {
    roaring::Roaring r;
    r.addMany(pos.size(), pos.data());
    r.shrinkToFit();
    return r;
}

// ----------------------------------------------------------------
// ComBit helpers (use ComBit::compress on bool-vector for the apples-to-apples
// "load then op" model the micro-bench already uses)
// ----------------------------------------------------------------
ComBit cb_build(const std::vector<uint32_t>& pos, size_t num_rows) {
    return ComBit::from_sparse_positions(pos, num_rows, 65536);
}

// ----------------------------------------------------------------
// WAH (FastBit) helpers
// ----------------------------------------------------------------
ibis::bitvector wah_build(const std::vector<uint32_t>& pos, size_t num_rows) {
    // FastBit packs 31 raw bits per appendWord (the 32nd bit is a fill flag).
    std::vector<bool> raw(num_rows, false);
    for (auto p : pos) raw[p] = true;
    ibis::bitvector bv;
    constexpr size_t BPW = 31;
    for (size_t i = 0; i < raw.size(); i += BPW) {
        uint32_t w = 0;
        size_t lim = std::min(raw.size(), i + BPW);
        for (size_t k = i; k < lim; k++)
            if (raw[k]) w |= 1u << (BPW - 1 - (k - i));   // MSB-first
        bv.appendWord(w);
    }
    bv.compress();
    bv.adjustSize(0, num_rows);
    return bv;
}

// ----------------------------------------------------------------
// EWAH helpers
// ----------------------------------------------------------------
ewah::EWAHBoolArray<uint64_t> ew_build(const std::vector<uint32_t>& pos, size_t num_rows) {
    ewah::EWAHBoolArray<uint64_t> bv;
    for (auto p : pos) bv.set(p);
    // pad to num_rows
    if (bv.sizeInBits() < num_rows) bv.padWithZeroes(num_rows);
    return bv;
}

// ----------------------------------------------------------------
// Concise helpers
// ----------------------------------------------------------------
ConciseSet<false> con_build(const std::vector<uint32_t>& pos) {
    ConciseSet<false> s;
    for (auto p : pos) s.add(p);
    return s;
}

// ================================================================
// Scenario runners
// ================================================================
struct OpResult {
    double median_ms;  // median of measured iters (warm-up excluded)
    double min_ms;     // min of measured iters
    double max_ms;     // max of measured iters
    double stddev_ms;  // stddev of measured iters
    uint64_t card;     // result cardinality
    double size_mb;    // result serialized / storage size after op
};

// Compute simple stats; assumes measured.size() > 0.
inline OpResult finalize(std::vector<double>& measured, uint64_t card, double size_mb) {
    std::vector<double> sorted = measured;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    double mn  = sorted.front();
    double mx  = sorted.back();
    double mean = 0;
    for (auto v : measured) mean += v;
    mean /= measured.size();
    double var = 0;
    for (auto v : measured) var += (v - mean) * (v - mean);
    double sd = std::sqrt(var / measured.size());
    return {med, mn, mx, sd, card, size_mb};
}

// All runners share the same shape:
//   build A, build B (outside timer)
//   N_WARMUP × { run op, throw away timing }
//   N_MEASURED × { run op, record timing }
//   return median / min / max / stddev of measured iters.

template <int N_WARMUP, int N_MEASURED>
OpResult run_cr_or(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b) {
    auto A = cr_build(pos_a);
    auto B = cr_build(pos_b);
    roaring::Roaring R;
    for (int i = 0; i < N_WARMUP; i++) { R = A | B; R.shrinkToFit(); }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R = A | B;                // ← container conversion happens inside if needed
        R.shrinkToFit();          // bill any post-op finalization
        t.push_back(ms_since(t0));
    }
    return finalize(t, R.cardinality(), R.getSizeInBytes() / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_cr_and(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b,
                    uint32_t logical_size) {
    // Apples-to-apples policy: CR pairwise op INCLUDES the to-bitset conversion
    // of the result (matches benchmark_main.cpp's AND_op cell).  Without this
    // CR appears artificially fast at sparse densities because it keeps the
    // result in array/container form while every other backend pays to emit a
    // bitset-shaped (or RLE-resident) result.
    auto A = cr_build(pos_a);
    auto B = cr_build(pos_b);
    roaring::Roaring R;
    auto do_op = [&]() {
        R = A & B;
        roaring::api::bitset_t* bs =
            roaring::api::bitset_create_with_capacity(logical_size);
        roaring::api::roaring_bitmap_to_bitset(&R.roaring, bs);
        roaring::api::bitset_free(bs);
    };
    for (int i = 0; i < N_WARMUP; i++) do_op();
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        do_op();
        t.push_back(ms_since(t0));
    }
    return finalize(t, R.cardinality(), R.getSizeInBytes() / 1e6);
}

// ComBit OR/AND has a global toggle: combit_compress_results.
//   false (default) → output is left in Decompressed state (L1 byte stream,
//                      no L2/L3/L4 rebuilt); fast but result memory is the
//                      uncompressed L1.
//   true            → output is Compressed (compact_l2_l3 rebuilds L2/L3/L4
//                      and trims L1 to literal-only); slower but apples-to-
//                      apples with CRoaring's "compressed in, compressed out".
template <int N_WARMUP, int N_MEASURED>
OpResult run_cb_or(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b,
                   size_t num_rows, bool compress_result) {
    combit_compress_results = compress_result;     // global flag, set per call
    auto A = cb_build(pos_a, num_rows);
    auto B = cb_build(pos_b, num_rows);
    ComBit R;
    for (int i = 0; i < N_WARMUP; i++) { R = A | B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R = A | B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.popcount()), R.compressed_size_bytes() / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_cb_and(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b,
                    size_t num_rows, bool compress_result) {
    combit_compress_results = compress_result;
    auto A = cb_build(pos_a, num_rows);
    auto B = cb_build(pos_b, num_rows);
    ComBit R;
    for (int i = 0; i < N_WARMUP; i++) { R = A & B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R = A & B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.popcount()), R.compressed_size_bytes() / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_wah_or(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b, size_t num_rows) {
    auto A = wah_build(pos_a, num_rows);
    auto B = wah_build(pos_b, num_rows);
    ibis::bitvector R;
    for (int i = 0; i < N_WARMUP; i++) { R.copy(A); R |= B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R.copy(A);
        R |= B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.cnt()), static_cast<double>(R.bytes()) / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_wah_and(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b, size_t num_rows) {
    auto A = wah_build(pos_a, num_rows);
    auto B = wah_build(pos_b, num_rows);
    ibis::bitvector R;
    for (int i = 0; i < N_WARMUP; i++) { R.copy(A); R &= B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R.copy(A);
        R &= B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.cnt()), static_cast<double>(R.bytes()) / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_ew_or(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b, size_t num_rows) {
    auto A = ew_build(pos_a, num_rows);
    auto B = ew_build(pos_b, num_rows);
    ewah::EWAHBoolArray<uint64_t> R;
    for (int i = 0; i < N_WARMUP; i++) { R.reset(); A.logicalor(B, R); }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R.reset();
        A.logicalor(B, R);
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.numberOfOnes()), R.sizeInBytes() / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_ew_and(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b, size_t num_rows) {
    auto A = ew_build(pos_a, num_rows);
    auto B = ew_build(pos_b, num_rows);
    ewah::EWAHBoolArray<uint64_t> R;
    for (int i = 0; i < N_WARMUP; i++) { R.reset(); A.logicaland(B, R); }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R.reset();
        A.logicaland(B, R);
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.numberOfOnes()), R.sizeInBytes() / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_con_or(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b) {
    auto A = con_build(pos_a);
    auto B = con_build(pos_b);
    ConciseSet<false> R;
    for (int i = 0; i < N_WARMUP; i++) { R = A | B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R = A | B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.size()), static_cast<double>(R.sizeInBytes()) / 1e6);
}

template <int N_WARMUP, int N_MEASURED>
OpResult run_con_and(const std::vector<uint32_t>& pos_a, const std::vector<uint32_t>& pos_b) {
    auto A = con_build(pos_a);
    auto B = con_build(pos_b);
    ConciseSet<false> R;
    for (int i = 0; i < N_WARMUP; i++) { R = A & B; }
    std::vector<double> t; t.reserve(N_MEASURED);
    for (int i = 0; i < N_MEASURED; i++) {
        auto t0 = clk::now();
        R = A & B;
        t.push_back(ms_since(t0));
    }
    return finalize(t, static_cast<uint64_t>(R.size()), static_cast<double>(R.sizeInBytes()) / 1e6);
}

void print_row(const std::string& label, const OpResult& r) {
    std::cout << std::left << std::setw(10) << label
              << std::right << std::setw(11) << std::fixed << std::setprecision(3) << r.median_ms << " ms"
              << " ±" << std::setw(5) << std::fixed << std::setprecision(2) << r.stddev_ms
              << "  [" << std::setw(6) << std::fixed << std::setprecision(2) << r.min_ms
              << " — " << std::setw(6) << std::fixed << std::setprecision(2) << r.max_ms << "]"
              << std::setw(13) << r.card
              << std::setw(11) << std::fixed << std::setprecision(2) << r.size_mb << " MB"
              << "\n";
}

// Append one CSV row: scenario, backend, median, min, max, stddev, card, size_mb.
void write_csv_row(std::ofstream& csv, const std::string& scenario,
                   const std::string& backend, const OpResult& r) {
    csv << scenario << "," << backend << ","
        << std::fixed << std::setprecision(4) << r.median_ms << ","
        << r.min_ms << "," << r.max_ms << "," << r.stddev_ms << ","
        << r.card << "," << std::setprecision(4) << r.size_mb << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int N_WARMUP   = 2;     // 2 untimed warm-up rounds per cell
    constexpr int N_MEASURED = 9;     // 9 measured rounds; median + stddev reported
    constexpr size_t N_SEG = 1024;    // 1024 × 65536 = 67M bits / bitmap (TPC-H-scale)
    const size_t NUM_ROWS = N_SEG * 65536;

    // Optional CSV output path (1st CLI arg) — used by the Excel writer.
    std::string csv_path = (argc > 1) ? argv[1] : "transition_results.csv";
    std::ofstream csv(csv_path);
    csv << "scenario,backend,median_ms,min_ms,max_ms,stddev_ms,result_card,result_size_mb\n";

    std::cout << "Transition Benchmark — CRoaring vs ComBit (+ WAH / EWAH / Concise)\n";
    std::cout << "Bitmap size: " << NUM_ROWS << " bits (" << N_SEG << " × 65536 segments)\n";
    std::cout << "Iterations per cell: " << N_WARMUP << " warm-up + " << N_MEASURED
              << " measured (median ± stddev reported)\n";
    std::cout << "CSV output: " << csv_path << "\n\n";

    // ============================================================
    // Scenario 1 — OR pushes array → bitset
    //   Each segment of A and B has 3000 set bits → array container (3000<4096).
    //   Disjoint sampling: A∪B ≈ 6000 set bits → bitset container (6000>4096).
    //   CR pays the array→bitset conversion on every result container.
    // ============================================================
    std::cout << "================================================================\n";
    std::cout << "Scenario 1: OR — array → bitset  (CR pays container conversion)\n";
    std::cout << "Inputs:  ~3000 bits / 65536 container (density ≈ 4.6%, array)\n";
    std::cout << "Result:  ~6000 bits / 65536 container (density ≈ 9.2%, bitset)\n";
    std::cout << "================================================================\n";
    {
        auto pair = build_pair(N_SEG, /*count_a=*/3000, /*count_b=*/3000, /*seed=*/0x12345);
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::right << std::setw(13) << "OR median"
                  << std::setw(7) << " ±sd"
                  << std::setw(20) << "[min — max]"
                  << std::setw(13) << "result.card"
                  << std::setw(13) << "result.size\n";
        std::cout << std::string(76, '-') << "\n";
        auto cr_r  = run_cr_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b);
        print_row("CR",        cr_r);    write_csv_row(csv, "S1_OR_a2b", "CR",            cr_r);
        auto cb_d  = run_cb_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS, /*compress=*/false);
        print_row("CB (dec)",  cb_d);    write_csv_row(csv, "S1_OR_a2b", "CB-decompressed", cb_d);
        auto cb_c  = run_cb_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS, /*compress=*/true);
        print_row("CB (cmp)",  cb_c);    write_csv_row(csv, "S1_OR_a2b", "CB-compressed",   cb_c);
        auto wah_r = run_wah_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        print_row("WAH",       wah_r);   write_csv_row(csv, "S1_OR_a2b", "WAH",            wah_r);
        auto ew_r  = run_ew_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        print_row("EWAH",      ew_r);    write_csv_row(csv, "S1_OR_a2b", "EWAH",           ew_r);
        auto con_r = run_con_or<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b);
        print_row("Concise",   con_r);   write_csv_row(csv, "S1_OR_a2b", "Concise",        con_r);

        std::cout << "\n  CR vs CB-decompressed cardinality diff: "
                  << std::llabs(static_cast<long long>(cr_r.card) - static_cast<long long>(cb_d.card)) << "\n";
        std::cout << "  CR vs CB-compressed   cardinality diff: "
                  << std::llabs(static_cast<long long>(cr_r.card) - static_cast<long long>(cb_c.card)) << "\n";
        std::cout << "  CR / CB-decompressed ratio: " << std::fixed << std::setprecision(2)
                  << (cr_r.median_ms / cb_d.median_ms) << "×   (fast path, output is L1-flat)\n";
        std::cout << "  CR / CB-compressed   ratio: " << std::fixed << std::setprecision(2)
                  << (cr_r.median_ms / cb_c.median_ms) << "×   (apples-to-apples, output is 4-level compressed)\n";
    }

    // ============================================================
    // Scenario 2 — AND pushes bitset → array
    //   Each segment of A and B has 16000 set bits → bitset container (>4096).
    //   Expected |A∩B| per segment = 16000 × 16000 / 65536 ≈ 3906 → array.
    //   CR pays the bitset→array conversion on every result container.
    // ============================================================
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Scenario 2: AND — bitset → array  (CR pays container conversion)\n";
    std::cout << "Inputs:  ~16000 bits / 65536 container (density ≈ 24.4%, bitset)\n";
    std::cout << "Result:  ~ 3900 bits / 65536 container (density ≈  6.0%, array)\n";
    std::cout << "================================================================\n";
    {
        auto pair = build_pair(N_SEG, /*count_a=*/16000, /*count_b=*/16000, /*seed=*/0xABCDE);
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::right << std::setw(13) << "AND median"
                  << std::setw(7) << " ±sd"
                  << std::setw(20) << "[min — max]"
                  << std::setw(13) << "result.card"
                  << std::setw(13) << "result.size\n";
        std::cout << std::string(76, '-') << "\n";
        auto cr_r  = run_cr_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        print_row("CR",        cr_r);   write_csv_row(csv, "S2_AND_b2a", "CR",              cr_r);
        auto cb_d  = run_cb_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS, /*compress=*/false);
        print_row("CB (dec)",  cb_d);   write_csv_row(csv, "S2_AND_b2a", "CB-decompressed", cb_d);
        auto cb_c  = run_cb_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS, /*compress=*/true);
        print_row("CB (cmp)",  cb_c);   write_csv_row(csv, "S2_AND_b2a", "CB-compressed",   cb_c);
        auto wah_r = run_wah_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        print_row("WAH",       wah_r);  write_csv_row(csv, "S2_AND_b2a", "WAH",             wah_r);
        auto ew_r  = run_ew_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        print_row("EWAH",      ew_r);   write_csv_row(csv, "S2_AND_b2a", "EWAH",            ew_r);
        auto con_r = run_con_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b);
        print_row("Concise",   con_r);  write_csv_row(csv, "S2_AND_b2a", "Concise",         con_r);

        std::cout << "\n  CR vs CB-decompressed cardinality diff: "
                  << std::llabs(static_cast<long long>(cr_r.card) - static_cast<long long>(cb_d.card)) << "\n";
        std::cout << "  CR vs CB-compressed   cardinality diff: "
                  << std::llabs(static_cast<long long>(cr_r.card) - static_cast<long long>(cb_c.card)) << "\n";
        std::cout << "  CR / CB-decompressed ratio: " << std::fixed << std::setprecision(2)
                  << (cr_r.median_ms / cb_d.median_ms) << "×   (fast path, output is L1-flat)\n";
        std::cout << "  CR / CB-compressed   ratio: " << std::fixed << std::setprecision(2)
                  << (cr_r.median_ms / cb_c.median_ms) << "×   (apples-to-apples, output is 4-level compressed)\n";
    }

    // ============================================================
    // Scenario 3 — AND density sweep
    //   Independent random A, B with count_a = count_b = X bits per
    //   65536-bit container.  Expected |A∩B| = X² / 65536.
    //
    //   CR regime map as X grows:
    //     X ≤ 4096           → both inputs array, result array.  Plain
    //                          array-array intersection.  Fast.
    //     4096 < X ≤ 16384   → both inputs bitset (X > DEFAULT_MAX_SIZE),
    //                          expected result ≤ 4096 → CR runs
    //                          bitset_extract_intersection_setbits to
    //                          rebuild a uint16 array container.  This
    //                          is the OBVIOUS bitset→array transition.
    //     X > 16384          → result_card > 4096 → CR keeps result as
    //                          bitset, no extraction cost.  Fast.
    //
    //   We sweep across all three regimes, including very sparse points
    //   (X = 50, density ≈ 1/1000) so the curve matches the existing
    //   micro-benchmark's "c=1000" reference column.
    //
    //   CSV rows tagged "S3_AND_density@d=0.XXXXX" — value is the
    //   *input* density d_in = X/65536, the more natural axis for AND
    //   (matches the existing c=N benchmark's "1 set bit per N rows").
    // ============================================================
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Scenario 3: AND — density sweep across all CR container regimes\n";
    std::cout << "Sweeping count_a from very sparse (50 → d≈1/1000) up to dense (48000 → d≈0.73)\n";
    std::cout << "================================================================\n";

    static const std::vector<uint32_t> S3_AND_COUNTS = {
        // Very sparse — corresponds to existing bench's c≈2000 down to c≈100.
        50,   100,   200,   400,   700,
        // Approaching CR's array-container boundary (X = 4096).
        1200, 2000,  3000,  4000,
        // ---- CR bitset→array transition zone ----
        // Inputs are bitset containers (X > 4096); expected result_card =
        // X² / 65536 is ≤ 4096 so CR has to call bitset_extract_intersection
        // _setbits to rebuild a uint16 array container — this is where CR's
        // AND tanks.  Densified sampling (every ~500 count) to chart the
        // shape of the cliff.
        4500, 5000, 5500,  6000,  6500,  7000,  7500,
        8000, 9000, 10000, 11000, 12000, 13000,
        14000, 15000, 15500, 16000,
        // ---- Cliff edge: result_card crosses 4096 (X ≈ 16384) ----
        // Once X > 16384, X² / 65536 > 4096 → result stays bitset → CR
        // skips the extract step → cliff drop.  Fine sampling around
        // 16384 so the cliff is visible.
        16300, 16400, 16500, 16700, 17000, 17500,
        // ---- Bitset-result regime ----
        18000, 20000, 24000, 32000, 48000
    };

    std::cout << std::left
              << std::setw(8)  << "count_a"
              << std::setw(12) << "d_in"
              << std::setw(12) << "exp.d_out"
              << std::setw(10) << "CR"
              << std::setw(10) << "CB(dec)"
              << std::setw(10) << "WAH"
              << std::setw(10) << "EWAH"
              << std::setw(10) << "Concise"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (uint32_t count_a : S3_AND_COUNTS) {
        // Independent random sampling — no overlap engineering: build_pair
        // uses disjoint seeds per segment so |A∩B| converges to the
        // hypergeometric expectation X² / 65536.
        auto pair = build_pair(N_SEG, count_a, count_a, /*seed=*/0xCAFE0000u + count_a);
        double d_in       = static_cast<double>(count_a) / 65536.0;
        double exp_d_out  = static_cast<double>(count_a) * count_a / (65536.0 * 65536.0);

        auto cr_r  = run_cr_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        // ComBit op-only: decompressed result (combit_compress_results=false).
        // The CB-compressed variant runs compact_l2_l3 inside the op so it is
        // not pure op-only — dropped from the chart per user policy.
        auto cb_d  = run_cb_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS, false);
        auto wah_r = run_wah_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        auto ew_r  = run_ew_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b, NUM_ROWS);
        auto con_r = run_con_and<N_WARMUP, N_MEASURED>(pair.pos_a, pair.pos_b);

        std::cout << std::left << std::setw(8) << count_a
                  << std::setw(12) << std::fixed << std::setprecision(5) << d_in
                  << std::setw(12) << std::fixed << std::setprecision(5) << exp_d_out
                  << std::setw(10) << std::fixed << std::setprecision(3) << cr_r.median_ms
                  << std::setw(10) << std::fixed << std::setprecision(3) << cb_d.median_ms
                  << std::setw(10) << std::fixed << std::setprecision(3) << wah_r.median_ms
                  << std::setw(10) << std::fixed << std::setprecision(3) << ew_r.median_ms
                  << std::setw(10) << std::fixed << std::setprecision(3) << con_r.median_ms
                  << "\n";

        // Tag carries d_in (input density per container) for the plot script.
        char tag[64];
        std::snprintf(tag, sizeof(tag), "S3_AND_density@d=%.5f", d_in);
        write_csv_row(csv, tag, "CR",              cr_r);
        write_csv_row(csv, tag, "CB-decompressed", cb_d);
        write_csv_row(csv, tag, "WAH",             wah_r);
        write_csv_row(csv, tag, "EWAH",            ew_r);
        write_csv_row(csv, tag, "Concise",         con_r);
    }

    csv.close();
    std::cout << "\n[CSV] Results written to: " << csv_path << "\n";
    return 0;
}
