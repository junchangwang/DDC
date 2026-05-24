// bench_L_ops.cpp
// =========================================================================
// Phase-2 sibling of bench_L_sizes.cpp.  For each cardinality c in the
// standard sweep:
//   1. Load the first two combit_w8 .bm files (depth=4 native compress)
//      and decompress them to bool vectors.
//   2. Re-compress each as a depth-N ComBitN for N ∈ {2,3,4,5}.
//   3. Measure sizes from the actual compressed representation.
//   4. Time AND / OR / XOR on the two depth-N inputs (op-only, median
//      of N_ITER).
//   5. Validate correctness against a reference produced by raw
//      bit-vector AND/OR/XOR.
//   6. Write a CSV row per (cardinality, variant) with sizes + op times.
//
// Output: tools/bench_L_ops.csv, schema:
//   cardinality,variant,total_bytes,total_MiB,
//   and_ms,or_ms,xor_ms,
//   and_ok,or_ok,xor_ok,
//   roundtrip_ok
// =========================================================================

#include "combit.h"      // ComBit (depth-4 native) for loading the .bm files
#include "combit_n.h"    // depth-parameterised ComBitN

#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <chrono>
namespace fs = std::filesystem;
using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Reference op on raw bit vectors (for correctness check).
static std::vector<bool> ref_and(const std::vector<bool>& a, const std::vector<bool>& b) {
    std::vector<bool> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] & b[i];
    return r;
}
static std::vector<bool> ref_or(const std::vector<bool>& a, const std::vector<bool>& b) {
    std::vector<bool> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] | b[i];
    return r;
}
static std::vector<bool> ref_xor(const std::vector<bool>& a, const std::vector<bool>& b) {
    std::vector<bool> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] ^ b[i];
    return r;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_L_ops <bitmap_root_dir> <out_csv>\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    std::vector<int> cards = {2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000};
    const int N_ITER = 5;

    std::ofstream out(out_path);
    out << "cardinality,variant,total_bytes,total_MiB,"
        << "and_ms,or_ms,xor_ms,"
        << "and_ok,or_ok,xor_ok,roundtrip_ok\n";

    for (int c : cards) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c) + "_combit_w8");
        if (!fs::is_directory(dir)) { std::cerr << "[skip] " << dir << "\n"; continue; }

        // Pick numerically-first two .bm files (matches benchmark_main's sort).
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".bm") files.push_back(e.path());
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
            return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
        });
        if (files.size() < 2) { std::cerr << "[skip] " << dir << " (<2 files)\n"; continue; }

        // Load via existing ComBit and decompress to bool vectors.
        std::ifstream ia(files[0], std::ios::binary), ib(files[1], std::ios::binary);
        ComBit A = ComBit::deserialize(ia);
        ComBit B = ComBit::deserialize(ib);
        std::vector<bool> ba(A.bit_count()), bb(B.bit_count());
        // ComBit::decompress returns segment-wise; concat all segments.
        // We use the depth-4 ComBit's segment-by-segment decompress.
        size_t base = 0;
        for (size_t s = 0; s < A.num_segments(); s++) {
            auto v = A.segment(s).decompress();
            for (size_t i = 0; i < v.size() && base + i < ba.size(); i++) ba[base + i] = v[i];
            base += v.size();
        }
        base = 0;
        for (size_t s = 0; s < B.num_segments(); s++) {
            auto v = B.segment(s).decompress();
            for (size_t i = 0; i < v.size() && base + i < bb.size(); i++) bb[base + i] = v[i];
            base += v.size();
        }

        // Reference results (raw bit-vector ops) — matches the
        // motivation-chart methodology: AND uses SELF (ba & ba) so the
        // output is non-trivial; OR / XOR use cross (ba vs bb) because
        // ha | ha = ha would be a degenerate copy.
        auto rA = ref_and(ba, ba);   // self-AND
        auto rO = ref_or (ba, bb);
        auto rX = ref_xor(ba, bb);

        // For each depth, compress + measure + time + validate.
        for (int depth : {2, 3, 4, 5}) {
            ComBitN cA = combit_n_compress(ba, depth);
            ComBitN cB = combit_n_compress(bb, depth);

            // Round-trip check on A (decompress(compress(bits)) == bits).
            auto rt = combit_n_decompress(cA);
            bool rt_ok = (rt == ba);

            // Per-bitmap compressed size of cA (cA and cB are within ~1%
            // of each other at the same density).
            size_t total = combit_n_total_bytes(cA);

            // Op-only timing — output is the raw decompressed L1 byte
            // stream (no recompression in the timed window).  AND is
            // self (cA & cA); OR / XOR are cross (cA vs cB) — same as
            // the production motivation-chart benchmark.
            std::vector<double> tA, tO, tX;
            { auto w = combit_n_and_dec(cA, cA); (void)w; }  // warm
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                auto r = combit_n_and_dec(cA, cA);
                auto t1 = clk::now();
                tA.push_back(ms(t0, t1));
                (void)r;
            }
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                auto r = combit_n_or_dec(cA, cB);
                auto t1 = clk::now();
                tO.push_back(ms(t0, t1));
                (void)r;
            }
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                auto r = combit_n_xor_dec(cA, cB);
                auto t1 = clk::now();
                tX.push_back(ms(t0, t1));
                (void)r;
            }

            // Validate: run op once more, compare to raw ref.  The
            // decompressed byte stream is converted to bits and trimmed
            // to bit_count for the equality check.
            auto chk = [&](const std::vector<uint8_t>& r,
                           const std::vector<bool>& ref) {
                auto rb = bytes_to_bits(r, ref.size());
                return rb == ref;
            };
            bool and_ok = chk(combit_n_and_dec(cA, cA), rA);
            bool or_ok  = chk(combit_n_or_dec (cA, cB), rO);
            bool xor_ok = chk(combit_n_xor_dec(cA, cB), rX);

            out << c << ",L" << depth << ","
                << total << "," << double(total) / (1024.0 * 1024.0) << ","
                << median(tA) << "," << median(tO) << "," << median(tX) << ","
                << (and_ok ? 1 : 0) << "," << (or_ok ? 1 : 0) << "," << (xor_ok ? 1 : 0) << ","
                << (rt_ok ? 1 : 0) << "\n";
            out.flush();
            std::cerr << "[c=" << c << " L" << depth << "] size=" << total
                      << "B and=" << median(tA) << "ms or=" << median(tO)
                      << "ms xor=" << median(tX) << "ms (correct: A="
                      << and_ok << " O=" << or_ok << " X=" << xor_ok
                      << " RT=" << rt_ok << ")\n";
        }
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
