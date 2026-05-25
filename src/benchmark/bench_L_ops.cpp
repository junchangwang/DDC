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

        // Reference results (raw bit-vector ops).  All three ops are
        // CROSS (ha vs hb) — switched from the previous self-AND setup
        // so AND doesn't get the production self-AND fast-path shortcut.
        // All variants now do the general cross-AND code path, apples
        // to apples.
        auto rA = ref_and(ba, bb);
        auto rO = ref_or (ba, bb);
        auto rX = ref_xor(ba, bb);

        // For each depth, compress + measure + time + validate.
        for (int depth : {2, 3, 4, 5}) {
            // -------- LOAD inputs at this depth ----------------------
            ComBitN cA, cB;
            // L4 uses production AVX-512 code path; A and B are reused
            // from the existing ComBit objects loaded above.  L2/L3/L5
            // use the scalar ComBitN reference (AVX-512 versions TBD).
            if (depth != 4) {
                cA = combit_n_compress(ba, depth);
                cB = combit_n_compress(bb, depth);
            }

            // Round-trip check (only meaningful for ComBitN depths).
            bool rt_ok = true;
            if (depth != 4) rt_ok = (combit_n_decompress(cA) == ba);

            // Compressed size at this depth.  For L4 use production
            // ComBit's total compressed bytes; for the others use
            // combit_n_total_bytes on the recompressed input.
            size_t total = 0;
            if (depth == 4) {
                for (const auto& seg : A.segments())
                    total += seg.size_breakdown().total_bits / 8;
            } else {
                total = combit_n_total_bytes(cA);
            }

            // -------- OP TIMING -------------------------------------
            // L4 path uses the PRODUCTION AVX-512 ComBit ops:
            //   AND: and_no_bypass (explicit no-bypass — matches the
            //        motivation-chart fast path)
            //   OR:  operator| (default, WITH bypass)
            //   XOR: operator^ (default)
            // All three run with combit_compress_results=false (op-only
            // decompressed output).  See and.cpp / or.cpp / xor.cpp for
            // the production AVX-512 inner loops.
            std::vector<double> tA, tO, tX;
            bool and_ok = true, or_ok = true, xor_ok = true;
            if (depth == 4) {
                // Warm-ups
                { ComBit w = A.and_no_bypass(B); (void)w; }
                { ComBit w = A | B; (void)w; }
                { ComBit w = A ^ B; (void)w; }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ComBit r = A.and_no_bypass(B);
                    auto t1 = clk::now();
                    tA.push_back(ms(t0, t1));
                    (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ComBit r = A | B;
                    auto t1 = clk::now();
                    tO.push_back(ms(t0, t1));
                    (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ComBit r = A ^ B;
                    auto t1 = clk::now();
                    tX.push_back(ms(t0, t1));
                    (void)r;
                }
                // Correctness: decompress each result and compare to raw.
                auto cb_to_bits = [&](const ComBit& cb) {
                    std::vector<bool> out(cb.bit_count(), false);
                    size_t off = 0;
                    for (size_t s = 0; s < cb.num_segments(); s++) {
                        auto v = cb.segment(s).decompress();
                        for (size_t i = 0; i < v.size() && off + i < out.size(); i++)
                            out[off + i] = v[i];
                        off += v.size();
                    }
                    return out;
                };
                and_ok = (cb_to_bits(A.and_no_bypass(B)) == rA);
                or_ok  = (cb_to_bits(A | B) == rO);
                xor_ok = (cb_to_bits(A ^ B) == rX);
            } else {
                // Cross-AND too (matches L4 above).  Goes through general
                // seg_op_l*<AND>, not seg_self_and_l*.
                { auto w = combit_n_and_dec_avx(cA, cB); (void)w; }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    auto r = combit_n_and_dec_avx(cA, cB);
                    auto t1 = clk::now();
                    tA.push_back(ms(t0, t1)); (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    auto r = combit_n_or_dec_avx(cA, cB);
                    auto t1 = clk::now();
                    tO.push_back(ms(t0, t1)); (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    auto r = combit_n_xor_dec_avx(cA, cB);
                    auto t1 = clk::now();
                    tX.push_back(ms(t0, t1)); (void)r;
                }
                auto chk = [&](const std::vector<uint8_t>& r,
                               const std::vector<bool>& ref) {
                    return bytes_to_bits(r, ref.size()) == ref;
                };
                and_ok = chk(combit_n_and_dec_avx(cA, cB), rA);
                or_ok  = chk(combit_n_or_dec_avx (cA, cB), rO);
                xor_ok = chk(combit_n_xor_dec_avx(cA, cB), rX);
            }

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
