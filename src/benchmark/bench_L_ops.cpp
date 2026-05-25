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
static std::vector<bool> ref_not(const std::vector<bool>& a) {
    std::vector<bool> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = !a[i];
    return r;
}

// (no extra helpers needed — NOT timing uses production operator~ for L4
//  and combit_n_not_inplace for L2/L3/L5; correctness uses decompress.)

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
        << "and_ms,or_ms,not_ms,"
        << "and_ok,or_ok,not_ok,roundtrip_ok\n";

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

        // Reference results (raw bit-vector ops).  AND and OR are
        // CROSS (ba vs bb).  NOT is unary on A.
        auto rA = ref_and(ba, bb);
        auto rO = ref_or (ba, bb);
        auto rN = ref_not(ba);

        // For each depth, load + measure + time + validate.
        for (int depth : {2, 3, 4, 5}) {
            // -------- LOAD inputs at this depth ----------------------
            ComBitN cA, cB;
            // L4 uses production AVX-512 code path; A and B are reused
            // from the existing ComBit objects loaded above.  L2/L3/L5
            // load real ComBitN .bm artefacts written by gen_bitmap
            // -L <depth>, mirroring how L4 is loaded from combit_w8/.
            // This keeps the comparison apples-to-apples — every variant
            // is deserialized fresh from disk in the timed window.
            if (depth != 4) {
                fs::path ndir = root / ("bm_100m_c" + std::to_string(c)
                                         + "_combit_L" + std::to_string(depth));
                if (!fs::is_directory(ndir)) {
                    std::cerr << "[skip] " << ndir << " (run gen_bitmap -L "
                              << depth << " first)\n";
                    continue;
                }
                std::vector<fs::path> nfiles;
                for (auto& e : fs::directory_iterator(ndir))
                    if (e.path().extension() == ".bm") nfiles.push_back(e.path());
                std::sort(nfiles.begin(), nfiles.end(),
                    [](const fs::path& a, const fs::path& b){
                        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
                    });
                if (nfiles.size() < 2) {
                    std::cerr << "[skip] " << ndir << " (<2 files)\n";
                    continue;
                }
                std::ifstream nia(nfiles[0], std::ios::binary);
                std::ifstream nib(nfiles[1], std::ios::binary);
                cA = combit_n_deserialize(nia);
                cB = combit_n_deserialize(nib);
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
            // AND / OR: cross (A op B) — no self-AND fast-path shortcut.
            // NOT: unary on A.  All produce decompressed byte streams
            // (combit_compress_results=false for the L4 OR; for L4 NOT
            // we manually decompress and pack so the output format is
            // the same vector<uint8_t> the combit_n walkers produce).
            std::vector<double> tA, tO, tN;
            bool and_ok = true, or_ok = true, not_ok = true;
            if (depth == 4) {
                // Warm-ups
                { ComBit w = A.and_no_bypass(B); (void)w; }
                { ComBit w = A | B; (void)w; }
                { ComBit w = ~A; (void)w; }
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
                    ComBit r = ~A;  // production in-place NOT
                    auto t1 = clk::now();
                    tN.push_back(ms(t0, t1));
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
                not_ok = (cb_to_bits(~A) == rN);
            } else {
                // Cross-AND too (matches L4 above).  Goes through general
                // seg_op_l*<AND>, not seg_self_and_l*.
                { auto w = combit_n_and_dec_avx(cA, cB); (void)w; }
                { auto w = combit_n_or_dec_avx(cA, cB); (void)w; }
                { auto w = combit_n_not_inplace(cA); (void)w; }
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
                    auto r = combit_n_not_inplace(cA);  // in-place NOT
                    auto t1 = clk::now();
                    tN.push_back(ms(t0, t1)); (void)r;
                }
                auto chk = [&](const std::vector<uint8_t>& r,
                               const std::vector<bool>& ref) {
                    return bytes_to_bits(r, ref.size()) == ref;
                };
                and_ok = chk(combit_n_and_dec_avx(cA, cB), rA);
                or_ok  = chk(combit_n_or_dec_avx (cA, cB), rO);
                // Verify NOT by decompressing the result and comparing
                // to the reference (decompress is NOT in the timed loop).
                not_ok = (combit_n_decompress(combit_n_not_inplace(cA)) == rN);
            }

            out << c << ",L" << depth << ","
                << total << "," << double(total) / (1024.0 * 1024.0) << ","
                << median(tA) << "," << median(tO) << "," << median(tN) << ","
                << (and_ok ? 1 : 0) << "," << (or_ok ? 1 : 0) << "," << (not_ok ? 1 : 0) << ","
                << (rt_ok ? 1 : 0) << "\n";
            out.flush();
            std::cerr << "[c=" << c << " L" << depth << "] size=" << total
                      << "B and=" << median(tA) << "ms or=" << median(tO)
                      << "ms not=" << median(tN) << "ms (correct: A="
                      << and_ok << " O=" << or_ok << " N=" << not_ok
                      << " RT=" << rt_ok << ")\n";
        }
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
