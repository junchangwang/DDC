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
        << "and_ms,or_ms,not_ms,comp_ms,"
        << "and_ok,or_ok,not_ok,comp_ok,roundtrip_ok\n";

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
        // A,B drive single-op AND/OR.  C joins B to drive the COMP
        // expression ~((A|B) & (B|C)) — same 3-input pattern as the
        // production motivation chart's COMP_op.  At c=2 only 2 .bm
        // files exist, so fall back to C = A (matches benchmark_main).
        const fs::path& fileC = (files.size() >= 3) ? files[2] : files[0];
        std::ifstream ia(files[0], std::ios::binary), ib(files[1], std::ios::binary), ic(fileC, std::ios::binary);
        ComBit A = ComBit::deserialize(ia);
        ComBit B = ComBit::deserialize(ib);
        ComBit C = ComBit::deserialize(ic);
        std::vector<bool> ba(A.bit_count()), bb(B.bit_count()), bc(C.bit_count());
        auto cb_decompress_to = [](const ComBit& cb, std::vector<bool>& out) {
            size_t base = 0;
            for (size_t s = 0; s < cb.num_segments(); s++) {
                auto v = cb.segment(s).decompress();
                for (size_t i = 0; i < v.size() && base + i < out.size(); i++) out[base + i] = v[i];
                base += v.size();
            }
        };
        cb_decompress_to(A, ba);
        cb_decompress_to(B, bb);
        cb_decompress_to(C, bc);

        // Reference results.  AND/OR are CROSS (ba vs bb).  NOT is unary on A.
        // COMP = ~((A|B) & (B|C)) — same as production benchmark_main COMP_op.
        auto rA = ref_and(ba, bb);
        auto rO = ref_or (ba, bb);
        auto rN = ref_not(ba);
        auto rCOMP = ref_not(ref_and(ref_or(ba, bb), ref_or(bb, bc)));

        // For each depth, load + measure + time + validate.
        for (int depth : {2, 3, 4, 5}) {
            // -------- LOAD inputs at this depth ----------------------
            ComBitN cA, cB, cC;
            // L4 uses production AVX-512 code path; A,B,C are reused
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
                // c=2: fall back to C = A (matches production motivation pattern).
                const fs::path& nfileC = (nfiles.size() >= 3) ? nfiles[2] : nfiles[0];
                std::ifstream nia(nfiles[0], std::ios::binary);
                std::ifstream nib(nfiles[1], std::ios::binary);
                std::ifstream nic(nfileC, std::ios::binary);
                cA = combit_n_deserialize(nia);
                cB = combit_n_deserialize(nib);
                cC = combit_n_deserialize(nic);
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
            std::vector<double> tA, tO, tN, tCOMP;
            bool and_ok = true, or_ok = true, not_ok = true, comp_ok = true;
            if (depth == 4) {
                // Warm-ups (A.negate_inplace mutates A — pair it to restore).
                { ComBit w = A.and_no_bypass(B); (void)w; }
                { ComBit w = A | B; (void)w; }
                A.negate_inplace(); A.negate_inplace();
                // COMP warm-up: ~((A|B) & (B|C)).  Production chain — every
                // stage returns compressed ComBit, no decompress in the chain.
                { ComBit t1 = A | B;
                  ComBit t2 = B | C;
                  ComBit t3 = t1.and_no_bypass(t2);
                  ComBit r  = ~t3; (void)r; }
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
                // NOT: time the in-place flip itself (no out-of-place copy).
                // NOT is self-inverse, so a second untimed call restores A.
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    A.negate_inplace();
                    auto t1 = clk::now();
                    tN.push_back(ms(t0, t1));
                    A.negate_inplace();  // restore for next iter
                }
                // COMP: ~((A|B) & (B|C)).  Time the full 4-stage compressed
                // chain — every intermediate ComBit is built with markers,
                // matching production motivation chart's COMP_op pattern.
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ComBit t1 = A | B;
                    ComBit t2 = B | C;
                    ComBit t3 = t1.and_no_bypass(t2);
                    ComBit r  = ~t3;
                    auto t1c = clk::now();
                    tCOMP.push_back(ms(t0, t1c));
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
                A.negate_inplace();
                not_ok = (cb_to_bits(A) == rN);
                A.negate_inplace();   // restore A to its original state
                {
                    ComBit t1 = A | B;
                    ComBit t2 = B | C;
                    ComBit t3 = t1.and_no_bypass(t2);
                    ComBit r  = ~t3;
                    comp_ok = (cb_to_bits(r) == rCOMP);
                }
            } else {
                // Cross-AND too (matches L4 above).  Goes through general
                // seg_op_l*<AND>, not seg_self_and_l*.  combit_n_not_inplace
                // mutates cA — pair it (NOT twice = identity) to restore.
                { auto w = combit_n_and_dec_avx(cA, cB); (void)w; }
                { auto w = combit_n_or_dec_avx(cA, cB); (void)w; }
                combit_n_not_inplace(cA); combit_n_not_inplace(cA);
                // COMP warm-up: compressed chain — each combit_n_or/and
                // returns a ComBitN at the same depth, so the next stage
                // gets a marker-aware input (matches L4's compressed chain).
                { ComBitN t1 = combit_n_or (cA, cB);
                  ComBitN t2 = combit_n_or (cB, cC);
                  ComBitN t3 = combit_n_and(t1, t2);
                  combit_n_not_inplace(t3); (void)t3; }
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
                    combit_n_not_inplace(cA);   // in-place flip (timed)
                    auto t1 = clk::now();
                    tN.push_back(ms(t0, t1));
                    combit_n_not_inplace(cA);   // restore (untimed)
                }
                // COMP: ~((A|B) & (B|C)) compressed chain.
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ComBitN t1 = combit_n_or (cA, cB);
                    ComBitN t2 = combit_n_or (cB, cC);
                    ComBitN t3 = combit_n_and(t1, t2);
                    combit_n_not_inplace(t3);
                    auto t1c = clk::now();
                    tCOMP.push_back(ms(t0, t1c));
                }
                auto chk = [&](const std::vector<uint8_t>& r,
                               const std::vector<bool>& ref) {
                    return bytes_to_bits(r, ref.size()) == ref;
                };
                and_ok = chk(combit_n_and_dec_avx(cA, cB), rA);
                or_ok  = chk(combit_n_or_dec_avx (cA, cB), rO);
                // Verify NOT by flipping, decompressing, and restoring.
                combit_n_not_inplace(cA);
                not_ok = (combit_n_decompress(cA) == rN);
                combit_n_not_inplace(cA);   // restore
                {
                    ComBitN t1 = combit_n_or (cA, cB);
                    ComBitN t2 = combit_n_or (cB, cC);
                    ComBitN t3 = combit_n_and(t1, t2);
                    combit_n_not_inplace(t3);
                    comp_ok = (combit_n_decompress(t3) == rCOMP);
                }
            }

            out << c << ",L" << depth << ","
                << total << "," << double(total) / (1024.0 * 1024.0) << ","
                << median(tA) << "," << median(tO) << "," << median(tN) << "," << median(tCOMP) << ","
                << (and_ok ? 1 : 0) << "," << (or_ok ? 1 : 0) << "," << (not_ok ? 1 : 0) << ","
                << (comp_ok ? 1 : 0) << "," << (rt_ok ? 1 : 0) << "\n";
            out.flush();
            std::cerr << "[c=" << c << " L" << depth << "] size=" << total
                      << "B and=" << median(tA) << "ms or=" << median(tO)
                      << "ms not=" << median(tN) << "ms comp=" << median(tCOMP)
                      << "ms (correct: A=" << and_ok << " O=" << or_ok
                      << " N=" << not_ok << " C=" << comp_ok
                      << " RT=" << rt_ok << ")\n";
        }
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
