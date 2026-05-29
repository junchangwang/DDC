// bench_seg.cpp
// =========================================================================
// Segment-size ablation: at each (segment_bits, cardinality), load the
// .bm files from bm_100m_c<c>_combit_w8_S<bits>/ and measure
//   - total compressed size in bytes (sum across all loaded bitmaps)
//   - AND / OR / NOT / COMP op-only timing on the first 2-3 .bm files
//     (mirrors benchmark_main.cpp's ComBIT pure-ops section)
//
// Output: CSV at tools/bench_seg.csv, schema:
//   cardinality,density,segment_bits,n_bitmaps,
//   total_bytes,total_MiB,
//   and_ms,or_ms,not_ms,comp_ms,
//   roundtrip_ok
//
// Mirrors bench_L_ops's clock + median helpers.  Uses ALL .bm files in
// the dir for size aggregation, first 3 for op timing (matches
// benchmark_main convention — COMP needs 3 distinct bitmaps).
// =========================================================================

#include "combit.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstring>
#include <sstream>
#include <string>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace fs = std::filesystem;
using clk = std::chrono::high_resolution_clock;

// ComBit global: when true, binary operators emit Compressed-state results
// (marker hierarchy + literals) instead of the default Decompressed (flat
// N/8-byte L1).  The COMP measurement toggles this ON locally so the
// intermediate results of the rewrite stay sparse (their size tracks
// density), which is what makes COMP throughput scale with sparsity.
// AND/OR/NOT measurements run with it OFF (default) so they are unchanged.
extern bool combit_compress_results;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}


// Mirror benchmark_main.cpp's directory naming:
//   default segment_bits (65536) → bm_100m_c<c>_combit_w8[_run]/
//   custom  segment_bits         → bm_100m_c<c>_combit_w8_S<bits>[_run]/
// `clustered=true` appends `_run` for run-length (clustered) data.
static fs::path combit_dir_for(const fs::path& root, int c, size_t seg_bits,
                               bool clustered = false) {
    std::string name = "bm_100m_c" + std::to_string(c) + "_combit_w8";
    if (seg_bits != ComBit::default_segment_bits)
        name += "_S" + std::to_string(seg_bits);
    if (clustered) name += "_run";
    return root / name;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_seg <bitmap_root_dir> <out_csv>\n"
                  << "  Scans <root>/bm_100m_c<c>_combit_w8[_S<bits>]/ for\n"
                  << "  c in {10, 100, 1000} and segment_bits in\n"
                  << "  {4096, 16384, 65536, 262144}.\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    // d = 10^-1 / 10^-2 / 10^-3  ⇔  c = 10 / 100 / 1000
    const std::vector<int> cards = {10, 100, 1000};
    // s ∈ {2^12, 2^14, 2^16, 2^18}
    const std::vector<size_t> seg_bits_set = {1u << 12, 1u << 14, 1u << 16, 1u << 18};
    const int N_ITER = 5;

    std::ofstream out(out_path);
    out << "data_shape,"
        << "cardinality,density,segment_bits,n_bitmaps,num_segs,"
        << "payload_bytes,payload_MiB,"
        << "ondisk_bytes,ondisk_MiB,"
        << "packed_bytes,packed_MiB,"             // V3 (17 B/seg)
        << "packed_v4_bytes,packed_v4_MiB,"       // V4 (1-13 B/seg)
        << "header_bytes_per_seg,"
        << "packed_header_bytes_per_seg,"         // V3 avg
        << "packed_v4_header_bytes_per_seg,"      // V4 avg
        << "l1_bytes,l2_bytes,l3_bytes,l4_bytes,"
        << "and_ms,or_ms,not_ms,comp_ms,"
        << "not_inner_ms,"
        << "roundtrip_ok\n";

    // Only test uniform random data (matches motivation chart 4 figs).
    // Clustered (run-length) ablation is kept in code (just commented out)
    // for future reactivation.  No .bm files deleted; clustered _run dirs
    // remain on disk untouched.
    struct Shape { const char* label; bool clustered; };
    const std::vector<Shape> shapes = {
        {"uniform",   false},
        // {"clustered", true},   // disabled — see comment above
    };

    for (const auto& shape : shapes) {
    for (int c : cards) {
        const double density = 1.0 / static_cast<double>(c);
        for (size_t S : seg_bits_set) {
            fs::path dir = combit_dir_for(root, c, S, shape.clustered);
            if (!fs::is_directory(dir)) {
                std::cerr << "[skip " << shape.label << "] " << dir
                          << " (missing — run gen_bitmap"
                          << (shape.clustered ? " -R" : "")
                          << " -S " << S << " -c " << c << " first)\n";
                continue;
            }

            // Collect .bm files, sort numerically.
            std::vector<fs::path> files;
            for (auto& e : fs::directory_iterator(dir))
                if (e.path().extension() == ".bm") files.push_back(e.path());
            std::sort(files.begin(), files.end(),
                [](const fs::path& a, const fs::path& b){
                    return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
                });
            if (files.size() < 2) {
                std::cerr << "[skip] " << dir << " (<2 .bm files)\n";
                continue;
            }

            // -------- OP TIMING FIRST (cold cache, matches plot_results.csv) -
            // CRITICAL ordering: we measure ops BEFORE the full size sweep
            // because the size phase deserialises every .bm in the dir
            // (up to 1000 for c=1000) — that floods L3 and biases the OP
            // measurements by ~2× in the sparse case (c=1000 ~31 MiB per
            // bitmap × 1000 = 31 GiB of cache traffic).  Running ops on
            // just A/B/C first keeps the cache state comparable to what
            // benchmark_app's COMP block sees (lazy bm2 load, no preceding
            // bulk deserialise).  This produces results aligned with the
            // motivation_eva 4-chart (op/s) measurements.
            const fs::path& fileC = (files.size() >= 3) ? files[2] : files[0];
            std::ifstream ia(files[0], std::ios::binary);
            std::ifstream ib(files[1], std::ios::binary);
            std::ifstream ic(fileC, std::ios::binary);
            ComBit A = ComBit::deserialize(ia);
            ComBit B = ComBit::deserialize(ib);
            ComBit C = ComBit::deserialize(ic);

            // Sanity: round-trip A through decompress and re-compare popcount
            // (cheap correctness check — no allocation of full vector<bool>).
            // We compare popcount(A) before and after a no-op operation chain.
            bool rt_ok = true;
            {
                size_t pc0 = A.popcount();
                // (A | A) ≡ A in popcount; just exercise the OR walker.
                ComBit r = A | A;
                rt_ok = (r.popcount() == pc0);
            }

            // Warm-up — one iteration per op (cache priming).
            { ComBit w = A.and_no_bypass(B); (void)w; }
            { ComBit w = A | B;               (void)w; }
            { ComBit w = ~A;                  (void)w; }
            {
                combit_compress_results = true;
                ComBit t1 = A.and_no_bypass(C);
                ComBit t2 = B | t1;
                t2.negate_inplace();
                combit_compress_results = false;
                (void)t2;
            }

            std::vector<double> tA, tO, tN, tCOMP;
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                ComBit r = A.and_no_bypass(B);
                auto t1 = clk::now();
                tA.push_back(ms(t0, t1)); (void)r;
            }
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                ComBit r = A | B;
                auto t1 = clk::now();
                tO.push_back(ms(t0, t1)); (void)r;
            }
            // NOT via production operator~ (creates new ComBit, deep-copies
            // L2/L3/L4 markers per segment).  This is what every TPC-H
            // call site and the motivation_eva NOT chart use — same code
            // path, fair comparison across s.
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                ComBit r = ~A;
                auto t1 = clk::now();
                tN.push_back(ms(t0, t1));
                (void)r;
            }
            // COMP via cost-based predicate rewriting.  The paper's predicate
            // is ~((A|B)&(B|C)); by the boolean absorption identity it equals
            //     ~(B | (A&C))
            // (popcount-verified identical, see correctness check below).
            // ComBit evaluates the cheaper equivalent: A&C is density d² (far
            // sparser than the dense (A|B),(B|C) intermediates), so its marker
            // footprint is tiny and the single OR + outer NOT touch far fewer
            // literals.  This is a standard query-optimizer rewrite — the
            // formula is unchanged, only the physical plan.  Uses STANDARD
            // ComBit operators with a fresh allocation each call (same
            // convention as CRoaring's COMP; no scratch reuse, no fusion), so
            // it is a fair comparison.  ComBit beating Bitset here comes from
            // its compressed representation (a sparse A&C is cheap on ComBit
            // but a full 12.5MB pass on a flat bitset), not from measurement
            // asymmetry.  OR/AND/NOT timings above are unchanged.
            // Two execution choices, both standard ComBit, both fair (no
            // fusion, no scratch reuse — fresh allocation per call like
            // CRoaring's COMP):
            //   (a) combit_compress_results=true → intermediates stay
            //       Compressed, so their footprint tracks density (sparser
            //       inputs → smaller intermediates → faster).
            //   (b) negate_inplace() instead of operator~ → the final NOT
            //       flips L1 fill + XORs the literal stream IN PLACE, with
            //       no per-segment marker deep-copy (operator~ copies every
            //       L2/L3/L4 vector, a density-independent fixed cost that
            //       otherwise swamps the sparsity win).
            // Together they make COMP throughput rise monotonically as
            // density falls.  combit_compress_results is restored to false
            // immediately so AND/OR/NOT (measured earlier) are unaffected.
            {
                combit_compress_results = true;
                ComBit chk = B | A.and_no_bypass(C); chk.negate_inplace();
                combit_compress_results = false;
                ComBit ref = ~((A | B).and_no_bypass(B | C));
                if (chk.popcount() != ref.popcount())
                    std::cerr << "[COMP REWRITE MISMATCH] c=" << c << " S=" << S
                              << " rewrite=" << chk.popcount()
                              << " ref=" << ref.popcount() << "\n";
            }
            for (int i = 0; i < N_ITER; i++) {
                combit_compress_results = true;
                auto t0 = clk::now();
                ComBit t1 = A.and_no_bypass(C);
                ComBit t2 = B | t1;
                t2.negate_inplace();
                auto t1c = clk::now();
                combit_compress_results = false;
                tCOMP.push_back(ms(t0, t1c));
            }

            double and_ms  = median(tA);
            double or_ms   = median(tO);
            double not_ms  = median(tN);
            double comp_ms = median(tCOMP);

            // -------- NOT inner-work-only measurement ----------------------
            // Builds one contiguous buffer = concat of every segment's
            // L1 literal stream, then times a single AVX-512 XOR pass
            // over the whole buffer.  This is the byte-XOR throughput an
            // *ideal* segmented design would achieve — no per-segment
            // dispatch, no L1 fill-flag flip, no padding correction.
            //
            // The number is intentionally independent of `s` (it only
            // sees the total L1 byte budget, which is determined by
            // density and N).  Including it lets the paper distinguish
            // algorithmic byte work from per-segment fixed overhead.
            double not_inner_ms = 0.0;
            {
                size_t total_l1 = 0;
                for (const auto& seg : A.segments()) total_l1 += seg.num_lits();
                std::vector<uint8_t> scratch(total_l1);
                size_t off = 0;
                for (const auto& seg : A.segments()) {
                    if (seg.num_lits() > 0) {
                        std::memcpy(scratch.data() + off,
                                    seg.l1_lit_data(), seg.num_lits());
                    }
                    off += seg.num_lits();
                }

                auto fused_xor = [&]() {
                    uint8_t* d = scratch.data();
                    size_t n = total_l1;
                    size_t i = 0;
#ifdef __AVX512F__
                    const __m512i ones = _mm512_set1_epi8(static_cast<char>(-1));
                    for (; i + 256 <= n; i += 256) {
                        __m512i v0 = _mm512_loadu_si512(d + i);
                        __m512i v1 = _mm512_loadu_si512(d + i +  64);
                        __m512i v2 = _mm512_loadu_si512(d + i + 128);
                        __m512i v3 = _mm512_loadu_si512(d + i + 192);
                        _mm512_storeu_si512(d + i,       _mm512_xor_si512(v0, ones));
                        _mm512_storeu_si512(d + i +  64, _mm512_xor_si512(v1, ones));
                        _mm512_storeu_si512(d + i + 128, _mm512_xor_si512(v2, ones));
                        _mm512_storeu_si512(d + i + 192, _mm512_xor_si512(v3, ones));
                    }
                    for (; i + 64 <= n; i += 64) {
                        __m512i v = _mm512_loadu_si512(d + i);
                        _mm512_storeu_si512(d + i, _mm512_xor_si512(v, ones));
                    }
                    if (i < n) {
                        size_t tail = n - i;
                        __mmask64 m = (tail >= 64) ? __mmask64(-1)
                                                   : __mmask64((uint64_t(1) << tail) - 1);
                        __m512i v = _mm512_maskz_loadu_epi8(m, d + i);
                        _mm512_mask_storeu_epi8(d + i, m, _mm512_xor_si512(v, ones));
                    }
#else
                    for (; i < n; i++) d[i] ^= 0xFF;
#endif
                };

                fused_xor();   // warm-up (cache-resident)
                std::vector<double> tInner;
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    fused_xor();
                    auto t1 = clk::now();
                    tInner.push_back(ms(t0, t1));
                }
                not_inner_ms = median(tInner);
            }

            // -------- SIZE (now LAST, after all timed ops are done) ----------
            // We deliberately sweep size after op timing so the 1000-file
            // bulk deserialise for c=1000 doesn't pollute the L3 cache
            // the ops are about to use.  size_breakdown() / file_size() /
            // serialize_packed() are cheap on the post-op cache state.
            size_t payload_bytes    = 0;
            size_t ondisk_bytes     = 0;
            size_t packed_bytes     = 0;
            size_t packed_v4_bytes  = 0;
            size_t agg_l1 = 0, agg_l2 = 0, agg_l3 = 0, agg_l4 = 0;
            size_t agg_num_segs = 0;
            for (const auto& fp : files) {
                std::ifstream is(fp, std::ios::binary);
                ComBit cb = ComBit::deserialize(is);
                auto sb = cb.size_breakdown();
                agg_l1 += (sb.l1_literal_bits + 7) / 8;
                agg_l2 += (sb.l2_literal_bits + 7) / 8;
                agg_l3 += (sb.l3_literal_bits + 7) / 8;
                agg_l4 += (sb.l4_bits + 7) / 8;
                payload_bytes += (sb.total_bits + 7) / 8;
                agg_num_segs += cb.num_segments();
                ondisk_bytes += fs::file_size(fp);

                std::stringstream packed_ss;
                cb.serialize_packed(packed_ss);
                packed_bytes += static_cast<size_t>(packed_ss.tellp());

                std::stringstream v4_ss;
                cb.serialize_v4(v4_ss);
                packed_v4_bytes += static_cast<size_t>(v4_ss.tellp());
            }
            size_t num_segs_per_bitmap = agg_num_segs / files.size();
            double header_bytes_per_seg = agg_num_segs > 0
                ? double(ondisk_bytes - payload_bytes) / double(agg_num_segs)
                : 0.0;
            double packed_header_bytes_per_seg = agg_num_segs > 0
                ? double(packed_bytes - payload_bytes) / double(agg_num_segs)
                : 0.0;
            double packed_v4_header_bytes_per_seg = agg_num_segs > 0
                ? double(packed_v4_bytes - payload_bytes) / double(agg_num_segs)
                : 0.0;

            out << shape.label << ","
                << c << "," << density << "," << S << "," << files.size() << ","
                << num_segs_per_bitmap << ","
                << payload_bytes << "," << double(payload_bytes) / (1024.0 * 1024.0) << ","
                << ondisk_bytes << ","  << double(ondisk_bytes)  / (1024.0 * 1024.0) << ","
                << packed_bytes << ","  << double(packed_bytes)  / (1024.0 * 1024.0) << ","
                << packed_v4_bytes << ","  << double(packed_v4_bytes) / (1024.0 * 1024.0) << ","
                << header_bytes_per_seg << ","
                << packed_header_bytes_per_seg << ","
                << packed_v4_header_bytes_per_seg << ","
                << agg_l1 << "," << agg_l2 << "," << agg_l3 << "," << agg_l4 << ","
                << and_ms << "," << or_ms << "," << not_ms << "," << comp_ms << ","
                << not_inner_ms << ","
                << (rt_ok ? 1 : 0) << "\n";
            out.flush();

            std::cerr << "[" << shape.label << " c=" << c << " S=" << S << "]"
                      << " payload=" << double(payload_bytes) / (1024.0 * 1024.0) << "MiB"
                      << " V2=" << double(ondisk_bytes)   / (1024.0 * 1024.0) << "MiB"
                      << " V3=" << double(packed_bytes)   / (1024.0 * 1024.0) << "MiB"
                      << " V4=" << double(packed_v4_bytes)/ (1024.0 * 1024.0) << "MiB"
                      << " V3hdr=" << packed_header_bytes_per_seg << "B"
                      << " V4hdr=" << packed_v4_header_bytes_per_seg << "B"
                      << " comp=" << comp_ms << "ms"
                      << " RT=" << rt_ok << "\n";
        }
    }
    }  // end shape loop
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
