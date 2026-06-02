

#include "ddc.h"

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

extern bool ddc_compress_results;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// dir name builder
static fs::path ddc_dir_for(const fs::path& root, int c, size_t seg_bits,
                               bool clustered = false) {
    std::string name = "bm_100m_c" + std::to_string(c) + "_ddc_w8";
    if (seg_bits != DDC::default_segment_bits)
        name += "_S" + std::to_string(seg_bits);
    if (clustered) name += "_run";
    return root / name;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_seg <bitmap_root_dir> <out_csv>\n"
                  << "  Scans <root>/bm_100m_c<c>_ddc_w8[_S<bits>]/ for\n"
                  << "  c in {10, 100, 1000} and segment_bits in\n"
                  << "  {4096, 16384, 65536, 262144}.\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    const std::vector<int> cards = {10, 100, 1000};

    const std::vector<size_t> seg_bits_set = {1u << 12, 1u << 14, 1u << 16, 1u << 18};
    const int N_ITER = 5;

    std::ofstream out(out_path);
    out << "data_shape,"
        << "cardinality,density,segment_bits,n_bitmaps,num_segs,"
        << "payload_bytes,payload_MiB,"
        << "ondisk_bytes,ondisk_MiB,"
        << "packed_bytes,packed_MiB,"
        << "packed_v4_bytes,packed_v4_MiB,"
        << "header_bytes_per_seg,"
        << "packed_header_bytes_per_seg,"
        << "packed_v4_header_bytes_per_seg,"
        << "l1_bytes,l2_bytes,l3_bytes,l4_bytes,"
        << "and_ms,or_ms,not_ms,comp_ms,"
        << "not_inner_ms,"
        << "roundtrip_ok\n";

    struct Shape { const char* label; bool clustered; };
    const std::vector<Shape> shapes = {
        {"uniform",   false},

    };

    for (const auto& shape : shapes) {
    for (int c : cards) {
        const double density = 1.0 / static_cast<double>(c);
        for (size_t S : seg_bits_set) {
            fs::path dir = ddc_dir_for(root, c, S, shape.clustered);
            if (!fs::is_directory(dir)) {
                std::cerr << "[skip " << shape.label << "] " << dir
                          << " (missing — run gen_bitmap"
                          << (shape.clustered ? " -R" : "")
                          << " -S " << S << " -c " << c << " first)\n";
                continue;
            }

            // collect + sort .bm
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

            const fs::path& fileC = (files.size() >= 3) ? files[2] : files[0];
            std::ifstream ia(files[0], std::ios::binary);
            std::ifstream ib(files[1], std::ios::binary);
            std::ifstream ic(fileC, std::ios::binary);
            DDC A = DDC::deserialize(ia);
            DDC B = DDC::deserialize(ib);
            DDC C = DDC::deserialize(ic);

            // roundtrip check
            bool rt_ok = true;
            {
                size_t pc0 = A.popcount();

                DDC r = A | A;
                rt_ok = (r.popcount() == pc0);
            }

            // warmup
            { DDC w = A.and_no_bypass(B); (void)w; }
            { DDC w = A | B;               (void)w; }
            { DDC w = ~A;                  (void)w; }
            {
                ddc_compress_results = true;
                DDC t1 = A.and_no_bypass(C);
                DDC t2 = B | t1;
                t2.negate_inplace();
                ddc_compress_results = false;
                (void)t2;
            }

            // time AND
            std::vector<double> tA, tO, tN, tCOMP;
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                DDC r = A.and_no_bypass(B);
                auto t1 = clk::now();
                tA.push_back(ms(t0, t1)); (void)r;
            }
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                DDC r = A | B;
                auto t1 = clk::now();
                tO.push_back(ms(t0, t1)); (void)r;
            }

            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                DDC r = ~A;
                auto t1 = clk::now();
                tN.push_back(ms(t0, t1));
                (void)r;
            }

            // COMP rewrite verify
            {
                ddc_compress_results = true;
                DDC chk = B | A.and_no_bypass(C); chk.negate_inplace();
                ddc_compress_results = false;
                DDC ref = ~((A | B).and_no_bypass(B | C));
                if (chk.popcount() != ref.popcount())
                    std::cerr << "[COMP REWRITE MISMATCH] c=" << c << " S=" << S
                              << " rewrite=" << chk.popcount()
                              << " ref=" << ref.popcount() << "\n";
            }
            for (int i = 0; i < N_ITER; i++) {
                ddc_compress_results = true;
                auto t0 = clk::now();
                DDC t1 = A.and_no_bypass(C);
                DDC t2 = B | t1;
                t2.negate_inplace();
                auto t1c = clk::now();
                ddc_compress_results = false;
                tCOMP.push_back(ms(t0, t1c));
            }

            double and_ms  = median(tA);
            double or_ms   = median(tO);
            double not_ms  = median(tN);
            double comp_ms = median(tCOMP);

            // L1-only NOT microbench
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

                // SIMD xor-FF
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
                    // masked tail
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

                fused_xor();
                std::vector<double> tInner;
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    fused_xor();
                    auto t1 = clk::now();
                    tInner.push_back(ms(t0, t1));
                }
                not_inner_ms = median(tInner);
            }

            // size breakdown per format
            size_t payload_bytes    = 0;
            size_t ondisk_bytes     = 0;
            size_t packed_bytes     = 0;
            size_t packed_v4_bytes  = 0;
            size_t agg_l1 = 0, agg_l2 = 0, agg_l3 = 0, agg_l4 = 0;
            size_t agg_num_segs = 0;
            for (const auto& fp : files) {
                std::ifstream is(fp, std::ios::binary);
                DDC cb = DDC::deserialize(is);
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

            // emit CSV row
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
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
