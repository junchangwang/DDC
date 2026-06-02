

#include "ddc.h"
#include "ddc_n.h"

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

// reference ops
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

    // per-cardinality sweep
    for (int c : cards) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c) + "_ddc_w8");
        if (!fs::is_directory(dir)) { std::cerr << "[skip] " << dir << "\n"; continue; }

        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".bm") files.push_back(e.path());
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
            return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
        });
        if (files.size() < 2) { std::cerr << "[skip] " << dir << " (<2 files)\n"; continue; }

        const fs::path& fileC = (files.size() >= 3) ? files[2] : files[0];
        std::ifstream ia(files[0], std::ios::binary), ib(files[1], std::ios::binary), ic(fileC, std::ios::binary);
        DDC A = DDC::deserialize(ia);
        DDC B = DDC::deserialize(ib);
        DDC C = DDC::deserialize(ic);
        std::vector<bool> ba(A.bit_count()), bb(B.bit_count()), bc(C.bit_count());
        // expand to bool ref
        auto cb_decompress_to = [](const DDC& cb, std::vector<bool>& out) {
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

        // expected results
        auto rA = ref_and(ba, bb);
        auto rO = ref_or (ba, bb);
        auto rN = ref_not(ba);
        auto rCOMP = ref_not(ref_and(ref_or(ba, bb), ref_or(bb, bc)));

        // per-level (L2..L5)
        for (int depth : {2, 3, 4, 5}) {

            DDCN cA, cB, cC;
            fs::path ndir = root / ("bm_100m_c" + std::to_string(c)
                                     + "_ddc_L" + std::to_string(depth));
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

            const fs::path& nfileC = (nfiles.size() >= 3) ? nfiles[2] : nfiles[0];
            std::ifstream nia(nfiles[0], std::ios::binary);
            std::ifstream nib(nfiles[1], std::ios::binary);
            std::ifstream nic(nfileC, std::ios::binary);
            cA = ddc_n_deserialize(nia);
            cB = ddc_n_deserialize(nib);
            cC = ddc_n_deserialize(nic);

            bool rt_ok = (ddc_n_decompress(cA) == ba);

            size_t total = ddc_n_total_bytes(cA);

            std::vector<double> tA, tO, tN, tCOMP;
            bool and_ok = true, or_ok = true, not_ok = true, comp_ok = true;
            {

                // warmup
                { auto w = ddc_n_and_dec_avx(cA, cB); (void)w; }
                { auto w = ddc_n_or_dec_avx(cA, cB); (void)w; }
                ddc_n_not_inplace(cA); ddc_n_not_inplace(cA);

                { DDCN t1 = ddc_n_or (cA, cB);
                  DDCN t2 = ddc_n_or (cB, cC);
                  DDCN t3 = ddc_n_and(t1, t2);
                  ddc_n_not_inplace(t3); (void)t3; }
                // timed AND
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    auto r = ddc_n_and_dec_avx(cA, cB);
                    auto t1 = clk::now();
                    tA.push_back(ms(t0, t1)); (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    auto r = ddc_n_or_dec_avx(cA, cB);
                    auto t1 = clk::now();
                    tO.push_back(ms(t0, t1)); (void)r;
                }
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    ddc_n_not_inplace(cA);
                    auto t1 = clk::now();
                    tN.push_back(ms(t0, t1));
                    ddc_n_not_inplace(cA);
                }

                // timed COMP (compound expr)
                for (int i = 0; i < N_ITER; i++) {
                    auto t0 = clk::now();
                    DDCN t1 = ddc_n_or (cA, cB);
                    DDCN t2 = ddc_n_or (cB, cC);
                    DDCN t3 = ddc_n_and(t1, t2);
                    ddc_n_not_inplace(t3);
                    auto t1c = clk::now();
                    tCOMP.push_back(ms(t0, t1c));
                }
                // correctness check
                auto chk = [&](const std::vector<uint8_t>& r,
                               const std::vector<bool>& ref) {
                    return bytes_to_bits(r, ref.size()) == ref;
                };
                and_ok = chk(ddc_n_and_dec_avx(cA, cB), rA);
                or_ok  = chk(ddc_n_or_dec_avx (cA, cB), rO);

                ddc_n_not_inplace(cA);
                not_ok = (ddc_n_decompress(cA) == rN);
                ddc_n_not_inplace(cA);
                {
                    DDCN t1 = ddc_n_or (cA, cB);
                    DDCN t2 = ddc_n_or (cB, cC);
                    DDCN t3 = ddc_n_and(t1, t2);
                    ddc_n_not_inplace(t3);
                    comp_ok = (ddc_n_decompress(t3) == rCOMP);
                }
            }

            // emit CSV row
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
