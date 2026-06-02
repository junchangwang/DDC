

#include "ddc_n.h"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>

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

struct Cfg { const char* name; BypassCfg cfg; };

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_bypass <bitmap_root_dir> <out_csv>\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    // cardinality sweep
    std::vector<int> cards = {2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000};
    const int N_ITER = 5;

    // bypass variants
    const Cfg configs[3] = {
        {"all",      BypassCfg::BP_ALL},
        {"L4_off",   BypassCfg::BP_NO_L4},
        {"L4L3_off", BypassCfg::BP_NONE},
    };

    std::ofstream out(out_path);
    out << "cardinality,config,or_ms,and_ms,comp_ms,"
        << "or_ok,and_ok,comp_ok,roundtrip_ok\n";

    for (int c : cards) {

        fs::path dir = root / ("bm_100m_c" + std::to_string(c) + "_ddc_L4");
        if (!fs::is_directory(dir)) { std::cerr << "[skip] " << dir << "\n"; continue; }

        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".bm") files.push_back(e.path());
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
            return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
        });
        if (files.size() < 2) { std::cerr << "[skip] " << dir << " (<2 files)\n"; continue; }

        const fs::path& fileC = (files.size() >= 3) ? files[2] : files[0];
        std::ifstream ia(files[0], std::ios::binary);
        std::ifstream ib(files[1], std::ios::binary);
        std::ifstream ic(fileC,    std::ios::binary);
        // load 3 bitvectors
        DDCN cA = ddc_n_deserialize(ia);
        DDCN cB = ddc_n_deserialize(ib);
        DDCN cC = ddc_n_deserialize(ic);

        std::vector<bool> ba = ddc_n_decompress(cA);
        std::vector<bool> bb = ddc_n_decompress(cB);
        std::vector<bool> bc = ddc_n_decompress(cC);
        bool rt_ok = (ba.size() == bb.size() && bb.size() == bc.size());

        // expected results
        auto rO    = ref_or(ba, bb);
        auto rA    = ref_and(ba, bb);
        auto rCOMP = ref_not(ref_and(ref_or(ba, bb), ref_or(bb, bc)));

        for (const auto& C : configs) {
            const BypassCfg cfg = C.cfg;

            // warmup
            { auto w = ddc_n_or_dec_l4_cfg (cA, cB, cfg); (void)w; }
            { auto w = ddc_n_and_dec_l4_cfg(cA, cB, cfg); (void)w; }
            { DDCN t1 = ddc_n_or_l4_cfg (cA, cB, cfg);
              DDCN t2 = ddc_n_or_l4_cfg (cB, cC, cfg);
              DDCN t3 = ddc_n_and_l4_cfg(t1, t2, cfg);
              ddc_n_not_inplace(t3); (void)t3; }

            // timed runs
            std::vector<double> tO, tA, tC;
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                auto r = ddc_n_or_dec_l4_cfg(cA, cB, cfg);
                auto t1 = clk::now();
                tO.push_back(ms(t0, t1)); (void)r;
            }
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                auto r = ddc_n_and_dec_l4_cfg(cA, cB, cfg);
                auto t1 = clk::now();
                tA.push_back(ms(t0, t1)); (void)r;
            }
            // composite: (A|B) AND (B|C), NOT
            for (int i = 0; i < N_ITER; i++) {
                auto t0 = clk::now();
                DDCN t1 = ddc_n_or_l4_cfg (cA, cB, cfg);
                DDCN t2 = ddc_n_or_l4_cfg (cB, cC, cfg);
                DDCN t3 = ddc_n_and_l4_cfg(t1, t2, cfg);
                ddc_n_not_inplace(t3);
                auto t1c = clk::now();
                tC.push_back(ms(t0, t1c));
            }

            // correctness check
            auto chk = [&](const std::vector<uint8_t>& r, const std::vector<bool>& ref) {
                return bytes_to_bits(r, ref.size()) == ref;
            };
            bool or_ok  = chk(ddc_n_or_dec_l4_cfg (cA, cB, cfg), rO);
            bool and_ok = chk(ddc_n_and_dec_l4_cfg(cA, cB, cfg), rA);
            bool comp_ok;
            {
                DDCN t1 = ddc_n_or_l4_cfg (cA, cB, cfg);
                DDCN t2 = ddc_n_or_l4_cfg (cB, cC, cfg);
                DDCN t3 = ddc_n_and_l4_cfg(t1, t2, cfg);
                ddc_n_not_inplace(t3);
                comp_ok = (ddc_n_decompress(t3) == rCOMP);
            }

            // emit row
            out << c << "," << C.name << ","
                << median(tO) << "," << median(tA) << "," << median(tC) << ","
                << (or_ok ? 1 : 0) << "," << (and_ok ? 1 : 0) << ","
                << (comp_ok ? 1 : 0) << "," << (rt_ok ? 1 : 0) << "\n";
            out.flush();
            std::cerr << "[c=" << c << " " << C.name << "] "
                      << "or=" << median(tO) << "ms and=" << median(tA)
                      << "ms comp=" << median(tC) << "ms (ok: O=" << or_ok
                      << " A=" << and_ok << " C=" << comp_ok << ")\n";
        }
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
