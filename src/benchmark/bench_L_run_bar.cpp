// bench_L_run_bar.cpp
// ============================================================================
// One-shot driver for the L-depth bar chart (clustered workload).
// Reads run-length .bm files (suffix `_run`) at a single cardinality,
// times the COMP expression ~((A|B) & (B|C)) for each depth L2/L3/L4/L5,
// and prints "<depth> <ops_per_sec>" per line.
//
// Used by tools/hierarchy_depth_runlength_bar.gnuplot.  The run-length
// data shape models a column store sorted by the indexed attribute
// (TPC-H Q6 / sorted shipdate), where every value's bitmap is a single
// contiguous run of set bits — L4's 32K-bit batch-skip's design target.
// ============================================================================

#include "combit.h"
#include "combit_n.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
using clk = std::chrono::high_resolution_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<fs::path> first3(const fs::path& dir) {
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".bm") files.push_back(e.path());
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });
    if (files.size() > 3) files.resize(3);
    return files;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: bench_L_run_bar <bitmap_root> <cardinality> <out_dat>\n";
        return 1;
    }
    fs::path root = argv[1];
    int c = std::stoi(argv[2]);
    fs::path out_path = argv[3];
    const int N_ITER = 5;

    // L4 = production ComBit at bm_100m_c<c>_combit_w8_run/
    fs::path l4_dir = root / ("bm_100m_c" + std::to_string(c) + "_combit_w8_run");
    auto l4_files = first3(l4_dir);
    if (l4_files.size() < 3) { std::cerr << "missing 3 .bm at " << l4_dir << "\n"; return 1; }

    std::ifstream ia(l4_files[0], std::ios::binary);
    std::ifstream ib(l4_files[1], std::ios::binary);
    std::ifstream ic(l4_files[2], std::ios::binary);
    ComBit A = ComBit::deserialize(ia);
    ComBit B = ComBit::deserialize(ib);
    ComBit C = ComBit::deserialize(ic);

    std::ofstream out(out_path);
    out << "# depth  ops_s   median_comp_ms\n";

    std::cerr << "[c=" << c << " run-length]\n";

    // L4 production: ~((A|B) & (B|C))
    {
        // Warm-up
        { ComBit t1 = A | B; ComBit t2 = B | C; ComBit t3 = t1.and_no_bypass(t2);
          ComBit r = ~t3; (void)r; }
        std::vector<double> t;
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = clk::now();
            ComBit t1 = A | B;
            ComBit t2 = B | C;
            ComBit t3 = t1.and_no_bypass(t2);
            ComBit r  = ~t3;
            auto t1c = clk::now();
            t.push_back(ms(t0, t1c));
            (void)r;
        }
        double m = median(t);
        out << "L4  " << (1000.0 / m) << "  " << m << "\n";
        std::cerr << "  L4 COMP=" << m << "ms\n";
    }

    // L2/L3/L5 ComBitN: load run-length .bm at the appropriate depth and run
    // the SAME 4-stage compressed chain via combit_n_or / combit_n_and /
    // combit_n_not_inplace (per-segment scratch + SIMD compress fusion).
    for (int depth : {2, 3, 5}) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c)
                              + "_combit_L" + std::to_string(depth) + "_run");
        auto files = first3(dir);
        if (files.size() < 3) { std::cerr << "missing 3 .bm at " << dir << "\n"; return 1; }

        std::ifstream nia(files[0], std::ios::binary);
        std::ifstream nib(files[1], std::ios::binary);
        std::ifstream nic(files[2], std::ios::binary);
        ComBitN cA = combit_n_deserialize(nia);
        ComBitN cB = combit_n_deserialize(nib);
        ComBitN cC = combit_n_deserialize(nic);

        // Warm-up
        { ComBitN t1 = combit_n_or (cA, cB);
          ComBitN t2 = combit_n_or (cB, cC);
          ComBitN t3 = combit_n_and(t1, t2);
          combit_n_not_inplace(t3); (void)t3; }

        std::vector<double> t;
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = clk::now();
            ComBitN t1 = combit_n_or (cA, cB);
            ComBitN t2 = combit_n_or (cB, cC);
            ComBitN t3 = combit_n_and(t1, t2);
            combit_n_not_inplace(t3);
            auto t1c = clk::now();
            t.push_back(ms(t0, t1c));
        }
        double m = median(t);
        out << "L" << depth << "  " << (1000.0 / m) << "  " << m << "\n";
        std::cerr << "  L" << depth << " COMP=" << m << "ms\n";
    }

    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
