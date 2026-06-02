

#include "ddc.h"
#include "ddc_n.h"

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

// pick 3 inputs
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

    fs::path l4_dir = root / ("bm_100m_c" + std::to_string(c) + "_ddc_w8_run");
    auto l4_files = first3(l4_dir);
    if (l4_files.size() < 3) { std::cerr << "missing 3 .bm at " << l4_dir << "\n"; return 1; }

    std::ifstream ia(l4_files[0], std::ios::binary);
    std::ifstream ib(l4_files[1], std::ios::binary);
    std::ifstream ic(l4_files[2], std::ios::binary);
    // load L4 inputs
    DDC A = DDC::deserialize(ia);
    DDC B = DDC::deserialize(ib);
    DDC C = DDC::deserialize(ic);

    std::ofstream out(out_path);
    out << "# depth  ops_s   median_comp_ms\n";

    std::cerr << "[c=" << c << " run-length]\n";

    {

        // warmup
        { DDC t1 = A | B; DDC t2 = B | C; DDC t3 = t1.and_no_bypass(t2);
          DDC r = ~t3; (void)r; }
        std::vector<double> t;
        // timed COMP
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = clk::now();
            DDC t1 = A | B;
            DDC t2 = B | C;
            DDC t3 = t1.and_no_bypass(t2);
            DDC r  = ~t3;
            auto t1c = clk::now();
            t.push_back(ms(t0, t1c));
            (void)r;
        }
        double m = median(t);
        out << "L4  " << (1000.0 / m) << "  " << m << "\n";
        std::cerr << "  L4 COMP=" << m << "ms\n";
    }

    // sweep depths
    for (int depth : {2, 3, 5}) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c)
                              + "_ddc_L" + std::to_string(depth) + "_run");
        auto files = first3(dir);
        if (files.size() < 3) { std::cerr << "missing 3 .bm at " << dir << "\n"; return 1; }

        std::ifstream nia(files[0], std::ios::binary);
        std::ifstream nib(files[1], std::ios::binary);
        std::ifstream nic(files[2], std::ios::binary);
        DDCN cA = ddc_n_deserialize(nia);
        DDCN cB = ddc_n_deserialize(nib);
        DDCN cC = ddc_n_deserialize(nic);

        // warmup
        { DDCN t1 = ddc_n_or (cA, cB);
          DDCN t2 = ddc_n_or (cB, cC);
          DDCN t3 = ddc_n_and(t1, t2);
          ddc_n_not_inplace(t3); (void)t3; }

        std::vector<double> t;
        for (int i = 0; i < N_ITER; i++) {
            auto t0 = clk::now();
            DDCN t1 = ddc_n_or (cA, cB);
            DDCN t2 = ddc_n_or (cB, cC);
            DDCN t3 = ddc_n_and(t1, t2);
            ddc_n_not_inplace(t3);
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
