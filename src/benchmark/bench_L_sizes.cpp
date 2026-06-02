

#include "ddc.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <cstdint>
namespace fs = std::filesystem;

struct SegmentLayers {
    size_t l1_lit;
    size_t l2_lit;
    size_t l3_lit;
    size_t l2_count;
    size_t l3_count;
    size_t l4_count;
    size_t l4_byte_cnt;
    size_t l4_lit;
    size_t l5_byte_cnt;
};

// per-segment layer sizes
static SegmentLayers analyse_segment(const DDCBtv& seg) {
    SegmentLayers s{};
    s.l1_lit      = seg.num_lits();
    s.l2_lit      = seg.l2_lit_count();
    s.l3_lit      = seg.l3_lit_count();
    s.l2_count    = seg.l2_count();
    s.l3_count    = seg.l3_count();
    s.l4_count    = seg.l4_count();
    s.l4_byte_cnt = (s.l4_count + 7) / 8;

    const uint8_t* l4 = seg.l4_data();
    size_t non_zero = 0, non_ff = 0;
    for (size_t i = 0; i < s.l4_byte_cnt; i++) {
        if (l4[i] != 0x00) non_zero++;
        if (l4[i] != 0xFF) non_ff++;
    }
    // pick fill byte
    uint8_t l4_fill = (non_ff < non_zero) ? 0xFF : 0x00;
    size_t l4_lit = 0;
    for (size_t i = 0; i < s.l4_byte_cnt; i++)
        if (l4[i] != l4_fill) l4_lit++;
    s.l4_lit      = l4_lit;
    s.l5_byte_cnt = (s.l4_byte_cnt + 7) / 8;
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_L_sizes <bitmap_root_dir> <out_csv>\n";
        std::cerr << "  Scans <root>/bm_100m_c*_ddc_w8/ for each c.\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    std::vector<int> cards = {2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                              3000, 5000, 10000, 20000, 50000};

    std::ofstream out(out_path);
    out << "cardinality,variant,l1_bytes,l2_bytes,l3_bytes,l4_bytes,l5_bytes,total_bytes,total_MiB\n";

    for (int c : cards) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c) + "_ddc_w8");
        if (!fs::is_directory(dir)) {
            std::cerr << "[skip] " << dir << " (missing)\n";
            continue;
        }

        size_t agg_l1=0, agg_l2_lit=0, agg_l3_lit=0, agg_l4_lit=0;
        size_t agg_l2_raw=0, agg_l3_raw=0, agg_l4_raw=0, agg_l5_raw=0;
        size_t file_count = 0;
        // scan .bm files
        for (auto& e : fs::directory_iterator(dir)) {
            if (e.path().extension() != ".bm") continue;
            std::ifstream in(e.path(), std::ios::binary);
            DDC cb = DDC::deserialize(in);
            file_count++;
            // aggregate per segment
            for (const auto& seg : cb.segments()) {
                SegmentLayers s = analyse_segment(seg);
                agg_l1     += s.l1_lit;
                agg_l2_lit += s.l2_lit;
                agg_l3_lit += s.l3_lit;
                agg_l4_lit += s.l4_lit;
                agg_l2_raw += (s.l2_count + 7) / 8;
                agg_l3_raw += (s.l3_count + 7) / 8;
                agg_l4_raw += s.l4_byte_cnt;
                agg_l5_raw += s.l5_byte_cnt;
            }
        }

        auto emit = [&](const char* variant, size_t l1, size_t l2, size_t l3,
                        size_t l4, size_t l5) {
            double total = double(l1 + l2 + l3 + l4 + l5);
            out << c << "," << variant << ","
                << double(l1) << "," << double(l2) << "," << double(l3) << ","
                << double(l4) << "," << double(l5) << ","
                << total << "," << total / (1024.0 * 1024.0) << "\n";
        };

        // emit per-depth variants
        emit("L2", agg_l1, agg_l2_raw, 0, 0, 0);

        emit("L3", agg_l1, agg_l2_lit, agg_l3_raw, 0, 0);

        emit("L4", agg_l1, agg_l2_lit, agg_l3_lit, agg_l4_raw, 0);

        emit("L5", agg_l1, agg_l2_lit, agg_l3_lit, agg_l4_lit, agg_l5_raw);

        std::cerr << "[c=" << c << "] " << file_count << " files\n";
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
