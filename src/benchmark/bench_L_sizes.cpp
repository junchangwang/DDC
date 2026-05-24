// bench_L_sizes.cpp
// =========================================================================
// Phase 1 of the L2/L3/L4/L5 depth study.
//
// Reads existing combit_w8 .bm files (which are compressed with the
// current depth=4 layout) and, for each cardinality, computes the
// PER-VARIANT total bytes that the marker hierarchy would occupy at
// depths 2, 3, 4, 5.  Sizes are analytical (derived from the L4
// breakdown) — we do NOT recompress the data:
//
//   L2 variant  : L1_lit + ⌈l2_count / 8⌉                    (no L3, no L4)
//   L3 variant  : L1_lit + L2_lit + ⌈l3_count / 8⌉           (no L4)
//   L4 variant  : L1_lit + L2_lit + L3_lit + ⌈l4_count / 8⌉  (current)
//   L5 variant  : L1_lit + L2_lit + L3_lit + l4_lit + ⌈l5_count / 8⌉
//
// l4_lit is computed by walking the L4 byte stream and counting bytes
// that differ from l4_fill (chosen as the more common of {0x00, 0xFF}).
// l5_count = number of L4 BYTES = ⌈l4_count / 8⌉.
//
// Output: CSV at tools/bench_L_sizes.csv, one row per (backend, c, layer)
// columns: c, variant, l1, l2, l3, l4, l5, total
// =========================================================================

#include "combit.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <cstdint>
namespace fs = std::filesystem;

struct SegmentLayers {
    size_t l1_lit;       // L1 literal bytes
    size_t l2_lit;       // L2 literal bytes (compressed via L3)
    size_t l3_lit;       // L3 literal bytes (compressed via L4)
    size_t l2_count;     // total L2 bits
    size_t l3_count;     // total L3 bits (= l2_byte_count = ⌈l2_count/8⌉)
    size_t l4_count;     // total L4 bits (= l3_byte_count = ⌈l3_count/8⌉)
    size_t l4_byte_cnt;  // = ⌈l4_count / 8⌉ — raw L4 byte stream size
    size_t l4_lit;       // L4 bytes that differ from l4_fill (used for L5)
    size_t l5_byte_cnt;  // = ⌈l4_byte_cnt / 8⌉ — raw L5 byte stream size
};

static SegmentLayers analyse_segment(const ComBitBtv& seg) {
    SegmentLayers s{};
    s.l1_lit      = seg.num_lits();
    s.l2_lit      = seg.l2_lit_count();
    s.l3_lit      = seg.l3_lit_count();
    s.l2_count    = seg.l2_count();
    s.l3_count    = seg.l3_count();
    s.l4_count    = seg.l4_count();
    s.l4_byte_cnt = (s.l4_count + 7) / 8;

    // Choose l4_fill that minimizes literal L4 bytes (mirrors the
    // greedy choice that compress_l3_to_l4 makes for L3 fill).
    const uint8_t* l4 = seg.l4_data();
    size_t non_zero = 0, non_ff = 0;
    for (size_t i = 0; i < s.l4_byte_cnt; i++) {
        if (l4[i] != 0x00) non_zero++;
        if (l4[i] != 0xFF) non_ff++;
    }
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
        std::cerr << "  Scans <root>/bm_100m_c*_combit_w8/ for each c.\n";
        return 1;
    }
    fs::path root = argv[1];
    fs::path out_path = argv[2];

    std::vector<int> cards = {2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000};

    std::ofstream out(out_path);
    out << "cardinality,variant,l1_bytes,l2_bytes,l3_bytes,l4_bytes,l5_bytes,total_bytes,total_MiB\n";

    for (int c : cards) {
        fs::path dir = root / ("bm_100m_c" + std::to_string(c) + "_combit_w8");
        if (!fs::is_directory(dir)) {
            std::cerr << "[skip] " << dir << " (missing)\n";
            continue;
        }
        // Aggregate sizes across ALL .bm files in this cardinality dir.
        size_t agg_l1=0, agg_l2_lit=0, agg_l3_lit=0, agg_l4_lit=0;
        size_t agg_l2_raw=0, agg_l3_raw=0, agg_l4_raw=0, agg_l5_raw=0;
        size_t file_count = 0;
        for (auto& e : fs::directory_iterator(dir)) {
            if (e.path().extension() != ".bm") continue;
            std::ifstream in(e.path(), std::ios::binary);
            ComBit cb = ComBit::deserialize(in);
            file_count++;
            for (const auto& seg : cb.segments()) {
                SegmentLayers s = analyse_segment(seg);
                agg_l1     += s.l1_lit;
                agg_l2_lit += s.l2_lit;
                agg_l3_lit += s.l3_lit;
                agg_l4_lit += s.l4_lit;
                agg_l2_raw += (s.l2_count + 7) / 8;     // ⌈l2_count/8⌉
                agg_l3_raw += (s.l3_count + 7) / 8;     // ⌈l3_count/8⌉ = ⌈l2_byte_cnt/8⌉
                agg_l4_raw += s.l4_byte_cnt;             // ⌈l4_count/8⌉
                agg_l5_raw += s.l5_byte_cnt;             // ⌈l4_byte_cnt/8⌉
            }
        }
        // Per-variant TOTALS across all C bitmaps loaded for this
        // cardinality (matches the motivation-chart memory reporting
        // format: "Density 0.001 | Loaded 1000 | L4: 23.33 MB" = sum of
        // L4 bytes across all 1000 bitmaps, NOT per-bitmap average).
        auto emit = [&](const char* variant, size_t l1, size_t l2, size_t l3,
                        size_t l4, size_t l5) {
            double total = double(l1 + l2 + l3 + l4 + l5);
            out << c << "," << variant << ","
                << double(l1) << "," << double(l2) << "," << double(l3) << ","
                << double(l4) << "," << double(l5) << ","
                << total << "," << total / (1024.0 * 1024.0) << "\n";
        };
        // L2 variant: L1 + L2_raw
        emit("L2", agg_l1, agg_l2_raw, 0, 0, 0);
        // L3 variant: L1 + L2_lit + L3_raw
        emit("L3", agg_l1, agg_l2_lit, agg_l3_raw, 0, 0);
        // L4 variant (current): L1 + L2_lit + L3_lit + L4_raw
        emit("L4", agg_l1, agg_l2_lit, agg_l3_lit, agg_l4_raw, 0);
        // L5 variant: L1 + L2_lit + L3_lit + L4_lit + L5_raw
        emit("L5", agg_l1, agg_l2_lit, agg_l3_lit, agg_l4_lit, agg_l5_raw);

        std::cerr << "[c=" << c << "] " << file_count << " files\n";
    }
    out.close();
    std::cerr << "[done] wrote " << out_path << "\n";
    return 0;
}
