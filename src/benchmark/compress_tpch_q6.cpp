

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cstdint>
#include <iomanip>

#include "benchmark/uti.h"
#include "benchmark/backends/ddc/ddc_backend.h"
#include "benchmark/backends/wah/wah_backend.h"
#include "benchmark/backends/croaring/croaring_backend.h"
#include "benchmark/backends/ewah/ewah_backend.h"

namespace fs = std::filesystem;

static size_t NUM_ROWS = 59986052;

struct CompressStats {
    long raw_total = 0;
    long ddc_total = 0;
    long wah_total = 0;
    long croaring_total = 0;
    long ewah_total = 0;
    int count = 0;
};

void compress_one(const std::string& raw_path,
                  const std::string& ddc_path,
                  const std::string& wah_path,
                  const std::string& croaring_path,
                  const std::string& ewah_path,
                  DDCBackend& ddc_be,
                  WahBackend& wah_be,
                  CroaringBackend& croaring_be,
                  EwahBackend& ewah_be,
                  CompressStats& stats,
                  bool verbose = true) {
    // compress one bitmap, all backends
    auto bits = read_raw_bm(raw_path, NUM_ROWS);
    if (bits.empty()) {
        std::cerr << "Error: failed to read " << raw_path << "\n";
        return;
    }

    fs::create_directories(fs::path(ddc_path).parent_path());
    fs::create_directories(fs::path(wah_path).parent_path());
    fs::create_directories(fs::path(croaring_path).parent_path());
    fs::create_directories(fs::path(ewah_path).parent_path());

    {
        auto handle = bits_to_bitmap(&ddc_be, bits);
        ddc_be.Serialize(*handle, ddc_path);
    }

    {
        auto handle = bits_to_bitmap(&wah_be, bits);
        wah_be.Serialize(*handle, wah_path);
    }

    {
        auto handle = bits_to_bitmap(&croaring_be, bits);
        croaring_be.Serialize(*handle, croaring_path);
    }

    {
        auto handle = bits_to_bitmap(&ewah_be, bits);
        ewah_be.Serialize(*handle, ewah_path);
    }

    // accumulate sizes
    long raw_size = get_file_size(raw_path);
    long cb_size = get_file_size(ddc_path);
    long wah_size = get_file_size(wah_path);
    long cr_size = get_file_size(croaring_path);
    long ew_size = get_file_size(ewah_path);

    stats.raw_total += raw_size;
    stats.ddc_total += cb_size;
    stats.wah_total += wah_size;
    stats.croaring_total += cr_size;
    stats.ewah_total += ew_size;
    stats.count++;

    if (verbose) {
        std::string name = fs::path(raw_path).parent_path().filename().string()
                         + "/" + fs::path(raw_path).filename().string();
        std::cout << "  " << name
                  << "  raw=" << raw_size
                  << "  ddc=" << cb_size
                  << "  wah=" << wah_size
                  << "  croaring=" << cr_size
                  << "  ewah=" << ew_size << "\n";
    }
}

void print_stats(const std::string& label, const CompressStats& s) {
    double cb_ratio = s.raw_total > 0 ? (double)s.ddc_total / s.raw_total : 0;
    double wah_ratio = s.raw_total > 0 ? (double)s.wah_total / s.raw_total : 0;
    double cr_ratio = s.raw_total > 0 ? (double)s.croaring_total / s.raw_total : 0;
    double ew_ratio = s.raw_total > 0 ? (double)s.ewah_total / s.raw_total : 0;
    std::cout << "  " << label << ": " << s.count << " bitmaps"
              << "  raw=" << s.raw_total / (1024*1024) << "MB"
              << "  ddc=" << s.ddc_total / (1024*1024) << "MB ("
              << std::fixed << std::setprecision(2) << cb_ratio << "x)"
              << "  wah=" << s.wah_total / (1024*1024) << "MB ("
              << std::fixed << std::setprecision(2) << wah_ratio << "x)"
              << "  croaring=" << s.croaring_total / (1024*1024) << "MB ("
              << std::fixed << std::setprecision(2) << cr_ratio << "x)"
              << "  ewah=" << s.ewah_total / (1024*1024) << "MB ("
              << std::fixed << std::setprecision(2) << ew_ratio << "x)\n";
}

int main(int argc, char* argv[]) {
    std::string input_dir = "bitmap/tpch_q6";
    std::string output_base = "bitmap/tpch_q6";

    if (argc >= 2) input_dir = argv[1];
    if (argc >= 3) output_base = argv[2];
    if (argc >= 4) NUM_ROWS = std::stoull(argv[3]);

    std::string ddc_dir = output_base + "_ddc";
    std::string wah_dir = output_base + "_wah";
    std::string croaring_dir = output_base + "_croaring";
    std::string ewah_dir = output_base + "_ewah";

    std::cout << "=========================================\n";
    std::cout << "  Compress TPC-H Q6 Bitmaps\n";
    std::cout << "  (Publication-grade: equality encoding)\n";
    std::cout << "=========================================\n";
    std::cout << "  Input:    " << input_dir << "\n";
    std::cout << "  DDC:   " << ddc_dir << "\n";
    std::cout << "  WAH:      " << wah_dir << "\n";
    std::cout << "  CRoaring: " << croaring_dir << "\n";
    std::cout << "  EWAH:     " << ewah_dir << "\n";
    std::cout << "  Rows:     " << NUM_ROWS << "\n\n";

    DDCBackend ddc_be;
    WahBackend wah_be;
    CroaringBackend croaring_be;
    EwahBackend ewah_be;

    auto t0 = std::chrono::high_resolution_clock::now();
    CompressStats disc_stats, qty_stats, ship_stats, total_stats;

    // discount predicate
    std::cout << "[Discount] Compressing 11 bitmaps...\n";
    for (int v = 0; v <= 10; v++) {
        std::string rel = "discount/" + std::to_string(v) + ".bm";
        compress_one(input_dir + "/" + rel,
                     ddc_dir + "/" + rel,
                     wah_dir + "/" + rel,
                     croaring_dir + "/" + rel,
                     ewah_dir + "/" + rel,
                     ddc_be, wah_be, croaring_be, ewah_be, disc_stats);
    }

    // quantity predicate
    std::cout << "\n[Quantity] Compressing 50 bitmaps...\n";
    for (int v = 1; v <= 50; v++) {
        std::string rel = "quantity/" + std::to_string(v) + ".bm";
        compress_one(input_dir + "/" + rel,
                     ddc_dir + "/" + rel,
                     wah_dir + "/" + rel,
                     croaring_dir + "/" + rel,
                     ewah_dir + "/" + rel,
                     ddc_be, wah_be, croaring_be, ewah_be, qty_stats);
    }

    // shipdate predicate
    std::string ship_input = input_dir + "/shipdate";
    std::vector<int> shipdate_ids;
    for (auto& entry : fs::directory_iterator(ship_input)) {
        if (entry.path().extension() == ".bm") {
            std::string stem = entry.path().stem().string();
            try { shipdate_ids.push_back(std::stoi(stem)); }
            catch (...) { continue; }
        }
    }
    std::sort(shipdate_ids.begin(), shipdate_ids.end());
    std::cout << "\n[Shipdate] Compressing " << shipdate_ids.size() << " bitmaps...\n";
    int ship_progress = 0;
    for (int id : shipdate_ids) {
        std::string rel = "shipdate/" + std::to_string(id) + ".bm";
        compress_one(input_dir + "/" + rel,
                     ddc_dir + "/" + rel,
                     wah_dir + "/" + rel,
                     croaring_dir + "/" + rel,
                     ewah_dir + "/" + rel,
                     ddc_be, wah_be, croaring_be, ewah_be, ship_stats, false);
        ship_progress++;
        if (ship_progress % 500 == 0 || ship_progress == (int)shipdate_ids.size()) {
            std::cout << "  ... " << ship_progress << "/" << shipdate_ids.size() << " done\n";
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t1 - t0).count();

    // aggregate totals
    total_stats.raw_total = disc_stats.raw_total + qty_stats.raw_total + ship_stats.raw_total;
    total_stats.ddc_total = disc_stats.ddc_total + qty_stats.ddc_total + ship_stats.ddc_total;
    total_stats.wah_total = disc_stats.wah_total + qty_stats.wah_total + ship_stats.wah_total;
    total_stats.croaring_total = disc_stats.croaring_total + qty_stats.croaring_total + ship_stats.croaring_total;
    total_stats.ewah_total = disc_stats.ewah_total + qty_stats.ewah_total + ship_stats.ewah_total;
    total_stats.count = disc_stats.count + qty_stats.count + ship_stats.count;

    // write manifests
    {
        std::ofstream f(ddc_dir + "/done.txt");
        f << "format=ddc\nnum_rows=" << NUM_ROWS
          << "\nnum_bitmaps=" << total_stats.count
          << "\ntotal_bytes=" << total_stats.ddc_total << "\n";
    }
    {
        std::ofstream f(wah_dir + "/done.txt");
        f << "format=wah\nnum_rows=" << NUM_ROWS
          << "\nnum_bitmaps=" << total_stats.count
          << "\ntotal_bytes=" << total_stats.wah_total << "\n";
    }
    {
        std::ofstream f(croaring_dir + "/done.txt");
        f << "format=croaring\nnum_rows=" << NUM_ROWS
          << "\nnum_bitmaps=" << total_stats.count
          << "\ntotal_bytes=" << total_stats.croaring_total << "\n";
    }
    {
        std::ofstream f(ewah_dir + "/done.txt");
        f << "format=ewah\nnum_rows=" << NUM_ROWS
          << "\nnum_bitmaps=" << total_stats.count
          << "\ntotal_bytes=" << total_stats.ewah_total << "\n";
    }

    std::cout << "\n=========================================\n";
    std::cout << "  Compression Summary\n";
    std::cout << "=========================================\n";
    print_stats("Discount (11)", disc_stats);
    print_stats("Quantity (50)", qty_stats);
    print_stats("Shipdate (" + std::to_string(ship_stats.count) + ")", ship_stats);
    std::cout << "  -----------------------------------------\n";
    print_stats("TOTAL (" + std::to_string(total_stats.count) + ")", total_stats);
    std::cout << "\n  Time: " << std::fixed << std::setprecision(1) << total_sec << "s\n";
    std::cout << "  DDC output:   " << ddc_dir << "\n";
    std::cout << "  WAH output:      " << wah_dir << "\n";
    std::cout << "  CRoaring output: " << croaring_dir << "\n";
    std::cout << "  EWAH output:     " << ewah_dir << "\n";
    std::cout << "=========================================\n";

    return 0;
}
