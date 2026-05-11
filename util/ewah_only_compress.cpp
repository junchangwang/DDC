// Quick EWAH-only compressor: reads raw .bm → EWAH, skipping already-done files
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdint>
#include <algorithm>

#include "ewah/ewah.h"

namespace fs = std::filesystem;

static void compress_one(const std::string& raw_path, const std::string& out_path, size_t num_rows) {
    if (fs::exists(out_path)) return;  // skip if already done

    std::ifstream in(raw_path, std::ios::binary);
    if (!in) { std::cerr << "Error: " << raw_path << "\n"; return; }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    ewah::EWAHBoolArray<uint64_t> btv;
    for (size_t i = 0; i < num_rows; i++) {
        bool bit = (bytes[i / 8] >> (i % 8)) & 1;
        if (bit) btv.set(i);
    }

    fs::create_directories(fs::path(out_path).parent_path());
    std::ofstream out(out_path, std::ios::binary);
    uint64_t current_bits = num_rows;
    out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
    btv.write(out);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: ewah_only_compress <raw_dir> <ewah_dir> <num_rows>\n";
        return 1;
    }
    std::string raw_dir = argv[1];
    std::string ewah_dir = argv[2];
    size_t num_rows = std::stoull(argv[3]);

    // Discount
    for (int v = 0; v <= 10; v++) {
        std::string rel = "discount/" + std::to_string(v) + ".bm";
        compress_one(raw_dir + "/" + rel, ewah_dir + "/" + rel, num_rows);
    }
    // Quantity
    for (int v = 1; v <= 50; v++) {
        std::string rel = "quantity/" + std::to_string(v) + ".bm";
        compress_one(raw_dir + "/" + rel, ewah_dir + "/" + rel, num_rows);
    }
    // Shipdate
    std::vector<int> ids;
    for (auto& e : fs::directory_iterator(raw_dir + "/shipdate")) {
        if (e.path().extension() == ".bm") {
            try { ids.push_back(std::stoi(e.path().stem().string())); }
            catch (...) {}
        }
    }
    std::sort(ids.begin(), ids.end());
    int done = 0;
    for (int id : ids) {
        std::string rel = "shipdate/" + std::to_string(id) + ".bm";
        compress_one(raw_dir + "/" + rel, ewah_dir + "/" + rel, num_rows);
        done++;
        if (done % 200 == 0) std::cout << "  shipdate: " << done << "/" << ids.size() << "\n";
    }
    std::cout << "Done! " << (11 + 50 + ids.size()) << " bitmaps processed.\n";
    return 0;
}
