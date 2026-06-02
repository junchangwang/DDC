
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "ddc.h"

namespace fs = std::filesystem;

static std::vector<uint8_t> read_raw_bm(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    auto sz = in.tellg();
    in.seekg(0);
    std::vector<uint8_t> buf(sz);
    in.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static void compress_one(const std::string& raw_path, const std::string& out_path, size_t total_rows) {
    auto raw = read_raw_bm(raw_path);
    if (raw.empty()) {
        std::cerr << "  SKIP (empty): " << raw_path << "\n";
        return;
    }

    std::vector<bool> bits(total_rows, false);
    for (size_t i = 0; i < total_rows; i++) {
        if ((raw[i / 8] >> (i % 8)) & 1)
            bits[i] = true;
    }
    auto cb = DDC::compress(bits);

    fs::create_directories(fs::path(out_path).parent_path());
    std::ofstream out(out_path, std::ios::binary);
    cb.serialize(out);
}

int main(int argc, char** argv) {
    std::string input_dir = "bitmap/tpch_q6";
    std::string output_dir = "bitmap/tpch_q6_ddc";
    size_t total_rows = 59986052;

    if (argc >= 2) input_dir = argv[1];
    if (argc >= 3) output_dir = argv[2];
    if (argc >= 4) total_rows = std::stoull(argv[3]);

    std::cout << "DDC-only compressor (L3 format)\n";
    std::cout << "  Input:  " << input_dir << "\n";
    std::cout << "  Output: " << output_dir << "\n";
    std::cout << "  Rows:   " << total_rows << "\n\n";

    size_t count = 0;
    for (auto& col_entry : fs::directory_iterator(input_dir)) {
        if (!col_entry.is_directory()) continue;
        std::string col = col_entry.path().filename().string();
        std::cout << "[" << col << "] ";

        for (auto& f : fs::directory_iterator(col_entry.path())) {
            std::string fname = f.path().filename().string();
            if (fname.size() < 4 || fname.substr(fname.size()-3) != ".bm") continue;

            std::string raw_path = f.path().string();
            std::string out_path = output_dir + "/" + col + "/" + fname;
            compress_one(raw_path, out_path, total_rows);
            count++;
        }
        std::cout << "done\n";
    }

    std::ofstream(output_dir + "/done.txt") << "L3 format, " << count << " bitmaps\n";
    std::cout << "\nTotal: " << count << " bitmaps compressed.\n";

    return 0;
}
