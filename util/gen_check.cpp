#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include "gen_bm.h"

namespace fs = std::filesystem;

// ---- Path helpers ----

std::string dataset_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/dataset_" + std::to_string(n) + "_" + std::to_string(c);
}

std::string index_file_path(const std::string& base_dir) {
    return base_dir + "/index.txt";
}

/// Format row count: 100000000 → "100m", 1000000 → "1m", 1000 → "1000"
static std::string format_rows(int n) {
    if (n >= 1000000 && n % 1000000 == 0)
        return std::to_string(n / 1000000) + "m";
    if (n >= 1000 && n % 1000 == 0)
        return std::to_string(n / 1000) + "k";
    return std::to_string(n);
}

/// Convert algorithm name to upper-case suffix: "wah" → "WAH"
static std::string algo_upper(const std::string& algo) {
    std::string r = algo;
    for (auto& ch : r) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return r;
}

/// Raw uncompressed bitmaps: src/core/bitset/bitmaps_100m_c100/
std::string raw_dir_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/src/core/bitset/bitmaps_" + format_rows(n)
         + "_c" + std::to_string(c);
}

/// Compressed bitmaps: bm_100m_c100_wah/ (root level)
std::string compressed_dir_path(const std::string& base_dir, int n, int c, const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bm_" + format_rows(n)
         + "_c" + std::to_string(c) + "_" + algo_lower;
}

std::string compressed_done_path(const std::string& compressed_dir) {
    return compressed_dir + "/done.txt";
}

// ---- Check index.txt for dataset entry ----

bool dataset_exists_in_index(const std::string& base_dir, int n, int c) {
    std::string index_path = index_file_path(base_dir);
    std::ifstream f(index_path);
    if (!f.is_open()) return false;

    std::string target = "n=" + std::to_string(n) + " c=" + std::to_string(c);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find(target) != std::string::npos) return true;
    }
    return false;
}

// ---- Check compressed done.txt ----

bool compression_exists(const std::string& compressed_dir) {
    return fs::exists(compressed_done_path(compressed_dir));
}

// ---- Call gen_dataset.sh ----

bool call_gen_dataset(const std::string& base_dir, int n, int c) {
    std::string cmd = base_dir + "/util/gen_dataset.sh -n " + std::to_string(n)
                    + " -c " + std::to_string(c)
                    + " -d " + base_dir;
    std::cout << "[gen_check] Calling: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

// ---- Call gen() to compress bitmaps ----

bool call_gen(int n, int c, const std::string& algorithm,
              const std::string& compressed_dir, const std::string& base_dir) {
    std::string ds_path = dataset_path(base_dir, n, c);

    if (!fs::exists(ds_path)) {
        std::cerr << "[gen_check] Error: dataset not found: " << ds_path << std::endl;
        return false;
    }

    // Read dataset once — shared across raw + compressed generation
    std::cout << "[gen_check] Reading dataset into buckets..." << std::endl;
    auto buckets = read_dataset_buckets(ds_path, static_cast<uint64_t>(n), c);
    if (buckets.empty()) return false;

    // Step 1.5: Generate raw bitmaps if not already present
    std::string raw_dir = raw_dir_path(base_dir, n, c);
    std::string raw_done = raw_dir + "/done.txt";
    if (!fs::exists(raw_done)) {
        std::cout << "[gen_check] Generating raw bitmaps → " << raw_dir << std::endl;
        if (!gen_raw(buckets, raw_dir, static_cast<uint64_t>(n), c)) return false;
        std::ofstream(raw_done) << "n=" << n << "\nc=" << c << "\n";
    } else {
        std::cout << "[gen_check] Raw bitmaps already exist: " << raw_dir << std::endl;
    }

    // Step 2: Generate compressed bitmaps
    std::string algo = algorithm;
    for (auto& ch : algo) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    bool ok = false;
    if (algo == "wah") {
        ok = gen_wah(buckets, compressed_dir, static_cast<uint64_t>(n), c);
    } else if (algo == "roaring") {
        ok = gen_roaring(buckets, compressed_dir, static_cast<uint64_t>(n), c);
    } else if (algo == "ewah") {
        ok = gen_ewah(buckets, compressed_dir, static_cast<uint64_t>(n), c);
    } else if (algo == "concise") {
        ok = gen_concise(buckets, compressed_dir, static_cast<uint64_t>(n), c);
    } else if (algo == "combit") {
        ok = gen_combit(buckets, compressed_dir, static_cast<uint64_t>(n), c);
    } else {
        std::cerr << "[gen_check] Error: unsupported algorithm '" << algorithm << "'\n"
                  << "[gen_check] Supported: wah, roaring, ewah, concise, combit\n";
        return false;
    }

    if (!ok) return false;

    // Write done.txt marker
    std::string done_path = compressed_done_path(compressed_dir);
    std::ofstream done(done_path);
    done << "algorithm=" << algorithm << "\n"
         << "n=" << n << "\n"
         << "c=" << c << "\n";
    return true;
}

// ---- Main ----

int main(int argc, char* argv[]) {
    int n = -1;
    int c = -1;
    std::string algorithm;
    std::string base_dir = ".";

    // Parse arguments: -n <n> -c <c> <algorithm> [-d <base_dir>]
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            c = std::stoi(argv[++i]);
        } else if (arg == "-d" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg[0] != '-') {
            algorithm = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    if (n <= 0 || c <= 0 || algorithm.empty()) {
        std::cerr << "Usage: " << argv[0] << " -n <n> -c <c> <algorithm> [-d <base_dir>]" << std::endl;
        std::cerr << "  -n <n>        : number of rows" << std::endl;
        std::cerr << "  -c <c>        : cardinality" << std::endl;
        std::cerr << "  <algorithm>   : compression algorithm (e.g. roaring, wah)" << std::endl;
        std::cerr << "  -d <base_dir> : base directory (default: .)" << std::endl;
        return 1;
    }

    std::string comp_dir = compressed_dir_path(base_dir, n, c, algorithm);

    std::cout << "[gen_check] n=" << n << " c=" << c
              << " algorithm=" << algorithm
              << " base_dir=" << base_dir << std::endl;

    // Step 1: Check if dataset exists (on disk or in index.txt)
    std::string ds_file = dataset_path(base_dir, n, c);
    if (fs::exists(ds_file)) {
        std::cout << "[gen_check] Dataset file exists: " << ds_file << std::endl;
    } else if (!dataset_exists_in_index(base_dir, n, c)) {
        std::cout << "[gen_check] Dataset not found in index.txt. Generating..." << std::endl;
        if (!call_gen_dataset(base_dir, n, c)) {
            std::cerr << "[gen_check] Error: gen_dataset.sh failed." << std::endl;
            return 1;
        }
        std::cout << "[gen_check] Dataset generated successfully." << std::endl;
    } else {
        std::cout << "[gen_check] Dataset already exists." << std::endl;
    }

    // Step 2: Check if compression already done
    if (!compression_exists(comp_dir)) {
        std::cout << "[gen_check] Compressed data not found. Calling gen()..." << std::endl;
        fs::create_directories(comp_dir);
        if (!call_gen(n, c, algorithm, comp_dir, base_dir)) {
            std::cerr << "[gen_check] Error: gen() failed or not yet implemented." << std::endl;
            return 1;
        }
        std::cout << "[gen_check] Compression completed." << std::endl;
    } else {
        std::cout << "[gen_check] Compressed data already exists: " << comp_dir << std::endl;
    }

    return 0;
}