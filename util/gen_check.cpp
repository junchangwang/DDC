#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

// ---- Path helpers ----

std::string dataset_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/dataset_" + std::to_string(n) + "_" + std::to_string(c);
}

std::string index_file_path(const std::string& base_dir) {
    return base_dir + "/index.txt";
}

std::string compressed_dir_path(const std::string& base_dir, int n, int c, const std::string& algorithm) {
    return base_dir + "/compressed_n" + std::to_string(n) + "_c" + std::to_string(c) + "_" + algorithm;
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
    std::string cmd = base_dir + "/gen_dataset.sh -n " + std::to_string(n)
                    + " -c " + std::to_string(c)
                    + " -d " + base_dir;
    std::cout << "[gen_check] Calling: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

// ---- Call gen() (to be implemented by teammate) ----

bool call_gen(int n, int c, const std::string& algorithm, const std::string& compressed_dir) {
    // TODO: replace with actual call to gen() once interface is confirmed
    std::cout << "[gen_check] gen() not yet implemented." << std::endl;
    std::cout << "[gen_check] Would compress dataset_" << n << "_" << c
              << " using " << algorithm
              << " into " << compressed_dir << std::endl;
    return false;
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

    // Step 1: Check if dataset exists in index.txt
    if (!dataset_exists_in_index(base_dir, n, c)) {
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
        if (!call_gen(n, c, algorithm, comp_dir)) {
            std::cerr << "[gen_check] Error: gen() failed or not yet implemented." << std::endl;
            return 1;
        }
        std::cout << "[gen_check] Compression completed." << std::endl;
    } else {
        std::cout << "[gen_check] Compressed data already exists: " << comp_dir << std::endl;
    }

    return 0;
}