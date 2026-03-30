// gen_bitmap.cpp — Unified bitmap generation tool
//
// Reads a binary dataset and generates compressed bitmaps for any
// supported algorithm (WAH, Roaring, EWAH, Concise, ComBit).
// Also generates raw uncompressed bitmaps as a baseline.
//
// Usage:
//   gen_bitmap -n <rows> -c <cardinality> <algorithm> [-d <base_dir>] [-z <zip_length>]
//
// Supported algorithms: wah, roaring, ewah, concise, combit, bitset

#include "gen_bitmap.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <filesystem>

// FastBit (WAH)
#include "fastbit/bitvector.h"
#include "utils/util.h"

// CRoaring
#include "croaring/roaring.hh"

// EWAH
#include "ewah/ewah.h"

// Concise
#include "Concise/concise.h"

// ComBit
#include <combit.h>

namespace fs = std::filesystem;

// ==================================================================
//  Dataset I/O
// ==================================================================

std::vector<std::vector<uint32_t>> read_dataset_buckets(
    const std::string& dataset_path, uint64_t rows, int cardinality)
{
    std::ifstream fin(dataset_path, std::ios::binary);
    if (!fin) {
        std::cerr << "[read_dataset] Error: cannot open " << dataset_path << "\n";
        return {};
    }
    std::vector<int32_t> data(rows);
    fin.read(reinterpret_cast<char*>(data.data()), rows * sizeof(int32_t));
    if (!fin) {
        std::cerr << "[read_dataset] Error: failed to read " << rows << " int32\n";
        return {};
    }
    fin.close();

    std::vector<std::vector<uint32_t>> buckets(cardinality + 1);
    for (uint64_t i = 0; i < rows; i++) {
        int32_t v = data[i];
        if (v >= 1 && v <= cardinality) {
            buckets[v].push_back(static_cast<uint32_t>(i));
        }
    }
    return buckets;
}

// ==================================================================
//  gen_raw: uncompressed little-endian packed bits
// ==================================================================

bool gen_raw(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    size_t packed_bytes = (rows + 7) / 8;

    for (int v = 1; v <= cardinality; v++) {
        std::vector<uint8_t> bitmap(packed_bytes, 0);
        for (uint32_t idx : buckets[v]) {
            bitmap[idx / 8] |= (1u << (idx % 8));
        }

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bitmap.data()), packed_bytes);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_raw] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_raw] All " << cardinality << " raw bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_bitset: packed bits, z bitmaps per file
// ==================================================================

bool gen_bitset(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality, int z)
{
    fs::create_directories(output_dir);

    size_t packed_bytes = (rows + 7) / 8;
    int num_files = (cardinality + z - 1) / z;

    for (int i = 0; i < num_files; i++) {
        int start_val = i * z + 1;
        int end_val   = std::min((i + 1) * z, cardinality);

        std::string out_path = (z == 1)
            ? output_dir + "/" + std::to_string(start_val) + ".bm"
            : output_dir + "/" + std::to_string(i) + ".bmz";

        std::ofstream out(out_path, std::ios::binary);
        for (int v = start_val; v <= end_val; v++) {
            std::vector<uint8_t> bitmap(packed_bytes, 0);
            for (uint32_t idx : buckets[v]) {
                bitmap[idx / 8] |= (1u << (idx % 8));
            }
            out.write(reinterpret_cast<const char*>(bitmap.data()), packed_bytes);
        }

        if (i % 100 == 0 || i == num_files - 1) {
            std::cout << "[gen_bitset] Written file " << (i + 1) << "/" << num_files
                      << " (values " << start_val << "-" << end_val << ")\n";
        }
    }
    std::cout << "[gen_bitset] All " << cardinality << " bitmaps written (z=" << z
              << ") to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_wah: FastBit WAH compression
// ==================================================================

bool gen_wah(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    Table_config config;
    config.enable_fence_pointer = false;

    for (int v = 1; v <= cardinality; v++) {
        ibis::bitvector btv;
        btv.adjustSize(0, static_cast<uint32_t>(rows));
        btv.decompress();

        for (uint32_t idx : buckets[v]) {
            btv.setBit(idx, 1, &config);
        }

        btv.compress();

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        btv.write(out_path.c_str());

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_wah] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_wah] All " << cardinality << " WAH bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_roaring: CRoaring compression
// ==================================================================

bool gen_roaring(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        roaring::Roaring bitmap;
        for (uint32_t idx : buckets[v]) {
            bitmap.add(idx);
        }
        bitmap.runOptimize();

        // Serialize: [current_size (uint32_t)] [roaring binary data]
        uint32_t current_size = static_cast<uint32_t>(rows);
        size_t roaring_size = bitmap.getSizeInBytes();
        std::vector<char> buffer(roaring_size);
        bitmap.write(buffer.data());

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_size), sizeof(current_size));
        out.write(buffer.data(), roaring_size);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_roaring] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_roaring] All " << cardinality << " Roaring bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_ewah: EWAH compression
// ==================================================================

bool gen_ewah(const std::vector<std::vector<uint32_t>>& buckets,
              const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ewah::EWAHBoolArray<uint64_t> btv;
        for (uint32_t idx : buckets[v]) {
            btv.set(idx);
        }

        // Serialize: [current_bits (uint64_t)] [ewah binary]
        uint64_t current_bits = rows;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
        btv.write(out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_ewah] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_ewah] All " << cardinality << " EWAH bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_concise: Concise compression
// ==================================================================

bool gen_concise(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ConciseSet<false> btv;
        for (uint32_t idx : buckets[v]) {
            btv.add(idx);
        }

        // Serialize: [current_bits (uint64_t)] [last] [lastWordIndex] [words data]
        uint64_t current_bits = rows;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
        out.write(reinterpret_cast<const char*>(&btv.last), sizeof(btv.last));
        out.write(reinterpret_cast<const char*>(&btv.lastWordIndex), sizeof(btv.lastWordIndex));
        if (btv.lastWordIndex >= 0) {
            uint32_t count = static_cast<uint32_t>(btv.lastWordIndex + 1);
            out.write(reinterpret_cast<const char*>(btv.words.data()), count * sizeof(uint32_t));
        }

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_concise] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_concise] All " << cardinality << " Concise bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_combit: ComBit compression
// ==================================================================

bool gen_combit(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        // Build vector<bool> from sorted position indices
        const auto& raw = buckets[v];
        std::vector<bool> bits(rows, false);
        for (size_t i = 0; i < raw.size() && raw[i] < rows; i++)
            bits[raw[i]] = true;

        // Compress and serialize
        ComBit cb = ComBit::compress<8>(bits);
        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        cb.serialize(out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_combit] All " << cardinality << " ComBit bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  Path helpers
// ==================================================================

/// Format row count: 1000000000 → "1b", 100000000 → "100m", 1000 → "1k"
static std::string format_rows(int n) {
    if (n >= 1000000000 && n % 1000000000 == 0)
        return std::to_string(n / 1000000000) + "b";
    if (n >= 1000000 && n % 1000000 == 0)
        return std::to_string(n / 1000000) + "m";
    if (n >= 1000 && n % 1000 == 0)
        return std::to_string(n / 1000) + "k";
    return std::to_string(n);
}

static std::string dataset_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/dataset_" + std::to_string(n) + "_" + std::to_string(c);
}

/// Raw uncompressed bitmaps: bitmap/bitmaps_100m_c100/
static std::string raw_dir_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/bitmap/bitmaps_" + format_rows(n)
         + "_c" + std::to_string(c);
}

/// Compressed bitmaps: bitmap/bm_100m_c100_wah/
static std::string compressed_dir_path(const std::string& base_dir, int n, int c,
                                       const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_c" + std::to_string(c) + "_" + algo_lower;
}

static std::string compressed_done_path(const std::string& dir) {
    return dir + "/done.txt";
}

// ==================================================================
//  Dataset & compression checks
// ==================================================================

static bool dataset_exists_in_index(const std::string& base_dir, int n, int c) {
    std::string index_path = base_dir + "/index.txt";
    std::ifstream f(index_path);
    if (!f.is_open()) return false;

    std::string target = "n=" + std::to_string(n) + " c=" + std::to_string(c);
    std::string line;
    while (std::getline(f, line)) {
        if (line.find(target) != std::string::npos) return true;
    }
    return false;
}

static bool compression_exists(const std::string& compressed_dir) {
    return fs::exists(compressed_done_path(compressed_dir));
}

static bool call_gen_dataset(const std::string& base_dir, int n, int c) {
    std::string cmd = base_dir + "/util/gen_dataset.sh -n " + std::to_string(n)
                    + " -c " + std::to_string(c)
                    + " -d " + base_dir;
    std::cout << "[gen_bitmap] Calling: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

/// Dispatch to the appropriate generation function for the given algorithm.
static bool generate_compressed(const std::string& algorithm,
                                const std::vector<std::vector<uint32_t>>& buckets,
                                const std::string& output_dir,
                                uint64_t rows, int cardinality, int z) {
    std::string algo = algorithm;
    for (auto& ch : algo)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (algo == "bitset")  return gen_bitset(buckets, output_dir, rows, cardinality, z);
    if (algo == "wah")     return gen_wah(buckets, output_dir, rows, cardinality);
    if (algo == "roaring") return gen_roaring(buckets, output_dir, rows, cardinality);
    if (algo == "ewah")    return gen_ewah(buckets, output_dir, rows, cardinality);
    if (algo == "concise") return gen_concise(buckets, output_dir, rows, cardinality);
    if (algo == "combit")  return gen_combit(buckets, output_dir, rows, cardinality);

    std::cerr << "[gen_bitmap] Error: unsupported algorithm '" << algorithm << "'\n"
              << "[gen_bitmap] Supported: bitset, wah, roaring, ewah, concise, combit\n";
    return false;
}

// ==================================================================
//  Main entry
// ==================================================================

int main(int argc, char* argv[]) {
    int n = -1;
    int c = -1;
    int z = 1;
    std::string algorithm;
    std::string base_dir = ".";

    // Parse arguments: -n <n> -c <c> <algorithm> [-d <base_dir>] [-z <zip_length>]
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            c = std::stoi(argv[++i]);
        } else if (arg == "-z" && i + 1 < argc) {
            z = std::stoi(argv[++i]);
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
        std::cerr << "Usage: " << argv[0] << " -n <n> -c <c> <algorithm> [-d <base_dir>] [-z <zip_length>]\n"
                  << "  -n <n>        : number of rows\n"
                  << "  -c <c>        : cardinality\n"
                  << "  <algorithm>   : compression algorithm (bitset, wah, roaring, ewah, concise, combit)\n"
                  << "  -d <base_dir> : base directory (default: .)\n"
                  << "  -z <z>        : zip length for bitset mode (default: 1)\n";
        return 1;
    }

    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (z != 1 && algo_lower != "bitset") {
        std::cerr << "Error: -z is only valid for bitset mode" << std::endl;
        return 1;
    }

    // Determine output directory
    std::string comp_dir;
    if (algo_lower == "bitset") {
        if (z == 1)
            comp_dir = compressed_dir_path(base_dir, n, c, "bitset");
        else
            comp_dir = base_dir + "/bitmap/bitmap_n" + format_rows(n)
                     + "_c" + std::to_string(c) + "_z" + std::to_string(z);
    } else {
        comp_dir = compressed_dir_path(base_dir, n, c, algorithm);
    }

    std::cout << "[gen_bitmap] n=" << n << " c=" << c
              << " algorithm=" << algorithm
              << " base_dir=" << base_dir << std::endl;

    // Step 1: Ensure dataset exists
    std::string ds_file = dataset_path(base_dir, n, c);
    if (fs::exists(ds_file)) {
        std::cout << "[gen_bitmap] Dataset file exists: " << ds_file << std::endl;
    } else if (!dataset_exists_in_index(base_dir, n, c)) {
        std::cout << "[gen_bitmap] Dataset not found. Generating..." << std::endl;
        if (!call_gen_dataset(base_dir, n, c)) {
            std::cerr << "[gen_bitmap] Error: gen_dataset.sh failed." << std::endl;
            return 1;
        }
        std::cout << "[gen_bitmap] Dataset generated successfully." << std::endl;
    } else {
        std::cout << "[gen_bitmap] Dataset already exists." << std::endl;
    }

    // Step 2: Check if compression already done
    if (compression_exists(comp_dir)) {
        std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
        return 0;
    }

    std::cout << "[gen_bitmap] Generating compressed bitmaps..." << std::endl;
    fs::create_directories(comp_dir);

    // Read dataset into buckets
    if (!fs::exists(ds_file)) {
        std::cerr << "[gen_bitmap] Error: dataset not found: " << ds_file << std::endl;
        return 1;
    }

    std::cout << "[gen_bitmap] Reading dataset into buckets..." << std::endl;
    auto buckets = read_dataset_buckets(ds_file, static_cast<uint64_t>(n), c);
    if (buckets.empty()) return 1;

    // Generate raw bitmaps if not already present
    std::string raw_dir = raw_dir_path(base_dir, n, c);
    std::string raw_done = raw_dir + "/done.txt";
    if (!fs::exists(raw_done)) {
        std::cout << "[gen_bitmap] Generating raw bitmaps → " << raw_dir << std::endl;
        if (!gen_raw(buckets, raw_dir, static_cast<uint64_t>(n), c)) return 1;
        std::ofstream(raw_done) << "n=" << n << "\nc=" << c << "\n";
    } else {
        std::cout << "[gen_bitmap] Raw bitmaps already exist: " << raw_dir << std::endl;
    }

    // Generate compressed bitmaps
    if (!generate_compressed(algorithm, buckets, comp_dir, static_cast<uint64_t>(n), c, z)) {
        std::cerr << "[gen_bitmap] Error: compression failed." << std::endl;
        return 1;
    }

    // Write done.txt marker
    std::ofstream done(compressed_done_path(comp_dir));
    done << "algorithm=" << algorithm << "\n"
         << "n=" << n << "\n"
         << "c=" << c << "\n";

    std::cout << "[gen_bitmap] Compression completed: " << comp_dir << std::endl;
    return 0;
}
