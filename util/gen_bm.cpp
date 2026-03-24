#include "gen_bm.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
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
#include <bitmap_vector.hpp>

namespace fs = std::filesystem;

// ==========================================
// Shared: read dataset into buckets
// ==========================================

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

// ==========================================
// gen_raw: uncompressed little-endian packed bits
// ==========================================

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

// ==========================================
// gen_wah: FastBit WAH compression
// ==========================================

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

// ==========================================
// gen_roaring: CRoaring compression
// ==========================================

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

// ==========================================
// gen_ewah: EWAH compression
// ==========================================

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

// ==========================================
// gen_concise: Concise compression
// ==========================================

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

// ==========================================
// gen_combit: ComBit compression
// ==========================================

bool gen_combit(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        combit::BitmapVector btv;
        for (uint32_t idx : buckets[v]) {
            btv.set_bit(idx);
        }

        // Serialize: [current_bits (uint64_t)] [num_indices (size_t)] [indices...]
        uint64_t current_bits = rows;
        std::vector<uint32_t> indices = btv.decode_positions();
        // Filter to valid range
        while (!indices.empty() && indices.back() >= current_bits) {
            indices.pop_back();
        }
        size_t num_indices = indices.size();

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
        out.write(reinterpret_cast<const char*>(&num_indices), sizeof(num_indices));
        if (num_indices > 0) {
            out.write(reinterpret_cast<const char*>(indices.data()), num_indices * sizeof(uint32_t));
        }

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_combit] All " << cardinality << " ComBit bitmaps written to " << output_dir << "\n";
    return true;
}