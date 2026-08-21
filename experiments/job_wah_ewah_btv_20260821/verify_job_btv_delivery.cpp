#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark/backends/croaring/croaring_backend.h"
#include "benchmark/backends/ewah/ewah_backend.h"
#include "benchmark/backends/wah/wah_backend.h"
#include "benchmark/dense_btv_delivery.h"

namespace fs = std::filesystem;

static std::vector<fs::path> numeric_bitmaps(const fs::path& directory) {
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".bm") continue;
        const std::string stem = entry.path().stem().string();
        if (!stem.empty() && std::all_of(stem.begin(), stem.end(), ::isdigit)) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return std::stoul(lhs.stem().string()) < std::stoul(rhs.stem().string());
    });
    return result;
}

static uint64_t fnv1a64(const roaring::api::bitset_t* bitset,
                        size_t logical_words) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < logical_words; ++i) {
        uint64_t word = i < bitset->arraysize ? bitset->array[i] : 0;
        for (unsigned byte = 0; byte < 8; ++byte) {
            hash ^= static_cast<uint8_t>(word >> (byte * 8));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

static uint64_t dense_cardinality(const roaring::api::bitset_t* bitset,
                                  size_t logical_words) {
    uint64_t result = 0;
    for (size_t i = 0; i < logical_words; ++i) {
        const uint64_t word = i < bitset->arraysize ? bitset->array[i] : 0;
        result += static_cast<uint64_t>(__builtin_popcountll(word));
    }
    return result;
}

static uint64_t word_at(const roaring::api::bitset_t* bitset, size_t index) {
    return index < bitset->arraysize ? bitset->array[index] : 0;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: verify_job_btv_delivery ROWS WAH_DIR EWAH_DIR CR_DIR\n";
        return 2;
    }
    const uint32_t rows = static_cast<uint32_t>(std::stoul(argv[1]));
    const auto wah_files = numeric_bitmaps(argv[2]);
    const auto ewah_files = numeric_bitmaps(argv[3]);
    const auto cr_files = numeric_bitmaps(argv[4]);
    if (wah_files.size() < 2 || ewah_files.size() < 2 || cr_files.size() < 2) {
        throw std::runtime_error("at least two bitmaps are required");
    }

    WahBackend wah_backend;
    EwahBackend ewah_backend;
    CroaringBackend cr_backend;
    auto load = [](IBitmapBackend& backend, const std::vector<fs::path>& files,
                   size_t index) {
        return backend.Load(files[index < files.size() ? index : 0].string());
    };

    auto wah_a_h = load(wah_backend, wah_files, 0);
    auto wah_b_h = load(wah_backend, wah_files, 1);
    auto wah_c_h = load(wah_backend, wah_files, wah_files.size() >= 3 ? 2 : 0);
    auto ewah_a_h = load(ewah_backend, ewah_files, 0);
    auto ewah_b_h = load(ewah_backend, ewah_files, 1);
    auto ewah_c_h = load(ewah_backend, ewah_files, ewah_files.size() >= 3 ? 2 : 0);
    auto cr_a_h = load(cr_backend, cr_files, 0);
    auto cr_b_h = load(cr_backend, cr_files, 1);
    auto cr_c_h = load(cr_backend, cr_files, cr_files.size() >= 3 ? 2 : 0);

    auto& wah_a = static_cast<WahHandle&>(*wah_a_h).btv;
    auto& wah_b = static_cast<WahHandle&>(*wah_b_h).btv;
    auto& wah_c = static_cast<WahHandle&>(*wah_c_h).btv;
    ibis::bitvector wah_t1; wah_t1.copy(wah_a); wah_t1 |= wah_b;
    ibis::bitvector wah_t2; wah_t2.copy(wah_b); wah_t2 |= wah_c;
    ibis::bitvector wah_t3; wah_t3.copy(wah_t1); wah_t3 &= wah_t2;
    ibis::bitvector wah_result; wah_result.copy(wah_t3); wah_result.flip();

    auto& ewah_a = static_cast<EwahHandle&>(*ewah_a_h).btv;
    auto& ewah_b = static_cast<EwahHandle&>(*ewah_b_h).btv;
    auto& ewah_c = static_cast<EwahHandle&>(*ewah_c_h).btv;
    ewah::EWAHBoolArray<uint64_t> ewah_t1, ewah_t2, ewah_t3, ewah_result;
    ewah_a.logicalor(ewah_b, ewah_t1);
    ewah_b.logicalor(ewah_c, ewah_t2);
    ewah_t1.logicaland(ewah_t2, ewah_t3);
    ewah_t3.logicalnot(ewah_result);

    auto& cr_a = static_cast<CroaringHandle&>(*cr_a_h).bitmap;
    auto& cr_b = static_cast<CroaringHandle&>(*cr_b_h).bitmap;
    auto& cr_c = static_cast<CroaringHandle&>(*cr_c_h).bitmap;
    roaring::Roaring cr_t1 = cr_a | cr_b;
    roaring::Roaring cr_t2 = cr_b | cr_c;
    roaring::Roaring cr_result = cr_t1 & cr_t2;
    cr_result.flip(0, rows);

    if (wah_result.size() != rows || ewah_result.sizeInBits() != rows) {
        throw std::runtime_error("native result length mismatch");
    }
    auto* wah_dense = dense_btv_delivery::from_wah(wah_result, rows);
    auto* ewah_dense = dense_btv_delivery::from_ewah(ewah_result, rows);
    auto* cr_dense = roaring::api::bitset_create_with_capacity(rows);
    roaring::api::roaring_bitmap_to_bitset(&cr_result.roaring, cr_dense);

    const size_t words = (static_cast<size_t>(rows) + 63) / 64;
    if (wah_dense->arraysize != words || ewah_dense->arraysize != words) {
        throw std::runtime_error("adapter output length mismatch");
    }
    for (size_t i = 0; i < words; ++i) {
        const uint64_t expected = word_at(cr_dense, i);
        if (wah_dense->array[i] != expected || ewah_dense->array[i] != expected) {
            std::cerr << "word mismatch at " << i << "\n";
            return 1;
        }
    }
    const unsigned tail = rows & 63U;
    if (tail != 0) {
        const uint64_t padding_mask = ~((uint64_t{1} << tail) - 1);
        if ((wah_dense->array[words - 1] & padding_mask) != 0 ||
            (ewah_dense->array[words - 1] & padding_mask) != 0) {
            throw std::runtime_error("nonzero tail padding");
        }
    }

    const uint64_t wah_card = dense_cardinality(wah_dense, words);
    const uint64_t ewah_card = dense_cardinality(ewah_dense, words);
    const uint64_t cr_card = dense_cardinality(cr_dense, words);
    if (wah_card != ewah_card || wah_card != cr_card) {
        throw std::runtime_error("cardinality mismatch");
    }
    const uint64_t hash = fnv1a64(wah_dense, words);
    if (hash != fnv1a64(ewah_dense, words) || hash != fnv1a64(cr_dense, words)) {
        throw std::runtime_error("hash mismatch");
    }

    roaring::api::bitset_free(wah_dense);
    roaring::api::bitset_free(ewah_dense);
    roaring::api::bitset_free(cr_dense);
    std::cout << "PASS rows=" << rows << " words=" << words
              << " cardinality=" << wah_card << " fnv1a64=" << std::hex
              << hash << std::dec << "\n";
    return 0;
}
