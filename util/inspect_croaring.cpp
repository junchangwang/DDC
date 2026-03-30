// inspect_croaring.cpp — Print CRoaring container type breakdown
//
// Usage: inspect_croaring -n <rows> -c <cardinality>
//
// Generates uniform random data, builds CRoaring bitmaps (one per bucket),
// calls runOptimize(), then prints container statistics for each bitmap.

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <random>
#include "croaring/roaring.hh"
#include "croaring/roaring.h"

int main(int argc, char** argv) {
    uint64_t n = 100000000;
    int c = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n = std::stoull(argv[++i]);
        if (arg == "-c" && i + 1 < argc) c = std::stoi(argv[++i]);
    }

    std::cout << "=== CRoaring Container Inspection ===\n";
    std::cout << "Rows: " << n << " | Cardinality: " << c << "\n\n";

    // Generate random data
    std::cout << "Generating random dataset...\n";
    std::vector<int32_t> data(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> dist(1, c);
    for (uint64_t i = 0; i < n; ++i) data[i] = dist(rng);

    // Build buckets
    std::vector<std::vector<uint32_t>> buckets(c + 1);
    for (uint64_t i = 0; i < n; ++i) {
        int32_t v = data[i];
        if (v >= 1 && v <= c) buckets[v].push_back(static_cast<uint32_t>(i));
    }

    // Summary totals
    uint32_t total_array = 0, total_bitset = 0, total_run = 0;

    for (int v = 1; v <= c; ++v) {
        roaring::Roaring bitmap;
        for (uint32_t idx : buckets[v]) bitmap.add(idx);
        bitmap.runOptimize();

        // Get statistics via public C API
        roaring_statistics_t stats;
        roaring_bitmap_statistics(&bitmap.roaring, &stats);

        std::cout << "Bitmap " << v << ": card=" << stats.cardinality
                  << " | containers=" << stats.n_containers
                  << " [array=" << stats.n_array_containers
                  << " bitset=" << stats.n_bitset_containers
                  << " run=" << stats.n_run_containers << "]"
                  << " | values: arr=" << stats.n_values_array_containers
                  << " bs=" << stats.n_values_bitset_containers
                  << " run=" << stats.n_values_run_containers
                  << "\n";

        total_array  += stats.n_array_containers;
        total_bitset += stats.n_bitset_containers;
        total_run    += stats.n_run_containers;
    }

    std::cout << "\n=== TOTALS across all " << c << " bitmaps ===\n";
    std::cout << "Array containers:  " << total_array << "\n";
    std::cout << "Bitset containers: " << total_bitset << "\n";
    std::cout << "Run containers:    " << total_run << "\n";

    // Theoretical analysis
    uint64_t containers_per_bitmap = (n + 65535) / 65536;
    uint64_t avg_card_per_bitmap = n / c;
    uint64_t avg_per_container = avg_card_per_bitmap / containers_per_bitmap;
    std::cout << "\n=== Theoretical ===\n";
    std::cout << "Containers per bitmap: " << containers_per_bitmap << "\n";
    std::cout << "Avg set bits per bitmap: " << avg_card_per_bitmap << "\n";
    std::cout << "Avg set bits per container: " << avg_per_container << "\n";
    std::cout << "Threshold (array<->bitset): 4096\n";
    std::cout << "Expected type: " << (avg_per_container < 4096 ? "ARRAY" : "BITSET") << "\n";

    return 0;
}
