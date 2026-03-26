#ifndef BENCHMARK_UTI_H
#define BENCHMARK_UTI_H

#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <map>
#include <memory>
#include <cstdint>
#include "bitmap_backend.h"

// ==========================================
// CSV output helper
// ==========================================
extern std::ofstream* g_csv;

void csv_write_header();
void csv_row(const std::string& backend, size_t rows, size_t card,
             const std::string& op, double time_ms,
             long compressed_bytes, double ratio,
             uint64_t result_card, int iteration);

double compute_median(std::vector<double>& v);

// ==========================================
// High-Resolution Timer
// ==========================================
class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() { reset(); }
    void reset() { start_time = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// ==========================================
// File / bitmap utilities
// ==========================================
long get_file_size(const std::string& filename);

std::vector<bool> read_raw_bm(const std::string& path, size_t num_rows);

std::unique_ptr<BitmapHandle> bits_to_bitmap(
    IBitmapBackend* backend, const std::vector<bool>& bits);

std::map<std::string, std::string> read_metadata(const std::string& path);

// ==========================================
// Compressed directory name parser
// ==========================================
struct CompressedDirInfo {
    size_t rows = 0;
    size_t cardinality = 0;
    std::string algo;
};

CompressedDirInfo parse_compressed_dir_name(const std::string& dir_path);
std::string algo_to_backend_key(const std::string& algo);

// ==========================================
// Print usage
// ==========================================
void print_usage(const char* prog);

#endif // BENCHMARK_UTI_H
