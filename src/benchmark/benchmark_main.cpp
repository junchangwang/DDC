#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <random>

#include "bitmap_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"
#include "backends/combit/combit_backend.h"

// ==========================================
// 1. High-Resolution Timer for Benchmarking
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
// 2. Utility to measure physical file size
// ==========================================
long get_file_size(const std::string& filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}

// ==========================================
// 3. Core Benchmark Pipeline
// ==========================================
void run_performance_benchmark(IBitmapBackend* backend, const std::string& backend_name, size_t num_bits) {
    std::cout << "\n=======================================\n";
    std::cout << "Running Benchmark for Backend: [" << backend_name << "]\n";
    std::cout << "Dataset Size: " << num_bits << " bits\n";
    std::cout << "=======================================\n";

    Timer timer;

    // 1. Test: Create & Append (Build Performance)
    auto btv1 = backend->Create();
    auto btv2 = backend->Create();
    
    // Simulate data ingestion using a pseudo-random number generator.
    // This isolates CPU algorithm performance from slow Disk I/O.
    std::mt19937 gen(42); // Fixed seed for reproducible and fair comparison
    std::bernoulli_distribution d(0.1); // Simulate a sparse bitmap (10% ones)

    timer.reset();
    for (size_t i = 0; i < num_bits; ++i) {
        backend->Append(*btv1, d(gen));
        backend->Append(*btv2, d(gen)); // Generate second bitmap for logical ops
    }
    double append_time = timer.elapsed_ms();
    std::cout << "[Build] Append " << num_bits << " bits took: " << append_time << " ms\n";

    // 2. Test: Serialization & Compression Ratio
    std::string tmp_file = backend_name + "_compressed.bin";
    timer.reset();
    backend->Serialize(*btv1, tmp_file);
    double serialize_time = timer.elapsed_ms();
    long file_size = get_file_size(tmp_file);
    std::cout << "[Storage] Serialization took: " << serialize_time << " ms\n";
    std::cout << "[Storage] Compressed File Size: " << file_size << " bytes (" 
              << (double)file_size / 1024.0 << " KB)\n";

    // 3. Test: Logical OR Performance
    timer.reset();
    auto or_res = backend->bitOr(*btv1, *btv2);
    double or_time = timer.elapsed_ms();
    std::cout << "[Compute] bitOr  took: " << or_time << " ms (Result Cardinality: " << backend->Cardinality(*or_res) << ")\n";

    // 4. Test: Logical AND Performance
    timer.reset();
    auto and_res = backend->bitAnd(*btv1, *btv2);
    double and_time = timer.elapsed_ms();
    std::cout << "[Compute] bitAnd took: " << and_time << " ms (Result Cardinality: " << backend->Cardinality(*and_res) << ")\n";

    // 5. Test: Logical XOR Performance
    timer.reset();
    auto xor_res = backend->bitXor(*btv1, *btv2);
    double xor_time = timer.elapsed_ms();
    std::cout << "[Compute] bitXor took: " << xor_time << " ms (Result Cardinality: " << backend->Cardinality(*xor_res) << ")\n";

    std::cout << "---------------------------------------\n";
}

// ==========================================
// Main function: CLI argument parsing and routing
// ==========================================
int main(int argc, char** argv) {
    std::string backend_type = "all";
    size_t test_size = 10000000; // Default: 10 Million bits

    // Simple Command Line Interface (CLI) parser
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend_type = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            test_size = std::stoull(argv[++i]);
        }
    }

    std::cout << "Initialization Complete. Starting CUBIT Benchmark...\n";

    WahBackend wah;
    CroaringBackend croaring;
    CombitBackend combit;

    if (backend_type == "wah" || backend_type == "all") {
        run_performance_benchmark(&wah, "WAH (FastBit)", test_size);
    }
    if (backend_type == "croaring" || backend_type == "all") {
        run_performance_benchmark(&croaring, "CRoaring", test_size);
    }
    if (backend_type == "combit" || backend_type == "all") {
        run_performance_benchmark(&combit, "ComBIT (New)", test_size);
    }

    return 0;
}