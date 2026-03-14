#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <random>
#include <filesystem>
#include <algorithm>
#include <map>
#include "backends/combit/combit_backend.h"
#include "bitmap_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"

namespace fs = std::filesystem;

// ==========================================
// 1. High-Resolution Timer
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
// 3. Read raw .bm file (little-endian packed bits)
// ==========================================
static std::vector<bool> read_raw_bm(const std::string& path, size_t num_rows) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) {
        std::cerr << "Error: cannot open " << path << "\n";
        return {};
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                std::istreambuf_iterator<char>());
    std::vector<bool> bits;
    bits.reserve(num_rows);
    for (size_t i = 0; i < bytes.size() && bits.size() < num_rows; ++i) {
        for (int bit = 0; bit < 8 && bits.size() < num_rows; ++bit) {
            bits.push_back((bytes[i] >> bit) & 1);
        }
    }
    return bits;
}

/// Build a backend bitmap from a bit vector.
static std::unique_ptr<BitmapHandle> bits_to_bitmap(
    IBitmapBackend* backend, const std::vector<bool>& bits)
{
    auto h = backend->Create();
    for (bool b : bits) backend->Append(*h, b);
    return h;
}

/// Parse metadata.txt key=value pairs.
static std::map<std::string, std::string> read_metadata(const std::string& path) {
    std::map<std::string, std::string> m;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq != std::string::npos)
            m[line.substr(0, eq)] = line.substr(eq + 1);
    }
    return m;
}

// ==========================================
// 4. Core Benchmark: Random Data
// ==========================================
void run_performance_benchmark(IBitmapBackend* backend, const std::string& backend_name, size_t num_bits) {
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — Random Data\n";
    std::cout << "Dataset Size: " << num_bits << " bits\n";
    std::cout << "=======================================\n";

    Timer timer;

    auto btv1 = backend->Create();
    auto btv2 = backend->Create();
    
    std::mt19937 gen(42); 
    std::bernoulli_distribution d(0.1); 

    timer.reset();
    for (size_t i = 0; i < num_bits; ++i) {
        backend->Append(*btv1, d(gen));
        backend->Append(*btv2, d(gen)); 
    }
    double append_time = timer.elapsed_ms();
    std::cout << "[Build] Append " << num_bits << " bits took: " << append_time << " ms\n";

    std::string tmp_file = backend_name + "_compressed.bin";
    timer.reset();
    backend->Serialize(*btv1, tmp_file);
    double serialize_time = timer.elapsed_ms();
    long file_size = get_file_size(tmp_file);
    double raw_size = num_bits / 8.0;
    std::cout << "[Storage] Serialization took: " << serialize_time << " ms\n";
    std::cout << "[Storage] Compressed Size: " << file_size << " bytes (" 
              << (double)file_size / 1024.0 << " KB)"
              << " | Ratio: " << (raw_size > 0 ? file_size / raw_size : 0) << "x\n";

    timer.reset();
    auto loaded = backend->Load(tmp_file);
    double load_time = timer.elapsed_ms();
    std::cout << "[Storage] Load took: " << load_time << " ms (Cardinality: "
              << backend->Cardinality(*loaded) << ")\n";

    timer.reset();
    auto or_res = backend->bitOr(*btv1, *btv2);
    double or_time = timer.elapsed_ms();
    std::cout << "[Compute] bitOr  took: " << or_time << " ms (Result Cardinality: " << backend->Cardinality(*or_res) << ")\n";

    timer.reset();
    auto and_res = backend->bitAnd(*btv1, *btv2);
    double and_time = timer.elapsed_ms();
    std::cout << "[Compute] bitAnd took: " << and_time << " ms (Result Cardinality: " << backend->Cardinality(*and_res) << ")\n";

    timer.reset();
    auto xor_res = backend->bitXor(*btv1, *btv2);
    double xor_time = timer.elapsed_ms();
    std::cout << "[Compute] bitXor took: " << xor_time << " ms (Result Cardinality: " << backend->Cardinality(*xor_res) << ")\n";

    std::cout << "---------------------------------------\n";
}

// ==========================================
// 5. BM File Benchmark: Load real .bm files
// ==========================================
void run_bm_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                      const std::string& bm_dir, size_t num_rows)
{
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — .bm Files\n";
    std::cout << "Source: " << bm_dir << " (" << num_rows << " rows)\n";
    std::cout << "=======================================\n";

    Timer timer;

    // Collect all .bm files in bm_dir (non-recursive first level)
    std::vector<std::string> bm_files;
    for (auto& entry : fs::directory_iterator(bm_dir)) {
        if (entry.path().extension() == ".bm")
            bm_files.push_back(entry.path().string());
    }
    // Also add column/*.bm if exists
    std::string col_dir = bm_dir + "/column";
    if (fs::is_directory(col_dir)) {
        for (auto& entry : fs::directory_iterator(col_dir)) {
            if (entry.path().extension() == ".bm")
                bm_files.push_back(entry.path().string());
        }
    }
    std::sort(bm_files.begin(), bm_files.end());

    if (bm_files.empty()) {
        std::cerr << "  No .bm files found in " << bm_dir << "\n";
        return;
    }
    std::cout << "  Found " << bm_files.size() << " .bm files\n\n";

    // --- Phase 1: Load (read raw .bm → Append bits) ---
    std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
    timer.reset();
    for (auto& path : bm_files) {
        auto bits = read_raw_bm(path, num_rows);
        bitmaps.push_back(bits_to_bitmap(backend, bits));
    }
    double load_time = timer.elapsed_ms();
    std::cout << "[Load] Read & build " << bm_files.size()
              << " bitmaps: " << load_time << " ms\n";

    // Print per-file cardinality
    for (size_t i = 0; i < bm_files.size(); ++i) {
        std::string name = fs::path(bm_files[i]).filename().string();
        if (bm_files[i].find("/column/") != std::string::npos)
            name = "column/" + name;
        std::cout << "  " << name << " → cardinality: "
                  << backend->Cardinality(*bitmaps[i]) << "\n";
    }

    // --- Phase 2: Serialize & compression ratio ---
    std::cout << "\n";
    double total_ser = 0;
    long total_compressed = 0;
    for (size_t i = 0; i < bm_files.size(); ++i) {
        std::string out_path = backend_name + "_ser_" + std::to_string(i) + ".bin";
        timer.reset();
        backend->Serialize(*bitmaps[i], out_path);
        total_ser += timer.elapsed_ms();
        total_compressed += get_file_size(out_path);
        std::remove(out_path.c_str());
    }
    double raw_total = (double)bm_files.size() * num_rows / 8.0;
    std::cout << "[Serialize] " << bm_files.size() << " bitmaps: "
              << total_ser << " ms | Total compressed: " << total_compressed
              << " bytes | Ratio: " << (raw_total > 0 ? total_compressed / raw_total : 0)
              << "x\n";

    // --- Phase 3: Logical ops (pairwise on first 2 bitmaps) ---
    if (bitmaps.size() >= 2) {
        auto& a = bitmaps[0];
        auto& b = bitmaps[1];
        std::string na = fs::path(bm_files[0]).filename().string();
        std::string nb = fs::path(bm_files[1]).filename().string();

        timer.reset();
        auto or_res = backend->bitOr(*a, *b);
        double or_t = timer.elapsed_ms();

        timer.reset();
        auto and_res = backend->bitAnd(*a, *b);
        double and_t = timer.elapsed_ms();

        timer.reset();
        auto xor_res = backend->bitXor(*a, *b);
        double xor_t = timer.elapsed_ms();

        std::cout << "\n[Compute] Pairwise ops on " << na << " & " << nb << ":\n";
        std::cout << "  bitOr:  " << or_t  << " ms (card: " << backend->Cardinality(*or_res)  << ")\n";
        std::cout << "  bitAnd: " << and_t << " ms (card: " << backend->Cardinality(*and_res) << ")\n";
        std::cout << "  bitXor: " << xor_t << " ms (card: " << backend->Cardinality(*xor_res) << ")\n";
    }

    // --- Phase 4: Multi-way OR over column bitmaps ---
    std::vector<size_t> col_indices;
    for (size_t i = 0; i < bm_files.size(); ++i) {
        if (bm_files[i].find("/column/") != std::string::npos)
            col_indices.push_back(i);
    }
    if (col_indices.size() >= 2) {
        timer.reset();
        auto acc = backend->bitOr(*bitmaps[col_indices[0]], *bitmaps[col_indices[1]]);
        for (size_t k = 2; k < col_indices.size(); ++k) {
            acc = backend->bitOr(*acc, *bitmaps[col_indices[k]]);
        }
        double multi_t = timer.elapsed_ms();
        std::cout << "\n[Compute] Multi-way OR of " << col_indices.size()
                  << " column bitmaps: " << multi_t << " ms (card: "
                  << backend->Cardinality(*acc) << ")\n";
    }

    std::cout << "---------------------------------------\n";
}

// ==========================================
// 6. Print usage
// ==========================================
static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --backend <wah|croaring|combit|all>  Backend to benchmark (default: all)\n"
              << "  --size <N>                           Number of random bits (default: 10000000)\n"
              << "  --bm-dir <path>                      Directory with .bm files (enables file mode)\n"
              << "  --num-rows <N>                       Number of rows in .bm files (default: from metadata.txt)\n"
              << "  --help                               Show this help\n\n"
              << "Examples:\n"
              << "  " << prog << "                                    # Random data, all backends\n"
              << "  " << prog << " --backend combit --size 1000000    # Random, ComBit only\n"
              << "  " << prog << " --bm-dir ../test_bitmaps           # .bm files, all backends\n"
              << "  " << prog << " --bm-dir /path/to/bitmaps --backend wah\n";
}

// ==========================================
// Main function: CLI argument parsing
// ==========================================
int main(int argc, char** argv) {
    std::string backend_type = "all";
    size_t test_size = 10000000;
    std::string bm_dir;
    size_t num_rows = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend_type = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            test_size = std::stoull(argv[++i]);
        } else if (arg == "--bm-dir" && i + 1 < argc) {
            bm_dir = argv[++i];
        } else if (arg == "--num-rows" && i + 1 < argc) {
            num_rows = std::stoull(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    WahBackend wah;
    CroaringBackend croaring;
    CombitBackend combit;

    // Build list of (backend_ptr, name) to run
    struct BackendEntry { IBitmapBackend* ptr; std::string name; std::string key; };
    std::vector<BackendEntry> backends = {
        {&wah,      "WAH (FastBit)", "wah"},
        {&croaring, "CRoaring",      "croaring"},
        {&combit,   "ComBIT (New)",   "combit"},
    };

    if (!bm_dir.empty()) {
        // === .bm file mode ===
        if (!fs::is_directory(bm_dir)) {
            std::cerr << "Error: " << bm_dir << " is not a directory\n";
            return 1;
        }
        // Auto-read num_rows from metadata.txt if not specified
        if (num_rows == 0) {
            auto meta = read_metadata(bm_dir + "/metadata.txt");
            if (meta.count("num_rows"))
                num_rows = std::stoull(meta["num_rows"]);
        }
        if (num_rows == 0) {
            std::cerr << "Error: cannot determine num_rows. Use --num-rows or provide metadata.txt\n";
            return 1;
        }

        std::cout << "=== .bm File Benchmark Mode ===\n";
        std::cout << "Directory: " << bm_dir << " | Rows: " << num_rows << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all")
                run_bm_benchmark(be.ptr, be.name, bm_dir, num_rows);
        }
    } else {
        // === Random data mode (original behavior) ===
        std::cout << "=== Random Data Benchmark Mode ===\n";
        std::cout << "Size: " << test_size << " bits\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all")
                run_performance_benchmark(be.ptr, be.name, test_size);
        }
    }

    return 0;
}