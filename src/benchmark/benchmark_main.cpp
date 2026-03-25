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
#include <numeric>
#include <cstdint>
#include "backends/combit/combit_backend.h"
#include "bitmap_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"
#include "backends/ewah/ewah_backend.h"
#include "backends/Concise/concise_backend.h"

namespace fs = std::filesystem;

// ==========================================
// CSV output helper
// ==========================================
static std::ofstream* g_csv = nullptr;

static void csv_write_header() {
    if (!g_csv) return;
    *g_csv << "backend,num_rows,cardinality,operation,time_ms,"
           << "compressed_bytes,compression_ratio,result_cardinality,iteration\n";
}

static void csv_row(const std::string& backend, size_t rows, size_t card,
                    const std::string& op, double time_ms,
                    long compressed_bytes, double ratio,
                    uint64_t result_card, int iteration) {
    if (!g_csv) return;
    *g_csv << backend << "," << rows << "," << card << ","
           << op << "," << time_ms << ","
           << compressed_bytes << "," << ratio << ","
           << result_card << "," << iteration << "\n";
}

static double compute_median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}

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
// 3b. Parse index.txt from bmz directory
// ==========================================
struct BmzIndex {
    size_t rows = 0;
    size_t cardinality = 0;
    size_t files_per_dir = 0;
    size_t num_dirs = 0;
    // bmz_idx -> (start_val, end_val)
    std::map<int, std::pair<int, int>> mapping;
};

static BmzIndex parse_bmz_index(const std::string& index_path) {
    BmzIndex idx;
    std::ifstream f(index_path);
    if (!f.good()) return idx;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = line.substr(0, eq);
            // trim spaces
            while (!key.empty() && key.back() == ' ') key.pop_back();
            std::string val = line.substr(eq + 1);
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());
            if (key == "rows") idx.rows = std::stoull(val);
            else if (key == "cardinality") idx.cardinality = std::stoull(val);
            else if (key == "files_per_dir") idx.files_per_dir = std::stoull(val);
            else if (key == "num_dirs") idx.num_dirs = std::stoull(val);
        } else if (line.find(".bmz:") != std::string::npos) {
            // e.g. "0.bmz: bitmaps 1-100"
            int bmz_idx = std::stoi(line.substr(0, line.find('.')));
            auto dash_pos = line.rfind('-');
            auto space_before_range = line.rfind(' ', dash_pos);
            int start_val = std::stoi(line.substr(space_before_range + 1, dash_pos - space_before_range - 1));
            int end_val = std::stoi(line.substr(dash_pos + 1));
            idx.mapping[bmz_idx] = {start_val, end_val};
        }
    }
    return idx;
}

// ==========================================
// 3c. Read a single bitmap from a .bmz file
// ==========================================
static std::vector<bool> read_bitmap_from_bmz(const std::string& bmz_path,
                                               int val, int start_val,
                                               size_t num_rows)
{
    size_t bitmap_size = (num_rows + 7) / 8;
    size_t offset = static_cast<size_t>(val - start_val) * bitmap_size;

    std::ifstream f(bmz_path, std::ios::binary);
    if (!f.good()) return {};
    f.seekg(static_cast<std::streamoff>(offset));

    std::vector<uint8_t> bytes(bitmap_size);
    f.read(reinterpret_cast<char*>(bytes.data()), bitmap_size);

    std::vector<bool> bits;
    bits.reserve(num_rows);
    for (size_t i = 0; i < bytes.size() && bits.size() < num_rows; ++i) {
        for (int bit = 0; bit < 8 && bits.size() < num_rows; ++bit) {
            bits.push_back((bytes[i] >> bit) & 1);
        }
    }
    return bits;
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
// 6. BMZ File Benchmark (with multi-iteration + CSV)
// ==========================================
void run_bmz_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                       const std::string& bmz_dir, const BmzIndex& idx,
                       size_t sample_count, size_t iterations = 1)
{
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — .bmz Files\n";
    std::cout << "Source: " << bmz_dir << "\n";
    std::cout << "Rows: " << idx.rows << " | Cardinality: " << idx.cardinality;
    if (iterations > 1) std::cout << " | Iterations: " << iterations;
    std::cout << "\n=======================================\n";

    Timer timer;
    size_t num_rows = idx.rows;

    // Determine which bitmaps to load
    std::vector<int> vals_to_load;
    if (sample_count > 0 && sample_count < idx.cardinality) {
        for (size_t i = 0; i < sample_count; ++i) {
            vals_to_load.push_back(static_cast<int>(1 + i * idx.cardinality / sample_count));
        }
    } else {
        for (size_t v = 1; v <= idx.cardinality; ++v) {
            vals_to_load.push_back(static_cast<int>(v));
        }
    }
    std::cout << "Loading " << vals_to_load.size() << " bitmaps...\n\n";

    // --- Pre-read raw bits (shared across iterations) ---
    std::vector<std::vector<bool>> all_raw_bits;
    for (int val : vals_to_load) {
        int bmz_idx = (val - 1) / static_cast<int>(idx.files_per_dir);
        auto it = idx.mapping.find(bmz_idx);
        if (it == idx.mapping.end()) { all_raw_bits.emplace_back(); continue; }
        int start_val = it->second.first;
        std::string bmz_path = bmz_dir + "/" + std::to_string(bmz_idx) + ".bmz";
        auto bits = read_bitmap_from_bmz(bmz_path, val, start_val, num_rows);
        if (bits.empty()) {
            std::cerr << "  Warning: failed to read bitmap for value " << val << "\n";
        }
        all_raw_bits.push_back(std::move(bits));
    }

    // Timing accumulators
    std::vector<double> build_times, ser_times, or_times, and_times, xor_times, multi_or_times;
    long total_compressed = 0;
    double raw_total = 0;
    uint64_t or_card = 0, and_card = 0, xor_card = 0, multi_or_card = 0;

    // --- Warm-up run (not measured) ---
    if (iterations > 1) {
        std::cout << "[Warm-up] Running 1 warm-up iteration...\n";
        std::vector<std::unique_ptr<BitmapHandle>> warmup_bm;
        for (auto& bits : all_raw_bits) {
            if (!bits.empty()) warmup_bm.push_back(bits_to_bitmap(backend, bits));
        }
        if (warmup_bm.size() >= 2) {
            auto w_or = backend->bitOr(*warmup_bm[0], *warmup_bm[1]);
            auto w_and = backend->bitAnd(*warmup_bm[0], *warmup_bm[1]);
            (void)w_or; (void)w_and;
        }
    }

    for (size_t iter = 0; iter < iterations; ++iter) {
        // --- Phase 1: Build bitmaps ---
        std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
        timer.reset();
        for (auto& bits : all_raw_bits) {
            if (!bits.empty()) bitmaps.push_back(bits_to_bitmap(backend, bits));
        }
        double build_t = timer.elapsed_ms();
        build_times.push_back(build_t);

        if (iter == 0) {
            std::cout << "[Load] Read & build " << bitmaps.size()
                      << " bitmaps from .bmz: " << build_t << " ms\n";
            for (size_t i = 0; i < bitmaps.size() && i < vals_to_load.size(); ++i) {
                std::cout << "  value " << vals_to_load[i]
                          << " → cardinality: " << backend->Cardinality(*bitmaps[i]) << "\n";
            }
        }

        if (bitmaps.size() < 2) {
            std::cout << "  Not enough bitmaps to run logical ops.\n";
            return;
        }

        // --- Phase 2: Serialize & compression ratio ---
        double iter_ser = 0;
        long iter_compressed = 0;
        for (size_t i = 0; i < bitmaps.size(); ++i) {
            std::string out_path = backend_name + "_bmz_ser_" + std::to_string(i) + ".bin";
            timer.reset();
            backend->Serialize(*bitmaps[i], out_path);
            iter_ser += timer.elapsed_ms();
            iter_compressed += get_file_size(out_path);
            std::remove(out_path.c_str());
        }
        ser_times.push_back(iter_ser);
        total_compressed = iter_compressed;
        raw_total = static_cast<double>(bitmaps.size()) * num_rows / 8.0;

        // --- Phase 3: Pairwise logical ops ---
        {
            auto& a = bitmaps[0];
            auto& b = bitmaps[1];

            timer.reset();
            auto or_res = backend->bitOr(*a, *b);
            or_times.push_back(timer.elapsed_ms());
            or_card = backend->Cardinality(*or_res);

            timer.reset();
            auto and_res = backend->bitAnd(*a, *b);
            and_times.push_back(timer.elapsed_ms());
            and_card = backend->Cardinality(*and_res);

            timer.reset();
            auto xor_res = backend->bitXor(*a, *b);
            xor_times.push_back(timer.elapsed_ms());
            xor_card = backend->Cardinality(*xor_res);
        }

        // --- Phase 4: Multi-way OR ---
        if (bitmaps.size() >= 3) {
            timer.reset();
            auto acc = backend->bitOr(*bitmaps[0], *bitmaps[1]);
            for (size_t k = 2; k < bitmaps.size(); ++k) {
                acc = backend->bitOr(*acc, *bitmaps[k]);
            }
            multi_or_times.push_back(timer.elapsed_ms());
            multi_or_card = backend->Cardinality(*acc);
        }

        // Write per-iteration CSV rows
        int it = static_cast<int>(iter + 1);
        double ratio = (raw_total > 0) ? iter_compressed / raw_total : 0;
        csv_row(backend_name, num_rows, idx.cardinality, "build", build_t, 0, 0, 0, it);
        csv_row(backend_name, num_rows, idx.cardinality, "serialize", iter_ser, iter_compressed, ratio, 0, it);
        csv_row(backend_name, num_rows, idx.cardinality, "OR", or_times.back(), 0, 0, or_card, it);
        csv_row(backend_name, num_rows, idx.cardinality, "AND", and_times.back(), 0, 0, and_card, it);
        csv_row(backend_name, num_rows, idx.cardinality, "XOR", xor_times.back(), 0, 0, xor_card, it);
        if (!multi_or_times.empty())
            csv_row(backend_name, num_rows, idx.cardinality, "multi-OR", multi_or_times.back(), 0, 0, multi_or_card, it);
    }

    // --- Print summary (median if multiple iterations) ---
    double ratio = (raw_total > 0) ? total_compressed / raw_total : 0;
    if (iterations > 1) {
        std::cout << "\n[Summary] Median over " << iterations << " iterations:\n";
        std::cout << "  Build:     " << compute_median(build_times) << " ms\n";
        std::cout << "  Serialize: " << compute_median(ser_times) << " ms"
                  << " | Compressed: " << total_compressed << " bytes"
                  << " | Ratio: " << ratio << "x\n";
        std::cout << "  bitOr:     " << compute_median(or_times) << " ms (card: " << or_card << ")\n";
        std::cout << "  bitAnd:    " << compute_median(and_times) << " ms (card: " << and_card << ")\n";
        std::cout << "  bitXor:    " << compute_median(xor_times) << " ms (card: " << xor_card << ")\n";
        if (!multi_or_times.empty())
            std::cout << "  multi-OR:  " << compute_median(multi_or_times) << " ms (card: " << multi_or_card << ")\n";
    } else {
        std::cout << "\n[Serialize] Compressed: " << total_compressed
                  << " bytes (" << total_compressed / 1024.0 << " KB)"
                  << " | Ratio: " << ratio << "x\n";
        std::cout << "\n[Compute] Pairwise ops on value " << vals_to_load[0]
                  << " & value " << vals_to_load[1] << ":\n";
        std::cout << "  bitOr:  " << or_times[0]  << " ms (card: " << or_card << ")\n";
        std::cout << "  bitAnd: " << and_times[0] << " ms (card: " << and_card << ")\n";
        std::cout << "  bitXor: " << xor_times[0] << " ms (card: " << xor_card << ")\n";
        if (!multi_or_times.empty())
            std::cout << "\n[Compute] Multi-way OR of " << all_raw_bits.size()
                      << " bitmaps: " << multi_or_times[0] << " ms (card: " << multi_or_card << ")\n";
    }

    std::cout << "---------------------------------------\n";
}

// ==========================================
// 6b. Pre-compressed .bm Benchmark
// ==========================================

/// Parse directory name: bm_{rows}_c{card}_{algo}
struct CompressedDirInfo {
    size_t rows = 0;
    size_t cardinality = 0;
    std::string algo;
};

static CompressedDirInfo parse_compressed_dir_name(const std::string& dir_path) {
    CompressedDirInfo info;
    std::string name = fs::path(dir_path).filename().string();
    // Expected: bm_{rows}_c{card}_{algo}
    if (name.substr(0, 3) != "bm_") return info;

    size_t pos = 3;
    size_t u1 = name.find('_', pos);
    if (u1 == std::string::npos) return info;
    std::string rows_str = name.substr(pos, u1 - pos);

    if (!rows_str.empty() && (rows_str.back() == 'm' || rows_str.back() == 'M'))
        info.rows = std::stoull(rows_str.substr(0, rows_str.size()-1)) * 1000000;
    else if (!rows_str.empty() && (rows_str.back() == 'k' || rows_str.back() == 'K'))
        info.rows = std::stoull(rows_str.substr(0, rows_str.size()-1)) * 1000;
    else
        info.rows = std::stoull(rows_str);

    pos = u1 + 1;
    if (pos >= name.size() || name[pos] != 'c') return info;
    pos++;
    size_t u2 = name.find('_', pos);
    if (u2 == std::string::npos) return info;
    info.cardinality = std::stoull(name.substr(pos, u2 - pos));
    info.algo = name.substr(u2 + 1);
    return info;
}

/// Map gen_check algorithm name to benchmark backend key
static std::string algo_to_backend_key(const std::string& algo) {
    if (algo == "wah") return "wah";
    if (algo == "roaring") return "croaring";
    if (algo == "ewah") return "ewah";
    if (algo == "concise") return "concise";
    if (algo == "combit") return "combit";
    return "";
}

void run_compressed_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                              const std::string& comp_dir, size_t num_rows,
                              size_t cardinality,
                              size_t sample_count, size_t iterations = 1)
{
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — Compressed .bm Files\n";
    std::cout << "Source: " << comp_dir << "\n";

    // Collect .bm files, sort numerically
    std::vector<std::string> bm_files;
    for (auto& entry : fs::directory_iterator(comp_dir)) {
        if (entry.path().extension() == ".bm")
            bm_files.push_back(entry.path().string());
    }
    std::sort(bm_files.begin(), bm_files.end(), [](const std::string& a, const std::string& b){
        return std::stoi(fs::path(a).stem().string()) < std::stoi(fs::path(b).stem().string());
    });

    if (bm_files.empty()) {
        std::cerr << "  No .bm files found in " << comp_dir << "\n";
        return;
    }

    // Sample if needed
    std::vector<std::string> selected;
    if (sample_count > 0 && sample_count < bm_files.size()) {
        for (size_t i = 0; i < sample_count; ++i)
            selected.push_back(bm_files[i * bm_files.size() / sample_count]);
    } else {
        selected = bm_files;
    }

    std::cout << "Rows: " << num_rows << " | Cardinality: " << cardinality
              << " | Loaded: " << selected.size();
    if (iterations > 1) std::cout << " | Iterations: " << iterations;
    std::cout << "\n=======================================\n";

    // File sizes (once)
    long total_compressed = 0;
    for (auto& path : selected) total_compressed += get_file_size(path);
    double raw_total = static_cast<double>(selected.size()) * num_rows / 8.0;
    double ratio = (raw_total > 0) ? total_compressed / raw_total : 0;

    // Timing accumulators
    std::vector<double> load_times, or_times, and_times, xor_times, multi_or_times;
    uint64_t or_card = 0, and_card = 0, xor_card = 0, multi_or_card = 0;
    Timer timer;

    // Warm-up
    if (iterations > 1 && selected.size() >= 2) {
        std::cout << "[Warm-up] Running 1 warm-up iteration...\n";
        auto w1 = backend->Load(selected[0]);
        auto w2 = backend->Load(selected[1]);
        if (w1 && w2) { auto wor = backend->bitOr(*w1, *w2); (void)wor; }
    }

    for (size_t iter = 0; iter < iterations; ++iter) {
        // Phase 1: Load compressed bitmaps
        std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
        timer.reset();
        for (auto& path : selected) {
            auto h = backend->Load(path);
            if (h) bitmaps.push_back(std::move(h));
        }
        double load_t = timer.elapsed_ms();
        load_times.push_back(load_t);

        if (iter == 0) {
            std::cout << "[Load] Loaded " << bitmaps.size()
                      << " compressed bitmaps: " << load_t << " ms\n";
            for (size_t i = 0; i < bitmaps.size() && i < 5; ++i) {
                std::string name = fs::path(selected[i]).filename().string();
                std::cout << "  " << name << " → cardinality: "
                          << backend->Cardinality(*bitmaps[i]) << "\n";
            }
            if (bitmaps.size() > 5)
                std::cout << "  ... (" << bitmaps.size() - 5 << " more)\n";
            std::cout << "\n[Storage] Total compressed: " << total_compressed
                      << " bytes (" << total_compressed / 1024.0 << " KB)"
                      << " | Ratio: " << ratio << "x\n";
        }

        if (bitmaps.size() < 2) {
            std::cout << "  Not enough bitmaps for logical ops.\n";
            return;
        }

        // Phase 2: Pairwise logical ops
        {
            auto& a = bitmaps[0]; auto& b = bitmaps[1];
            timer.reset();
            auto or_res = backend->bitOr(*a, *b);
            or_times.push_back(timer.elapsed_ms());
            or_card = backend->Cardinality(*or_res);

            timer.reset();
            auto and_res = backend->bitAnd(*a, *b);
            and_times.push_back(timer.elapsed_ms());
            and_card = backend->Cardinality(*and_res);

            timer.reset();
            auto xor_res = backend->bitXor(*a, *b);
            xor_times.push_back(timer.elapsed_ms());
            xor_card = backend->Cardinality(*xor_res);
        }

        // Phase 3: Multi-way OR (all bitmaps)
        if (bitmaps.size() >= 3) {
            timer.reset();
            auto acc = backend->bitOr(*bitmaps[0], *bitmaps[1]);
            for (size_t k = 2; k < bitmaps.size(); ++k)
                acc = backend->bitOr(*acc, *bitmaps[k]);
            multi_or_times.push_back(timer.elapsed_ms());
            multi_or_card = backend->Cardinality(*acc);
        }

        // CSV per-iteration
        int it = static_cast<int>(iter + 1);
        csv_row(backend_name, num_rows, cardinality, "load", load_t, total_compressed, ratio, 0, it);
        csv_row(backend_name, num_rows, cardinality, "OR", or_times.back(), 0, 0, or_card, it);
        csv_row(backend_name, num_rows, cardinality, "AND", and_times.back(), 0, 0, and_card, it);
        csv_row(backend_name, num_rows, cardinality, "XOR", xor_times.back(), 0, 0, xor_card, it);
        if (!multi_or_times.empty())
            csv_row(backend_name, num_rows, cardinality, "multi-OR", multi_or_times.back(), 0, 0, multi_or_card, it);
    }

    // Summary
    if (iterations > 1) {
        std::cout << "\n[Summary] Median over " << iterations << " iterations:\n";
        std::cout << "  Load:      " << compute_median(load_times) << " ms\n";
        std::cout << "  bitOr:     " << compute_median(or_times) << " ms (card: " << or_card << ")\n";
        std::cout << "  bitAnd:    " << compute_median(and_times) << " ms (card: " << and_card << ")\n";
        std::cout << "  bitXor:    " << compute_median(xor_times) << " ms (card: " << xor_card << ")\n";
        if (!multi_or_times.empty())
            std::cout << "  multi-OR:  " << compute_median(multi_or_times) << " ms (card: " << multi_or_card << ")\n";
    } else {
        std::cout << "\n[Compute] Pairwise ops on first 2 bitmaps:\n";
        std::cout << "  bitOr:  " << or_times[0]  << " ms (card: " << or_card << ")\n";
        std::cout << "  bitAnd: " << and_times[0] << " ms (card: " << and_card << ")\n";
        std::cout << "  bitXor: " << xor_times[0] << " ms (card: " << xor_card << ")\n";
        if (!multi_or_times.empty())
            std::cout << "\n[Compute] Multi-way OR of " << selected.size()
                      << " bitmaps: " << multi_or_times[0] << " ms (card: " << multi_or_card << ")\n";
    }
    std::cout << "---------------------------------------\n";
}

// ==========================================
// 7. Print usage
// ==========================================
static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --backend <wah|croaring|combit|ewah|concise|all>  Backend to benchmark (default: all)\n"
              << "  --size <N>                           Number of random bits (default: 10000000)\n"
              << "  --bm-dir <path>                      Directory with raw .bm files\n"
              << "  --compressed-dir <path>              Directory with pre-compressed .bm files (from gen_check)\n"
              << "  --num-rows <N>                       Number of rows (default: auto-detect from dir name)\n"
              << "  --bmz-dir <path>                     Directory with .bmz files from gen_bitmap.sh\n"
              << "  --sample <N>                         Number of bitmaps to sample (default: all)\n"
              << "  --csv <path>                         Write results to CSV file (appends if exists)\n"
              << "  --iterations <N>                     Run N iterations, report median (default: 1)\n"
              << "  --help                               Show this help\n\n"
              << "Examples:\n"
              << "  " << prog << "                                    # Random data, all backends\n"
              << "  " << prog << " --backend combit --size 1000000    # Random, ComBit only\n"
              << "  " << prog << " --bm-dir ../test_bitmaps           # Raw .bm files, all backends\n"
              << "  " << prog << " --compressed-dir bm_100m_c100_wah --iterations 5 --csv results.csv\n"
              << "  " << prog << " --bmz-dir ./bitmaps --backend combit --sample 20\n";
}

// ==========================================
// Main function: CLI argument parsing
// ==========================================
int main(int argc, char** argv) {
    std::string backend_type = "all";
    size_t test_size = 10000000;
    std::string bm_dir;
    std::string bmz_dir;
    std::string compressed_dir;
    size_t num_rows = 0;
    size_t sample_count = 0;  // 0 = all
    std::string csv_path;
    size_t iterations = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend_type = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            test_size = std::stoull(argv[++i]);
        } else if (arg == "--bm-dir" && i + 1 < argc) {
            bm_dir = argv[++i];
        } else if (arg == "--bmz-dir" && i + 1 < argc) {
            bmz_dir = argv[++i];
        } else if (arg == "--compressed-dir" && i + 1 < argc) {
            compressed_dir = argv[++i];
        } else if (arg == "--num-rows" && i + 1 < argc) {
            num_rows = std::stoull(argv[++i]);
        } else if (arg == "--sample" && i + 1 < argc) {
            sample_count = std::stoull(argv[++i]);
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoull(argv[++i]);
            if (iterations == 0) iterations = 1;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Set up CSV output if requested
    std::ofstream csv_file;
    if (!csv_path.empty()) {
        bool csv_exists = fs::exists(csv_path) && fs::file_size(csv_path) > 0;
        csv_file.open(csv_path, csv_exists ? std::ios::app : std::ios::out);
        if (!csv_file) {
            std::cerr << "Error: cannot open CSV file " << csv_path << "\n";
            return 1;
        }
        g_csv = &csv_file;
        if (!csv_exists) csv_write_header();
        std::cout << "CSV output: " << csv_path << (csv_exists ? " (appending)\n" : "\n");
    }

    WahBackend wah;
    CroaringBackend croaring;
    CombitBackend combit;
    EwahBackend ewah;
    ConciseBackend concise;

    // Build list of (backend_ptr, name) to run
    struct BackendEntry { IBitmapBackend* ptr; std::string name; std::string key; };
    std::vector<BackendEntry> backends = {
        {&wah,      "WAH (FastBit)", "wah"},
        {&croaring, "CRoaring",      "croaring"},
        {&combit,   "ComBIT (New)",   "combit"},
        {&ewah,     "EWAH",          "ewah"},
        {&concise,  "Concise",       "concise"},
    };

    if (!compressed_dir.empty()) {
        // === Pre-compressed .bm file mode ===
        if (!fs::is_directory(compressed_dir)) {
            std::cerr << "Error: " << compressed_dir << " is not a directory\n";
            return 1;
        }
        auto dir_info = parse_compressed_dir_name(compressed_dir);
        size_t rows = (num_rows > 0) ? num_rows : dir_info.rows;
        size_t card = dir_info.cardinality;
        std::string detected_key = algo_to_backend_key(dir_info.algo);

        if (rows == 0) {
            std::cerr << "Error: cannot determine num_rows. Use --num-rows or follow naming bm_{rows}_c{card}_{algo}\n";
            return 1;
        }
        // Auto-select backend from directory name if --backend not specified
        if (backend_type == "all" && !detected_key.empty())
            backend_type = detected_key;

        std::cout << "=== Compressed .bm Benchmark Mode ===\n";
        std::cout << "Directory: " << compressed_dir << "\n";
        std::cout << "Rows: " << rows << " | Cardinality: " << card << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all")
                run_compressed_benchmark(be.ptr, be.name, compressed_dir, rows, card, sample_count, iterations);
        }
    } else if (!bmz_dir.empty()) {
        // === .bmz file mode ===
        if (!fs::is_directory(bmz_dir)) {
            std::cerr << "Error: " << bmz_dir << " is not a directory\n";
            return 1;
        }
        std::string index_path = bmz_dir + "/index.txt";
        BmzIndex idx = parse_bmz_index(index_path);
        if (idx.rows == 0 || idx.cardinality == 0) {
            std::cerr << "Error: cannot parse index.txt in " << bmz_dir << "\n";
            return 1;
        }

        std::cout << "=== .bmz File Benchmark Mode ===\n";
        std::cout << "Directory: " << bmz_dir << "\n";
        std::cout << "Rows: " << idx.rows << " | Cardinality: " << idx.cardinality
                  << " | Sample: " << sample_count << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all")
                run_bmz_benchmark(be.ptr, be.name, bmz_dir, idx, sample_count, iterations);
        }
    } else if (!bm_dir.empty()) {
        // === .bm file mode ===
        if (!fs::is_directory(bm_dir)) {
            std::cerr << "Error: " << bm_dir << " is not a directory\n";
            return 1;
        }
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