#include "uti.h"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ==========================================
// CSV output helper
// ==========================================
std::ofstream* g_csv = nullptr;

void csv_write_header() {
    if (!g_csv) return;
    *g_csv << "backend,num_rows,cardinality,operation,time_ms,"
           << "compressed_bytes,compression_ratio,result_cardinality,iteration\n";
}

void csv_row(const std::string& backend, size_t rows, size_t card,
             const std::string& op, double time_ms,
             long compressed_bytes, double ratio,
             uint64_t result_card, int iteration) {
    if (!g_csv) return;
    *g_csv << backend << "," << rows << "," << card << ","
           << op << "," << time_ms << ","
           << compressed_bytes << "," << ratio << ","
           << result_card << "," << iteration << "\n";
}

double compute_median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}

// ==========================================
// File / bitmap utilities
// ==========================================
long get_file_size(const std::string& filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

std::vector<bool> read_raw_bm(const std::string& path, size_t num_rows) {
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

std::unique_ptr<BitmapHandle> bits_to_bitmap(
    IBitmapBackend* backend, const std::vector<bool>& bits)
{
    auto h = backend->Create();
    for (bool b : bits) backend->Append(*h, b);
    return h;
}

std::map<std::string, std::string> read_metadata(const std::string& path) {
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
// Compressed directory name parser
// ==========================================
CompressedDirInfo parse_compressed_dir_name(const std::string& dir_path) {
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

std::string algo_to_backend_key(const std::string& algo) {
    if (algo == "wah") return "wah";
    if (algo == "roaring") return "croaring";
    if (algo == "ewah") return "ewah";
    if (algo == "concise") return "concise";
    if (algo == "combit") return "combit";
    if (algo.substr(0, 8) == "combit_w") return "combit";
    if (algo == "bitset") return "bitset";
    if (algo == "bitset_avx512") return "bitset_avx512";
    return "";
}

// ==========================================
// Print usage
// ==========================================
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  --backend <wah|croaring|combit|ewah|concise|bitset|bitset_avx512|all>  Backend to benchmark (default: all)\n"
              << "  --bm-dir <path>                      Directory with raw .bm files\n"
              << "  --compressed-dir <path>              Directory with pre-compressed .bm files\n"
              << "  --cross-or <dir_a> <dir_b>           Cross-cardinality OR: load bitmap 1 from each dir\n"
              << "  --num-rows <N>                       Number of rows (default: auto-detect from dir name)\n"
              << "  --sample <N>                         Number of bitmaps to sample (default: all)\n"
              << "  --csv <path>                         Write results to CSV file (appends if exists)\n"
              << "  --iterations <N>                     Run N iterations, report median (default: 1)\n"
              << "  --help                               Show this help\n\n"
              << "Examples:\n"
              << "  " << prog << " --bm-dir ../test_bitmaps --num-rows 100000000\n"
              << "  " << prog << " --compressed-dir bitmap/bm_100m_c100_wah --iterations 5 --csv results.csv\n"
              << "  " << prog << " --compressed-dir bitmap/bm_100m_c100_ewah --backend ewah\n"
              << "  " << prog << " --compressed-dir bitmap/bm_100m_c100_bitset --backend bitset\n"
              << "  " << prog << " --compressed-dir bitmap/bm_100m_c100_bitset --backend bitset_avx512\n"
              << "  " << prog << " --cross-or bitmap/bm_100m_c10_roaring bitmap/bm_100m_c20_roaring\n"
              << "  " << prog << " --cross-or bitmap/bm_100m_c10_bitset bitmap/bm_100m_c100_bitset --backend bitset\n";
}
