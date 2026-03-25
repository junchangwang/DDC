/**
 * verify_bitmap.cpp
 *
 * Two experiments to verify bitmap correctness across multiple modes.
 *
 * Experiment 1 (popcount):
 *   Load all bitmaps, sum popcounts, verify sum == n.
 *
 * Experiment 2 (getBit):
 *   Check first min(10000, n) rows against original dataset.
 *
 * Usage:
 *   ./verify_bitmap -mode <mode> -n <rows> -c <cardinality> [-d <base_dir>] [-z <zip_length>]
 *
 * Modes: raw, wah, roaring, ewah, concise, combit, bitset
 *
 * Example:
 *   ./verify_bitmap -mode wah     -n 10000 -c 100 -d .
 *   ./verify_bitmap -mode bitset  -n 10000 -c 100 -z 10 -d .
 *   ./verify_bitmap -mode roaring -n 10000 -c 100 -d .
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <filesystem>
#include <algorithm>

// WAH (FastBit)
#include "benchmark/backends/wah/wah_backend.h"
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

// ============================================================
// Path helpers — must match gen_bitmap.cpp naming conventions
// ============================================================

static std::string format_rows(int n) {
    if (n >= 1000000000 && n % 1000000000 == 0) return std::to_string(n / 1000000000) + "b";
    if (n >= 1000000    && n % 1000000    == 0) return std::to_string(n / 1000000)    + "m";
    if (n >= 1000       && n % 1000       == 0) return std::to_string(n / 1000)       + "k";
    return std::to_string(n);
}

/// wah/roaring/ewah/concise/combit: base_dir/bm_<rows>_c<c>_<algo>/
static std::string compressed_dir(const std::string& base_dir, int n, int c,
                                   const std::string& algo) {
    return base_dir + "/bm_" + format_rows(n) + "_c" + std::to_string(c) + "_" + algo;
}

/// bitset: base_dir/bitmap_n<n>_c<c>_z<z>/
static std::string bitset_dir(const std::string& base_dir, int n, int c, int z) {
    return base_dir + "/bitmap_n" + format_rows(n)
         + "_c" + std::to_string(c) + "_z" + std::to_string(z);
}

static std::string dataset_file(const std::string& base_dir, int n, int c) {
    return base_dir + "/dataset_" + std::to_string(n) + "_" + std::to_string(c);
}

static std::string bm_path(const std::string& dir, int v) {
    return dir + "/" + std::to_string(v) + ".bm";
}

// ============================================================
// Dataset loader
// ============================================================

static std::vector<int32_t> load_dataset(const std::string& path, int n) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "[verify] Error: cannot open dataset: " << path << "\n";
        return {};
    }
    std::vector<int32_t> data(n);
    fin.read(reinterpret_cast<char*>(data.data()), n * sizeof(int32_t));
    if (!fin) {
        std::cerr << "[verify] Error: failed to read dataset\n";
        return {};
    }
    return data;
}

// ============================================================
// Per-mode: popcount and getBit helpers
// ============================================================

// ---------- RAW / BITSET (little-endian packed bits) ----------

static int packed_bytes_for(int n) { return (n + 7) / 8; }

static bool raw_get_bit(const std::vector<uint8_t>& buf, int row) {
    return (buf[row / 8] >> (row % 8)) & 1;
}

static uint64_t raw_popcount(const std::vector<uint8_t>& buf, int n) {
    uint64_t cnt = 0;
    int packed_bytes = (n + 7) / 8;
    for (int i = 0; i < packed_bytes; i++)
        cnt += __builtin_popcount(buf[i]);
    return cnt;
}

static std::vector<uint8_t> load_raw(const std::string& dir, int n, int v) {
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return {}; }
    int pb = packed_bytes_for(n);
    std::ifstream fin(path, std::ios::binary);
    std::vector<uint8_t> buf(pb);
    fin.read(reinterpret_cast<char*>(buf.data()), pb);
    return buf;
}

// ---------- BITSET (.bmz offset-based) ----------

static std::vector<uint8_t> load_bitset(const std::string& dir, int n, int z, int v) {
    int bmz_idx   = (v - 1) / z;
    int local_idx = (v - 1) % z;
    std::string path = dir + "/" + std::to_string(bmz_idx) + ".bmz";
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return {}; }
    int pb = packed_bytes_for(n);
    std::ifstream fin(path, std::ios::binary);
    fin.seekg(static_cast<std::streamoff>(local_idx) * pb);
    std::vector<uint8_t> buf(pb);
    fin.read(reinterpret_cast<char*>(buf.data()), pb);
    return buf;
}

// ---------- WAH (FastBit) ----------

static std::unique_ptr<BitmapHandle> load_wah(WahBackend& backend,
                                               const std::string& dir, int v) {
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return nullptr; }
    return backend.Load(path);
}

// ---------- ROARING ----------

struct RoaringBM {
    roaring::Roaring bm;
    uint32_t n_rows = 0;
};

static RoaringBM load_roaring(const std::string& dir, int v) {
    RoaringBM result;
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return result; }
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&result.n_rows), sizeof(result.n_rows));
    std::vector<char> buf((std::istreambuf_iterator<char>(fin)), {});
    result.bm = roaring::Roaring::read(buf.data());
    return result;
}

// ---------- EWAH ----------

struct EwahBM {
    ewah::EWAHBoolArray<uint64_t> bm;
    uint64_t n_rows = 0;
};

static EwahBM load_ewah(const std::string& dir, int v) {
    EwahBM result;
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return result; }
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&result.n_rows), sizeof(result.n_rows));
    result.bm.read(fin);
    return result;
}

// ---------- CONCISE ----------

struct ConciseBM {
    ConciseSet<false> bm;
    uint64_t n_rows = 0;
};

static ConciseBM load_concise(const std::string& dir, int v) {
    ConciseBM result;
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return result; }
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&result.n_rows), sizeof(result.n_rows));
    fin.read(reinterpret_cast<char*>(&result.bm.last), sizeof(result.bm.last));
    fin.read(reinterpret_cast<char*>(&result.bm.lastWordIndex), sizeof(result.bm.lastWordIndex));
    if (result.bm.lastWordIndex >= 0) {
        uint32_t count = static_cast<uint32_t>(result.bm.lastWordIndex + 1);
        result.bm.words.resize(count);
        fin.read(reinterpret_cast<char*>(result.bm.words.data()), count * sizeof(uint32_t));
    }
    return result;
}

// ---------- COMBIT ----------

struct CombitBM {
    std::vector<uint32_t> set_positions;
    uint64_t n_rows = 0;
};

static CombitBM load_combit(const std::string& dir, int v) {
    CombitBM result;
    std::string path = bm_path(dir, v);
    if (!fs::exists(path)) { std::cerr << "[verify] not found: " << path << "\n"; return result; }
    std::ifstream fin(path, std::ios::binary);
    fin.read(reinterpret_cast<char*>(&result.n_rows), sizeof(result.n_rows));
    size_t num;
    fin.read(reinterpret_cast<char*>(&num), sizeof(num));
    result.set_positions.resize(num);
    if (num > 0)
        fin.read(reinterpret_cast<char*>(result.set_positions.data()), num * sizeof(uint32_t));
    return result;
}

// ============================================================
// Generic experiment runner
// ============================================================

struct ModeContext {
    std::string mode;
    std::string dir;
    int n, c, z;

    // WAH backend (only used for wah mode)
    WahBackend wah_backend;

    // Pre-loaded bitmaps
    std::vector<std::unique_ptr<BitmapHandle>> wah_bitmaps;
    std::vector<std::vector<uint8_t>>          raw_bitmaps;
    std::vector<RoaringBM>                     roaring_bitmaps;
    std::vector<EwahBM>                        ewah_bitmaps;
    std::vector<ConciseBM>                     concise_bitmaps;
    std::vector<CombitBM>                      combit_bitmaps;
};

/// Returns popcount for value v in given mode
static uint64_t get_popcount(ModeContext& ctx, int v) {
    if (ctx.mode == "raw" || ctx.mode == "bitset")
        return raw_popcount(ctx.raw_bitmaps[v], ctx.n);
    else if (ctx.mode == "wah")
        return ctx.wah_bitmaps[v] ? ctx.wah_backend.Cardinality(*ctx.wah_bitmaps[v]) : 0;
    else if (ctx.mode == "roaring")
        return ctx.roaring_bitmaps[v].bm.cardinality();
    else if (ctx.mode == "ewah")
        return ctx.ewah_bitmaps[v].bm.numberOfOnes();
    else if (ctx.mode == "concise")
        return ctx.concise_bitmaps[v].bm.size();
    else if (ctx.mode == "combit")
        return ctx.combit_bitmaps[v].set_positions.size();
    return 0;
}

/// Returns true if bit at row is set for value v in given mode
static bool get_bit(ModeContext& ctx, int v, int row) {
    if (ctx.mode == "raw" || ctx.mode == "bitset") {
        return raw_get_bit(ctx.raw_bitmaps[v], row);
    } else if (ctx.mode == "wah") {
        const auto& btv = static_cast<WahHandle&>(*ctx.wah_bitmaps[v]).btv;
        Table_config config;
        config.enable_fence_pointer = false;
        return btv.getBit(row, &config);
    } else if (ctx.mode == "roaring") {
        return ctx.roaring_bitmaps[v].bm.contains(row);
    } else if (ctx.mode == "ewah") {
        auto ones = ctx.ewah_bitmaps[v].bm.toArray();
        return std::binary_search(ones.begin(), ones.end(), (size_t)row);
    } else if (ctx.mode == "concise") {
        return ctx.concise_bitmaps[v].bm.contains(row);
    } else if (ctx.mode == "combit") {
        const auto& pos = ctx.combit_bitmaps[v].set_positions;
        return std::binary_search(pos.begin(), pos.end(), (uint32_t)row);
    }
    return false;
}

// ============================================================
// Preload all bitmaps into ModeContext
// ============================================================

static void preload_bitmaps(ModeContext& ctx) {
    std::cout << "[verify] Preloading " << ctx.c << " bitmaps...\n";
    if (ctx.mode == "wah") {
        ctx.wah_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.wah_bitmaps[v] = load_wah(ctx.wah_backend, ctx.dir, v);
    } else if (ctx.mode == "raw") {
        ctx.raw_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.raw_bitmaps[v] = load_raw(ctx.dir, ctx.n, v);
    } else if (ctx.mode == "bitset") {
        ctx.raw_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++) {
            if (ctx.z == 1)
                ctx.raw_bitmaps[v] = load_raw(ctx.dir, ctx.n, v);
            else
                ctx.raw_bitmaps[v] = load_bitset(ctx.dir, ctx.n, ctx.z, v);
        }
    } else if (ctx.mode == "roaring") {
        ctx.roaring_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.roaring_bitmaps[v] = load_roaring(ctx.dir, v);
    } else if (ctx.mode == "ewah") {
        ctx.ewah_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.ewah_bitmaps[v] = load_ewah(ctx.dir, v);
    } else if (ctx.mode == "concise") {
        ctx.concise_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.concise_bitmaps[v] = load_concise(ctx.dir, v);
    } else if (ctx.mode == "combit") {
        ctx.combit_bitmaps.resize(ctx.c + 1);
        for (int v = 1; v <= ctx.c; v++)
            ctx.combit_bitmaps[v] = load_combit(ctx.dir, v);
    }
    std::cout << "[verify] Preload complete.\n";
}

// ============================================================
// Experiment 1 — popcount
// ============================================================

static void experiment1(ModeContext& ctx) {
    std::cout << "\n========================================\n";
    std::cout << "Experiment 1 [" << ctx.mode << "] — Popcount Verification\n";
    std::cout << "========================================\n";
    std::cout << "n=" << ctx.n << "  c=" << ctx.c << "\n\n";

    if (!fs::is_directory(ctx.dir)) {
        std::cerr << "[verify] Directory not found: " << ctx.dir << "\n";
        return;
    }

    uint64_t total = 0;

    for (int v = 1; v <= ctx.c; v++) {
        uint64_t pop = get_popcount(ctx, v);
        total += pop;
        std::cout << "  value=" << v << "  popcount=" << pop << "\n";
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "Bitmaps processed: " << ctx.c << "/" << ctx.c << "\n";
    std::cout << "Total 1s (sum of popcounts): " << total << "\n";
    std::cout << "Expected (cnt = n):          " << ctx.n << "\n";
    if ((int64_t)total == ctx.n)
        std::cout << "Result: PASS — sum matches n\n";
    else
        std::cout << "Result: FAIL — sum differs by " << (int64_t)total - ctx.n << "\n";
}

// ============================================================
// Batch bit collector
// ============================================================

static std::vector<bool> collect_bits_for_rows(ModeContext& ctx, int v, int check_rows) {
    std::vector<bool> bits(check_rows, false);
    if (ctx.mode == "raw" || ctx.mode == "bitset") {
        for (int row = 0; row < check_rows; row++)
            bits[row] = raw_get_bit(ctx.raw_bitmaps[v], row);
    } else if (ctx.mode == "wah") {
        Table_config cfg; cfg.enable_fence_pointer = false;
        const auto& btv = static_cast<WahHandle&>(*ctx.wah_bitmaps[v]).btv;
        for (int row = 0; row < check_rows; row++)
            bits[row] = btv.getBit(row, &cfg);
    } else if (ctx.mode == "roaring") {
        for (int row = 0; row < check_rows; row++)
            bits[row] = ctx.roaring_bitmaps[v].bm.contains(row);
    } else if (ctx.mode == "ewah") {
        auto ones = ctx.ewah_bitmaps[v].bm.toArray();
        for (int row = 0; row < check_rows; row++)
            bits[row] = std::binary_search(ones.begin(), ones.end(), (size_t)row);
    } else if (ctx.mode == "concise") {
        for (int row = 0; row < check_rows; row++)
            bits[row] = ctx.concise_bitmaps[v].bm.contains(row);
    } else if (ctx.mode == "combit") {
        const auto& pos = ctx.combit_bitmaps[v].set_positions;
        for (int row = 0; row < check_rows; row++)
            bits[row] = std::binary_search(pos.begin(), pos.end(), (uint32_t)row);
    }
    return bits;
}

// ============================================================
// Experiment 2 — getBit
// ============================================================

static void experiment2(ModeContext& ctx, const std::string& base_dir) {
    std::cout << "\n========================================\n";
    std::cout << "Experiment 2 [" << ctx.mode << "] — getBit Verification\n";
    std::cout << "========================================\n";

    int check_rows = std::min(ctx.n, 10000);
    std::cout << "n=" << ctx.n << "  c=" << ctx.c
              << "  checking first " << check_rows << " rows\n\n";

    // Load dataset
    auto data = load_dataset(dataset_file(base_dir, ctx.n, ctx.c), ctx.n);
    if (data.empty()) return;

    if (!fs::is_directory(ctx.dir)) {
        std::cerr << "[verify] Directory not found: " << ctx.dir << "\n";
        return;
    }

    // Pass 1: true positive — each row's expected bitmap must have bit=1
    int pass = 0, fail = 0;
    for (int row = 0; row < check_rows; row++) {
        int expected_val = data[row];
        bool bit = get_bit(ctx, expected_val, row);
        if (bit) {
            pass++;
        } else {
            std::cerr << "  [TP-FAIL] row=" << row << " expected value=" << expected_val
                      << " but bit is 0\n";
            fail++;
        }
    }

    // Pass 2: false positive — no other bitmap should have bit=1 for these rows
    int fp_fail = 0;
    for (int v = 1; v <= ctx.c; v++) {
        auto bits = collect_bits_for_rows(ctx, v, check_rows);
        for (int row = 0; row < check_rows; row++) {
            if (data[row] == v) continue;
            if (bits[row]) {
                std::cerr << "  [FP-FAIL] row=" << row << " value=" << v
                          << " should be 0 but is 1\n";
                fp_fail++;
            }
        }
    }

    std::cout << "--- Summary ---\n";
    std::cout << "Rows checked: " << check_rows << "\n";
    std::cout << "True positive  — Pass: " << pass << "  Fail: " << fail << "\n";
    std::cout << "False positive — Fail: " << fp_fail << "\n";
    std::cout << (fail == 0 && fp_fail == 0 ? "Result: PASS\n" : "Result: FAIL\n");
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    std::string mode;
    int n = -1, c = -1, z = 1;
    std::string base_dir = ".";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "-mode" && i+1 < argc) mode     = argv[++i];
        else if (arg == "-n"    && i+1 < argc) n        = std::stoi(argv[++i]);
        else if (arg == "-c"    && i+1 < argc) c        = std::stoi(argv[++i]);
        else if (arg == "-z"    && i+1 < argc) z        = std::stoi(argv[++i]);
        else if (arg == "-d"    && i+1 < argc) base_dir = argv[++i];
        else { std::cerr << "Unknown option: " << arg << "\n"; return 1; }
    }

    if (mode.empty() || n <= 0 || c <= 0) {
        std::cerr << "Usage: " << argv[0]
                  << " -mode <raw|wah|roaring|ewah|concise|combit|bitset>"
                  << " -n <rows> -c <cardinality>"
                  << " [-z <zip_length>] [-d <base_dir>]\n";
        return 1;
    }

    // Normalise mode to lowercase
    for (auto& ch : mode) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    // Build context
    ModeContext ctx;
    ctx.mode = mode;
    ctx.n    = n;
    ctx.c    = c;
    ctx.z    = z;

    if (mode == "raw")
        ctx.dir = base_dir + "/src/core/bitset/bitmaps_" + format_rows(n) + "_c" + std::to_string(c);
    else if (mode == "bitset")
        ctx.dir = bitset_dir(base_dir, n, c, z);
    else
        ctx.dir = compressed_dir(base_dir, n, c, mode);

    std::cout << "[verify] mode=" << mode << " n=" << n << " c=" << c
              << " z=" << z << " base_dir=" << base_dir << "\n"
              << "[verify] dir=" << ctx.dir << "\n";

    preload_bitmaps(ctx);
    experiment1(ctx);
    experiment2(ctx, base_dir);

    return 0;
}
