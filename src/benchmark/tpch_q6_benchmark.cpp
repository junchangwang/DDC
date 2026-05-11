// tpch_q6_benchmark.cpp — TPC-H Q6 Benchmark: ComBit vs WAH
//
// Evaluates TPC-H Query 6 using bitmap index operations:
//   WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
//     AND l_discount BETWEEN 0.05 AND 0.07
//     AND l_quantity < 24
//
// Bitmap layout (under <base_dir>):
//   discount/  : 11 equality-encoded bitmaps (0.bm..10.bm)
//   quantity/  : 50 equality-encoded bitmaps (1.bm..50.bm)
//   shipdate/  : 1 pre-computed range bitmap (range_1994.bm)
//
// Usage:
//   tpch_q6_benchmark [base_dir] [iterations]
//
// Defaults:
//   base_dir   = bitmap/tpch_q6
//   iterations = 10

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <cstdint>

#include "benchmark/uti.h"
#include "benchmark/backends/combit/combit_backend.h"
#include "benchmark/backends/wah/wah_backend.h"

namespace fs = std::filesystem;

// ============================================================
// Constants
// ============================================================
static const size_t NUM_ROWS = 59986052;  // lineitem TPC-H SF10

// Q6 predicate parameters
static const int DISCOUNT_MIN = 5;   // l_discount >= 0.05  → bitmap value 5
static const int DISCOUNT_MAX = 7;   // l_discount <= 0.07  → bitmap value 7
static const int QUANTITY_MAX = 23;   // l_quantity < 24     → bitmaps 1..23

// Expected Q6 intermediate and final cardinalities (for verification)
static const uint64_t EXPECTED_SHIPDATE_CARD = 9099165;
static const uint64_t EXPECTED_FINAL_CARD    = 1139264;

// ============================================================
// Statistics helper
// ============================================================
struct Stats {
    double min_ms = 1e18, max_ms = 0, sum_ms = 0;
    int count = 0;
    std::vector<double> values;

    void add(double ms) {
        min_ms = std::min(min_ms, ms);
        max_ms = std::max(max_ms, ms);
        sum_ms += ms;
        values.push_back(ms);
        count++;
    }

    double avg() const { return count > 0 ? sum_ms / count : 0; }

    double median() const {
        if (values.empty()) return 0;
        auto v = values;
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
    }
};

// ============================================================
// Load raw .bm files → backend bitmaps
// ============================================================
struct Q6Bitmaps {
    // Discount bitmaps: indices 0..10; Q6 uses [5], [6], [7]
    std::vector<std::unique_ptr<BitmapHandle>> discount;
    // Quantity bitmaps: indices 1..50; Q6 uses [1]..[23]
    std::vector<std::unique_ptr<BitmapHandle>> quantity;
    // Shipdate range bitmap
    std::unique_ptr<BitmapHandle> shipdate_range;
};

Q6Bitmaps load_q6_bitmaps(IBitmapBackend* backend, const std::string& base_dir) {
    Q6Bitmaps bm;

    // Discount: 0.bm .. 10.bm (11 files)
    bm.discount.resize(11);
    for (int v = 0; v <= 10; v++) {
        std::string path = base_dir + "/discount/" + std::to_string(v) + ".bm";
        auto bits = read_raw_bm(path, NUM_ROWS);
        bm.discount[v] = bits_to_bitmap(backend, bits);
    }

    // Quantity: 1.bm .. 50.bm (50 files)
    bm.quantity.resize(51); // index 0 unused
    for (int v = 1; v <= 50; v++) {
        std::string path = base_dir + "/quantity/" + std::to_string(v) + ".bm";
        auto bits = read_raw_bm(path, NUM_ROWS);
        bm.quantity[v] = bits_to_bitmap(backend, bits);
    }

    // Shipdate range bitmap
    {
        std::string path = base_dir + "/shipdate/range_1994.bm";
        auto bits = read_raw_bm(path, NUM_ROWS);
        bm.shipdate_range = bits_to_bitmap(backend, bits);
    }

    return bm;
}

// ============================================================
// Q6 execution: perform OR and AND operations
// ============================================================
struct Q6Result {
    double discount_or_ms = 0;
    double quantity_or_ms = 0;
    double and_ms = 0;
    double decode_ms = 0;
    double total_op_ms = 0;

    uint64_t discount_or_card = 0;
    uint64_t quantity_or_card = 0;
    uint64_t final_card = 0;
    size_t decoded_rows = 0;
};

Q6Result run_q6(IBitmapBackend* backend, const Q6Bitmaps& bm) {
    Q6Result r;
    Timer timer;

    // Phase 1: OR discount bitmaps (5, 6, 7)
    timer.reset();
    auto disc_result = backend->bitOr(*bm.discount[DISCOUNT_MIN],
                                       *bm.discount[DISCOUNT_MIN + 1]);
    for (int v = DISCOUNT_MIN + 2; v <= DISCOUNT_MAX; v++) {
        disc_result = backend->bitOr(*disc_result, *bm.discount[v]);
    }
    r.discount_or_ms = timer.elapsed_ms();
    r.discount_or_card = backend->Cardinality(*disc_result);

    // Phase 2: OR quantity bitmaps (1..23)
    timer.reset();
    auto qty_result = backend->bitOr(*bm.quantity[1], *bm.quantity[2]);
    for (int v = 3; v <= QUANTITY_MAX; v++) {
        qty_result = backend->bitOr(*qty_result, *bm.quantity[v]);
    }
    r.quantity_or_ms = timer.elapsed_ms();
    r.quantity_or_card = backend->Cardinality(*qty_result);

    // Phase 3: AND all three conditions
    timer.reset();
    auto and_result = backend->bitAnd(*disc_result, *qty_result);
    and_result = backend->bitAnd(*and_result, *bm.shipdate_range);
    r.and_ms = timer.elapsed_ms();
    r.final_card = backend->Cardinality(*and_result);

    // Phase 4: Decode (extract row IDs)
    timer.reset();
    auto rows = backend->Decode(*and_result);
    r.decode_ms = timer.elapsed_ms();
    r.decoded_rows = rows.size();

    r.total_op_ms = r.discount_or_ms + r.quantity_or_ms + r.and_ms + r.decode_ms;
    return r;
}

// ============================================================
// Print results
// ============================================================
void print_separator() {
    std::cout << std::string(70, '-') << "\n";
}

void print_stats_line(const std::string& label, const Stats& s) {
    std::cout << "  " << std::left << std::setw(22) << label
              << "median: " << std::right << std::setw(8) << std::fixed
              << std::setprecision(3) << s.median() << " ms"
              << "  avg: " << std::setw(8) << s.avg() << " ms"
              << "  [" << std::setw(8) << s.min_ms
              << " - " << std::setw(8) << s.max_ms << "]\n";
}

void run_benchmark(IBitmapBackend* backend, const std::string& name,
                   const std::string& base_dir, int iterations,
                   std::ofstream& csv) {
    std::cout << "\n";
    print_separator();
    std::cout << "  Backend: " << name << "\n";
    print_separator();

    // Phase A: Load bitmaps (timed once)
    Timer load_timer;
    auto bm = load_q6_bitmaps(backend, base_dir);
    double load_ms = load_timer.elapsed_ms();
    std::cout << "  Load (11+50+1 bitmaps): " << std::fixed
              << std::setprecision(1) << load_ms << " ms\n";

    // Verify shipdate cardinality
    uint64_t ship_card = backend->Cardinality(*bm.shipdate_range);
    std::cout << "  Shipdate range card:    " << ship_card;
    if (ship_card == EXPECTED_SHIPDATE_CARD)
        std::cout << "  [OK]\n";
    else
        std::cout << "  [MISMATCH! expected " << EXPECTED_SHIPDATE_CARD << "]\n";

    // Phase B: Warmup (1 iteration, not timed)
    std::cout << "  Warming up ...\n";
    run_q6(backend, bm);

    // Phase C: Measured iterations
    Stats s_disc_or, s_qty_or, s_and, s_decode, s_total;

    for (int i = 0; i < iterations; i++) {
        auto r = run_q6(backend, bm);

        s_disc_or.add(r.discount_or_ms);
        s_qty_or .add(r.quantity_or_ms);
        s_and    .add(r.and_ms);
        s_decode .add(r.decode_ms);
        s_total  .add(r.total_op_ms);

        // Verify on first iteration
        if (i == 0) {
            std::cout << "  Discount OR card: " << r.discount_or_card << "\n";
            std::cout << "  Quantity OR card: " << r.quantity_or_card << "\n";
            std::cout << "  Final AND card:   " << r.final_card;
            if (r.final_card == EXPECTED_FINAL_CARD)
                std::cout << "  [OK]\n";
            else
                std::cout << "  [MISMATCH! expected " << EXPECTED_FINAL_CARD << "]\n";
            std::cout << "  Decoded rows:     " << r.decoded_rows << "\n";
        }

        // Write CSV row
        csv << name << ","
            << i + 1 << ","
            << r.discount_or_ms << ","
            << r.quantity_or_ms << ","
            << r.and_ms << ","
            << r.decode_ms << ","
            << r.total_op_ms << ","
            << r.final_card << "\n";
    }

    // Print summary
    std::cout << "\n  Results (" << iterations << " iterations):\n";
    print_stats_line("OR discount (3)", s_disc_or);
    print_stats_line("OR quantity (23)", s_qty_or);
    print_stats_line("AND (3-way)", s_and);
    print_stats_line("Decode", s_decode);
    print_stats_line("Total (ops+decode)", s_total);
    print_separator();
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    std::string base_dir = "bitmap/tpch_q6";
    int iterations = 10;

    if (argc >= 2) base_dir = argv[1];
    if (argc >= 3) iterations = std::stoi(argv[2]);

    std::cout << "========================================\n";
    std::cout << "  TPC-H Q6 Bitmap Benchmark\n";
    std::cout << "  ComBit vs WAH\n";
    std::cout << "========================================\n";
    std::cout << "  Rows:       " << NUM_ROWS << "\n";
    std::cout << "  Bitmap dir: " << base_dir << "\n";
    std::cout << "  Iterations: " << iterations << "\n";

    // Check bitmap directory exists
    if (!fs::is_directory(base_dir)) {
        std::cerr << "Error: bitmap directory not found: " << base_dir << "\n";
        std::cerr << "Run: python3 util/export_tpch_q6.py <duckdb> <db> " << base_dir << "\n";
        return 1;
    }

    // Open CSV output
    std::string csv_path = "results_tpch_q6.csv";
    std::ofstream csv(csv_path);
    csv << "backend,iteration,discount_or_ms,quantity_or_ms,and_ms,"
        << "decode_ms,total_op_ms,result_card\n";

    // Run WAH benchmark
    {
        WahBackend wah;
        run_benchmark(&wah, "WAH", base_dir, iterations, csv);
    }

    // Run ComBit benchmark
    {
        CombitBackend combit;
        run_benchmark(&combit, "ComBit", base_dir, iterations, csv);
    }

    csv.close();
    std::cout << "\nCSV results written to: " << csv_path << "\n";

    return 0;
}
