#include "combit.h"
#include "combit_util.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// 1. Correctness Tests (ComBitBtv - segment level)
// ---------------------------------------------------------------------------

static void run_correctness_tests() {
    std::cout << "========================================\n"
              << "  Correctness Tests (ComBitBtv)\n"
              << "========================================\n\n";

    int pass = 0, fail = 0;
    auto check = [&](const std::string& name, bool ok) {
        std::cout << (ok ? "  PASS: " : "  FAIL: ") << name << "\n";
        if (ok) pass++; else fail++;
    };

    // --- Specification example -----------------------------------------
    {
        std::cout << "--- Specification Example ---\n";
        auto cb = ComBitBtv::from_string(
            "00000000 00000000 00001000 00000000 00000000 00000001");
        cb.print();
        std::cout << "\n";

        bool ok = true;
        ok &= (cb.leading_bits_count() == 6);
        ok &= (cb.is_fill(0) == true  && cb.is_fill(1) == true  &&
                cb.is_fill(2) == false && cb.is_fill(3) == true  &&
                cb.is_fill(4) == true  && cb.is_fill(5) == false);
        ok &= (cb.num_literals() == 2 &&
                cb.get_literal(0) == 0x08 &&
                cb.get_literal(1) == 0x01);
        ok &= (cb.to_string() ==
                "00000000 00000000 00001000 00000000 00000000 00000001");
        check("Specification example matches", ok);
    }
    std::cout << "\n";

    // --- Round-trip tests ----------------------------------------------
    std::cout << "--- Round-trip Tests ---\n";
    auto roundtrip = [](const std::vector<bool>& bits, const auto& cb) {
        auto dec = cb.decompress();
        if (bits.size() != dec.size()) return false;
        for (size_t i = 0; i < bits.size(); i++)
            if (bool(bits[i]) != bool(dec[i])) return false;
        return true;
    };

    {
        auto b = generate_uniform(1000, 0.01);
        check("1K  d=0.01", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.10);
        check("1K  d=0.10", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.50);
        check("1K  d=0.50", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(10000, 0.10);
        check("10K d=0.10", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        std::vector<bool> b;
        check("empty", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.0);
        check("all-zeros", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 1.0);
        check("all-ones", roundtrip(b, ComBitBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1003, 0.1);
        check("non-aligned (1003 bits)",
              roundtrip(b, ComBitBtv::compress(b)));
    }
    std::cout << "\n";

    // --- Bitwise operations -------------------------------------------
    std::cout << "--- Bitwise Operation Tests ---\n";
    {
        auto a = generate_uniform(1000, 0.3, 1);
        auto b = generate_uniform(1000, 0.3, 2);
        auto a_cb = ComBitBtv::compress(a);
        auto b_cb = ComBitBtv::compress(b);

        bool ok = true;
        auto check_vec = [&](const std::vector<bool>& got,
                             auto pred) {
            for (size_t i = 0; i < a.size(); i++)
                if (bool(got[i]) != pred(i)) { ok = false; break; }
        };
        check_vec((a_cb & b_cb).decompress(),
                  [&](size_t i) { return bool(a[i]) && bool(b[i]); });
        check_vec((a_cb | b_cb).decompress(),
                  [&](size_t i) { return bool(a[i]) || bool(b[i]); });
        check_vec((a_cb ^ b_cb).decompress(),
                  [&](size_t i) { return bool(a[i]) != bool(b[i]); });
        check_vec((~a_cb).decompress(),
                  [&](size_t i) { return !bool(a[i]); });
        check("AND/OR/XOR/NOT", ok);
    }
    std::cout << "\n";

    // --- Popcount -----------------------------------------------------
    std::cout << "--- Popcount Tests ---\n";
    {
        auto b = generate_uniform(10000, 0.25, 123);
        size_t expected = 0;
        for (size_t i = 0; i < b.size(); i++)
            if (b[i]) expected++;
        check("popcount", ComBitBtv::compress(b).popcount() == expected);
    }
    std::cout << "\n";

    std::cout << "Results: " << pass << " passed, " << fail << " failed\n\n";
}

// ---------------------------------------------------------------------------
// 2. Segmented ComBit Correctness Tests
// ---------------------------------------------------------------------------

static void run_segmented_correctness_tests() {
    std::cout << "========================================\n"
              << "  Segmented ComBit Correctness Tests\n"
              << "========================================\n\n";

    int pass = 0, fail = 0;
    auto check = [&](const std::string& name, bool ok) {
        std::cout << (ok ? "  PASS: " : "  FAIL: ") << name << "\n";
        if (ok) pass++; else fail++;
    };

    auto roundtrip = [](const std::vector<bool>& bits, const ComBit& cb) {
        auto dec = cb.decompress();
        if (bits.size() != dec.size()) return false;
        for (size_t i = 0; i < bits.size(); i++)
            if (bool(bits[i]) != bool(dec[i])) return false;
        return true;
    };

    std::cout << "--- Segmented Round-trip Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.10);
        check("200K seg=64K", roundtrip(b, ComBit::compress(b)));
    }
    {
        auto b = generate_uniform(200000, 0.10);
        check("200K seg=1K",
              roundtrip(b, ComBit::compress(b, false, 1024)));
    }
    {
        auto b = generate_uniform(1003, 0.10);
        check("non-aligned 1003 seg=256",
              roundtrip(b, ComBit::compress(b, false, 256)));
    }
    {
        std::vector<bool> b;
        check("empty", roundtrip(b, ComBit::compress(b)));
    }
    {
        auto b = generate_uniform(100, 0.0);
        check("all-zeros seg=32",
              roundtrip(b, ComBit::compress(b, false, 32)));
    }
    std::cout << "\n";

    std::cout << "--- Segment Count Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.10);
        auto cb = ComBit::compress(b, false, 65536);
        check("200K / 64K = 4 segments", cb.num_segments() == 4);
    }
    {
        auto b = generate_uniform(65536, 0.10);
        auto cb = ComBit::compress(b, false, 65536);
        check("64K / 64K = 1 segment", cb.num_segments() == 1);
    }
    {
        auto b = generate_uniform(65537, 0.10);
        auto cb = ComBit::compress(b, false, 65536);
        check("64K+1 / 64K = 2 segments", cb.num_segments() == 2);
    }
    std::cout << "\n";

    std::cout << "--- Segmented Bitwise Operation Tests ---\n";
    {
        auto a = generate_uniform(200000, 0.3, 1);
        auto b = generate_uniform(200000, 0.3, 2);

        auto test_seg_ops = [&](const char* tag, auto cb_a, auto cb_b) {
            bool ok = true;
            auto and_bits = (cb_a & cb_b).decompress();
            auto or_bits  = (cb_a | cb_b).decompress();
            auto xor_bits = (cb_a ^ cb_b).decompress();
            auto not_bits = (~cb_a).decompress();

            for (size_t i = 0; i < a.size(); i++) {
                if (bool(and_bits[i]) != (bool(a[i]) && bool(b[i]))) { ok = false; break; }
                if (bool(or_bits[i])  != (bool(a[i]) || bool(b[i]))) { ok = false; break; }
                if (bool(xor_bits[i]) != (bool(a[i]) != bool(b[i]))) { ok = false; break; }
                if (bool(not_bits[i]) != !bool(a[i]))                { ok = false; break; }
            }
            check(std::string(tag) + " AND/OR/XOR/NOT", ok);
        };

        test_seg_ops("seg=64K",
                     ComBit::compress(a),  ComBit::compress(b));
        test_seg_ops("seg=1K",
                     ComBit::compress(a, false, 1024),
                     ComBit::compress(b, false, 1024));
    }
    std::cout << "\n";

    std::cout << "--- Segmented Popcount Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.25, 123);
        size_t expected = 0;
        for (size_t i = 0; i < b.size(); i++)
            if (b[i]) expected++;

        check("seg=64K popcount",
              ComBit::compress(b).popcount() == expected);
        check("seg=1K  popcount",
              ComBit::compress(b, false, 1024).popcount() == expected);
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit Print Example ---\n";
    {
        auto b = generate_uniform(200000, 0.05);
        auto cb = ComBit::compress(b);
        cb.print();
    }
    std::cout << "\n";

    std::cout << "Results: " << pass << " passed, " << fail << " failed\n\n";
}

// ---------------------------------------------------------------------------
// 3. Compression Ratio Analysis
// ---------------------------------------------------------------------------

static void run_compression_analysis() {
    std::cout << "========================================\n"
              << "  Compression Ratio Analysis (ComBitBtv)\n"
              << "========================================\n\n";

    const size_t N = 1'000'000;
    const std::vector<double> densities =
        {0.001, 0.005, 0.01, 0.05, 0.10, 0.25, 0.50};

    std::cout << "--- Uniform Random Distribution (" << N << " bits) ---\n";
    std::cout << std::setw(10) << "Density"
              << std::setw(14) << "Ratio" << "\n";
    std::cout << std::string(24, '-') << "\n";

    for (double d : densities) {
        auto bits = generate_uniform(N, d);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << ComBitBtv::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Clustered Distribution (" << N << " bits) ---\n";
    std::cout << std::setw(12) << "Clusters"
              << std::setw(12) << "ClustSz"
              << std::setw(14) << "Ratio" << "\n";
    std::cout << std::string(38, '-') << "\n";

    struct CC { size_t n; size_t sz; };
    for (auto [n, sz] : std::vector<CC>{{10,100},{10,1000},{100,100},
                                         {100,1000},{1000,100}}) {
        auto bits = generate_clustered(N, n, sz);
        std::cout << std::setw(12) << n
                  << std::setw(12) << sz
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << ComBitBtv::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Size Breakdown (Uniform, d=0.05, " << N << " bits) ---\n";
    auto bits = generate_uniform(N, 0.05);
    auto cb = ComBitBtv::compress(bits);
    auto sb = cb.size_breakdown();
    std::cout << "  ComBitBtv:\n"
              << "    Leading bits: " << std::setw(10) << sb.leading_bits_count  << " bits\n"
              << "    Literals:   " << std::setw(10) << sb.literal_bits   << " bits\n"
              << "    Total:      " << std::setw(10) << sb.total_bits     << " bits ("
              << std::fixed << std::setprecision(2)
              << cb.compression_ratio() << "x)\n"
              << "    Fill words: " << cb.num_fills()
              << "  Literal words: " << cb.num_literals() << "\n";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 4. AND Performance Benchmarks
// ---------------------------------------------------------------------------

static void run_AND_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  AND Performance Benchmarks (ComBitBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- AND Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << "\n" << std::string(42, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = ComBitBtv::compress(ba), b = ComBitBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a & b;
                                   g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_and(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit AND Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << "\n" << std::string(24, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = ComBit::compress(ba), sb = ComBit::compress(bb);

        double st = time_ms([&]{ auto r = sa & sb;
                                  g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 5. OR Performance Benchmarks
// ---------------------------------------------------------------------------

static void run_OR_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  OR Performance Benchmarks (ComBitBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- OR Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << "\n" << std::string(42, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = ComBitBtv::compress(ba), b = ComBitBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a | b;
                                   g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_or(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit OR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << "\n" << std::string(24, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = ComBit::compress(ba), sb = ComBit::compress(bb);

        double st = time_ms([&]{ auto r = sa | sb;
                                  g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// 6. XOR Performance Benchmarks
// ---------------------------------------------------------------------------

static void run_XOR_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  XOR Performance Benchmarks (ComBitBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- XOR Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << "\n" << std::string(42, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = ComBitBtv::compress(ba), b = ComBitBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a ^ b;
                                   g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_xor(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit XOR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "ComBit (ms)"
              << "\n" << std::string(24, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = ComBit::compress(ba), sb = ComBit::compress(bb);

        double st = time_ms([&]{ auto r = sa ^ sb;
                                  g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st << "\n";
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "\n"
              << "========================================\n"
              << "  ComBit Compression Evaluation\n"
              << "========================================\n\n";

    run_correctness_tests();
    run_segmented_correctness_tests();
    run_compression_analysis();
    run_AND_performance_benchmarks();
    run_OR_performance_benchmarks();
    run_XOR_performance_benchmarks();

    return 0;
}
