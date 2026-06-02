#include "ddc.h"
#include "ddc_util.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdint>

// roundtrip + ops correctness
static void run_correctness_tests() {
    std::cout << "========================================\n"
              << "  Correctness Tests (DDCBtv)\n"
              << "========================================\n\n";

    int pass = 0, fail = 0;
    auto check = [&](const std::string& name, bool ok) {
        std::cout << (ok ? "  PASS: " : "  FAIL: ") << name << "\n";
        if (ok) pass++; else fail++;
    };

    {
        std::cout << "--- Specification Example ---\n";
        auto cb = DDCBtv::from_string(
            "00000000 00000000 00001000 00000000 00000000 00000001");
        cb.print();
        std::cout << "\n";

        bool ok = true;
        ok &= (cb.l2_count() == 6);
        ok &= (cb.num_lits() == 2 &&
                cb.get_literal(0) == 0x08 &&
                cb.get_literal(1) == 0x01);
        ok &= (cb.to_string() ==
                "00000000 00000000 00001000 00000000 00000000 00000001");
        check("Specification example matches", ok);
    }
    std::cout << "\n";

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
        check("1K  d=0.01", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.10);
        check("1K  d=0.10", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.50);
        check("1K  d=0.50", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(10000, 0.10);
        check("10K d=0.10", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        std::vector<bool> b;
        check("empty", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.0);
        check("all-zeros", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 1.0);
        check("all-ones", roundtrip(b, DDCBtv::compress(b)));
    }
    {
        auto b = generate_uniform(1003, 0.1);
        check("non-aligned (1003 bits)",
              roundtrip(b, DDCBtv::compress(b)));
    }
    std::cout << "\n";

    // AND/OR/XOR/NOT vs reference
    std::cout << "--- Bitwise Operation Tests ---\n";
    {
        auto a = generate_uniform(1000, 0.3, 1);
        auto b = generate_uniform(1000, 0.3, 2);
        auto a_cb = DDCBtv::compress(a);
        auto b_cb = DDCBtv::compress(b);

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

    std::cout << "--- Popcount Tests ---\n";
    {
        auto b = generate_uniform(10000, 0.25, 123);
        size_t expected = 0;
        for (size_t i = 0; i < b.size(); i++)
            if (b[i]) expected++;
        check("popcount", DDCBtv::compress(b).popcount() == expected);
    }
    std::cout << "\n";

    std::cout << "Results: " << pass << " passed, " << fail << " failed\n\n";
}

// segmented container correctness
static void run_segmented_correctness_tests() {
    std::cout << "========================================\n"
              << "  Segmented DDC Correctness Tests\n"
              << "========================================\n\n";

    int pass = 0, fail = 0;
    auto check = [&](const std::string& name, bool ok) {
        std::cout << (ok ? "  PASS: " : "  FAIL: ") << name << "\n";
        if (ok) pass++; else fail++;
    };

    auto roundtrip = [](const std::vector<bool>& bits, const DDC& cb) {
        auto dec = cb.decompress();
        if (bits.size() != dec.size()) return false;
        for (size_t i = 0; i < bits.size(); i++)
            if (bool(bits[i]) != bool(dec[i])) return false;
        return true;
    };

    std::cout << "--- Segmented Round-trip Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.10);
        check("200K seg=64K", roundtrip(b, DDC::compress(b)));
    }
    {
        auto b = generate_uniform(200000, 0.10);
        check("200K seg=1K",
              roundtrip(b, DDC::compress(b, false, 1024)));
    }
    {
        auto b = generate_uniform(1003, 0.10);
        check("non-aligned 1003 seg=256",
              roundtrip(b, DDC::compress(b, false, 256)));
    }
    {
        std::vector<bool> b;
        check("empty", roundtrip(b, DDC::compress(b)));
    }
    {
        auto b = generate_uniform(100, 0.0);
        check("all-zeros seg=32",
              roundtrip(b, DDC::compress(b, false, 32)));
    }
    std::cout << "\n";

    std::cout << "--- Segment Count Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.10);
        auto cb = DDC::compress(b, false, 65536);
        check("200K / 64K = 4 segments", cb.num_segments() == 4);
    }
    {
        auto b = generate_uniform(65536, 0.10);
        auto cb = DDC::compress(b, false, 65536);
        check("64K / 64K = 1 segment", cb.num_segments() == 1);
    }
    {
        auto b = generate_uniform(65537, 0.10);
        auto cb = DDC::compress(b, false, 65536);
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
                     DDC::compress(a),  DDC::compress(b));
        test_seg_ops("seg=1K",
                     DDC::compress(a, false, 1024),
                     DDC::compress(b, false, 1024));
    }
    std::cout << "\n";

    std::cout << "--- Segmented Popcount Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.25, 123);
        size_t expected = 0;
        for (size_t i = 0; i < b.size(); i++)
            if (b[i]) expected++;

        check("seg=64K popcount",
              DDC::compress(b).popcount() == expected);
        check("seg=1K  popcount",
              DDC::compress(b, false, 1024).popcount() == expected);
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC Print Example ---\n";
    {
        auto b = generate_uniform(200000, 0.05);
        auto cb = DDC::compress(b);
        cb.print();
    }
    std::cout << "\n";

    std::cout << "Results: " << pass << " passed, " << fail << " failed\n\n";
}

// compression ratio sweep
static void run_compression_analysis() {
    std::cout << "========================================\n"
              << "  Compression Ratio Analysis (DDCBtv)\n"
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
                  << std::setw(14) << DDCBtv::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    // clustered distribution
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
                  << std::setw(14) << DDCBtv::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    // per-level size breakdown
    std::cout << "--- Size Breakdown (Uniform, d=0.05, " << N << " bits) ---\n";
    auto bits = generate_uniform(N, 0.05);
    auto cb = DDCBtv::compress(bits);
    auto sb = cb.size_breakdown();
    std::cout << "  DDCBtv (4-level: L4/L3-literals/L2-literals/L1-literals):\n"
              << "    L4 bits:    " << std::setw(10) << sb.l4_bits          << " bits\n"
              << "    L3 literals:" << std::setw(10) << sb.l3_literal_bits  << " bits\n"
              << "    L2 literals:" << std::setw(10) << sb.l2_literal_bits  << " bits\n"
              << "    L1 literals:" << std::setw(10) << sb.l1_literal_bits  << " bits\n"
              << "    Total:      " << std::setw(10) << sb.total_bits       << " bits ("
              << std::fixed << std::setprecision(2)
              << cb.compression_ratio() << "x)\n"
              << "    Fill words: " << cb.num_fills()
              << "  Literal words: " << cb.num_lits() << "\n";
    std::cout << "\n";
}

// AND speed vs raw words
static void run_AND_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  AND Performance Benchmarks (DDCBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- AND Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a & b;
                                   g_ddc_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_and(wa, wb);
                                    g_ddc_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC AND Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(48, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        // segmented AND
        double st = time_ms([&]{ auto r = sa & sb;
                                  g_ddc_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// OR speed vs raw words
static void run_OR_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  OR Performance Benchmarks (DDCBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- OR Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a | b;
                                   g_ddc_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_or(wa, wb);
                                    g_ddc_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC OR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(48, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        double st = time_ms([&]{ auto r = sa | sb;
                                  g_ddc_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// XOR speed vs raw words
static void run_XOR_performance_benchmarks() {
    std::cout << "========================================\n"
              << "  XOR Performance Benchmarks (DDCBtv)\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- XOR Operation Speed (100M bits, 3 iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(18) << "Uncompr. (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double tc  = time_ms([&]{ auto r = a ^ b;
                                   g_ddc_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_xor(wa, wb);
                                    g_ddc_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << tc
                  << std::setw(18) << traw
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC XOR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "DDC (ms)"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(48, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        double st = time_ms([&]{ auto r = sa ^ sb;
                                  g_ddc_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// AND: raw vs recompressed result
static void run_AND_compressed_result_benchmarks() {
    std::cout << "========================================\n"
              << "  AND -> Compressed Result Benchmarks\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- AND (100M bits, 3 iter): uncompressed vs compressed result ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "R Ratio"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(92, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);

        // result left expanded
        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = a & b;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        // result recompressed
        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = a & b;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        auto r_comp = a & b;
        double r_ratio = r_comp.compression_ratio();

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(10) << r_ratio
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC AND: compressed result (" << N << " bits, "
              << ITER << " iter, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(82, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = sa & sb;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = sa & sb;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// OR: raw vs recompressed result
static void run_OR_compressed_result_benchmarks() {
    std::cout << "========================================\n"
              << "  OR -> Compressed Result Benchmarks\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- OR (100M bits, 3 iter): uncompressed vs compressed result ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "R Ratio"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(92, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);

        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = a | b;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = a | b;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        auto r_comp = a | b;
        double r_ratio = r_comp.compression_ratio();

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(10) << r_ratio
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC OR: compressed result (" << N << " bits, "
              << ITER << " iter, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(82, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = sa | sb;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = sa | sb;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// XOR: raw vs recompressed result
static void run_XOR_compressed_result_benchmarks() {
    std::cout << "========================================\n"
              << "  XOR -> Compressed Result Benchmarks\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};

    std::cout << "--- XOR (100M bits, 3 iter): uncompressed vs compressed result ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(10) << "R Ratio"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(92, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a = DDCBtv::compress(ba), b = DDCBtv::compress(bb);

        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = a ^ b;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = a ^ b;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        auto r_comp = a ^ b;
        double r_ratio = r_comp.compression_ratio();

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(10) << r_ratio
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (a.l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (a.l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented DDC XOR: compressed result (" << N << " bits, "
              << ITER << " iter, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(18) << "Uncompr.R (ms)"
              << std::setw(18) << "Compr.R (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(8)  << "L3"
              << std::setw(8)  << "L1Fill"
              << std::setw(8)  << "L2Fill"
              << "\n" << std::string(82, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa = DDC::compress(ba), sb = DDC::compress(bb);

        ddc_compress_results = false;
        double t_uncomp = time_ms([&]{ auto r = sa ^ sb;
                                        g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = true;
        double t_comp = time_ms([&]{ auto r = sa ^ sb;
                                      g_ddc_sink += r.compressed_size_bits(); }, ITER);

        ddc_compress_results = false;

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(18) << t_uncomp
                  << std::setw(18) << t_comp
                  << std::setw(12) << (t_uncomp / t_comp)
                  << std::setw(8)  << "yes"
                  << std::setw(8)  << (sa.segment(0).l1_fill_ones() ? "1" : "0")
                  << std::setw(8)  << (sa.segment(0).l2_fill_ones() ? "Literal" : "Fill")
                  << "\n";
    }
    std::cout << "\n";
}

// OR_many vs chained OR
static void run_OR_many_benchmarks() {
    std::cout << "========================================\n"
              << "  OR_many vs Sequential OR Benchmarks\n"
              << "========================================\n\n";

    const size_t N = 100'000'000;
    const int ITER = 3;
    const std::vector<double> densities = {0.0001, 0.001, 0.01, 0.10, 0.50};
    const std::vector<size_t> counts = {2, 4, 8, 16, 32};

    for (size_t count : counts) {
        std::cout << "--- OR of " << count << " bitvectors ("
                  << N << " bits, " << ITER << " iter, seg=64K) ---\n"
                  << std::setw(10) << "Density"
                  << std::setw(18) << "Sequential (ms)"
                  << std::setw(18) << "OR_many (ms)"
                  << std::setw(12) << "Speedup"
                  << "\n" << std::string(58, '-') << "\n";

        for (double d : densities) {
            std::vector<DDC> btvs;
            std::vector<const DDC*> ptrs;
            btvs.reserve(count);
            for (size_t k = 0; k < count; k++)
                btvs.push_back(DDC::compress(
                    generate_uniform(N, d, k + 1)));
            for (size_t k = 0; k < count; k++)
                ptrs.push_back(&btvs[k]);

            // chained pairwise OR
            double t_seq = time_ms([&]{
                DDC acc = btvs[0] | btvs[1];
                for (size_t k = 2; k < count; k++)
                    acc |= btvs[k];
                g_ddc_sink += acc.compressed_size_bits();
            }, ITER);

            // k-way merge
            double t_many = time_ms([&]{
                auto r = DDC::OR_many(count, ptrs.data());
                g_ddc_sink += r.compressed_size_bits();
            }, ITER);

            std::cout << std::fixed << std::setprecision(3)
                      << std::setw(10) << d
                      << std::setw(18) << t_seq
                      << std::setw(18) << t_many
                      << std::setw(12) << (t_seq / t_many)
                      << "\n";
        }
        std::cout << "\n";
    }
}

int main(int  , char**  ) {
    std::cout << "\n"
              << "========================================\n"
              << "  DDC Compression Evaluation\n"
              << "========================================\n\n";

    run_correctness_tests();
    run_segmented_correctness_tests();
    run_compression_analysis();

    run_AND_performance_benchmarks();
    run_AND_compressed_result_benchmarks();

    run_OR_performance_benchmarks();
    run_OR_compressed_result_benchmarks();

    run_XOR_performance_benchmarks();
    run_XOR_compressed_result_benchmarks();

    run_OR_many_benchmarks();

    return 0;
}
