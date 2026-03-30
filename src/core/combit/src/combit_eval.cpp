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
        auto cb = ComBitBtv<8>::from_string(
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
        check("WS=8  1K  d=0.01", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.10);
        check("WS=8  1K  d=0.10", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.50);
        check("WS=8  1K  d=0.50", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(10000, 0.10);
        check("WS=8  10K d=0.10", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(10000, 0.10);
        check("WS=16 10K d=0.10", roundtrip(b, ComBitBtv<16>::compress(b)));
    }
    {
        auto b = generate_uniform(10000, 0.10);
        check("WS=32 10K d=0.10", roundtrip(b, ComBitBtv<32>::compress(b)));
    }
    {
        std::vector<bool> b;
        check("WS=8  empty", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 0.0);
        check("WS=8  all-zeros", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1000, 1.0);
        check("WS=8  all-ones", roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1003, 0.1);
        check("WS=8  non-aligned (1003 bits)",
              roundtrip(b, ComBitBtv<8>::compress(b)));
    }
    {
        auto b = generate_uniform(1003, 0.1);
        check("WS=16 non-aligned (1003 bits)",
              roundtrip(b, ComBitBtv<16>::compress(b)));
    }
    {
        auto b = generate_uniform(1003, 0.1);
        check("WS=32 non-aligned (1003 bits)",
              roundtrip(b, ComBitBtv<32>::compress(b)));
    }
    std::cout << "\n";

    // --- Bitwise operations -------------------------------------------
    std::cout << "--- Bitwise Operation Tests ---\n";
    auto test_ops = [&](const char* tag, const auto& a_cb, const auto& b_cb,
                        const std::vector<bool>& a_bits,
                        const std::vector<bool>& b_bits) {
        bool ok = true;
        auto check_vec = [&](const std::vector<bool>& got,
                             auto pred) {
            for (size_t i = 0; i < a_bits.size(); i++)
                if (bool(got[i]) != pred(i)) { ok = false; break; }
        };
        check_vec((a_cb & b_cb).decompress(),
                  [&](size_t i) { return bool(a_bits[i]) && bool(b_bits[i]); });
        check_vec((a_cb | b_cb).decompress(),
                  [&](size_t i) { return bool(a_bits[i]) || bool(b_bits[i]); });
        check_vec((a_cb ^ b_cb).decompress(),
                  [&](size_t i) { return bool(a_bits[i]) != bool(b_bits[i]); });
        check_vec((~a_cb).decompress(),
                  [&](size_t i) { return !bool(a_bits[i]); });
        check(std::string(tag) + " AND/OR/XOR/NOT", ok);
    };

    {
        auto a = generate_uniform(1000, 0.3, 1);
        auto b = generate_uniform(1000, 0.3, 2);
        test_ops("WS=8 ", ComBitBtv<8>::compress(a),
                 ComBitBtv<8>::compress(b), a, b);
        test_ops("WS=16", ComBitBtv<16>::compress(a),
                 ComBitBtv<16>::compress(b), a, b);
        test_ops("WS=32", ComBitBtv<32>::compress(a),
                 ComBitBtv<32>::compress(b), a, b);
    }
    std::cout << "\n";

    // --- Popcount -----------------------------------------------------
    std::cout << "--- Popcount Tests ---\n";
    auto test_pop = [&](const char* label, const auto& cb,
                        const std::vector<bool>& bits) {
        size_t expected = 0;
        for (size_t i = 0; i < bits.size(); i++)
            if (bits[i]) expected++;
        check(label, cb.popcount() == expected);
    };
    {
        auto b = generate_uniform(10000, 0.25, 123);
        test_pop("WS=8  popcount", ComBitBtv<8>::compress(b), b);
        test_pop("WS=16 popcount", ComBitBtv<16>::compress(b), b);
        test_pop("WS=32 popcount", ComBitBtv<32>::compress(b), b);
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
        check("WS=8  200K seg=64K", roundtrip(b, ComBit::compress<8>(b)));
        check("WS=16 200K seg=64K", roundtrip(b, ComBit::compress<16>(b)));
        check("WS=32 200K seg=64K", roundtrip(b, ComBit::compress<32>(b)));
        check("WS=64 200K seg=64K", roundtrip(b, ComBit::compress<64>(b)));
    }
    {
        auto b = generate_uniform(200000, 0.10);
        check("WS=8  200K seg=1K",
              roundtrip(b, ComBit::compress<8>(b, false, 1024)));
        check("WS=16 200K seg=1K",
              roundtrip(b, ComBit::compress<16>(b, false, 1024)));
    }
    {
        auto b = generate_uniform(1003, 0.10);
        check("WS=8  non-aligned 1003 seg=256",
              roundtrip(b, ComBit::compress<8>(b, false, 256)));
    }
    {
        std::vector<bool> b;
        check("WS=8  empty", roundtrip(b, ComBit::compress<8>(b)));
    }
    {
        auto b = generate_uniform(100, 0.0);
        check("WS=8  all-zeros seg=32",
              roundtrip(b, ComBit::compress<8>(b, false, 32)));
    }
    std::cout << "\n";

    std::cout << "--- Segment Count Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.10);
        auto cb = ComBit::compress<8>(b, false, 65536);
        check("200K / 64K = 4 segments", cb.num_segments() == 4);
    }
    {
        auto b = generate_uniform(65536, 0.10);
        auto cb = ComBit::compress<8>(b, false, 65536);
        check("64K / 64K = 1 segment", cb.num_segments() == 1);
    }
    {
        auto b = generate_uniform(65537, 0.10);
        auto cb = ComBit::compress<8>(b, false, 65536);
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

        test_seg_ops("WS=8  seg=64K",
                     ComBit::compress<8>(a),  ComBit::compress<8>(b));
        test_seg_ops("WS=16 seg=64K",
                     ComBit::compress<16>(a), ComBit::compress<16>(b));
        test_seg_ops("WS=32 seg=64K",
                     ComBit::compress<32>(a), ComBit::compress<32>(b));
        test_seg_ops("WS=8  seg=1K",
                     ComBit::compress<8>(a, false, 1024),
                     ComBit::compress<8>(b, false, 1024));
    }
    std::cout << "\n";

    std::cout << "--- Segmented Cross-WS AND Tests ---\n";
    {
        auto a = generate_uniform(200000, 0.3, 1);
        auto b = generate_uniform(200000, 0.3, 2);

        auto cb_a8  = ComBit::compress<8>(a);
        auto cb_b16 = ComBit::compress<16>(b);
        auto cb_b32 = ComBit::compress<32>(b);

        auto and_8_16 = (cb_a8 & cb_b16).decompress();
        auto and_8_32 = (cb_a8 & cb_b32).decompress();

        bool ok = true;
        for (size_t i = 0; i < a.size(); i++) {
            if (bool(and_8_16[i]) != (bool(a[i]) && bool(b[i]))) { ok = false; break; }
        }
        check("Cross WS=8 & WS=16 seg=64K", ok);

        ok = true;
        for (size_t i = 0; i < a.size(); i++) {
            if (bool(and_8_32[i]) != (bool(a[i]) && bool(b[i]))) { ok = false; break; }
        }
        check("Cross WS=8 & WS=32 seg=64K", ok);
    }
    std::cout << "\n";

    std::cout << "--- Segmented Popcount Tests ---\n";
    {
        auto b = generate_uniform(200000, 0.25, 123);
        size_t expected = 0;
        for (size_t i = 0; i < b.size(); i++)
            if (b[i]) expected++;

        check("WS=8  seg=64K popcount",
              ComBit::compress<8>(b).popcount() == expected);
        check("WS=16 seg=64K popcount",
              ComBit::compress<16>(b).popcount() == expected);
        check("WS=8  seg=1K  popcount",
              ComBit::compress<8>(b, false, 1024).popcount() == expected);
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit Print Example ---\n";
    {
        auto b = generate_uniform(200000, 0.05);
        auto cb = ComBit::compress<8>(b);
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
              << std::setw(14) << "WS=8"
              << std::setw(14) << "WS=16"
              << std::setw(14) << "WS=32"
              << std::setw(14) << "WS=64" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto bits = generate_uniform(N, d);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << ComBitBtv<8>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<16>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<32>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<64>::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Clustered Distribution (" << N << " bits) ---\n";
    std::cout << std::setw(12) << "Clusters"
              << std::setw(12) << "ClustSz"
              << std::setw(14) << "WS=8"
              << std::setw(14) << "WS=16"
              << std::setw(14) << "WS=32"
              << std::setw(14) << "WS=64" << "\n";
    std::cout << std::string(80, '-') << "\n";

    struct CC { size_t n; size_t sz; };
    for (auto [n, sz] : std::vector<CC>{{10,100},{10,1000},{100,100},
                                         {100,1000},{1000,100}}) {
        auto bits = generate_clustered(N, n, sz);
        std::cout << std::setw(12) << n
                  << std::setw(12) << sz
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << ComBitBtv<8>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<16>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<32>::compress(bits).compression_ratio()
                  << std::setw(14) << ComBitBtv<64>::compress(bits).compression_ratio()
                  << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Size Breakdown (Uniform, d=0.05, " << N << " bits) ---\n";
    auto bits = generate_uniform(N, 0.05);

    auto print_breakdown = [](const auto& cb, unsigned ws) {
        auto sb = cb.size_breakdown();
        std::cout << "  ComBitBtv<" << ws << ">:\n"
                  << "    Leading bits: " << std::setw(10) << sb.leading_bits_count  << " bits\n"
                  << "    Literals:   " << std::setw(10) << sb.literal_bits   << " bits\n"
                  << "    Total:      " << std::setw(10) << sb.total_bits     << " bits ("
                  << std::fixed << std::setprecision(2)
                  << cb.compression_ratio() << "x)\n"
                  << "    Fill words: " << cb.num_fills()
                  << "  Literal words: " << cb.num_literals() << "\n";
    };
    print_breakdown(ComBitBtv<8>::compress(bits), 8);
    print_breakdown(ComBitBtv<16>::compress(bits), 16);
    print_breakdown(ComBitBtv<32>::compress(bits), 32);
    print_breakdown(ComBitBtv<64>::compress(bits), 64);
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

    auto header = [](const char* title) {
        std::cout << "--- " << title << " (100M bits, 3 iterations) ---\n"
                  << std::setw(10) << "Density"
                  << std::setw(14) << "WS=8 (ms)"
                  << std::setw(14) << "WS=16 (ms)"
                  << std::setw(14) << "WS=32 (ms)"
                  << std::setw(14) << "WS=64 (ms)"
                  << std::setw(18) << "Uncompr. (ms)"
                  << "\n" << std::string(84, '-') << "\n";
    };

    header("AND Operation Speed");
    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba),   b8  = ComBitBtv<8>::compress(bb);
        auto a16 = ComBitBtv<16>::compress(ba),  b16 = ComBitBtv<16>::compress(bb);
        auto a32 = ComBitBtv<32>::compress(ba),  b32 = ComBitBtv<32>::compress(bb);
        auto a64 = ComBitBtv<64>::compress(ba),  b64 = ComBitBtv<64>::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double t8   = time_ms([&]{ auto r = a8  & b8;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16  = time_ms([&]{ auto r = a16 & b16;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32  = time_ms([&]{ auto r = a32 & b32;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t64  = time_ms([&]{ auto r = a64 & b64;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_and(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << t8
                  << std::setw(14) << t16
                  << std::setw(14) << t32
                  << std::setw(14) << t64
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Cross-Word-Size AND Speed (" << N << " bits, "
              << ITER << " iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(16) << "8&16 (ms)"
              << std::setw(16) << "8&32 (ms)"
              << std::setw(16) << "8&64 (ms)"
              << std::setw(16) << "16&32 (ms)"
              << std::setw(16) << "16&64 (ms)"
              << std::setw(16) << "32&64 (ms)"
              << "\n" << std::string(106, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba);
        auto a16 = ComBitBtv<16>::compress(ba);
        auto a32 = ComBitBtv<32>::compress(ba);
        auto b16 = ComBitBtv<16>::compress(bb);
        auto b32 = ComBitBtv<32>::compress(bb);
        auto b64 = ComBitBtv<64>::compress(bb);

        double t8_16  = time_ms([&]{ auto r = cross_and(a8, b16);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_32  = time_ms([&]{ auto r = cross_and(a8, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_64  = time_ms([&]{ auto r = cross_and(a8, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_32 = time_ms([&]{ auto r = cross_and(a16, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_64 = time_ms([&]{ auto r = cross_and(a16, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32_64 = time_ms([&]{ auto r = cross_and(a32, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(16) << t8_16
                  << std::setw(16) << t8_32
                  << std::setw(16) << t8_64
                  << std::setw(16) << t16_32
                  << std::setw(16) << t16_64
                  << std::setw(16) << t32_64 << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit AND Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "WS=8 (ms)"
              << std::setw(14) << "WS=16 (ms)"
              << std::setw(14) << "WS=32 (ms)"
              << std::setw(14) << "WS=64 (ms)"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa8  = ComBit::compress<8>(ba),   sb8  = ComBit::compress<8>(bb);
        auto sa16 = ComBit::compress<16>(ba),  sb16 = ComBit::compress<16>(bb);
        auto sa32 = ComBit::compress<32>(ba),  sb32 = ComBit::compress<32>(bb);
        auto sa64 = ComBit::compress<64>(ba),  sb64 = ComBit::compress<64>(bb);

        double st8   = time_ms([&]{ auto r = sa8  & sb8;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st16  = time_ms([&]{ auto r = sa16 & sb16;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st32  = time_ms([&]{ auto r = sa32 & sb32;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st64  = time_ms([&]{ auto r = sa64 & sb64;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st8
                  << std::setw(14) << st16
                  << std::setw(14) << st32
                  << std::setw(14) << st64 << "\n";
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

    auto header = [](const char* title) {
        std::cout << "--- " << title << " (100M bits, 3 iterations) ---\n"
                  << std::setw(10) << "Density"
                  << std::setw(14) << "WS=8 (ms)"
                  << std::setw(14) << "WS=16 (ms)"
                  << std::setw(14) << "WS=32 (ms)"
                  << std::setw(14) << "WS=64 (ms)"
                  << std::setw(18) << "Uncompr. (ms)"
                  << "\n" << std::string(84, '-') << "\n";
    };

    header("OR Operation Speed");
    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba),   b8  = ComBitBtv<8>::compress(bb);
        auto a16 = ComBitBtv<16>::compress(ba),  b16 = ComBitBtv<16>::compress(bb);
        auto a32 = ComBitBtv<32>::compress(ba),  b32 = ComBitBtv<32>::compress(bb);
        auto a64 = ComBitBtv<64>::compress(ba),  b64 = ComBitBtv<64>::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double t8   = time_ms([&]{ auto r = a8  | b8;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16  = time_ms([&]{ auto r = a16 | b16;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32  = time_ms([&]{ auto r = a32 | b32;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t64  = time_ms([&]{ auto r = a64 | b64;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_or(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << t8
                  << std::setw(14) << t16
                  << std::setw(14) << t32
                  << std::setw(14) << t64
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Cross-Word-Size OR Speed (" << N << " bits, "
              << ITER << " iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(16) << "8|16 (ms)"
              << std::setw(16) << "8|32 (ms)"
              << std::setw(16) << "8|64 (ms)"
              << std::setw(16) << "16|32 (ms)"
              << std::setw(16) << "16|64 (ms)"
              << std::setw(16) << "32|64 (ms)"
              << "\n" << std::string(106, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba);
        auto a16 = ComBitBtv<16>::compress(ba);
        auto a32 = ComBitBtv<32>::compress(ba);
        auto b16 = ComBitBtv<16>::compress(bb);
        auto b32 = ComBitBtv<32>::compress(bb);
        auto b64 = ComBitBtv<64>::compress(bb);

        double t8_16  = time_ms([&]{ auto r = cross_or(a8, b16);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_32  = time_ms([&]{ auto r = cross_or(a8, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_64  = time_ms([&]{ auto r = cross_or(a8, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_32 = time_ms([&]{ auto r = cross_or(a16, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_64 = time_ms([&]{ auto r = cross_or(a16, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32_64 = time_ms([&]{ auto r = cross_or(a32, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(16) << t8_16
                  << std::setw(16) << t8_32
                  << std::setw(16) << t8_64
                  << std::setw(16) << t16_32
                  << std::setw(16) << t16_64
                  << std::setw(16) << t32_64 << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit OR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "WS=8 (ms)"
              << std::setw(14) << "WS=16 (ms)"
              << std::setw(14) << "WS=32 (ms)"
              << std::setw(14) << "WS=64 (ms)"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa8  = ComBit::compress<8>(ba),   sb8  = ComBit::compress<8>(bb);
        auto sa16 = ComBit::compress<16>(ba),  sb16 = ComBit::compress<16>(bb);
        auto sa32 = ComBit::compress<32>(ba),  sb32 = ComBit::compress<32>(bb);
        auto sa64 = ComBit::compress<64>(ba),  sb64 = ComBit::compress<64>(bb);

        double st8   = time_ms([&]{ auto r = sa8  | sb8;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st16  = time_ms([&]{ auto r = sa16 | sb16;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st32  = time_ms([&]{ auto r = sa32 | sb32;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st64  = time_ms([&]{ auto r = sa64 | sb64;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st8
                  << std::setw(14) << st16
                  << std::setw(14) << st32
                  << std::setw(14) << st64 << "\n";
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

    auto header = [](const char* title) {
        std::cout << "--- " << title << " (100M bits, 3 iterations) ---\n"
                  << std::setw(10) << "Density"
                  << std::setw(14) << "WS=8 (ms)"
                  << std::setw(14) << "WS=16 (ms)"
                  << std::setw(14) << "WS=32 (ms)"
                  << std::setw(14) << "WS=64 (ms)"
                  << std::setw(18) << "Uncompr. (ms)"
                  << "\n" << std::string(84, '-') << "\n";
    };

    header("XOR Operation Speed");
    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba),   b8  = ComBitBtv<8>::compress(bb);
        auto a16 = ComBitBtv<16>::compress(ba),  b16 = ComBitBtv<16>::compress(bb);
        auto a32 = ComBitBtv<32>::compress(ba),  b32 = ComBitBtv<32>::compress(bb);
        auto a64 = ComBitBtv<64>::compress(ba),  b64 = ComBitBtv<64>::compress(bb);
        auto wa = bools_to_words(ba), wb = bools_to_words(bb);

        double t8   = time_ms([&]{ auto r = a8  ^ b8;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16  = time_ms([&]{ auto r = a16 ^ b16;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32  = time_ms([&]{ auto r = a32 ^ b32;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t64  = time_ms([&]{ auto r = a64 ^ b64;
                                    g_combit_sink += r.compressed_size_bits(); }, ITER);
        double traw = time_ms([&]{ auto r = raw_xor(wa, wb);
                                    g_combit_sink += r.size(); }, ITER);
        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << t8
                  << std::setw(14) << t16
                  << std::setw(14) << t32
                  << std::setw(14) << t64
                  << std::setw(18) << traw << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Cross-Word-Size XOR Speed (" << N << " bits, "
              << ITER << " iterations) ---\n"
              << std::setw(10) << "Density"
              << std::setw(16) << "8^16 (ms)"
              << std::setw(16) << "8^32 (ms)"
              << std::setw(16) << "8^64 (ms)"
              << std::setw(16) << "16^32 (ms)"
              << std::setw(16) << "16^64 (ms)"
              << std::setw(16) << "32^64 (ms)"
              << "\n" << std::string(106, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto a8  = ComBitBtv<8>::compress(ba);
        auto a16 = ComBitBtv<16>::compress(ba);
        auto a32 = ComBitBtv<32>::compress(ba);
        auto b16 = ComBitBtv<16>::compress(bb);
        auto b32 = ComBitBtv<32>::compress(bb);
        auto b64 = ComBitBtv<64>::compress(bb);

        double t8_16  = time_ms([&]{ auto r = cross_xor(a8, b16);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_32  = time_ms([&]{ auto r = cross_xor(a8, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t8_64  = time_ms([&]{ auto r = cross_xor(a8, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_32 = time_ms([&]{ auto r = cross_xor(a16, b32);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t16_64 = time_ms([&]{ auto r = cross_xor(a16, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);
        double t32_64 = time_ms([&]{ auto r = cross_xor(a32, b64);
                                      g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(16) << t8_16
                  << std::setw(16) << t8_32
                  << std::setw(16) << t8_64
                  << std::setw(16) << t16_32
                  << std::setw(16) << t16_64
                  << std::setw(16) << t32_64 << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Segmented ComBit XOR Speed (" << N << " bits, "
              << ITER << " iterations, seg=64K) ---\n"
              << std::setw(10) << "Density"
              << std::setw(14) << "WS=8 (ms)"
              << std::setw(14) << "WS=16 (ms)"
              << std::setw(14) << "WS=32 (ms)"
              << std::setw(14) << "WS=64 (ms)"
              << "\n" << std::string(66, '-') << "\n";

    for (double d : densities) {
        auto ba = generate_uniform(N, d, 1);
        auto bb = generate_uniform(N, d, 2);
        auto sa8  = ComBit::compress<8>(ba),   sb8  = ComBit::compress<8>(bb);
        auto sa16 = ComBit::compress<16>(ba),  sb16 = ComBit::compress<16>(bb);
        auto sa32 = ComBit::compress<32>(ba),  sb32 = ComBit::compress<32>(bb);
        auto sa64 = ComBit::compress<64>(ba),  sb64 = ComBit::compress<64>(bb);

        double st8   = time_ms([&]{ auto r = sa8  ^ sb8;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st16  = time_ms([&]{ auto r = sa16 ^ sb16;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st32  = time_ms([&]{ auto r = sa32 ^ sb32;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);
        double st64  = time_ms([&]{ auto r = sa64 ^ sb64;
                                     g_combit_sink += r.compressed_size_bits(); }, ITER);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(10) << d
                  << std::setw(14) << st8
                  << std::setw(14) << st16
                  << std::setw(14) << st32
                  << std::setw(14) << st64 << "\n";
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
