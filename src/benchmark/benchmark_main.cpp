#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include "uti.h"
#include "backends/combit/combit_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"
#include "backends/ewah/ewah_backend.h"
#include "backends/Concise/concise_backend.h"
#include "backends/bitset/bitset_backend.h"
#include "backends/bitset_avx512/bitset_avx512_backend.h"
#include <bitset_simd.hpp>
#include <segmented_bitset.hpp>

namespace fs = std::filesystem;

// ==========================================
// CRoaring helper: ensure result is in bitset form
// ==========================================
// After a logical operation, if the result has array or run containers,
// convert to a flat bitset (simulating "I need the result as btv").
// Returns the conversion time in ms (0 if already all-bitset).
static double croaring_to_bitset_if_needed(const roaring::Roaring& r,
                                           uint32_t logical_size) {
    roaring::api::roaring_statistics_t stats;
    roaring::api::roaring_bitmap_statistics(&r.roaring, &stats);
    if (stats.n_array_containers == 0 && stats.n_run_containers == 0)
        return 0.0;  // already all bitset containers, no conversion needed

    Timer conv_timer;
    conv_timer.reset();
    roaring::api::bitset_t* bs = roaring::api::bitset_create_with_capacity(
        logical_size);
    roaring::api::roaring_bitmap_to_bitset(&r.roaring, bs);
    roaring::api::bitset_free(bs);
    return conv_timer.elapsed_ms();
}

// Print container breakdown for a CRoaring result
static void croaring_print_containers(const std::string& label,
                                      const roaring::Roaring& r) {
    roaring::api::roaring_statistics_t s;
    roaring::api::roaring_bitmap_statistics(&r.roaring, &s);
    std::cout << "  [" << label << " containers] array=" << s.n_array_containers
              << " bitset=" << s.n_bitset_containers
              << " run=" << s.n_run_containers << "\n";
}

// ==========================================
// 1. BM File Benchmark: Load raw .bm files
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
    std::sort(bm_files.begin(), bm_files.end(), [](const std::string& a, const std::string& b){
        auto sa = fs::path(a).stem().string();
        auto sb = fs::path(b).stem().string();
        // column/ files sort after top-level files
        bool ca = a.find("/column/") != std::string::npos;
        bool cb = b.find("/column/") != std::string::npos;
        if (ca != cb) return !ca;
        return std::stoi(sa) < std::stoi(sb);
    });

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

    // --- Phase 2: Logical ops (pairwise on first 2 bitmaps) ---
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

        // For CRoaring: add to-bitset conversion time if result has array/run containers
        if (backend_name == "CRoaring") {
            auto* ha_c = dynamic_cast<CroaringHandle*>(a.get());
            auto* hb_c = dynamic_cast<CroaringHandle*>(b.get());
            if (ha_c && hb_c) {
                uint32_t logical_sz = std::max(ha_c->current_size, hb_c->current_size);
                auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
                auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
                auto* xor_h = dynamic_cast<CroaringHandle*>(xor_res.get());
                if (or_h)  or_t  += croaring_to_bitset_if_needed(or_h->bitmap, logical_sz);
                if (and_h) and_t += croaring_to_bitset_if_needed(and_h->bitmap, logical_sz);
                if (xor_h) xor_t += croaring_to_bitset_if_needed(xor_h->bitmap, logical_sz);
            }
        }

        std::cout << "\n[Compute] Pairwise ops on " << na << " & " << nb << ":\n";
        std::cout << "  bitOr:  " << or_t  << " ms (card: " << backend->Cardinality(*or_res)  << ")\n";
        std::cout << "  bitAnd: " << and_t << " ms (card: " << backend->Cardinality(*and_res) << ")\n";
        std::cout << "  bitXor: " << xor_t << " ms (card: " << backend->Cardinality(*xor_res) << ")\n";

        // --- Pure ops timing (operation-only, no allocation overhead) ---
        // Bitset (Plain)
        if (backend_name.find("Plain") != std::string::npos) {
            auto* ha = dynamic_cast<BitsetHandle*>(a.get());
            auto* hb = dynamic_cast<BitsetHandle*>(b.get());
            if (ha && hb) {
                size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());
                bitset::BitsetVector result_buf;
                result_buf.allocate_nozero(n);
                bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);

                timer.reset();
                bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                double and_pre = timer.elapsed_ms();
                timer.reset();
                bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                double or_pre = timer.elapsed_ms();

                bitset::BitsetVector a_copy = ha->btv;
                timer.reset();
                bitset::simd::words_and_inplace_scalar(a_copy.words_mut(), hb->btv.words(), n);
                double and_inplace = timer.elapsed_ms();

                std::cout << "\n[Pure Ops] Bitset (Plain) operation-only:\n";
                std::cout << "  AND (pre-alloc): " << and_pre << " ms\n";
                std::cout << "  OR  (pre-alloc): " << or_pre  << " ms\n";
                std::cout << "  AND (in-place):  " << and_inplace << " ms\n";
            }
        }
        // CRoaring
        if (backend_name == "CRoaring") {
            auto* ha = dynamic_cast<CroaringHandle*>(a.get());
            auto* hb = dynamic_cast<CroaringHandle*>(b.get());
            if (ha && hb) {
                uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                { roaring::Roaring w = ha->bitmap & hb->bitmap; (void)w; }

                timer.reset();
                roaring::Roaring and_r = ha->bitmap & hb->bitmap;
                double and_pure = timer.elapsed_ms();
                and_pure += croaring_to_bitset_if_needed(and_r, logical_sz);

                timer.reset();
                roaring::Roaring or_r = ha->bitmap | hb->bitmap;
                double or_pure = timer.elapsed_ms();
                or_pure += croaring_to_bitset_if_needed(or_r, logical_sz);

                timer.reset();
                roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
                double xor_pure = timer.elapsed_ms();
                xor_pure += croaring_to_bitset_if_needed(xor_r, logical_sz);

                roaring::Roaring a_copy = ha->bitmap;
                timer.reset();
                a_copy &= hb->bitmap;
                double and_inplace = timer.elapsed_ms();
                and_inplace += croaring_to_bitset_if_needed(a_copy, logical_sz);

                std::cout << "\n[Pure Ops] CRoaring operation-only (+ to-bitset if needed):\n";
                std::cout << "  AND: " << and_pure << " ms (card: " << and_r.cardinality() << ")\n";
                std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.cardinality() << ")\n";
                std::cout << "  XOR: " << xor_pure << " ms (card: " << xor_r.cardinality() << ")\n";
                std::cout << "  AND (in-place): " << and_inplace << " ms\n";
                croaring_print_containers("AND result", and_r);
                croaring_print_containers("OR  result", or_r);
                croaring_print_containers("XOR result", xor_r);
                croaring_print_containers("Input A", ha->bitmap);
                croaring_print_containers("Input B", hb->bitmap);
            }
        }
    }

    // --- Phase 3: Multi-way OR over column bitmaps ---
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
// 2. Pre-compressed .bm Benchmark
// ==========================================
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

            // ComBit size breakdown: leading bits + literal words
            if (backend_name.find("ComBIT") != std::string::npos) {
                size_t total_lb = 0, total_lit = 0, total_cb = 0;
                for (auto& bm : bitmaps) {
                    auto* ch = dynamic_cast<CombitHandle*>(bm.get());
                    if (ch) {
                        auto sb = ch->compressed.size_breakdown();
                        total_lb  += sb.l3_bits;
                        total_lit += sb.l1_literal_bits;
                        total_cb  += sb.total_bits;
                    }
                }
                std::cout << "[ComBit Storage] leading bits: " << total_lb
                          << " bits (" << total_lb / 8.0 / 1024.0 << " KB)"
                          << " | literal words: " << total_lit
                          << " bits (" << total_lit / 8.0 / 1024.0 << " KB)"
                          << " | total: " << total_cb
                          << " bits (" << total_cb / 8.0 / 1024.0 << " KB)\n";
            }
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
            double or_elapsed = timer.elapsed_ms();

            timer.reset();
            auto and_res = backend->bitAnd(*a, *b);
            double and_elapsed = timer.elapsed_ms();

            timer.reset();
            auto xor_res = backend->bitXor(*a, *b);
            double xor_elapsed = timer.elapsed_ms();

            // For CRoaring: add to-bitset conversion time if result has array/run containers
            if (backend_name == "CRoaring") {
                auto* ha = dynamic_cast<CroaringHandle*>(a.get());
                auto* hb = dynamic_cast<CroaringHandle*>(b.get());
                if (ha && hb) {
                    uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                    auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
                    auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
                    auto* xor_h = dynamic_cast<CroaringHandle*>(xor_res.get());
                    if (or_h)  or_elapsed  += croaring_to_bitset_if_needed(or_h->bitmap, logical_sz);
                    if (and_h) and_elapsed += croaring_to_bitset_if_needed(and_h->bitmap, logical_sz);
                    if (xor_h) xor_elapsed += croaring_to_bitset_if_needed(xor_h->bitmap, logical_sz);
                }
            }

            or_times.push_back(or_elapsed);
            or_card = backend->Cardinality(*or_res);

            and_times.push_back(and_elapsed);
            and_card = backend->Cardinality(*and_res);

            xor_times.push_back(xor_elapsed);
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

    // ============================================================
    // Phase 4: Pure operation timing (operation-only, no alloc)
    // ============================================================
    {
        auto bm0 = backend->Load(selected[0]);
        auto bm1 = backend->Load(selected[1]);
        if (bm0 && bm1) {
            // --- Bitset (Plain) pure ops ---
            if (backend_name.find("Plain") != std::string::npos) {
                auto* ha = dynamic_cast<BitsetHandle*>(bm0.get());
                auto* hb = dynamic_cast<BitsetHandle*>(bm1.get());
                if (ha && hb) {
                    size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());
                    bitset::BitsetVector result_buf;
                    result_buf.allocate_nozero(n);

                    // Warm up
                    bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);

                    timer.reset();
                    bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double and_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double or_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_xor_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double xor_pre = timer.elapsed_ms();

                    // In-place AND (a[i] &= b[i])
                    bitset::BitsetVector a_copy = ha->btv;
                    timer.reset();
                    bitset::simd::words_and_inplace_scalar(a_copy.words_mut(), hb->btv.words(), n);
                    double and_inplace = timer.elapsed_ms();

                    std::cout << "\n[Pure Ops] Bitset (Plain) operation-only (no allocation):\n";
                    std::cout << "  AND (pre-alloc):  " << and_pre << " ms\n";
                    std::cout << "  OR  (pre-alloc):  " << or_pre  << " ms\n";
                    std::cout << "  XOR (pre-alloc):  " << xor_pre << " ms\n";
                    std::cout << "  AND (in-place):   " << and_inplace << " ms\n";

                    // --- Segmented Bitset (8KB segments) ---
                    bitset::SegmentedBitset seg_a, seg_b;
                    seg_a.build_from(ha->btv);
                    seg_b.build_from(hb->btv);

                    // warm up
                    { auto w = bitset::SegmentedBitset::seg_or(seg_a, seg_b, false); (void)w; }

                    timer.reset();
                    auto seg_or_res = bitset::SegmentedBitset::seg_or(seg_a, seg_b, false);
                    double seg_or_t = timer.elapsed_ms();

                    timer.reset();
                    auto seg_and_res = bitset::SegmentedBitset::seg_and(seg_a, seg_b, false);
                    double seg_and_t = timer.elapsed_ms();

                    std::cout << "\n[Segmented] Bitset (8KB segs, " << seg_a.num_segments() << " segs):\n";
                    std::cout << "  OR:  " << seg_or_t  << " ms (card: " << seg_or_res.popcount(false) << ")\n";
                    std::cout << "  AND: " << seg_and_t << " ms (card: " << seg_and_res.popcount(false) << ")\n";
                }
            }

            // --- Bitset (AVX512) pure ops ---
            if (backend_name.find("AVX") != std::string::npos) {
                auto* ha = dynamic_cast<BitsetAVX512Handle*>(bm0.get());
                auto* hb = dynamic_cast<BitsetAVX512Handle*>(bm1.get());
                if (ha && hb) {
                    size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());
                    bitset::BitsetVector result_buf;
                    result_buf.allocate_nozero(n);

                    bitset::simd::words_and_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);

                    timer.reset();
                    bitset::simd::words_and_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double and_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_or_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double or_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_xor_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
                    double xor_pre = timer.elapsed_ms();

                    bitset::BitsetVector a_copy = ha->btv;
                    timer.reset();
                    bitset::simd::words_and_inplace_simd(a_copy.words_mut(), hb->btv.words(), n);
                    double and_inplace = timer.elapsed_ms();

                    std::cout << "\n[Pure Ops] Bitset (AVX512) operation-only (no allocation):\n";
                    std::cout << "  AND (pre-alloc):  " << and_pre << " ms\n";
                    std::cout << "  OR  (pre-alloc):  " << or_pre  << " ms\n";
                    std::cout << "  XOR (pre-alloc):  " << xor_pre << " ms\n";
                    std::cout << "  AND (in-place):   " << and_inplace << " ms\n";
                }
            }

            // --- CRoaring pure ops ---
            if (backend_name == "CRoaring") {
                auto* ha = dynamic_cast<CroaringHandle*>(bm0.get());
                auto* hb = dynamic_cast<CroaringHandle*>(bm1.get());
                if (ha && hb) {
                    uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                    // Warm up
                    { roaring::Roaring w = ha->bitmap & hb->bitmap; (void)w; }

                    timer.reset();
                    roaring::Roaring and_r = ha->bitmap & hb->bitmap;
                    double and_pure = timer.elapsed_ms();
                    and_pure += croaring_to_bitset_if_needed(and_r, logical_sz);

                    timer.reset();
                    roaring::Roaring or_r = ha->bitmap | hb->bitmap;
                    double or_pure = timer.elapsed_ms();
                    or_pure += croaring_to_bitset_if_needed(or_r, logical_sz);

                    timer.reset();
                    roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
                    double xor_pure = timer.elapsed_ms();
                    xor_pure += croaring_to_bitset_if_needed(xor_r, logical_sz);

                    roaring::Roaring a_copy = ha->bitmap;
                    timer.reset();
                    a_copy &= hb->bitmap;
                    double and_inplace = timer.elapsed_ms();
                    and_inplace += croaring_to_bitset_if_needed(a_copy, logical_sz);

                    std::cout << "\n[Pure Ops] CRoaring operation-only (+ to-bitset if needed):\n";
                    std::cout << "  AND: " << and_pure << " ms (card: " << and_r.cardinality() << ")\n";
                    std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.cardinality() << ")\n";
                    std::cout << "  XOR: " << xor_pure << " ms (card: " << xor_r.cardinality() << ")\n";
                    std::cout << "  AND (in-place): " << and_inplace << " ms\n";
                    croaring_print_containers("AND result", and_r);
                    croaring_print_containers("OR  result", or_r);
                    croaring_print_containers("XOR result", xor_r);
                    croaring_print_containers("Input A", ha->bitmap);
                    croaring_print_containers("Input B", hb->bitmap);
                }
            }

            // --- ComBit pure ops ---
            if (backend_name.find("ComBIT") != std::string::npos) {
                auto* ha = dynamic_cast<CombitHandle*>(bm0.get());
                auto* hb = dynamic_cast<CombitHandle*>(bm1.get());
                if (ha && hb) {
                    // Warm up
                    { ComBit w = ha->compressed & hb->compressed; (void)w; }

                    timer.reset();
                    ComBit and_r = ha->compressed & hb->compressed;
                    double and_pure = timer.elapsed_ms();

                    timer.reset();
                    ComBit or_r = ha->compressed | hb->compressed;
                    double or_pure = timer.elapsed_ms();

                    timer.reset();
                    ComBit xor_r = ha->compressed ^ hb->compressed;
                    double xor_pure = timer.elapsed_ms();

                    std::cout << "\n[Pure Ops] ComBIT operation-only timing:\n";
                    std::cout << "  AND: " << and_pure << " ms (card: " << and_r.popcount() << ")\n";
                    std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.popcount() << ")\n";
                    std::cout << "  XOR: " << xor_pure << " ms (card: " << xor_r.popcount() << ")\n";
                }
            }
        }
    }

    std::cout << "---------------------------------------\n";
}

// ==========================================
// 3. Cross-cardinality OR benchmark
// ==========================================
void run_cross_or_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                            const std::string& dir_a, const std::string& dir_b,
                            size_t num_rows)
{
    auto info_a = parse_compressed_dir_name(dir_a);
    auto info_b = parse_compressed_dir_name(dir_b);

    std::cout << "\n=======================================\n";
    std::cout << "Cross-Cardinality OR [" << backend_name << "]\n";
    std::cout << "  A: " << dir_a << " (c=" << info_a.cardinality << ")\n";
    std::cout << "  B: " << dir_b << " (c=" << info_b.cardinality << ")\n";
    std::cout << "  Rows: " << num_rows << "\n";
    std::cout << "=======================================\n";

    Timer timer;

    // Load first bitmap from each directory
    std::vector<std::string> files_a, files_b;
    for (auto& e : fs::directory_iterator(dir_a))
        if (e.path().extension() == ".bm") files_a.push_back(e.path().string());
    for (auto& e : fs::directory_iterator(dir_b))
        if (e.path().extension() == ".bm") files_b.push_back(e.path().string());

    std::sort(files_a.begin(), files_a.end(), [](const std::string& a, const std::string& b){
        return std::stoi(fs::path(a).stem().string()) < std::stoi(fs::path(b).stem().string());
    });
    std::sort(files_b.begin(), files_b.end(), [](const std::string& a, const std::string& b){
        return std::stoi(fs::path(a).stem().string()) < std::stoi(fs::path(b).stem().string());
    });

    if (files_a.empty() || files_b.empty()) {
        std::cerr << "  Error: no .bm files found\n";
        return;
    }

    // Load bitmap 1 from each directory
    auto bm_a = backend->Load(files_a[0]);
    auto bm_b = backend->Load(files_b[0]);
    if (!bm_a || !bm_b) {
        std::cerr << "  Error: failed to load bitmaps\n";
        return;
    }
    uint64_t card_a = backend->Cardinality(*bm_a);
    uint64_t card_b = backend->Cardinality(*bm_b);
    std::cout << "  " << fs::path(files_a[0]).filename().string()
              << " (c=" << info_a.cardinality << ") card=" << card_a << "\n";
    std::cout << "  " << fs::path(files_b[0]).filename().string()
              << " (c=" << info_b.cardinality << ") card=" << card_b << "\n";

    // ComBit size breakdown
    if (backend_name.find("ComBIT") != std::string::npos) {
        auto print_sb = [](const std::string& label, BitmapHandle* bm) {
            auto* ch = dynamic_cast<CombitHandle*>(bm);
            if (ch) {
                auto sb = ch->compressed.size_breakdown();
                std::cout << "  [ComBit " << label << "] leading: " << sb.l3_bits
                          << " bits | literal: " << sb.l1_literal_bits
                          << " bits | total: " << sb.total_bits << " bits ("
                          << sb.total_bits / 8.0 / 1024.0 << " KB)\n";
            }
        };
        print_sb("A", bm_a.get());
        print_sb("B", bm_b.get());
    }

    // OR
    { auto w = backend->bitOr(*bm_a, *bm_b); (void)w; }  // warm-up
    timer.reset();
    auto or_res = backend->bitOr(*bm_a, *bm_b);
    double or_t = timer.elapsed_ms();

    // AND
    timer.reset();
    auto and_res = backend->bitAnd(*bm_a, *bm_b);
    double and_t = timer.elapsed_ms();

    // For CRoaring: add to-bitset conversion time if result has array/run containers
    if (backend_name == "CRoaring") {
        auto* ha = dynamic_cast<CroaringHandle*>(bm_a.get());
        auto* hb = dynamic_cast<CroaringHandle*>(bm_b.get());
        if (ha && hb) {
            uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
            auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
            auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
            if (or_h)  or_t  += croaring_to_bitset_if_needed(or_h->bitmap, logical_sz);
            if (and_h) and_t += croaring_to_bitset_if_needed(and_h->bitmap, logical_sz);
        }
    }

    uint64_t or_card = backend->Cardinality(*or_res);
    uint64_t and_card = backend->Cardinality(*and_res);

    std::cout << "\n[Result] c=" << info_a.cardinality << " OR c=" << info_b.cardinality << ":\n";
    std::cout << "  OR:  " << or_t  << " ms (card: " << or_card  << ")\n";
    std::cout << "  AND: " << and_t << " ms (card: " << and_card << ")\n";

    // Pure ops for CRoaring
    if (backend_name == "CRoaring") {
        auto* ha = dynamic_cast<CroaringHandle*>(bm_a.get());
        auto* hb = dynamic_cast<CroaringHandle*>(bm_b.get());
        if (ha && hb) {
            uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
            { roaring::Roaring w = ha->bitmap | hb->bitmap; (void)w; }

            timer.reset();
            roaring::Roaring or_r = ha->bitmap | hb->bitmap;
            double or_pure = timer.elapsed_ms();
            or_pure += croaring_to_bitset_if_needed(or_r, logical_sz);

            timer.reset();
            roaring::Roaring and_r = ha->bitmap & hb->bitmap;
            double and_pure = timer.elapsed_ms();
            and_pure += croaring_to_bitset_if_needed(and_r, logical_sz);

            timer.reset();
            roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
            double xor_pure = timer.elapsed_ms();
            xor_pure += croaring_to_bitset_if_needed(xor_r, logical_sz);

            roaring::Roaring a_copy = ha->bitmap;
            timer.reset();
            a_copy |= hb->bitmap;
            double or_inplace = timer.elapsed_ms();
            or_inplace += croaring_to_bitset_if_needed(a_copy, logical_sz);

            std::cout << "\n[Pure Ops] CRoaring (+ to-bitset if needed):\n";
            std::cout << "  OR:            " << or_pure << " ms (card: " << or_r.cardinality() << ")\n";
            std::cout << "  AND:           " << and_pure << " ms (card: " << and_r.cardinality() << ")\n";
            std::cout << "  XOR:           " << xor_pure << " ms (card: " << xor_r.cardinality() << ")\n";
            std::cout << "  OR (in-place): " << or_inplace << " ms\n";
            croaring_print_containers("AND result", and_r);
            croaring_print_containers("OR  result", or_r);
            croaring_print_containers("XOR result", xor_r);
            croaring_print_containers("Input A", ha->bitmap);
            croaring_print_containers("Input B", hb->bitmap);
        }
    }

    // Pure ops for Bitset (Plain)
    if (backend_name.find("Plain") != std::string::npos) {
        auto* ha = dynamic_cast<BitsetHandle*>(bm_a.get());
        auto* hb = dynamic_cast<BitsetHandle*>(bm_b.get());
        if (ha && hb) {
            size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());
            bitset::BitsetVector result_buf;
            result_buf.allocate_nozero(n);
            bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);

            timer.reset();
            bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
            double or_pre = timer.elapsed_ms();
            timer.reset();
            bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
            double and_pre = timer.elapsed_ms();

            std::cout << "\n[Pure Ops] Bitset (Plain):\n";
            std::cout << "  OR  (pre-alloc): " << or_pre  << " ms\n";
            std::cout << "  AND (pre-alloc): " << and_pre << " ms\n";

            // Segmented
            bitset::SegmentedBitset seg_a, seg_b;
            seg_a.build_from(ha->btv);
            seg_b.build_from(hb->btv);
            { auto w = bitset::SegmentedBitset::seg_or(seg_a, seg_b, false); (void)w; }
            timer.reset();
            auto seg_or_res = bitset::SegmentedBitset::seg_or(seg_a, seg_b, false);
            double seg_or_t = timer.elapsed_ms();
            timer.reset();
            auto seg_and_res = bitset::SegmentedBitset::seg_and(seg_a, seg_b, false);
            double seg_and_t = timer.elapsed_ms();
            std::cout << "\n[Segmented] Bitset (8KB segs, " << seg_a.num_segments() << " segs):\n";
            std::cout << "  OR:  " << seg_or_t  << " ms (card: " << seg_or_res.popcount(false) << ")\n";
            std::cout << "  AND: " << seg_and_t << " ms (card: " << seg_and_res.popcount(false) << ")\n";
        }
    }

    // Pure ops for Bitset (AVX512)
    if (backend_name.find("AVX") != std::string::npos) {
        auto* ha = dynamic_cast<BitsetAVX512Handle*>(bm_a.get());
        auto* hb = dynamic_cast<BitsetAVX512Handle*>(bm_b.get());
        if (ha && hb) {
            size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());
            bitset::BitsetVector result_buf;
            result_buf.allocate_nozero(n);
            bitset::simd::words_or_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);

            timer.reset();
            bitset::simd::words_or_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
            double or_pre = timer.elapsed_ms();
            timer.reset();
            bitset::simd::words_and_simd(ha->btv.words(), hb->btv.words(), result_buf.words_mut(), n);
            double and_pre = timer.elapsed_ms();

            std::cout << "\n[Pure Ops] Bitset (AVX512):\n";
            std::cout << "  OR  (pre-alloc): " << or_pre  << " ms\n";
            std::cout << "  AND (pre-alloc): " << and_pre << " ms\n";
        }
    }

    // Pure ops for ComBit
    if (backend_name.find("ComBIT") != std::string::npos) {
        auto* ha = dynamic_cast<CombitHandle*>(bm_a.get());
        auto* hb = dynamic_cast<CombitHandle*>(bm_b.get());
        if (ha && hb) {
            { ComBit w = ha->compressed | hb->compressed; (void)w; }
            timer.reset();
            ComBit or_r = ha->compressed | hb->compressed;
            double or_pure = timer.elapsed_ms();
            timer.reset();
            ComBit and_r = ha->compressed & hb->compressed;
            double and_pure = timer.elapsed_ms();

            std::cout << "\n[Pure Ops] ComBIT:\n";
            std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.popcount() << ")\n";
            std::cout << "  AND: " << and_pure << " ms (card: " << and_r.popcount() << ")\n";
        }
    }

    // CSV
    std::string label = "c" + std::to_string(info_a.cardinality) + ":c" + std::to_string(info_b.cardinality);
    csv_row(backend_name, num_rows, 0, "cross-OR(" + label + ")", or_t, 0, 0, or_card, 1);
    csv_row(backend_name, num_rows, 0, "cross-AND(" + label + ")", and_t, 0, 0, and_card, 1);

    std::cout << "---------------------------------------\n";
}

// ==========================================
// Main function: CLI argument parsing
// ==========================================
int main(int argc, char** argv) {
    std::string backend_type = "all";
    std::string bm_dir;
    std::string compressed_dir;
    std::string cross_or_dir_a, cross_or_dir_b;
    size_t num_rows = 0;
    size_t sample_count = 0;  // 0 = all
    std::string csv_path;
    size_t iterations = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--backend" && i + 1 < argc) {
            backend_type = argv[++i];
        } else if (arg == "--bm-dir" && i + 1 < argc) {
            bm_dir = argv[++i];
        } else if (arg == "--compressed-dir" && i + 1 < argc) {
            compressed_dir = argv[++i];
        } else if (arg == "--cross-or" && i + 2 < argc) {
            cross_or_dir_a = argv[++i];
            cross_or_dir_b = argv[++i];
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
    BitsetBackend bitset;
    BitsetAVX512Backend bitset_avx512;

    // Build list of (backend_ptr, name) to run
    struct BackendEntry { IBitmapBackend* ptr; std::string name; std::string key; };
    std::vector<BackendEntry> backends = {
        {&wah,            "WAH (FastBit)",       "wah"},
        {&croaring,       "CRoaring",            "croaring"},
        {&combit,         "ComBIT (New)",         "combit"},
        {&ewah,           "EWAH",                "ewah"},
        {&concise,        "Concise",             "concise"},
        {&bitset,         "Bitset (Plain)",      "bitset"},
        {&bitset_avx512,  "Bitset (AVX512)",     "bitset_avx512"},
    };

    if (!cross_or_dir_a.empty() && !cross_or_dir_b.empty()) {
        // === Cross-cardinality OR mode ===
        if (!fs::is_directory(cross_or_dir_a) || !fs::is_directory(cross_or_dir_b)) {
            std::cerr << "Error: --cross-or directories must exist\n";
            return 1;
        }
        auto info_a = parse_compressed_dir_name(cross_or_dir_a);
        auto info_b = parse_compressed_dir_name(cross_or_dir_b);
        size_t rows = (num_rows > 0) ? num_rows : info_a.rows;
        if (rows == 0) rows = info_b.rows;
        if (rows == 0) {
            std::cerr << "Error: cannot determine num_rows. Use --num-rows\n";
            return 1;
        }
        std::string detected_key = algo_to_backend_key(info_a.algo);

        if (backend_type == "all" && !detected_key.empty()) {
            if (detected_key == "bitset")
                backend_type = "bitset_both";
            else
                backend_type = detected_key;
        }

        std::cout << "=== Cross-Cardinality OR Benchmark Mode ===\n";
        std::cout << "Dir A: " << cross_or_dir_a << " (c=" << info_a.cardinality << ")\n";
        std::cout << "Dir B: " << cross_or_dir_b << " (c=" << info_b.cardinality << ")\n";
        std::cout << "Rows: " << rows << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all"
                || (backend_type == "bitset_both" && (be.key == "bitset" || be.key == "bitset_avx512")))
                run_cross_or_benchmark(be.ptr, be.name, cross_or_dir_a, cross_or_dir_b, rows);
        }
    } else if (!compressed_dir.empty()) {
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
        // For bitset directories, run both plain and AVX512 backends
        if (backend_type == "all" && !detected_key.empty()) {
            if (detected_key == "bitset")
                backend_type = "bitset_both";
            else
                backend_type = detected_key;
        }

        std::cout << "=== Compressed .bm Benchmark Mode ===\n";
        std::cout << "Directory: " << compressed_dir << "\n";
        std::cout << "Rows: " << rows << " | Cardinality: " << card << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all"
                || (backend_type == "bitset_both" && (be.key == "bitset" || be.key == "bitset_avx512")))
                run_compressed_benchmark(be.ptr, be.name, compressed_dir, rows, card, sample_count, iterations);
        }
    } else if (!bm_dir.empty()) {
        // === Raw .bm file mode ===
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

        std::cout << "=== Raw .bm File Benchmark Mode ===\n";
        std::cout << "Directory: " << bm_dir << " | Rows: " << num_rows << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all")
                run_bm_benchmark(be.ptr, be.name, bm_dir, num_rows);
        }
    } else {
        std::cerr << "Error: must specify --compressed-dir or --bm-dir\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}