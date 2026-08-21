#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstdint>
#include "uti.h"
#include "dense_btv_delivery.h"
#include "backends/ddc/ddc_backend.h"
#include "backends/wah/wah_backend.h"
#include "backends/croaring/croaring_backend.h"
#include "backends/ewah/ewah_backend.h"
#include "backends/Concise/concise_backend.h"
#include "backends/bitset/bitset_backend.h"
#include "backends/bitset_avx512/bitset_avx512_backend.h"
#include "backends/bsr/bsr_backend.h"
#include "backends/hbi/hbi_backend.h"
#include <bitset_simd.hpp>
#include <segmented_bitset.hpp>

namespace fs = std::filesystem;

// perf mode (hbi branch): loop ONLY the compressed operation so `perf stat -D`
// can measure a pure op region (no load, no build, no conversion, no CSV).
static std::string g_perf_op;     // "and" | "not"
static long g_perf_loops = 0;

// roaring -> dense bitset cost
static double croaring_to_bitset(const roaring::Roaring& r,
                                 uint32_t logical_size) {
    Timer conv_timer;
    conv_timer.reset();
    roaring::api::bitset_t* bs = roaring::api::bitset_create_with_capacity(
        logical_size);
    roaring::api::roaring_bitmap_to_bitset(&r.roaring, bs);
    roaring::api::bitset_free(bs);
    return conv_timer.elapsed_ms();
}

static void croaring_print_containers(const std::string& label,
                                      const roaring::Roaring& r) {
    roaring::api::roaring_statistics_t s;
    roaring::api::roaring_bitmap_statistics(&r.roaring, &s);
    double arr_mb = s.n_bytes_array_containers / (1024.0 * 1024.0);
    double bs_mb  = s.n_bytes_bitset_containers / (1024.0 * 1024.0);
    double run_mb = s.n_bytes_run_containers / (1024.0 * 1024.0);
    size_t total  = s.n_bytes_array_containers + s.n_bytes_bitset_containers
                  + s.n_bytes_run_containers;
    std::cout << "  [" << label << " containers]"
              << " array=" << s.n_array_containers << " (" << arr_mb << " MB)"
              << " bitset=" << s.n_bitset_containers << " (" << bs_mb << " MB)"
              << " run=" << s.n_run_containers << " (" << run_mb << " MB)"
              << " total_bytes=" << total << "\n";
}

static bool and_pair_mode() {
    static const bool v = [] {
        const char* e = std::getenv("DDC_BENCH_AND_MODE");
        return e != nullptr && std::string(e) == "pair";
    }();
    return v;
}

// raw .bm bench
void run_bm_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                      const std::string& bm_dir, size_t num_rows)
{
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — .bm Files\n";
    std::cout << "Source: " << bm_dir << " (" << num_rows << " rows)\n";
    std::cout << "=======================================\n";

    Timer timer;

    std::vector<std::string> bm_files;
    for (auto& entry : fs::directory_iterator(bm_dir)) {
        if (entry.path().extension() == ".bm")
            bm_files.push_back(entry.path().string());
    }

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

    // load + build
    std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
    timer.reset();
    for (auto& path : bm_files) {
        auto bits = read_raw_bm(path, num_rows);
        bitmaps.push_back(bits_to_bitmap(backend, bits));
    }
    double load_time = timer.elapsed_ms();
    std::cout << "[Load] Read & build " << bm_files.size()
              << " bitmaps: " << load_time << " ms\n";

    for (size_t i = 0; i < bm_files.size(); ++i) {
        std::string name = fs::path(bm_files[i]).filename().string();
        if (bm_files[i].find("/column/") != std::string::npos)
            name = "column/" + name;
        std::cout << "  " << name << " → cardinality: "
                  << backend->Cardinality(*bitmaps[i]) << "\n";
    }

    // pairwise ops
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

        if (backend_name == "CRoaring") {
            auto* ha_c = dynamic_cast<CroaringHandle*>(a.get());
            auto* hb_c = dynamic_cast<CroaringHandle*>(b.get());
            if (ha_c && hb_c) {
                uint32_t logical_sz = std::max(ha_c->current_size, hb_c->current_size);
                auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
                auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
                auto* xor_h = dynamic_cast<CroaringHandle*>(xor_res.get());
                if (or_h)  or_t  += croaring_to_bitset(or_h->bitmap, logical_sz);
                if (and_h) and_t += croaring_to_bitset(and_h->bitmap, logical_sz);
                if (xor_h) xor_t += croaring_to_bitset(xor_h->bitmap, logical_sz);
            }
        }

        std::cout << "\n[Compute] Pairwise ops on " << na << " & " << nb << ":\n";
        std::cout << "  bitOr:  " << or_t  << " ms (card: " << backend->Cardinality(*or_res)  << ")\n";
        std::cout << "  bitAnd: " << and_t << " ms (card: " << backend->Cardinality(*and_res) << ")\n";
        std::cout << "  bitXor: " << xor_t << " ms (card: " << backend->Cardinality(*xor_res) << ")\n";

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

        if (backend_name == "CRoaring") {
            auto* ha = dynamic_cast<CroaringHandle*>(a.get());
            auto* hb = dynamic_cast<CroaringHandle*>(b.get());
            if (ha && hb) {
                uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                { roaring::Roaring w = ha->bitmap & hb->bitmap; (void)w; }

                timer.reset();
                roaring::Roaring and_r = ha->bitmap & hb->bitmap;
                double and_pure = timer.elapsed_ms();
                and_pure += croaring_to_bitset(and_r, logical_sz);

                timer.reset();
                roaring::Roaring or_r = ha->bitmap | hb->bitmap;
                double or_pure = timer.elapsed_ms();
                or_pure += croaring_to_bitset(or_r, logical_sz);

                timer.reset();
                roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
                double xor_pure = timer.elapsed_ms();
                xor_pure += croaring_to_bitset(xor_r, logical_sz);

                roaring::Roaring a_copy = ha->bitmap;
                timer.reset();
                a_copy &= hb->bitmap;
                double and_inplace = timer.elapsed_ms();
                and_inplace += croaring_to_bitset(a_copy, logical_sz);

                std::cout << "\n[Pure Ops] CRoaring operation-only (+ to-btv conversion):\n";
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

    // multi-way OR
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

// compressed .bm bench
void run_compressed_benchmark(IBitmapBackend* backend, const std::string& backend_name,
                              const std::string& comp_dir, size_t num_rows,
                              size_t cardinality,
                              size_t sample_count, size_t iterations = 1)
{
    std::cout << "\n=======================================\n";
    std::cout << "Benchmark [" << backend_name << "] — Compressed .bm Files\n";
    std::cout << "Source: " << comp_dir << "\n";

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

    // Sum serialized logical file lengths. This includes every byte written
    // inside each .bm file, but excludes RAM and filesystem block slack.
    long total_compressed = 0;
    for (auto& path : selected) total_compressed += get_file_size(path);
    double raw_total = static_cast<double>(selected.size()) * num_rows / 8.0;
    double ratio = (raw_total > 0) ? total_compressed / raw_total : 0;

    std::vector<double> load_times, or_times, and_times, xor_times, multi_or_times;
    uint64_t or_card = 0, and_card = 0, xor_card = 0, multi_or_card = 0;
    Timer timer;

    // warm-up
    if (iterations > 1 && selected.size() >= 2) {
        std::cout << "[Warm-up] Running 1 warm-up iteration...\n";
        auto w1 = backend->Load(selected[0]);
        auto w2 = backend->Load(selected[1]);
        if (w1 && w2) { auto wor = backend->bitOr(*w1, *w2); (void)wor; }
    }

    for (size_t iter = 0; iter < iterations; ++iter) {

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

            // DDC per-level size
            if (backend_name.find("DDC") != std::string::npos) {
                size_t total_l4 = 0, total_l3 = 0, total_l2 = 0, total_l1 = 0, total_cb = 0;
                for (auto& bm : bitmaps) {
                    auto* ch = dynamic_cast<DDCHandle*>(bm.get());
                    if (ch) {
                        auto sb = ch->compressed.size_breakdown();
                        total_l4 += sb.l4_bits;
                        total_l3 += sb.l3_literal_bits;
                        total_l2 += sb.l2_literal_bits;
                        total_l1 += sb.l1_literal_bits;
                        total_cb += sb.total_bits;
                    }
                }
                std::cout << "[DDC Storage] l4_bytes: "
                          << total_l4 / 8.0 / 1024.0 / 1024.0 << " MB"
                          << " | l3_bytes: "
                          << total_l3 / 8.0 / 1024.0 / 1024.0 << " MB"
                          << " | l2_bytes: "
                          << total_l2 / 8.0 / 1024.0 / 1024.0 << " MB"
                          << " | l1_bytes: "
                          << total_l1 / 8.0 / 1024.0 / 1024.0 << " MB"
                          << " | total_bytes: "
                          << total_cb / 8.0 / 1024.0 / 1024.0 << " MB\n";
            }

            if (backend_name == "CRoaring") {
                size_t tot_arr = 0, tot_bs = 0, tot_run = 0;
                size_t bytes_arr = 0, bytes_bs = 0, bytes_run = 0;
                for (auto& bm : bitmaps) {
                    auto* ch = dynamic_cast<CroaringHandle*>(bm.get());
                    if (ch) {
                        roaring::api::roaring_statistics_t s;
                        roaring::api::roaring_bitmap_statistics(&ch->bitmap.roaring, &s);
                        tot_arr  += s.n_array_containers;
                        tot_bs   += s.n_bitset_containers;
                        tot_run  += s.n_run_containers;
                        bytes_arr += s.n_bytes_array_containers;
                        bytes_bs  += s.n_bytes_bitset_containers;
                        bytes_run += s.n_bytes_run_containers;
                    }
                }
                size_t total_bytes = bytes_arr + bytes_bs + bytes_run;
                std::cout << "[CRoaring Storage]"
                          << " array=" << tot_arr << " (" << bytes_arr / (1024.0*1024.0) << " MB)"
                          << " | run=" << tot_run << " (" << bytes_run / (1024.0*1024.0) << " MB)"
                          << " | bitset=" << tot_bs << " (" << bytes_bs / (1024.0*1024.0) << " MB)"
                          << " | total_bytes=" << total_bytes << "\n";
            }

            // BSR in-memory payload: 32-bit base + 32-bit state per chunk
            if (backend_name == "QFilter-BSR") {
                uint64_t chunks = 0;
                for (auto& bm : bitmaps) {
                    auto* h = dynamic_cast<BsrHandle*>(bm.get());
                    if (h) chunks += static_cast<uint64_t>(h->size);
                }
                std::cout << "[BSR Storage] chunks=" << chunks
                          << " | bases_bytes=" << chunks * 4.0 / (1024.0*1024.0) << " MB"
                          << " | states_bytes=" << chunks * 4.0 / (1024.0*1024.0) << " MB"
                          << " | total_bytes=" << chunks * 8.0 / (1024.0*1024.0) << " MB\n";
            }
        }

        if (bitmaps.size() < 2) {
            std::cout << "  Not enough bitmaps for logical ops.\n";
            return;
        }

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

            if (backend_name == "CRoaring") {
                auto* ha = dynamic_cast<CroaringHandle*>(a.get());
                auto* hb = dynamic_cast<CroaringHandle*>(b.get());
                if (ha && hb) {
                    uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                    auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
                    auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
                    auto* xor_h = dynamic_cast<CroaringHandle*>(xor_res.get());
                    if (or_h)  or_elapsed  += croaring_to_bitset(or_h->bitmap, logical_sz);
                    if (and_h) and_elapsed += croaring_to_bitset(and_h->bitmap, logical_sz);
                    if (xor_h) xor_elapsed += croaring_to_bitset(xor_h->bitmap, logical_sz);
                }
            }

            or_times.push_back(or_elapsed);
            or_card = backend->Cardinality(*or_res);

            and_times.push_back(and_elapsed);
            and_card = backend->Cardinality(*and_res);

            xor_times.push_back(xor_elapsed);
            xor_card = backend->Cardinality(*xor_res);
        }

        // multi-way OR (skipped in perf mode: not part of the measured
        // region and costs ~10s of pre-marker time per run)
        if (bitmaps.size() >= 3 && g_perf_loops == 0) {
            timer.reset();
            auto acc = backend->bitOr(*bitmaps[0], *bitmaps[1]);
            for (size_t k = 2; k < bitmaps.size(); ++k)
                acc = backend->bitOr(*acc, *bitmaps[k]);
            multi_or_times.push_back(timer.elapsed_ms());
            multi_or_card = backend->Cardinality(*acc);
        }

        int it = static_cast<int>(iter + 1);
        csv_row(backend_name, num_rows, cardinality, "load", load_t, total_compressed, ratio, 0, it);
        csv_row(backend_name, num_rows, cardinality, "OR", or_times.back(), 0, 0, or_card, it);
        csv_row(backend_name, num_rows, cardinality, "AND", and_times.back(), 0, 0, and_card, it);
        csv_row(backend_name, num_rows, cardinality, "XOR", xor_times.back(), 0, 0, xor_card, it);
        if (!multi_or_times.empty())
            csv_row(backend_name, num_rows, cardinality, "multi-OR", multi_or_times.back(), 0, 0, multi_or_card, it);
    }

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

    {
        auto bm0 = backend->Load(selected[0]);
        auto bm1 = backend->Load(selected[1]);

        decltype(bm0) bm2;
        if (bm0 && bm1) {

            // scalar bitset pure ops
            if (backend_name.find("Plain") != std::string::npos) {
                auto* ha = dynamic_cast<BitsetHandle*>(bm0.get());
                auto* hb = dynamic_cast<BitsetHandle*>(bm1.get());
                if (ha && hb) {
                    size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());

                    // FIX (warm-page symmetry): pre-allocate AND pre-touch every result
                    // buffer, then time ONLY the op. The old code warmed just AND & NOT,
                    // so OR/XOR paid ~3ms of first-touch page faults and looked 4x slower.
                    bitset::BitsetVector and_buf, or_buf, xor_buf, not_buf, a_copy = ha->btv;
                    and_buf.allocate_nozero(n); or_buf.allocate_nozero(n);
                    xor_buf.allocate_nozero(n); not_buf.allocate_nozero(n);
                    bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), and_buf.words_mut(), n); // warm
                    bitset::simd::words_or_scalar (ha->btv.words(), hb->btv.words(), or_buf.words_mut(),  n); // warm
                    bitset::simd::words_xor_scalar(ha->btv.words(), hb->btv.words(), xor_buf.words_mut(), n); // warm
                    bitset::simd::words_not_scalar(ha->btv.words(), not_buf.words_mut(), n);                  // warm

                    timer.reset();
                    bitset::simd::words_and_scalar(ha->btv.words(), hb->btv.words(), and_buf.words_mut(), n);
                    double and_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), or_buf.words_mut(), n);
                    double or_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_xor_scalar(ha->btv.words(), hb->btv.words(), xor_buf.words_mut(), n);
                    double xor_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_and_inplace_scalar(a_copy.words_mut(), ha->btv.words(), n);
                    double and_inplace = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_not_scalar(ha->btv.words(), not_buf.words_mut(), n);
                    double not_pre = timer.elapsed_ms();

                    std::cout << "\n[Pure Ops] Bitset (Plain) operation-only (no allocation):\n";
                    std::cout << "  AND (pre-alloc):  " << and_pre << " ms\n";
                    std::cout << "  OR  (pre-alloc):  " << or_pre  << " ms\n";
                    std::cout << "  XOR (pre-alloc):  " << xor_pre << " ms\n";
                    std::cout << "  NOT (pre-alloc):  " << not_pre << " ms\n";
                    std::cout << "  AND (in-place):   " << and_inplace << " ms\n";

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pre,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pre, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pre, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pre, 0, 0, 0, 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    // COMP: ~((A|B)&(B|C))
                    {
                        auto* hc = bm2 ? dynamic_cast<BitsetHandle*>(bm2.get()) : ha;
                        if (hc) {
                            bitset::BitsetVector t1, t2, t3;
                            t1.allocate_nozero(n); t2.allocate_nozero(n); t3.allocate_nozero(n);

                            bitset::simd::words_or_scalar(ha->btv.words(), hb->btv.words(), t1.words_mut(), n);
                            bitset::simd::words_or_scalar(hb->btv.words(), hc->btv.words(), t2.words_mut(), n);
                            bitset::simd::words_and_scalar(t1.words(), t2.words(), t3.words_mut(), n);
                            bitset::simd::words_not_scalar(t3.words(), t1.words_mut(), n);
                            timer.reset();
                            bitset::simd::words_or_scalar (ha->btv.words(), hb->btv.words(), t1.words_mut(), n);
                            bitset::simd::words_or_scalar (hb->btv.words(), hc->btv.words(), t2.words_mut(), n);
                            bitset::simd::words_and_scalar(t1.words(),       t2.words(),       t3.words_mut(), n);
                            bitset::simd::words_not_scalar(t3.words(),       t1.words_mut(),                   n);
                            double comp_t = timer.elapsed_ms();
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_t << " ms\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_t, 0, 0, 0, 1);
                        }
                    }

                    // segmented variant
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

            // AVX512 bitset pure ops
            if (backend_name.find("AVX") != std::string::npos) {
                auto* ha = dynamic_cast<BitsetAVX512Handle*>(bm0.get());
                auto* hb = dynamic_cast<BitsetAVX512Handle*>(bm1.get());
                if (ha && hb) {
                    size_t n = std::min(ha->btv.words_cnt(), hb->btv.words_cnt());

                    // FIX (warm-page symmetry): pre-allocate AND pre-touch every result
                    // buffer, then time ONLY the op. The old code warmed just AND & NOT,
                    // so OR/XOR paid ~3ms of first-touch page faults and looked 4x slower.
                    bitset::BitsetVector and_buf, or_buf, xor_buf, not_buf, a_copy = ha->btv;
                    and_buf.allocate_nozero(n); or_buf.allocate_nozero(n);
                    xor_buf.allocate_nozero(n); not_buf.allocate_nozero(n);
                    bitset::simd::words_and_simd(ha->btv.words(), hb->btv.words(), and_buf.words_mut(), n); // warm
                    bitset::simd::words_or_simd (ha->btv.words(), hb->btv.words(), or_buf.words_mut(),  n); // warm
                    bitset::simd::words_xor_simd(ha->btv.words(), hb->btv.words(), xor_buf.words_mut(), n); // warm
                    bitset::simd::words_not_simd(ha->btv.words(), not_buf.words_mut(), n);                  // warm

                    timer.reset();
                    bitset::simd::words_and_simd(ha->btv.words(), hb->btv.words(), and_buf.words_mut(), n);
                    double and_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_or_simd(ha->btv.words(), hb->btv.words(), or_buf.words_mut(), n);
                    double or_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_xor_simd(ha->btv.words(), hb->btv.words(), xor_buf.words_mut(), n);
                    double xor_pre = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_and_inplace_simd(a_copy.words_mut(), ha->btv.words(), n);
                    double and_inplace = timer.elapsed_ms();

                    timer.reset();
                    bitset::simd::words_not_simd(ha->btv.words(), not_buf.words_mut(), n);
                    double not_pre = timer.elapsed_ms();

                    std::cout << "\n[Pure Ops] Bitset (AVX512) operation+alloc:\n";
                    std::cout << "  AND (with alloc): " << and_pre << " ms\n";
                    std::cout << "  OR  (with alloc): " << or_pre  << " ms\n";
                    std::cout << "  XOR (with alloc): " << xor_pre << " ms\n";
                    std::cout << "  NOT (with alloc): " << not_pre << " ms\n";
                    std::cout << "  AND (in-place):   " << and_inplace << " ms\n";

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pre,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pre, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pre, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pre, 0, 0, 0, 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<BitsetAVX512Handle*>(bm2.get()) : ha;
                        if (hc) {
                            bitset::BitsetVector t1, t2, t3;
                            t1.allocate_nozero(n); t2.allocate_nozero(n); t3.allocate_nozero(n);
                            bitset::simd::words_or_simd(ha->btv.words(), hb->btv.words(), t1.words_mut(), n);
                            bitset::simd::words_or_simd(hb->btv.words(), hc->btv.words(), t2.words_mut(), n);
                            bitset::simd::words_and_simd(t1.words(), t2.words(), t3.words_mut(), n);
                            bitset::simd::words_not_simd(t3.words(), t1.words_mut(), n);
                            timer.reset();
                            bitset::simd::words_or_simd (ha->btv.words(), hb->btv.words(), t1.words_mut(), n);
                            bitset::simd::words_or_simd (hb->btv.words(), hc->btv.words(), t2.words_mut(), n);
                            bitset::simd::words_and_simd(t1.words(),       t2.words(),       t3.words_mut(), n);
                            bitset::simd::words_not_simd(t3.words(),       t1.words_mut(),                   n);
                            double comp_t = timer.elapsed_ms();
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_t << " ms\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_t, 0, 0, 0, 1);
                        }
                    }
                }
            }

            // CRoaring pure ops (median of N)
            if (backend_name == "CRoaring") {
                auto* ha = dynamic_cast<CroaringHandle*>(bm0.get());
                auto* hb = dynamic_cast<CroaringHandle*>(bm1.get());
                if (ha && hb) {
                    uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };

                    const roaring::Roaring& and_rhs = and_pair_mode() ? hb->bitmap : ha->bitmap;
                    { roaring::Roaring w = ha->bitmap & and_rhs; (void)w; }
                    { roaring::Roaring w = ha->bitmap | hb->bitmap; (void)w; }
                    { roaring::Roaring w = ha->bitmap ^ hb->bitmap; (void)w; }

                    std::vector<double> and_t, or_t, xor_t;
                    std::vector<double> and_nc_t, xor_nc_t, not_nc_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        roaring::Roaring r = ha->bitmap & and_rhs;
                        double t = timer.elapsed_ms();
                        and_nc_t.push_back(t);
                        t += croaring_to_bitset(r, logical_sz);
                        and_t.push_back(t);
                    }

                    std::vector<double> or_nc_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        roaring::Roaring r = ha->bitmap | hb->bitmap;
                        double t = timer.elapsed_ms();
                        or_nc_t.push_back(t);
                        (void)r;
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        roaring::Roaring r = ha->bitmap | hb->bitmap;
                        double t = timer.elapsed_ms();
                        t += croaring_to_bitset(r, logical_sz);
                        or_t.push_back(t);
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        roaring::Roaring r = ha->bitmap ^ hb->bitmap;
                        double t = timer.elapsed_ms();
                        xor_nc_t.push_back(t);
                        t += croaring_to_bitset(r, logical_sz);
                        xor_t.push_back(t);
                    }

                    { roaring::Roaring w = ha->bitmap; w.flip(0, logical_sz); (void)w; }
                    std::vector<double> not_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        roaring::Roaring r = ha->bitmap;
                        r.flip(0, logical_sz);
                        double t = timer.elapsed_ms();
                        not_nc_t.push_back(t);
                        t += croaring_to_bitset(r, logical_sz);
                        not_t.push_back(t);
                    }
                    double not_pure = median(not_t);

                    double and_pure   = median(and_t);
                    double or_pure    = median(or_t);
                    double or_no_conv = median(or_nc_t);
                    double xor_pure   = median(xor_t);
                    // audit 2026-08-01: CSV *_op rows switch to the pure-op
                    // caliber (native Roaring result, NO to-bitset conversion)
                    // to match every other backend's *_op row; the
                    // conversion-included numbers stay on stdout.
                    double and_no_conv = median(and_nc_t);
                    double xor_no_conv = median(xor_nc_t);
                    double not_no_conv = median(not_nc_t);

                    roaring::Roaring and_r = ha->bitmap & and_rhs;
                    roaring::Roaring or_r  = ha->bitmap | hb->bitmap;
                    roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
                    roaring::Roaring not_r = ha->bitmap; not_r.flip(0, logical_sz);

                    roaring::Roaring a_copy = ha->bitmap;
                    timer.reset();
                    a_copy &= hb->bitmap;
                    double and_inplace = timer.elapsed_ms();
                    and_inplace += croaring_to_bitset(a_copy, logical_sz);

                    std::cout << "\n[Pure Ops] CRoaring operation-only (+ to-btv conversion):\n";
                    std::cout << "  AND: " << and_pure << " ms (card: " << and_r.cardinality() << ")\n";
                    std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.cardinality() << ")\n";
                    std::cout << "  OR (CR internal only, no to-btv): " << or_no_conv << " ms\n";
                    std::cout << "  XOR: " << xor_pure << " ms (card: " << xor_r.cardinality() << ")\n";
                    std::cout << "  NOT: " << not_pure << " ms (card: " << not_r.cardinality() << ")\n";
                    std::cout << "  AND (in-place): " << and_inplace << " ms\n";
                    croaring_print_containers("AND result", and_r);
                    croaring_print_containers("OR  result", or_r);
                    croaring_print_containers("XOR result", xor_r);
                    croaring_print_containers("NOT result", not_r);
                    croaring_print_containers("Input A", ha->bitmap);
                    croaring_print_containers("Input B", hb->bitmap);

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_no_conv,  0, 0, or_r.cardinality(),  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_no_conv, 0, 0, and_r.cardinality(), 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_no_conv, 0, 0, xor_r.cardinality(), 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_no_conv, 0, 0, not_r.cardinality(), 1);
                    csv_row(backend_name, num_rows, cardinality, "OR_op_conv",  or_pure,  0, 0, or_r.cardinality(),  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op_conv", and_pure, 0, 0, and_r.cardinality(), 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op_conv", xor_pure, 0, 0, xor_r.cardinality(), 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op_conv", not_pure, 0, 0, not_r.cardinality(), 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<CroaringHandle*>(bm2.get()) : ha;
                        if (hc) {
                            uint32_t lsz3 = std::max(logical_sz, hc->current_size);
                            { roaring::Roaring t1 = ha->bitmap | hb->bitmap;
                              roaring::Roaring t2 = hb->bitmap | hc->bitmap;
                              roaring::Roaring t3 = t1 & t2; t3.flip(0, lsz3); (void)t3; }

                            // Native-only COMP is timed in a separate loop so
                            // dense-delivery allocation and cache state cannot
                            // contaminate the native measurement.
                            std::vector<double> comp_native_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                roaring::Roaring t1 = ha->bitmap | hb->bitmap;
                                roaring::Roaring t2 = hb->bitmap | hc->bitmap;
                                roaring::Roaring t3 = t1 & t2;
                                t3.flip(0, lsz3);
                                double t = timer.elapsed_ms();
                                asm volatile("" : : "r"(&t3) : "memory");
                                comp_native_t.push_back(t);
                            }
                            double comp_native = median(comp_native_t);
                            roaring::Roaring native_t1 = ha->bitmap | hb->bitmap;
                            roaring::Roaring native_t2 = hb->bitmap | hc->bitmap;
                            roaring::Roaring native_result = native_t1 & native_t2;
                            native_result.flip(0, lsz3);
                            uint64_t comp_native_cardinality = native_result.cardinality();

                            std::vector<double> comp_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                roaring::Roaring t1 = ha->bitmap | hb->bitmap;
                                roaring::Roaring t2 = hb->bitmap | hc->bitmap;
                                roaring::Roaring t3 = t1 & t2;
                                t3.flip(0, lsz3);
                                double t = timer.elapsed_ms();
                                t += croaring_to_bitset(t3, lsz3);
                                comp_t.push_back(t);
                            }
                            double comp_pure = median(comp_t);
                            std::cout << "  COMP native (~((A|B)&(B|C))): "
                                      << comp_native << " ms\n";
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_pure << " ms\n";
                            csv_row(backend_name, num_rows, cardinality,
                                    "COMP_op_native", comp_native, 0, 0,
                                    comp_native_cardinality, 1);
                            csv_row(backend_name, num_rows, cardinality,
                                    "COMP_op", comp_pure, 0, 0,
                                    comp_native_cardinality, 1);

                        }
                    }
                }
            }

            // ---- perf mode: pure compressed-op loop, then exit ----
            if (g_perf_loops > 0 && !g_perf_op.empty()) {
                std::cout << "[perf-mode] op=" << g_perf_op << " loops="
                          << g_perf_loops << " backend=" << backend_name << "\n";
                if (backend_name.rfind("HBI", 0) == 0) {
                    auto* ha = dynamic_cast<HbiHandle*>(bm0.get());
                    std::unique_ptr<BitmapHandle> dup = backend->Load(selected[0]);
                    auto* ha2 = dynamic_cast<HbiHandle*>(dup.get());
                    if (g_perf_op == "and") {
                        { hbi::HbiBitmap w = hbi::HbiBitmap::and_(ha->bm, ha2->bm); (void)w; }
                        std::cout << "[perf-mode] warmed, measuring region begins\n" << std::flush;
                        for (long i = 0; i < g_perf_loops; i++) {
                            hbi::HbiBitmap r = hbi::HbiBitmap::and_(ha->bm, ha2->bm);
                            asm volatile("" : : "r"(&r) : "memory");
                        }
                    } else if (g_perf_op == "not") {
                        { hbi::HbiBitmap w = hbi::HbiBitmap::not_(ha->bm); (void)w; }
                        std::cout << "[perf-mode] warmed, measuring region begins\n" << std::flush;
                        for (long i = 0; i < g_perf_loops; i++) {
                            hbi::HbiBitmap r = hbi::HbiBitmap::not_(ha->bm);
                            asm volatile("" : : "r"(&r) : "memory");
                        }
                    }
                    return;
                }
                if (backend_name.find("DDC") != std::string::npos) {
                    auto* ha = dynamic_cast<DDCHandle*>(bm0.get());
                    if (g_perf_op == "and") {
                        { DDC w = ha->compressed.and_no_bypass(ha->compressed); (void)w; }
                        std::cout << "[perf-mode] warmed, measuring region begins\n" << std::flush;
                        for (long i = 0; i < g_perf_loops; i++) {
                            DDC r = ha->compressed.and_no_bypass(ha->compressed);
                            asm volatile("" : : "r"(&r) : "memory");
                        }
                    } else if (g_perf_op == "not") {
                        { DDC w = ~ha->compressed; (void)w; }
                        std::cout << "[perf-mode] warmed, measuring region begins\n" << std::flush;
                        for (long i = 0; i < g_perf_loops; i++) {
                            DDC r = ~ha->compressed;
                            asm volatile("" : : "r"(&r) : "memory");
                        }
                    }
                    return;
                }
                std::cout << "[perf-mode] backend not supported\n";
                return;
            }

            // DDC pure ops (median of N)
            if (backend_name.find("DDC") != std::string::npos) {
                auto* ha = dynamic_cast<DDCHandle*>(bm0.get());
                auto* hb = dynamic_cast<DDCHandle*>(bm1.get());
                if (ha && hb) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };

                    auto ddc_and_once = [&]() {
                        return and_pair_mode() ? (ha->compressed & hb->compressed)
                                               : ha->compressed.and_no_bypass(ha->compressed);
                    };
                    { DDC w = ddc_and_once(); (void)w; }
                    { DDC w = ha->compressed | hb->compressed; (void)w; }
                    { DDC w = ha->compressed ^ hb->compressed; (void)w; }
                    { DDC w = ~ha->compressed;                 (void)w; }

                    std::vector<double> and_t, or_t, xor_t, not_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        DDC and_r = ddc_and_once();
                        and_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        DDC or_r = ha->compressed | hb->compressed;
                        or_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        DDC xor_r = ha->compressed ^ hb->compressed;
                        xor_t.push_back(timer.elapsed_ms());
                    }

                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        DDC not_r = ~ha->compressed;
                        not_t.push_back(timer.elapsed_ms());
                    }
                    double and_pure = median(and_t);
                    double or_pure  = median(or_t);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t);

                    {
                        std::vector<double> mirror_or;
                        for (int i = 0; i < N_ITER; i++) {
                            auto t0 = std::chrono::high_resolution_clock::now();
                            DDC or_r = ha->compressed | hb->compressed;
                            auto t1 = std::chrono::high_resolution_clock::now();
                            mirror_or.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                        }
                        std::sort(mirror_or.begin(), mirror_or.end());
                        std::cout << "  [MIRROR-IN-PROC] OR raw: ";
                        for (double x : mirror_or) std::cout << x << " ";
                        std::cout << " median=" << mirror_or[N_ITER/2] << "\n";
                    }

                    DDC and_r = ddc_and_once();
                    DDC or_r  = ha->compressed | hb->compressed;
                    DDC xor_r = ha->compressed ^ hb->compressed;
                    DDC not_r = ~ha->compressed;

                    std::cout << "\n[Pure Ops] DDC operation-only timing:\n";
                    std::cout << "  AND: " << and_pure << " ms (card: " << and_r.popcount() << ")\n";
                    std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.popcount() << ")\n";
                    std::cout << "  XOR: " << xor_pure << " ms (card: " << xor_r.popcount() << ")\n";
                    std::cout << "  NOT: " << not_pure << " ms (card: " << not_r.popcount() << ")\n";

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, or_r.popcount(),  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, and_r.popcount(), 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, xor_r.popcount(), 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, not_r.popcount(), 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<DDCHandle*>(bm2.get()) : ha;
                        if (hc) {

                            // COMP: literal 4-op plan
                            auto run_comp = [&]() {
                                DDC t1 = ha->compressed | hb->compressed;
                                DDC t2 = hb->compressed | hc->compressed;
                                t1 &= t2;
                                t1.negate_inplace();
                                return t1;
                            };
                            { DDC r = run_comp(); (void)r; }
                            std::vector<double> comp_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                DDC t1 = ha->compressed | hb->compressed;
                                DDC t2 = hb->compressed | hc->compressed;
                                t1 &= t2;
                                t1.negate_inplace();
                                double t = timer.elapsed_ms();
                                asm volatile("" : : "r"(&t1) : "memory");
                                comp_t.push_back(t);
                            }
                            double comp_pure = median(comp_t);
                            DDC final_r = run_comp();

                            DDC ref = ~((ha->compressed | hb->compressed).and_no_bypass(hb->compressed | hc->compressed));
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_pure
                                      << " ms (card: " << final_r.popcount()
                                      << (final_r.popcount()==ref.popcount()?" OK":" MISMATCH!") << ")\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, final_r.popcount(), 1);
                        }
                    }
                }
            }

            // WAH pure ops (copy + in-place)
            if (backend_name.find("WAH") != std::string::npos) {
                auto* ha = dynamic_cast<WahHandle*>(bm0.get());
                auto* hb = dynamic_cast<WahHandle*>(bm1.get());
                if (ha && hb) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };
                    { ibis::bitvector w; w.copy(ha->btv); w |= hb->btv; (void)w; }

                    const ibis::bitvector& wah_and_rhs = and_pair_mode() ? hb->btv : ha->btv;
                    { ibis::bitvector w; w.copy(ha->btv); w &= wah_and_rhs; (void)w; }
                    { ibis::bitvector w; w.copy(ha->btv); w ^= hb->btv; (void)w; }
                    { ibis::bitvector w; w.copy(ha->btv); w.flip();    (void)w; }

                    std::vector<double> and_t, or_t, xor_t, not_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ibis::bitvector r; r.copy(ha->btv); r |= hb->btv;
                        or_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ibis::bitvector r; r.copy(ha->btv); r &= wah_and_rhs;
                        and_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ibis::bitvector r; r.copy(ha->btv); r ^= hb->btv;
                        xor_t.push_back(timer.elapsed_ms());
                    }

                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ibis::bitvector r; r.copy(ha->btv); r.flip();
                        not_t.push_back(timer.elapsed_ms());
                    }
                    double or_pure  = median(or_t);
                    double and_pure = median(and_t);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t);
                    std::cout << "\n[Pure Ops] WAH (FastBit) operation-only:\n";
                    std::cout << "  AND: " << and_pure << " ms\n";
                    std::cout << "  OR:  " << or_pure  << " ms\n";
                    std::cout << "  XOR: " << xor_pure << " ms\n";
                    std::cout << "  NOT: " << not_pure << " ms\n";
                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, 0, 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<WahHandle*>(bm2.get()) : ha;
                        if (hc) {
                            { ibis::bitvector t1; t1.copy(ha->btv); t1 |= hb->btv;
                              ibis::bitvector t2; t2.copy(hb->btv); t2 |= hc->btv;
                              ibis::bitvector t3; t3.copy(t1);      t3 &= t2;
                              ibis::bitvector r;  r.copy(t3);       r.flip(); (void)r; }
                            std::vector<double> comp_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                ibis::bitvector t1; t1.copy(ha->btv); t1 |= hb->btv;
                                ibis::bitvector t2; t2.copy(hb->btv); t2 |= hc->btv;
                                ibis::bitvector t3; t3.copy(t1);      t3 &= t2;
                                ibis::bitvector r;  r.copy(t3);       r.flip();
                                const double t = timer.elapsed_ms();
                                asm volatile("" : : "r"(&r) : "memory");
                                comp_t.push_back(t);
                            }
                            double comp_pure = median(comp_t);
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_pure << " ms\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, 0, 1);

                            std::vector<double> comp_conv_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                ibis::bitvector t1; t1.copy(ha->btv); t1 |= hb->btv;
                                ibis::bitvector t2; t2.copy(hb->btv); t2 |= hc->btv;
                                ibis::bitvector t3; t3.copy(t1);      t3 &= t2;
                                ibis::bitvector r;  r.copy(t3);       r.flip();
                                double t = timer.elapsed_ms();
                                t += dense_btv_delivery::wah_time(
                                    r, static_cast<uint32_t>(num_rows));
                                comp_conv_t.push_back(t);
                            }
                            double comp_conv = median(comp_conv_t);
                            ibis::bitvector verify_t1; verify_t1.copy(ha->btv); verify_t1 |= hb->btv;
                            ibis::bitvector verify_t2; verify_t2.copy(hb->btv); verify_t2 |= hc->btv;
                            ibis::bitvector verify_t3; verify_t3.copy(verify_t1); verify_t3 &= verify_t2;
                            ibis::bitvector verify_r; verify_r.copy(verify_t3); verify_r.flip();
                            auto* verify_dense = dense_btv_delivery::from_wah(
                                verify_r, static_cast<uint32_t>(num_rows));
                            const uint64_t verify_cardinality =
                                roaring::api::bitset_count(verify_dense);
                            roaring::api::bitset_free(verify_dense);
                            std::cout << "  COMP (+ dense BTV): " << comp_conv << " ms\n";
                            csv_row(backend_name, num_rows, cardinality,
                                    "COMP_op_conv", comp_conv, 0, 0,
                                    verify_cardinality, 1);

                        }
                    }
                }
            }

            // EWAH pure ops
            if (backend_name == "EWAH") {
                auto* ha = dynamic_cast<EwahHandle*>(bm0.get());
                auto* hb = dynamic_cast<EwahHandle*>(bm1.get());
                if (ha && hb) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };
                    { ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalor(hb->btv, r); (void)r; }

                    const auto& ewah_and_rhs = and_pair_mode() ? hb->btv : ha->btv;
                    { ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicaland(ewah_and_rhs, r); (void)r; }
                    { ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalxor(hb->btv, r); (void)r; }
                    { ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalnot(r);          (void)r; }

                    std::vector<double> and_t, or_t, xor_t, not_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalor(hb->btv, r);
                        or_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicaland(ewah_and_rhs, r);
                        and_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalxor(hb->btv, r);
                        xor_t.push_back(timer.elapsed_ms());
                    }

                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ewah::EWAHBoolArray<uint64_t> r; ha->btv.logicalnot(r);
                        not_t.push_back(timer.elapsed_ms());
                    }
                    double or_pure  = median(or_t);
                    double and_pure = median(and_t);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t);
                    std::cout << "\n[Pure Ops] EWAH operation-only:\n";
                    std::cout << "  AND: " << and_pure << " ms\n";
                    std::cout << "  OR:  " << or_pure  << " ms\n";
                    std::cout << "  XOR: " << xor_pure << " ms\n";
                    std::cout << "  NOT: " << not_pure << " ms\n";
                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, 0, 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<EwahHandle*>(bm2.get()) : ha;
                        if (hc) {
                            { ewah::EWAHBoolArray<uint64_t> t1; ha->btv.logicalor(hb->btv, t1);
                              ewah::EWAHBoolArray<uint64_t> t2; hb->btv.logicalor(hc->btv, t2);
                              ewah::EWAHBoolArray<uint64_t> t3; t1.logicaland(t2, t3);
                              ewah::EWAHBoolArray<uint64_t> r;  t3.logicalnot(r); (void)r; }
                            std::vector<double> comp_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                ewah::EWAHBoolArray<uint64_t> t1; ha->btv.logicalor(hb->btv, t1);
                                ewah::EWAHBoolArray<uint64_t> t2; hb->btv.logicalor(hc->btv, t2);
                                ewah::EWAHBoolArray<uint64_t> t3; t1.logicaland(t2, t3);
                                ewah::EWAHBoolArray<uint64_t> r;  t3.logicalnot(r);
                                const double t = timer.elapsed_ms();
                                asm volatile("" : : "r"(&r) : "memory");
                                comp_t.push_back(t);
                            }
                            double comp_pure = median(comp_t);
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_pure << " ms\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, 0, 1);

                            std::vector<double> comp_conv_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                ewah::EWAHBoolArray<uint64_t> t1; ha->btv.logicalor(hb->btv, t1);
                                ewah::EWAHBoolArray<uint64_t> t2; hb->btv.logicalor(hc->btv, t2);
                                ewah::EWAHBoolArray<uint64_t> t3; t1.logicaland(t2, t3);
                                ewah::EWAHBoolArray<uint64_t> r;  t3.logicalnot(r);
                                double t = timer.elapsed_ms();
                                t += dense_btv_delivery::ewah_time(
                                    r, static_cast<uint32_t>(num_rows));
                                comp_conv_t.push_back(t);
                            }
                            double comp_conv = median(comp_conv_t);
                            ewah::EWAHBoolArray<uint64_t> verify_t1;
                            ewah::EWAHBoolArray<uint64_t> verify_t2;
                            ewah::EWAHBoolArray<uint64_t> verify_t3;
                            ewah::EWAHBoolArray<uint64_t> verify_r;
                            ha->btv.logicalor(hb->btv, verify_t1);
                            hb->btv.logicalor(hc->btv, verify_t2);
                            verify_t1.logicaland(verify_t2, verify_t3);
                            verify_t3.logicalnot(verify_r);
                            auto* verify_dense = dense_btv_delivery::from_ewah(
                                verify_r, static_cast<uint32_t>(num_rows));
                            const uint64_t verify_cardinality =
                                roaring::api::bitset_count(verify_dense);
                            roaring::api::bitset_free(verify_dense);
                            std::cout << "  COMP (+ dense BTV): " << comp_conv << " ms\n";
                            csv_row(backend_name, num_rows, cardinality,
                                    "COMP_op_conv", comp_conv, 0, 0,
                                    verify_cardinality, 1);

                        }
                    }
                }
            }

            // Concise pure ops (NOT = XOR all-ones)
            if (backend_name == "Concise") {
                auto* ha = dynamic_cast<ConciseHandle*>(bm0.get());
                auto* hb = dynamic_cast<ConciseHandle*>(bm1.get());
                if (ha && hb) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };

                    ConciseSet<false> all_ones;
                    {
                        uint64_t n = std::max(ha->current_bits, hb->current_bits);
                        for (uint32_t i = 0; i < static_cast<uint32_t>(n); i++)
                            all_ones.append(i);
                    }
                    { ConciseSet<false> w = ha->btv | hb->btv; (void)w; }

                    const ConciseSet<false>& con_and_rhs = and_pair_mode() ? hb->btv : ha->btv;
                    { ConciseSet<false> w = ha->btv & con_and_rhs; (void)w; }
                    { ConciseSet<false> w = ha->btv ^ hb->btv; (void)w; }
                    { ConciseSet<false> w = ha->btv ^ all_ones; (void)w; }

                    std::vector<double> and_t, or_t, xor_t, not_t;
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ConciseSet<false> r = ha->btv | hb->btv;
                        or_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ConciseSet<false> r = ha->btv & con_and_rhs;
                        and_t.push_back(timer.elapsed_ms());
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ConciseSet<false> r = ha->btv ^ hb->btv;
                        xor_t.push_back(timer.elapsed_ms());
                    }

                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        ConciseSet<false> r = ha->btv ^ all_ones;
                        not_t.push_back(timer.elapsed_ms());
                    }
                    double or_pure  = median(or_t);
                    double and_pure = median(and_t);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t);
                    std::cout << "\n[Pure Ops] Concise operation-only:\n";
                    std::cout << "  AND: " << and_pure << " ms\n";
                    std::cout << "  OR:  " << or_pure  << " ms\n";
                    std::cout << "  XOR: " << xor_pure << " ms\n";
                    std::cout << "  NOT: " << not_pure << " ms (XOR with all-ones)\n";
                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, 0, 1);

                    if (!bm2 && selected.size() >= 3) bm2 = backend->Load(selected[2]);
                    {
                        auto* hc = bm2 ? dynamic_cast<ConciseHandle*>(bm2.get()) : ha;
                        if (hc) {
                            { ConciseSet<false> t1 = ha->btv | hb->btv;
                              ConciseSet<false> t2 = hb->btv | hc->btv;
                              ConciseSet<false> t3 = t1 & t2;
                              ConciseSet<false> r  = t3 ^ all_ones; (void)r; }
                            std::vector<double> comp_t;
                            for (int i = 0; i < N_ITER; i++) {
                                timer.reset();
                                ConciseSet<false> t1 = ha->btv | hb->btv;
                                ConciseSet<false> t2 = hb->btv | hc->btv;
                                ConciseSet<false> t3 = t1 & t2;
                                ConciseSet<false> r  = t3 ^ all_ones;
                                comp_t.push_back(timer.elapsed_ms());
                            }
                            double comp_pure = median(comp_t);
                            std::cout << "  COMP (~((A|B)&(B|C))): " << comp_pure << " ms\n";
                            csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, 0, 1);
                        }
                    }
                }
            }

            // QFilter/BSR pure ops (median of N).
            // AND runs the vendored SIGMOD'18 kernels (adaptive QFilter /
            // SIMDGalloping, per the paper's own switching rule).
            // OR/XOR are OUR scalar BSR merges — upstream implements
            // intersection only — and must be labelled as such in figures.
            // NOT is OUR semantically-correct complement over the universe
            // (upstream has none): omitted chunks materialize as all-one,
            // all-one chunks vanish, tail chunk masked — the result is a
            // real BSR, so the measured cost is the representational cost
            // (O(num_rows/32), ~2N bits for sparse inputs).  COMP composes
            // our OR + upstream AND + our NOT.  Label both as ours.
            if (backend_name == "QFilter-BSR") {
                auto* ha = dynamic_cast<BsrHandle*>(bm0.get());
                auto* hb = dynamic_cast<BsrHandle*>(bm1.get());
                if (ha && hb) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };
                    // Result buffers are caller-provided in the upstream API,
                    // but DDC/CR construct their result containers INSIDE the
                    // timed region (DDC resizes ~12.5MB zeroed L1 per op).
                    // For a same-caliber comparison we allocate and free the
                    // BSR output arrays inside each timed iteration.
                    //
                    // Delivery contract, mirroring the CRoaring block above
                    // (git 7bc9f2f "add CRoaring to-bitset conversion timing"):
                    // right after each timed op, the SAME iteration's own
                    // result is converted to a dense bitset and that time is
                    // added on top.  *_op rows = op alone; *_op_conv rows =
                    // op + own-result conversion, exactly like CR's
                    // `t += croaring_to_bitset(r, ...)` accounting.
                    const size_t num_words = (num_rows + 63) / 64;
                    std::vector<uint64_t> dense(num_words);
                    auto conv_ms_of = [&](const int* b, const int* s, int n) {
                        Timer ct;
                        ct.reset();
                        bsr_util::to_dense_bitset(b, s, n, dense.data(), num_words);
                        return ct.elapsed_ms();
                    };

                    int *bc = nullptr, *sc = nullptr;

                    // ---- AND ----
                    const int* and_bases = and_pair_mode() ? hb->bases : ha->bases;
                    const int* and_states = and_pair_mode() ? hb->states : ha->states;
                    const int and_size = and_pair_mode() ? hb->size : ha->size;
                    const int and_alloc = and_pair_mode() ? std::min(ha->size, hb->size) : ha->size;
                    std::vector<double> and_t, and_cv;
                    int and_n = 0; uint64_t and_card = 0;
                    {
                        bsr_util::alloc_arrays(and_alloc, &bc, &sc);   // warm-up
                        and_n = bsr_util::intersect_adaptive(
                            ha->bases, ha->states, ha->size,
                            and_bases, and_states, and_size, bc, sc);
                        conv_ms_of(bc, sc, and_n);
                        free(bc); free(sc); bc = sc = nullptr;
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        bsr_util::alloc_arrays(and_alloc, &bc, &sc);
                        and_n = bsr_util::intersect_adaptive(
                            ha->bases, ha->states, ha->size,
                            and_bases, and_states, and_size, bc, sc);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(bc, sc, and_n);
                        and_t.push_back(t_op);
                        and_cv.push_back(t_op + t_cv);
                        and_card = bsr_util::popcount(sc, and_n);
                        free(bc); free(sc); bc = sc = nullptr;
                    }

                    // ---- OR = A | B, our scalar merge ----
                    std::vector<double> or_t, or_cv;
                    int or_n = 0; uint64_t or_card = 0;
                    {
                        bsr_util::alloc_arrays(ha->size + hb->size, &bc, &sc);  // warm-up
                        or_n = bsr_util::union_scalar(
                            ha->bases, ha->states, ha->size,
                            hb->bases, hb->states, hb->size, bc, sc);
                        conv_ms_of(bc, sc, or_n);
                        free(bc); free(sc); bc = sc = nullptr;
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        bsr_util::alloc_arrays(ha->size + hb->size, &bc, &sc);
                        or_n = bsr_util::union_scalar(
                            ha->bases, ha->states, ha->size,
                            hb->bases, hb->states, hb->size, bc, sc);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(bc, sc, or_n);
                        or_t.push_back(t_op);
                        or_cv.push_back(t_op + t_cv);
                        or_card = bsr_util::popcount(sc, or_n);
                        free(bc); free(sc); bc = sc = nullptr;
                    }

                    // ---- XOR = A ^ B (no figure; op only, uncharged like the
                    // RLE backends) ----
                    std::vector<double> xor_t;
                    volatile int sink = 0;
                    {
                        bsr_util::alloc_arrays(ha->size + hb->size, &bc, &sc);  // warm-up
                        sink = bsr_util::xor_scalar(
                            ha->bases, ha->states, ha->size,
                            hb->bases, hb->states, hb->size, bc, sc);
                        free(bc); free(sc); bc = sc = nullptr;
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        bsr_util::alloc_arrays(ha->size + hb->size, &bc, &sc);
                        sink = bsr_util::xor_scalar(
                            ha->bases, ha->states, ha->size,
                            hb->bases, hb->states, hb->size, bc, sc);
                        xor_t.push_back(timer.elapsed_ms());
                        free(bc); free(sc); bc = sc = nullptr;
                    }
                    (void)sink;

                    // ---- NOT = our semantic complement of A over [0, num_rows) ----
                    const int not_cap = static_cast<int>((num_rows + 31) / 32);
                    std::vector<double> not_t, not_cv;
                    int not_n = 0; uint64_t not_card = 0, not_bytes = 0;
                    {
                        bsr_util::alloc_arrays(not_cap, &bc, &sc);   // warm-up
                        not_n = bsr_util::not_scalar(ha->bases, ha->states, ha->size,
                                                     num_rows, bc, sc);
                        conv_ms_of(bc, sc, not_n);
                        free(bc); free(sc); bc = sc = nullptr;
                    }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        bsr_util::alloc_arrays(not_cap, &bc, &sc);
                        not_n = bsr_util::not_scalar(ha->bases, ha->states, ha->size,
                                                     num_rows, bc, sc);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(bc, sc, not_n);
                        not_t.push_back(t_op);
                        not_cv.push_back(t_op + t_cv);
                        not_card = bsr_util::popcount(sc, not_n);
                        not_bytes = static_cast<uint64_t>(not_n) * 8;
                        free(bc); free(sc); bc = sc = nullptr;
                    }

                    // ---- COMP = ~((A|B) & (B|C)); C falls back to A at c=2,
                    // matching the generic path above ----
                    double comp_pure = -1.0, comp_conv = -1.0;
                    if (!bm2) bm2 = (selected.size() >= 3)
                                        ? backend->Load(selected[2])
                                        : backend->Load(selected[0]);
                    auto* hc = bm2 ? dynamic_cast<BsrHandle*>(bm2.get()) : nullptr;
                    if (hc) {
                        std::vector<double> comp_t, comp_cv;
                        for (int i = 0; i < N_ITER + 1; i++) {   // i==0 is warm-up
                            int *b1=nullptr,*s1=nullptr,*b2=nullptr,*s2=nullptr;
                            int *b3=nullptr,*s3=nullptr,*b4=nullptr,*s4=nullptr;
                            timer.reset();
                            bsr_util::alloc_arrays(ha->size + hb->size, &b1, &s1);
                            int n1 = bsr_util::union_scalar(ha->bases, ha->states, ha->size,
                                                            hb->bases, hb->states, hb->size, b1, s1);
                            bsr_util::alloc_arrays(hb->size + hc->size, &b2, &s2);
                            int n2 = bsr_util::union_scalar(hb->bases, hb->states, hb->size,
                                                            hc->bases, hc->states, hc->size, b2, s2);
                            bsr_util::alloc_arrays(std::min(n1, n2), &b3, &s3);
                            int n3 = bsr_util::intersect_adaptive(b1, s1, n1, b2, s2, n2, b3, s3);
                            bsr_util::alloc_arrays(not_cap, &b4, &s4);
                            int n4 = bsr_util::not_scalar(b3, s3, n3, num_rows, b4, s4);
                            double t_op = timer.elapsed_ms();
                            double t_cv = conv_ms_of(b4, s4, n4);
                            if (i > 0) { comp_t.push_back(t_op); comp_cv.push_back(t_op + t_cv); }
                            free(b1); free(s1); free(b2); free(s2);
                            free(b3); free(s3); free(b4); free(s4);
                        }
                        comp_pure = median(comp_t);
                        comp_conv = median(comp_cv);
                    }

                    double and_pure = median(and_t), and_full = median(and_cv);
                    double or_pure  = median(or_t),  or_full  = median(or_cv);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t), not_full = median(not_cv);

                    std::cout << "\n[Pure Ops] QFilter-BSR (SIGMOD'18 kernels; OR/XOR = our scalar BSR merge):\n";
                    std::cout << "  AND (QFilter, " << (and_pair_mode() ? "A&B" : "A&A") << "): " << and_pure << " ms, +conv -> "
                              << and_full << " ms (card: " << and_card << ")\n";
                    std::cout << "  OR  (scalar merge): " << or_pure << " ms, +conv -> "
                              << or_full << " ms (card: " << or_card << ")\n";
                    std::cout << "  XOR (scalar merge): " << xor_pure << " ms\n";
                    std::cout << "  NOT (our semantic complement, materialized): " << not_pure
                              << " ms, +conv -> " << not_full << " ms (card: " << not_card
                              << ", result " << not_n << " chunks = " << not_bytes / 1e6 << " MB)\n";
                    if (comp_pure >= 0)
                        std::cout << "  COMP (~((A|B)&(B|C)), ours+QFilter): " << comp_pure
                                  << " ms, +conv -> " << comp_conv << " ms\n";

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, or_card,  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, and_card, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, not_card, 1);
                    if (comp_pure >= 0)
                        csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, 0, 1);
                    // *_op_conv = median over iterations of (op + own-result
                    // -> dense-bitset conversion): the delivery contract
                    // CRoaring's *_op rows pay via croaring_to_bitset and DDC
                    // pays natively via its decompressed result
                    csv_row(backend_name, num_rows, cardinality, "OR_op_conv",  or_full,  0, 0, or_card,  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op_conv", and_full, 0, 0, and_card, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op_conv", not_full, 0, 0, not_card, 1);
                    if (comp_pure >= 0)
                        csv_row(backend_name, num_rows, cardinality, "COMP_op_conv", comp_conv, 0, 0, 0, 1);
                }
            }

            // HBI-Pointer pure ops (median of N) — hbi branch, R1.W1/R3-W2
            // rebuttal.  Node-oriented tree baseline (Morzy et al., ADBIS'03)
            // extended with compressed-to-compressed Boolean algebra; see
            // src/core/hbi/hbi.h for the faithful-but-fair rules.
            // Delivery contract mirrors CRoaring/QFilter-BSR: per timed
            // iteration the op runs first (INCLUDING result-tree construction
            // and child-block allocation), then the same iteration's own
            // result is converted to a dense bitset.  *_op rows = op alone,
            // *_op_conv rows = op + conversion.  COMP keeps every intermediate
            // as HBI and converts only the final expression result, once.
            if (backend_name.rfind("HBI", 0) == 0) {
                auto* ha = dynamic_cast<HbiHandle*>(bm0.get());
                auto* hb = dynamic_cast<HbiHandle*>(bm1.get());
                // independent second copy of A: the harness A&A convention must
                // not shortcut via pointer identity (2026-07-22 spec)
                std::unique_ptr<BitmapHandle> bm0_dup = backend->Load(selected[0]);
                auto* ha2 = dynamic_cast<HbiHandle*>(bm0_dup.get());
                if (ha && hb && ha2) {
                    constexpr int N_ITER = 100;   // D2: formal caliber = tight-loop 100-median (2026-07-31)
                    auto median = [](std::vector<double>& v) {
                        std::sort(v.begin(), v.end());
                        return v[v.size() / 2];
                    };
                    const size_t num_words = (num_rows + 63) / 64;
                    std::vector<uint64_t> dense_out(num_words);
                    auto conv_ms_of = [&](const hbi::HbiBitmap& r) {
                        Timer ct;
                        ct.reset();
                        r.to_dense(dense_out.data(), num_words);
                        return ct.elapsed_ms();
                    };

                    // ---- AND ----
                    const hbi::HbiBitmap& hbi_and_rhs = and_pair_mode() ? hb->bm : ha2->bm;
                    std::vector<double> and_t, and_cv;
                    uint64_t and_card = 0;
                    { hbi::HbiBitmap w = hbi::HbiBitmap::and_(ha->bm, hbi_and_rhs);
                      conv_ms_of(w); }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        hbi::HbiBitmap r = hbi::HbiBitmap::and_(ha->bm, hbi_and_rhs);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(r);
                        and_t.push_back(t_op);
                        and_cv.push_back(t_op + t_cv);
                        and_card = r.popcount();
                    }

                    // ---- OR = A | B ----
                    std::vector<double> or_t, or_cv;
                    uint64_t or_card = 0;
                    { hbi::HbiBitmap w = hbi::HbiBitmap::or_(ha->bm, hb->bm);
                      conv_ms_of(w); }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        hbi::HbiBitmap r = hbi::HbiBitmap::or_(ha->bm, hb->bm);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(r);
                        or_t.push_back(t_op);
                        or_cv.push_back(t_op + t_cv);
                        or_card = r.popcount();
                    }

                    // ---- XOR = A ^ B (no figure: op only, uncharged) ----
                    std::vector<double> xor_t;
                    { hbi::HbiBitmap w = hbi::HbiBitmap::xor_(ha->bm, hb->bm); (void)w; }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        hbi::HbiBitmap r = hbi::HbiBitmap::xor_(ha->bm, hb->bm);
                        xor_t.push_back(timer.elapsed_ms());
                        volatile uint64_t sink = r.node_count(); (void)sink;
                    }

                    // ---- NOT = ~A (materializes elided zero regions) ----
                    std::vector<double> not_t, not_cv;
                    uint64_t not_card = 0, not_nodes = 0, not_allocs = 0, not_bytes = 0;
                    { hbi::HbiBitmap w = hbi::HbiBitmap::not_(ha->bm);
                      conv_ms_of(w); }
                    for (int i = 0; i < N_ITER; i++) {
                        timer.reset();
                        hbi::HbiBitmap r = hbi::HbiBitmap::not_(ha->bm);
                        double t_op = timer.elapsed_ms();
                        double t_cv = conv_ms_of(r);
                        not_t.push_back(t_op);
                        not_cv.push_back(t_op + t_cv);
                        not_card  = r.popcount();
                        not_nodes = r.node_count();
                        not_allocs = r.alloc_calls();
                        not_bytes = r.allocated_bytes();
                    }

                    // ---- COMP = ~((A|B) & (B|C)); C falls back to A at c=2 ----
                    double comp_pure = -1.0, comp_full = -1.0;
                    if (!bm2) bm2 = (selected.size() >= 3)
                                        ? backend->Load(selected[2])
                                        : backend->Load(selected[0]);
                    auto* hc = bm2 ? dynamic_cast<HbiHandle*>(bm2.get()) : nullptr;
                    if (hc) {
                        std::vector<double> comp_t, comp_cv;
                        for (int i = 0; i < N_ITER + 1; i++) {   // i==0 warm-up
                            timer.reset();
                            hbi::HbiBitmap t1 = hbi::HbiBitmap::or_(ha->bm, hb->bm);
                            hbi::HbiBitmap t2 = hbi::HbiBitmap::or_(hb->bm, hc->bm);
                            hbi::HbiBitmap t3 = hbi::HbiBitmap::and_(t1, t2);
                            hbi::HbiBitmap r  = hbi::HbiBitmap::not_(t3);
                            double t_op = timer.elapsed_ms();
                            double t_cv = conv_ms_of(r);   // final result only
                            if (i > 0) { comp_t.push_back(t_op); comp_cv.push_back(t_op + t_cv); }
                        }
                        comp_pure = median(comp_t);
                        comp_full = median(comp_cv);
                    }

                    double and_pure = median(and_t), and_full = median(and_cv);
                    double or_pure  = median(or_t),  or_full  = median(or_cv);
                    double xor_pure = median(xor_t);
                    double not_pure = median(not_t), not_full = median(not_cv);

                    std::cout << "\n[Pure Ops] " << backend_name
                              << " (pointer tree, node_bits=" << ha->bm.node_bits()
                              << ", depth=" << ha->bm.depth() << "):\n";
                    std::cout << "  AND (A&A\', independent objects): " << and_pure
                              << " ms, +conv -> " << and_full << " ms (card: " << and_card << ")\n";
                    std::cout << "  OR : " << or_pure << " ms, +conv -> " << or_full
                              << " ms (card: " << or_card << ")\n";
                    std::cout << "  XOR: " << xor_pure << " ms\n";
                    std::cout << "  NOT: " << not_pure << " ms, +conv -> " << not_full
                              << " ms (card: " << not_card
                              << ", result nodes: " << not_nodes
                              << ", child-block allocs: " << not_allocs
                              << ", arena bytes: " << not_bytes << ")\n";
                    if (comp_pure >= 0)
                        std::cout << "  COMP (~((A|B)&(B|C)), HBI intermediates): "
                                  << comp_pure << " ms, +conv -> " << comp_full << " ms\n";

                    csv_row(backend_name, num_rows, cardinality, "OR_op",  or_pure,  0, 0, or_card,  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op", and_pure, 0, 0, and_card, 1);
                    csv_row(backend_name, num_rows, cardinality, "XOR_op", xor_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op", not_pure, 0, 0, not_card, 1);
                    if (comp_pure >= 0)
                        csv_row(backend_name, num_rows, cardinality, "COMP_op", comp_pure, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "OR_op_conv",  or_full,  0, 0, or_card,  1);
                    csv_row(backend_name, num_rows, cardinality, "AND_op_conv", and_full, 0, 0, and_card, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_op_conv", not_full, 0, 0, not_card, 1);
                    if (comp_pure >= 0)
                        csv_row(backend_name, num_rows, cardinality, "COMP_op_conv", comp_full, 0, 0, 0, 1);
                    // NOT materialization metrics (machine-readable; time_ms
                    // column carries the raw counter value)
                    csv_row(backend_name, num_rows, cardinality, "NOT_nodes",       (double)not_nodes,  0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_alloc_calls", (double)not_allocs, 0, 0, 0, 1);
                    csv_row(backend_name, num_rows, cardinality, "NOT_alloc_bytes", (double)not_bytes,  0, 0, 0, 1);
                }
            }
        }
    }

    std::cout << "---------------------------------------\n";
}

// cross-cardinality OR/AND bench
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

    if (backend_name.find("DDC") != std::string::npos) {
        auto print_sb = [](const std::string& label, BitmapHandle* bm) {
            auto* ch = dynamic_cast<DDCHandle*>(bm);
            if (ch) {
                auto sb = ch->compressed.size_breakdown();
                std::cout << "  [DDC " << label << "] leading: " << sb.l3_bits
                          << " bits | literal: " << sb.l1_literal_bits
                          << " bits | total: " << sb.total_bits << " bits ("
                          << sb.total_bits / 8.0 / 1024.0 << " KB)\n";
            }
        };
        print_sb("A", bm_a.get());
        print_sb("B", bm_b.get());
    }

    { auto w = backend->bitOr(*bm_a, *bm_b); (void)w; }
    timer.reset();
    auto or_res = backend->bitOr(*bm_a, *bm_b);
    double or_t = timer.elapsed_ms();

    timer.reset();
    auto and_res = backend->bitAnd(*bm_a, *bm_b);
    double and_t = timer.elapsed_ms();

    if (backend_name == "CRoaring") {
        auto* ha = dynamic_cast<CroaringHandle*>(bm_a.get());
        auto* hb = dynamic_cast<CroaringHandle*>(bm_b.get());
        if (ha && hb) {
            uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
            auto* or_h  = dynamic_cast<CroaringHandle*>(or_res.get());
            auto* and_h = dynamic_cast<CroaringHandle*>(and_res.get());
            if (or_h)  or_t  += croaring_to_bitset(or_h->bitmap, logical_sz);
            if (and_h) and_t += croaring_to_bitset(and_h->bitmap, logical_sz);
        }
    }

    uint64_t or_card = backend->Cardinality(*or_res);
    uint64_t and_card = backend->Cardinality(*and_res);

    std::cout << "\n[Result] c=" << info_a.cardinality << " OR c=" << info_b.cardinality << ":\n";
    std::cout << "  OR:  " << or_t  << " ms (card: " << or_card  << ")\n";
    std::cout << "  AND: " << and_t << " ms (card: " << and_card << ")\n";

    if (backend_name == "CRoaring") {
        auto* ha = dynamic_cast<CroaringHandle*>(bm_a.get());
        auto* hb = dynamic_cast<CroaringHandle*>(bm_b.get());
        if (ha && hb) {
            uint32_t logical_sz = std::max(ha->current_size, hb->current_size);
            { roaring::Roaring w = ha->bitmap | hb->bitmap; (void)w; }

            timer.reset();
            roaring::Roaring or_r = ha->bitmap | hb->bitmap;
            double or_pure = timer.elapsed_ms();
            or_pure += croaring_to_bitset(or_r, logical_sz);

            timer.reset();
            roaring::Roaring and_r = ha->bitmap & hb->bitmap;
            double and_pure = timer.elapsed_ms();
            and_pure += croaring_to_bitset(and_r, logical_sz);

            timer.reset();
            roaring::Roaring xor_r = ha->bitmap ^ hb->bitmap;
            double xor_pure = timer.elapsed_ms();
            xor_pure += croaring_to_bitset(xor_r, logical_sz);

            roaring::Roaring a_copy = ha->bitmap;
            timer.reset();
            a_copy |= hb->bitmap;
            double or_inplace = timer.elapsed_ms();
            or_inplace += croaring_to_bitset(a_copy, logical_sz);

            std::cout << "\n[Pure Ops] CRoaring (+ to-btv conversion):\n";
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

    if (backend_name.find("DDC") != std::string::npos) {
        auto* ha = dynamic_cast<DDCHandle*>(bm_a.get());
        auto* hb = dynamic_cast<DDCHandle*>(bm_b.get());
        if (ha && hb) {
            { DDC w = ha->compressed | hb->compressed; (void)w; }
            timer.reset();
            DDC or_r = ha->compressed | hb->compressed;
            double or_pure = timer.elapsed_ms();
            timer.reset();
            DDC and_r = ha->compressed & hb->compressed;
            double and_pure = timer.elapsed_ms();

            std::cout << "\n[Pure Ops] DDC:\n";
            std::cout << "  OR:  " << or_pure  << " ms (card: " << or_r.popcount() << ")\n";
            std::cout << "  AND: " << and_pure << " ms (card: " << and_r.popcount() << ")\n";
        }
    }

    std::string label = "c" + std::to_string(info_a.cardinality) + ":c" + std::to_string(info_b.cardinality);
    csv_row(backend_name, num_rows, 0, "cross-OR(" + label + ")", or_t, 0, 0, or_card, 1);
    csv_row(backend_name, num_rows, 0, "cross-AND(" + label + ")", and_t, 0, 0, and_card, 1);

    std::cout << "---------------------------------------\n";
}

// arg parse + dispatch
int main(int argc, char** argv) {
    std::string backend_type = "all";
    std::string bm_dir;
    std::string compressed_dir;
    std::string cross_or_dir_a, cross_or_dir_b;
    size_t num_rows = 0;
    size_t sample_count = 0;
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
        } else if (arg == "--perf-op" && i + 1 < argc) {
            g_perf_op = argv[++i];
        } else if (arg == "--perf-loops" && i + 1 < argc) {
            g_perf_loops = std::stol(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoull(argv[++i]);
            if (iterations == 0) iterations = 1;
        } else if (arg == "--compress-results") {

            ddc_compress_results = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

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
    DDCBackend ddc;
    EwahBackend ewah;
    ConciseBackend concise;
    BitsetBackend bitset;
    BitsetAVX512Backend bitset_avx512;
    BsrBackend bsr;
    HbiBackend hbi_backend32(32), hbi_backend16(16), hbi_backend8(8), hbi_backend4(4);

    // backend registry
    struct BackendEntry { IBitmapBackend* ptr; std::string name; std::string key; };
    const char* ddc_label = ddc_compress_results
                                 ? "DDC (compress)"
                                 : "DDC (New)";
    std::vector<BackendEntry> backends = {
        {&wah,            "WAH (FastBit)",       "wah"},
        {&croaring,       "CRoaring",            "croaring"},
        {&ddc,          ddc_label,         "ddc"},
        {&ewah,           "EWAH",                "ewah"},
        {&concise,        "Concise",             "concise"},
        {&bitset,         "Bitset (Plain)",      "bitset"},
        {&bitset_avx512,  "Bitset (AVX512)",     "bitset_avx512"},
        {&bsr,            "QFilter-BSR",         "bsr"},
        {&hbi_backend32,  "HBI-32",              "hbi32"},
        {&hbi_backend16,  "HBI-16",              "hbi16"},
        {&hbi_backend8,   "HBI-8",               "hbi8"},
        {&hbi_backend4,   "HBI-4",               "hbi4"},
    };

    if (!cross_or_dir_a.empty() && !cross_or_dir_b.empty()) {

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

        if (backend_type == "all" && !detected_key.empty()) {
            if (detected_key == "bitset")
                backend_type = "bitset_both";
            else
                backend_type = detected_key;
        }

        std::cout << "=== Compressed .bm Benchmark Mode ===\n";
        std::cout << "Directory: " << compressed_dir << "\n";
        std::cout << "Rows: " << rows << " | Cardinality: " << card << "\n";
        std::cout << "AND caliber: " << (and_pair_mode() ? "PAIR (A&B, real op)" : "frozen (self-AND convention)") << "\n";

        for (auto& be : backends) {
            if (backend_type == be.key || backend_type == "all"
                || (backend_type == "bitset_both" && (be.key == "bitset" || be.key == "bitset_avx512")))
                run_compressed_benchmark(be.ptr, be.name, compressed_dir, rows, card, sample_count, iterations);
        }
    } else if (!bm_dir.empty()) {

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
