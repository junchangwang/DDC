// gen_bitmap.cpp — Unified bitmap generation tool
//
// Reads a binary dataset and generates compressed bitmaps for any
// supported algorithm (WAH, Roaring, EWAH, Concise, ComBit).
// Also generates raw uncompressed bitmaps as a baseline.
//
// Usage:
//   gen_bitmap -n <rows> -c <cardinality> <algorithm> [-d <base_dir>] [-z <zip_length>] [-w <word_size>]
//
// Supported algorithms: wah, roaring, ewah, concise, combit, bitset

#include "gen_bitmap.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <numeric>
#include <algorithm>

// FastBit (WAH)
#include "fastbit/bitvector.h"
#include "utils/util.h"

// CRoaring
#include "croaring/roaring.hh"

// EWAH
#include "ewah/ewah.h"

// Concise
#include "Concise/concise.h"

// ComBit
#include <combit.h>
#include <combit_n.h>

namespace fs = std::filesystem;

// ==================================================================
//  Dataset I/O
// ==================================================================

std::vector<std::vector<uint32_t>> read_dataset_buckets(
    const std::string& dataset_path, uint64_t rows, int cardinality)
{
    std::ifstream fin(dataset_path, std::ios::binary);
    if (!fin) {
        std::cerr << "[read_dataset] Error: cannot open " << dataset_path << "\n";
        return {};
    }
    std::vector<int32_t> data(rows);
    fin.read(reinterpret_cast<char*>(data.data()), rows * sizeof(int32_t));
    if (!fin) {
        std::cerr << "[read_dataset] Error: failed to read " << rows << " int32\n";
        return {};
    }
    fin.close();

    std::vector<std::vector<uint32_t>> buckets(cardinality + 1);
    for (uint64_t i = 0; i < rows; i++) {
        int32_t v = data[i];
        if (v >= 1 && v <= cardinality) {
            buckets[v].push_back(static_cast<uint32_t>(i));
        }
    }
    return buckets;
}

// Run-length per-value buckets — models a column store table sorted by
// the indexed attribute, so every distinct value's bitmap is a single
// contiguous run of set bits.  This is the TPC-H Q6 / sorted-shipdate
// shape: value v ∈ [1..c] owns rows [(v-1)·N/c, v·N/c).  The resulting
// per-value bitmap has cardinality N/c set bits all clustered, which is
// what L4's 32K-bit batch-skip is designed for.  Bench_L_ops + the
// bar-chart driver use this to validate the L-depth design choice on
// its target workload (the uniform-random sweep is the worst case).
std::vector<std::vector<uint32_t>> make_runlength_buckets(
    uint64_t num_rows, int cardinality)
{
    std::vector<std::vector<uint32_t>> buckets(cardinality + 1);
    uint64_t run = num_rows / cardinality;
    for (int v = 1; v <= cardinality; v++) {
        uint64_t start = static_cast<uint64_t>(v - 1) * run;
        uint64_t end   = (v == cardinality) ? num_rows : start + run;
        buckets[v].reserve(end - start);
        for (uint64_t i = start; i < end; i++)
            buckets[v].push_back(static_cast<uint32_t>(i));
    }
    return buckets;
}

// Generate INDEPENDENT random bitmap buckets — used to expose CRoaring's
// bitset→array AND-result transition, which only fires when the two input
// bitmaps overlap (which dataset-bucket bitmaps never do, since dataset
// values partition the rows).
//
// Each bitmap (v = 1..num_bitmaps) is sampled independently with `count_a`
// set bits per 65 536-bit segment, drawn via Fisher–Yates without
// replacement.  Returned buckets[v] = sorted positions of bitmap v;
// buckets[0] is unused, matching read_dataset_buckets's convention.
//
// Different bitmaps share set bits at random — A ∩ B has expected
// cardinality count_a² / 65 536 per segment.
std::vector<std::vector<uint32_t>> make_transition_buckets(
    uint64_t num_rows, int num_bitmaps, int count_a, uint64_t seed)
{
    constexpr size_t SEG_BITS = 65536;
    if (count_a < 0 || static_cast<size_t>(count_a) > SEG_BITS) {
        std::cerr << "[transition] Error: count_a must be in [0, " << SEG_BITS
                  << "], got " << count_a << "\n";
        return {};
    }
    const size_t num_segs = (num_rows + SEG_BITS - 1) / SEG_BITS;

    std::vector<std::vector<uint32_t>> buckets(num_bitmaps + 1);
    // Fisher-Yates scratch reused across segments to avoid per-segment alloc.
    std::vector<uint32_t> ix(SEG_BITS);
    for (int v = 1; v <= num_bitmaps; v++) {
        buckets[v].reserve(num_segs * static_cast<size_t>(count_a));
        for (size_t s = 0; s < num_segs; s++) {
            std::iota(ix.begin(), ix.end(), 0);
            // Unique seed per (bitmap, segment) → A and B are statistically
            // independent across bitmaps, deterministic across runs.
            std::mt19937 rng(seed + (s * static_cast<uint64_t>(num_bitmaps))
                                  + static_cast<uint64_t>(v - 1));
            for (int i = 0; i < count_a; i++) {
                std::uniform_int_distribution<uint32_t> pick(i, SEG_BITS - 1);
                std::swap(ix[i], ix[pick(rng)]);
            }
            const uint32_t base = static_cast<uint32_t>(s * SEG_BITS);
            std::vector<uint32_t> seg_pos(count_a);
            for (int i = 0; i < count_a; i++) seg_pos[i] = base + ix[i];
            std::sort(seg_pos.begin(), seg_pos.end());
            // Drop positions past num_rows (final segment may be truncated).
            for (uint32_t p : seg_pos) {
                if (static_cast<uint64_t>(p) < num_rows)
                    buckets[v].push_back(p);
            }
        }
    }
    return buckets;
}

// Generate ASYMMETRIC disjoint bitmap buckets — A has count_a set bits per
// 65 536-segment, B has count_b set bits, with A ∩ B = ∅ guaranteed.
//
// CR's "fast normal" zone for OR: when count_a + count_b ≤ 4096 (= CR's
// DEFAULT_MAX_SIZE), the array_array_container_union stays in the direct
// array merge path with no transient bitset.  Used as the MIDDLE point in
// the OR density plot, sitting between the regret point (o3500) and the
// no-regret bitset point (t2200) — see plot_or_density.py.
//
// Always produces 2 bitmaps (buckets size = 3; buckets[0] unused).
std::vector<std::vector<uint32_t>> make_asymmetric_disjoint_buckets(
    uint64_t num_rows, int count_a, int count_b, uint64_t seed)
{
    constexpr size_t SEG_BITS = 65536;
    if (count_a <= 0 || static_cast<size_t>(count_a) > SEG_BITS
     || count_b <= 0 || static_cast<size_t>(count_b) > SEG_BITS
     || static_cast<size_t>(count_a + count_b) > SEG_BITS) {
        std::cerr << "[asym] Error: invalid count_a=" << count_a
                  << ", count_b=" << count_b
                  << " (require both > 0 and a+b ≤ " << SEG_BITS << ")\n";
        return {};
    }
    const size_t num_segs = (num_rows + SEG_BITS - 1) / SEG_BITS;

    std::vector<std::vector<uint32_t>> buckets(3);  // [0] unused, [1]=A, [2]=B
    buckets[1].reserve(num_segs * static_cast<size_t>(count_a));
    buckets[2].reserve(num_segs * static_cast<size_t>(count_b));

    std::vector<uint32_t> ix(SEG_BITS);
    const int prefix_len = count_a + count_b;
    for (size_t s = 0; s < num_segs; s++) {
        // Fisher-Yates partial shuffle of the first (count_a + count_b)
        // positions.  After this:
        //   ix[0..count_a)            → A's positions in this segment
        //   ix[count_a..count_a+count_b) → B's positions (disjoint with A)
        std::iota(ix.begin(), ix.end(), 0);
        std::mt19937 rng(seed + s);
        for (int i = 0; i < prefix_len; i++) {
            std::uniform_int_distribution<uint32_t> pick(i, SEG_BITS - 1);
            std::swap(ix[i], ix[pick(rng)]);
        }
        const uint32_t base = static_cast<uint32_t>(s * SEG_BITS);

        std::vector<uint32_t> a_pos(count_a);
        for (int i = 0; i < count_a; i++) a_pos[i] = base + ix[i];
        std::sort(a_pos.begin(), a_pos.end());
        for (uint32_t p : a_pos)
            if (static_cast<uint64_t>(p) < num_rows) buckets[1].push_back(p);

        std::vector<uint32_t> b_pos(count_b);
        for (int i = 0; i < count_b; i++) b_pos[i] = base + ix[count_a + i];
        std::sort(b_pos.begin(), b_pos.end());
        for (uint32_t p : b_pos)
            if (static_cast<uint64_t>(p) < num_rows) buckets[2].push_back(p);
    }
    return buckets;
}

// Generate OVERLAPPING bitmap buckets — A and B share `overlap_ratio · count_a`
// set bits per segment, plus B has `count_a − overlap_ratio · count_a` fresh
// disjoint bits.  Used to expose CRoaring's OR-result `array → bitset → array`
// worst path: each input is an array container (count_a < 4096 per segment),
// |A|+|B|=2·count_a > 4096 triggers a transient bitset allocation, but
// |A∪B|=count_a·(2 − overlap_ratio) ≤ 4096 forces CR to convert that bitset
// back to an array (the regret path).
//
// At overlap_ratio = 0.95, count_a = 3500 → |A∪B| ≈ 3675 per segment, sitting
// just under the 4096 threshold for maximum CR damage (matches Scenario 3 in
// the transition_bench micro-benchmark).
//
// Always produces 2 bitmaps (buckets size = 3; buckets[0] unused).
std::vector<std::vector<uint32_t>> make_overlap_buckets(
    uint64_t num_rows, int count_a, double overlap_ratio, uint64_t seed)
{
    constexpr size_t SEG_BITS = 65536;
    if (count_a <= 0 || static_cast<size_t>(count_a) > SEG_BITS) {
        std::cerr << "[overlap] Error: count_a must be in (0, " << SEG_BITS
                  << "], got " << count_a << "\n";
        return {};
    }
    if (overlap_ratio < 0.0 || overlap_ratio > 1.0) {
        std::cerr << "[overlap] Error: overlap_ratio must be in [0, 1], got "
                  << overlap_ratio << "\n";
        return {};
    }
    const uint32_t carry = static_cast<uint32_t>(overlap_ratio * count_a);
    const uint32_t fresh = static_cast<uint32_t>(count_a) - carry;
    const size_t   num_segs = (num_rows + SEG_BITS - 1) / SEG_BITS;

    std::vector<std::vector<uint32_t>> buckets(3);   // [0] unused, [1]=A, [2]=B
    buckets[1].reserve(num_segs * static_cast<size_t>(count_a));
    buckets[2].reserve(num_segs * static_cast<size_t>(count_a));

    std::vector<uint32_t> ix(SEG_BITS);
    std::vector<uint8_t>  in_a(SEG_BITS);
    for (size_t s = 0; s < num_segs; s++) {
        // --- Sample A: count_a positions via Fisher-Yates ---
        std::iota(ix.begin(), ix.end(), 0);
        std::mt19937 rng_a(seed + 2 * s + 0);
        for (int i = 0; i < count_a; i++) {
            std::uniform_int_distribution<uint32_t> pick(i, SEG_BITS - 1);
            std::swap(ix[i], ix[pick(rng_a)]);
        }
        const uint32_t base = static_cast<uint32_t>(s * SEG_BITS);
        std::fill(in_a.begin(), in_a.end(), 0);
        for (int i = 0; i < count_a; i++) in_a[ix[i]] = 1;

        // --- Build B: first `carry` of A's positions + `fresh` new disjoint ---
        std::vector<uint32_t> b_pos;
        b_pos.reserve(count_a);
        for (uint32_t i = 0; i < carry; i++) b_pos.push_back(base + ix[i]);
        std::mt19937 rng_b(seed + 2 * s + 1);
        uint32_t fresh_added = 0;
        while (fresh_added < fresh) {
            uint32_t p = std::uniform_int_distribution<uint32_t>(0, SEG_BITS - 1)(rng_b);
            if (!in_a[p]) { in_a[p] = 2; b_pos.push_back(base + p); fresh_added++; }
        }

        // --- Emit A and B (sorted, dropping positions past num_rows) ---
        std::vector<uint32_t> a_pos(count_a);
        for (int i = 0; i < count_a; i++) a_pos[i] = base + ix[i];
        std::sort(a_pos.begin(), a_pos.end());
        std::sort(b_pos.begin(), b_pos.end());
        for (uint32_t p : a_pos)
            if (static_cast<uint64_t>(p) < num_rows) buckets[1].push_back(p);
        for (uint32_t p : b_pos)
            if (static_cast<uint64_t>(p) < num_rows) buckets[2].push_back(p);
    }
    return buckets;
}

// ==================================================================
//  gen_raw: uncompressed little-endian packed bits
// ==================================================================

bool gen_raw(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    size_t packed_bytes = (rows + 7) / 8;

    for (int v = 1; v <= cardinality; v++) {
        std::vector<uint8_t> bitmap(packed_bytes, 0);
        for (uint32_t idx : buckets[v]) {
            bitmap[idx / 8] |= (1u << (idx % 8));
        }

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bitmap.data()), packed_bytes);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_raw] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_raw] All " << cardinality << " raw bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_bitset: packed bits, z bitmaps per file
// ==================================================================

bool gen_bitset(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality, int z)
{
    fs::create_directories(output_dir);

    size_t packed_bytes = (rows + 7) / 8;
    int num_files = (cardinality + z - 1) / z;

    for (int i = 0; i < num_files; i++) {
        int start_val = i * z + 1;
        int end_val   = std::min((i + 1) * z, cardinality);

        std::string out_path = (z == 1)
            ? output_dir + "/" + std::to_string(start_val) + ".bm"
            : output_dir + "/" + std::to_string(i) + ".bmz";

        std::ofstream out(out_path, std::ios::binary);
        for (int v = start_val; v <= end_val; v++) {
            std::vector<uint8_t> bitmap(packed_bytes, 0);
            for (uint32_t idx : buckets[v]) {
                bitmap[idx / 8] |= (1u << (idx % 8));
            }
            out.write(reinterpret_cast<const char*>(bitmap.data()), packed_bytes);
        }

        if (i % 100 == 0 || i == num_files - 1) {
            std::cout << "[gen_bitset] Written file " << (i + 1) << "/" << num_files
                      << " (values " << start_val << "-" << end_val << ")\n";
        }
    }
    std::cout << "[gen_bitset] All " << cardinality << " bitmaps written (z=" << z
              << ") to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_wah: FastBit WAH compression
// ==================================================================

bool gen_wah(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    Table_config config;
    config.enable_fence_pointer = false;

    for (int v = 1; v <= cardinality; v++) {
        ibis::bitvector btv;
        btv.adjustSize(0, static_cast<uint32_t>(rows));
        btv.decompress();

        for (uint32_t idx : buckets[v]) {
            btv.setBit(idx, 1, &config);
        }

        btv.compress();

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        btv.write(out_path.c_str());

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_wah] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_wah] All " << cardinality << " WAH bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_roaring: CRoaring compression
// ==================================================================

bool gen_roaring(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        roaring::Roaring bitmap;
        for (uint32_t idx : buckets[v]) {
            bitmap.add(idx);
        }
        bitmap.runOptimize();

        // Serialize: [current_size (uint32_t)] [roaring binary data]
        uint32_t current_size = static_cast<uint32_t>(rows);
        size_t roaring_size = bitmap.getSizeInBytes();
        std::vector<char> buffer(roaring_size);
        bitmap.write(buffer.data());

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_size), sizeof(current_size));
        out.write(buffer.data(), roaring_size);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_roaring] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_roaring] All " << cardinality << " Roaring bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_ewah: EWAH compression
// ==================================================================

bool gen_ewah(const std::vector<std::vector<uint32_t>>& buckets,
              const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ewah::EWAHBoolArray<uint64_t> btv;
        for (uint32_t idx : buckets[v]) {
            btv.set(idx);
        }

        // Serialize: [current_bits (uint64_t)] [ewah binary]
        uint64_t current_bits = rows;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
        btv.write(out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_ewah] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_ewah] All " << cardinality << " EWAH bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_concise: Concise compression
// ==================================================================

bool gen_concise(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ConciseSet<false> btv;
        for (uint32_t idx : buckets[v]) {
            btv.add(idx);
        }

        // Serialize: [current_bits (uint64_t)] [last] [lastWordIndex] [words data]
        uint64_t current_bits = rows;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&current_bits), sizeof(current_bits));
        out.write(reinterpret_cast<const char*>(&btv.last), sizeof(btv.last));
        out.write(reinterpret_cast<const char*>(&btv.lastWordIndex), sizeof(btv.lastWordIndex));
        if (btv.lastWordIndex >= 0) {
            uint32_t count = static_cast<uint32_t>(btv.lastWordIndex + 1);
            out.write(reinterpret_cast<const char*>(btv.words.data()), count * sizeof(uint32_t));
        }

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_concise] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_concise] All " << cardinality << " Concise bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_combit: ComBit compression
// ==================================================================

bool gen_combit(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality,
                int word_size, size_t segment_bits = ComBit::default_segment_bits)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        // Build vector<bool> from sorted position indices
        const auto& raw = buckets[v];
        std::vector<bool> bits(rows, false);
        for (size_t i = 0; i < raw.size() && raw[i] < rows; i++)
            bits[raw[i]] = true;

        // Compress with the selected word size + segment_bits and serialize
        ComBit cb = ComBit::compress(bits, /*l1_fill_ones=*/false, segment_bits);
        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        cb.serialize(out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_combit] All " << cardinality << " ComBit (w" << word_size
              << ", S=" << segment_bits << ") bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_combit_n: ComBitN compression at a fixed depth (2/3/4/5)
// ==================================================================

bool gen_combit_n(const std::vector<std::vector<uint32_t>>& buckets,
                  const std::string& output_dir, uint64_t rows, int cardinality,
                  int depth)
{
    if (depth < 2 || depth > 5) {
        std::cerr << "[gen_combit_n] Error: depth must be in {2,3,4,5}, got "
                  << depth << "\n";
        return false;
    }
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        const auto& raw = buckets[v];
        std::vector<bool> bits(rows, false);
        for (size_t i = 0; i < raw.size() && raw[i] < rows; i++)
            bits[raw[i]] = true;

        ComBitN cb = combit_n_compress(bits, depth);
        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        combit_n_serialize(cb, out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit_n] Written " << v << "/" << cardinality
                      << " bitmaps (depth=" << depth << ")\n";
        }
    }
    std::cout << "[gen_combit_n] All " << cardinality << " ComBitN (L"
              << depth << ") bitmaps written to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_combit_n_tile: tile-mode ComBitN (matches gen_combit_tile)
// ==================================================================
//
// Compresses a small_n-bit bool vector at the requested depth, then
// byte-concatenates the per-segment payload tile_factor× and rewrites
// the top-level header to bit_count = small_n * tile_factor,
// num_segments = orig_num_segs * tile_factor.  Each segment retains
// its own (small) bit_count so combit_n_decompress handles them
// independently.  Used to match the c=2000 L4 .bm files which were
// generated with tile=100 small_n=1M.

bool gen_combit_n_tile(const std::vector<std::vector<uint32_t>>& buckets,
                       const std::string& output_dir, uint64_t small_n,
                       int cardinality, int tile_factor, int depth) {
    if (depth < 2 || depth > 5) {
        std::cerr << "[gen_combit_n_tile] Error: depth must be in {2,3,4,5}, got "
                  << depth << "\n";
        return false;
    }
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        std::vector<bool> bits(small_n, false);
        for (uint32_t idx : buckets[v])
            if (idx < small_n) bits[idx] = true;

        ComBitN small_cb = combit_n_compress(bits, depth);

        std::stringstream ss;
        combit_n_serialize(small_cb, ss);
        std::string buf = ss.str();

        // Top-level header layout (must match combit_n_serialize):
        //   uint8_t  fmt_tag       offset  0  (1 byte)
        //   uint8_t  depth         offset  1  (1 byte)
        //   uint64_t bit_count     offset  2  (8 bytes)
        //   uint64_t segment_bits  offset 10  (8 bytes)
        //   uint64_t num_segments  offset 18  (8 bytes)
        //   ------------------------------------ = 26 bytes
        constexpr size_t HEADER_BYTES = 26;
        if (buf.size() < HEADER_BYTES) {
            std::cerr << "[gen_combit_n_tile] Error: bad serialize size\n";
            return false;
        }
        uint64_t bit_count, seg_bits, num_segs;
        std::memcpy(&bit_count, buf.data() +  2, 8);
        std::memcpy(&seg_bits,  buf.data() + 10, 8);
        std::memcpy(&num_segs,  buf.data() + 18, 8);

        uint64_t new_bit_count = bit_count * tile_factor;
        uint64_t new_num_segs  = num_segs  * tile_factor;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(buf.data(), 2);  // fmt_tag + depth
        out.write(reinterpret_cast<const char*>(&new_bit_count), 8);
        out.write(reinterpret_cast<const char*>(&seg_bits),      8);
        out.write(reinterpret_cast<const char*>(&new_num_segs),  8);

        const char* payload = buf.data() + HEADER_BYTES;
        size_t payload_size = buf.size() - HEADER_BYTES;
        for (int i = 0; i < tile_factor; i++)
            out.write(payload, payload_size);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit_n_tile] Written " << v << "/" << cardinality
                      << " (depth=" << depth << ", small_n=" << small_n
                      << ", tile=" << tile_factor
                      << " → n=" << new_bit_count << ")\n";
        }
    }
    std::cout << "[gen_combit_n_tile] All " << cardinality
              << " ComBitN (L" << depth << ") bitmaps written (tile mode) to "
              << output_dir << "\n";
    return true;
}

// ==================================================================
//  gen_combit_tile: compress small bitmap once, tile T× via byte concat
// ==================================================================
//
// For high-cardinality benchmarks (card >= 2000) the per-bitmap compress
// over n=100M bits is the bottleneck.  Tile mode generates a small
// bitmap (size = small_n = n/tile) with default segment_bits=65536 (so
// the per-segment metadata layout matches vanilla gen on a real n-bit
// dataset), then byte-concatenates the segment payload `tile` times and
// rewrites the 24-byte header.  All resulting segments stay 65536-bit
// aligned to give the same op-side performance as vanilla gen.

bool gen_combit_tile(const std::vector<std::vector<uint32_t>>& buckets,
                     const std::string& output_dir,
                     uint64_t small_n, int cardinality, int tile_factor) {
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        // Build the small (small_n-bit) bit vector.
        std::vector<bool> bits(small_n, false);
        for (uint32_t idx : buckets[v])
            if (idx < small_n) bits[idx] = true;

        // Compress with the DEFAULT segment_bits (65536) — same per-segment
        // layout as vanilla gen, so the tiled bitmap behaves like a real
        // n-bit ComBit at op time.
        ComBit small_cb = ComBit::compress(bits);

        // Serialize to memory so we can manipulate the 24-byte header.
        std::stringstream ss;
        small_cb.serialize(ss);
        std::string buf = ss.str();
        if (buf.size() < 24) {
            std::cerr << "[gen_combit_tile] Error: bad serialize size\n";
            return false;
        }

        uint64_t bit_count, seg_bits, num_segs;
        std::memcpy(&bit_count, buf.data() +  0, 8);
        std::memcpy(&seg_bits,  buf.data() +  8, 8);
        std::memcpy(&num_segs,  buf.data() + 16, 8);

        uint64_t new_bit_count = bit_count * tile_factor;
        uint64_t new_num_segs  = num_segs  * tile_factor;

        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&new_bit_count), 8);
        out.write(reinterpret_cast<const char*>(&seg_bits),      8);
        out.write(reinterpret_cast<const char*>(&new_num_segs),  8);

        const char* payload = buf.data() + 24;
        size_t payload_size = buf.size() - 24;
        for (int i = 0; i < tile_factor; i++)
            out.write(payload, payload_size);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_combit_tile] Written " << v << "/" << cardinality
                      << " (small_n=" << small_n << ", tile=" << tile_factor
                      << " → n=" << new_bit_count << ")\n";
        }
    }
    std::cout << "[gen_combit_tile] All " << cardinality
              << " ComBit bitmaps written (tile mode) to " << output_dir << "\n";
    return true;
}

// ==================================================================
//  Path helpers
// ==================================================================

/// Format row count: 1000000000 → "1b", 100000000 → "100m", 1000 → "1k"
static std::string format_rows(int n) {
    if (n >= 1000000000 && n % 1000000000 == 0)
        return std::to_string(n / 1000000000) + "b";
    if (n >= 1000000 && n % 1000000 == 0)
        return std::to_string(n / 1000000) + "m";
    if (n >= 1000 && n % 1000 == 0)
        return std::to_string(n / 1000) + "k";
    return std::to_string(n);
}

static std::string dataset_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/dataset_" + std::to_string(n) + "_" + std::to_string(c);
}

/// Raw uncompressed bitmaps: bitmap/bitmaps_100m_c100/
static std::string raw_dir_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/bitmap/bitmaps_" + format_rows(n)
         + "_c" + std::to_string(c);
}

/// Compressed bitmaps: bitmap/bm_100m_c100_wah/
static std::string compressed_dir_path(const std::string& base_dir, int n, int c,
                                       const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_c" + std::to_string(c) + "_" + algo_lower;
}

/// Transition-mode bitmaps (independent random A, B):
///   bitmap/bm_100m_t6000_wah/  ← count_a = 6000 bits per 65 536-segment
/// Same directory layout as compressed_dir_path so benchmark_app --compressed-dir
/// can read it without modification.
static std::string transition_dir_path(const std::string& base_dir, int n,
                                       int count_a, const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_t" + std::to_string(count_a) + "_" + algo_lower;
}

/// Overlap-mode bitmaps (A + B with high overlap → CR OR worst path):
///   bitmap/bm_100m_o3500_wah/   ← count_a = 3500, overlap = 95% (hard-coded)
/// Same directory layout as compressed_dir_path so benchmark_app --compressed-dir
/// reads it without modification.
static std::string overlap_dir_path(const std::string& base_dir, int n,
                                    int count_a, const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_o" + std::to_string(count_a) + "_" + algo_lower;
}

/// Asymmetric-disjoint bitmaps (A with count_a bits, B with count_b bits,
/// guaranteed A∩B=∅):
///   bitmap/bm_100m_A2700_B1300_wah/   ← MIDDLE point in OR density plot
/// Used as a "CR fast" reference between the two CR slow points (o3500 +
/// t2200); A+B ≤ 4096 keeps CR in the direct array merge path.
static std::string asym_dir_path(const std::string& base_dir, int n,
                                 int count_a, int count_b,
                                 const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_A" + std::to_string(count_a)
         + "_B" + std::to_string(count_b)
         + "_" + algo_lower;
}

static std::string compressed_done_path(const std::string& dir) {
    return dir + "/done.txt";
}

// ==================================================================
//  Dataset & compression checks
// ==================================================================

static bool dataset_exists_in_index(const std::string& base_dir, int n, int c) {
    std::string index_path = base_dir + "/index.txt";
    std::ifstream f(index_path);
    if (!f.is_open()) return false;

    // Match the WHOLE "n=N c=C " token, so c=2 doesn't match c=2000.
    std::string target = "n=" + std::to_string(n) + " c=" + std::to_string(c) + " ";
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, target.size(), target) == 0) return true;
    }
    return false;
}

static bool compression_exists(const std::string& compressed_dir) {
    return fs::exists(compressed_done_path(compressed_dir));
}

static bool call_gen_dataset(const std::string& base_dir, int n, int c) {
    std::string cmd = base_dir + "/util/gen_dataset.sh -n " + std::to_string(n)
                    + " -c " + std::to_string(c)
                    + " -d " + base_dir;
    std::cout << "[gen_bitmap] Calling: " << cmd << std::endl;
    int ret = std::system(cmd.c_str());
    return ret == 0;
}

/// Dispatch to the appropriate generation function for the given algorithm.
/// `depth_n` is the ComBitN depth (2..5); -1 means "use word-size ComBit".
static bool generate_compressed(const std::string& algorithm,
                                const std::vector<std::vector<uint32_t>>& buckets,
                                const std::string& output_dir,
                                uint64_t rows, int cardinality, int z,
                                int word_size, int depth_n,
                                size_t segment_bits = ComBit::default_segment_bits) {
    std::string algo = algorithm;
    for (auto& ch : algo)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (algo == "bitset")  return gen_bitset(buckets, output_dir, rows, cardinality, z);
    if (algo == "wah")     return gen_wah(buckets, output_dir, rows, cardinality);
    if (algo == "roaring") return gen_roaring(buckets, output_dir, rows, cardinality);
    if (algo == "ewah")    return gen_ewah(buckets, output_dir, rows, cardinality);
    if (algo == "concise") return gen_concise(buckets, output_dir, rows, cardinality);
    if (algo == "combit") {
        if (depth_n > 0)
            return gen_combit_n(buckets, output_dir, rows, cardinality, depth_n);
        return gen_combit(buckets, output_dir, rows, cardinality, word_size, segment_bits);
    }

    std::cerr << "[gen_bitmap] Error: unsupported algorithm '" << algorithm << "'\n"
              << "[gen_bitmap] Supported: bitset, wah, roaring, ewah, concise, combit\n";
    return false;
}

// ==================================================================
//  Main entry
// ==================================================================

int main(int argc, char* argv[]) {
    int n = -1;
    int c = -1;
    int t = -1;   // -t <count_a> : transition mode (independent random bitmaps)
    int o = -1;   // -o <count_a> : overlap mode (95% overlap, fresh = count_a*0.05)
    int asym_a = -1, asym_b = -1;  // -A <count_a> -B <count_b> : asymmetric disjoint
    int z = 1;
    int w = 8;
    int tile = 1;
    int L_depth = -1;   // -L <depth> : combit ComBitN depth-N variant (2/3/4/5)
    long long seg_bits_arg = -1;   // -S <bits> : combit segment_bits (default 65536)
    bool run_length = false;   // -R : run-length per-value bitmaps (clustered)
    std::string algorithm;
    std::string base_dir = ".";

    // Parse arguments: -n <n> -c <c> <algorithm> [-d <base_dir>] [-z <zip_length>]
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            c = std::stoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            t = std::stoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            o = std::stoi(argv[++i]);
        } else if (arg == "-A" && i + 1 < argc) {
            asym_a = std::stoi(argv[++i]);
        } else if (arg == "-B" && i + 1 < argc) {
            asym_b = std::stoi(argv[++i]);
        } else if (arg == "-z" && i + 1 < argc) {
            z = std::stoi(argv[++i]);
        } else if (arg == "-w" && i + 1 < argc) {
            w = std::stoi(argv[++i]);
        } else if (arg == "-L" && i + 1 < argc) {
            L_depth = std::stoi(argv[++i]);
        } else if (arg == "-T" && i + 1 < argc) {
            tile = std::stoi(argv[++i]);
        } else if (arg == "-S" && i + 1 < argc) {
            seg_bits_arg = std::stoll(argv[++i]);
        } else if (arg == "-R") {
            run_length = true;
        } else if (arg == "-d" && i + 1 < argc) {
            base_dir = argv[++i];
        } else if (arg[0] != '-') {
            algorithm = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }

    // Transition / overlap / asymmetric modes force 2 bitmaps and ignore -c.
    // They are mutually exclusive.
    const bool transition_mode = (t > 0);
    const bool overlap_mode    = (o > 0);
    const bool asym_mode       = (asym_a > 0 && asym_b > 0);
    if (transition_mode || overlap_mode || asym_mode) c = 2;

    int special_modes = transition_mode + overlap_mode + asym_mode;
    if (special_modes > 1) {
        std::cerr << "Error: -t, -o, and -A/-B are mutually exclusive\n";
        return 1;
    }
    if ((asym_a > 0) != (asym_b > 0)) {
        std::cerr << "Error: -A and -B must be used together\n";
        return 1;
    }

    if (n <= 0 || c <= 0 || algorithm.empty()) {
        std::cerr << "Usage: " << argv[0] << " -n <n> {-c <c> | -t <count_a> | -o <count_a>} <algorithm> [options]\n"
                  << "  -n <n>        : number of rows (final size)\n"
                  << "  -c <c>        : cardinality (column-style mutually-disjoint bitmaps)\n"
                  << "  -t <count_a>  : transition mode — generate 2 independent random\n"
                  << "                  bitmaps with count_a set bits per 65 536-segment.\n"
                  << "                  Mutually exclusive with -c.  Output goes to\n"
                  << "                  bitmap/bm_<n>_t<count_a>_<algo>/\n"
                  << "  -o <count_a>  : overlap mode — 2 bitmaps with 95% overlap and 5% fresh\n"
                  << "                  bits (count_a · 0.05 disjoint).  Exposes CRoaring's\n"
                  << "                  OR array→bitset→array worst path.  Output goes to\n"
                  << "                  bitmap/bm_<n>_o<count_a>_<algo>/\n"
                  << "  <algorithm>   : compression algorithm (bitset, wah, roaring, ewah, concise, combit)\n"
                  << "  -d <base_dir> : base directory (default: .)\n"
                  << "  -z <z>        : zip length for bitset mode (default: 1)\n"
                  << "  -w <w>        : ComBit word size: 8, 16, 32, or 64 (default: 8)\n"
                  << "  -L <depth>    : ComBitN depth (combit only): 2, 3, 4, or 5.\n"
                  << "                  Overrides -w.  Produces bitmap/bm_<n>_c<c>_combit_L<depth>/\n"
                  << "                  files in the on-disk ComBitN format.  Used by the\n"
                  << "                  bench_L_ops fairness study.  Mutually exclusive with -T.\n"
                  << "  -S <bits>     : ComBit segment_bits (combit only, L4 only): power of 2 in\n"
                  << "                  [64, 2^24].  Default 65536 (2^16).  Output dir gets the\n"
                  << "                  suffix _S<bits> when non-default.  Mutually exclusive with -L\n"
                  << "                  and -T.  Used by the segment-size ablation study.\n"
                  << "  -T <T>        : tile factor (combit only).  Generate small dataset of\n"
                  << "                  size n/T with segment_bits = n/T, compress once, then\n"
                  << "                  byte-concat T copies.  Saves the n-scale compress pass.\n"
                  << "                  Requires n % T == 0.\n";
        return 1;
    }

    if ((transition_mode || overlap_mode || asym_mode) && tile > 1) {
        std::cerr << "Error: -t/-o/-A and -T (tile mode) are mutually exclusive\n";
        return 1;
    }

    if (tile > 1) {
        if (n % tile != 0) {
            std::cerr << "Error: -n (" << n << ") must be divisible by -T (" << tile << ")\n";
            return 1;
        }
    }

    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (z != 1 && algo_lower != "bitset") {
        std::cerr << "Error: -z is only valid for bitset mode" << std::endl;
        return 1;
    }

    if (w != 8 && w != 16 && w != 32 && w != 64) {
        std::cerr << "Error: -w must be 8, 16, 32, or 64" << std::endl;
        return 1;
    }

    if (L_depth != -1) {
        if (algo_lower != "combit") {
            std::cerr << "Error: -L is only valid with combit algorithm" << std::endl;
            return 1;
        }
        if (L_depth < 2 || L_depth > 5) {
            std::cerr << "Error: -L must be 2, 3, 4, or 5" << std::endl;
            return 1;
        }
    }

    // -S validation: power-of-2, multiple of 64 bits, combit-only, L4 only
    // (combit_n_compress takes its own segment_bits arg via default 1<<16).
    size_t segment_bits_val = ComBit::default_segment_bits;
    if (seg_bits_arg > 0) {
        if (algo_lower != "combit") {
            std::cerr << "Error: -S is only valid with combit algorithm" << std::endl;
            return 1;
        }
        if (L_depth > 0) {
            std::cerr << "Error: -S currently only supported with native ComBit (L4); not with -L" << std::endl;
            return 1;
        }
        if (tile > 1) {
            std::cerr << "Error: -S and -T are mutually exclusive (tile mode hard-codes seg_bits=n/T)" << std::endl;
            return 1;
        }
        // Power-of-2 + ≥64 + ≤ 2^24 for sanity.
        if (seg_bits_arg < 64 || (seg_bits_arg & (seg_bits_arg - 1)) != 0
            || seg_bits_arg > (1LL << 24)) {
            std::cerr << "Error: -S must be a power of 2 in [64, 2^24], got "
                      << seg_bits_arg << std::endl;
            return 1;
        }
        segment_bits_val = static_cast<size_t>(seg_bits_arg);
    }

    // Suffix for combit output dir: -L overrides -w (ComBitN serialised
    // format lives in bm_..._combit_L<N>/ rather than _w<W>/).
    // -S appends _S<bits> after the word/depth suffix (omitted when default).
    auto combit_suffix = [&]() {
        std::string base = (L_depth > 0)
            ? std::string("_L") + std::to_string(L_depth)
            : std::string("_w") + std::to_string(w);
        if (seg_bits_arg > 0 && static_cast<size_t>(seg_bits_arg) != ComBit::default_segment_bits)
            base += "_S" + std::to_string(seg_bits_arg);
        return base;
    };

    // Determine output directory
    std::string comp_dir;
    if (transition_mode) {
        // Transition mode: bitmap/bm_<n>_t<count_a>_<algo>[_w<W>|_L<N>]/
        comp_dir = transition_dir_path(base_dir, n, t, algorithm);
        if (algo_lower == "combit") comp_dir += combit_suffix();
    } else if (overlap_mode) {
        // Overlap mode: bitmap/bm_<n>_o<count_a>_<algo>[_w<W>|_L<N>]/
        comp_dir = overlap_dir_path(base_dir, n, o, algorithm);
        if (algo_lower == "combit") comp_dir += combit_suffix();
    } else if (asym_mode) {
        // Asymmetric mode: bitmap/bm_<n>_A<a>_B<b>_<algo>[_w<W>|_L<N>]/
        comp_dir = asym_dir_path(base_dir, n, asym_a, asym_b, algorithm);
        if (algo_lower == "combit") comp_dir += combit_suffix();
    } else if (algo_lower == "bitset") {
        if (z == 1)
            comp_dir = compressed_dir_path(base_dir, n, c, "bitset");
        else
            comp_dir = base_dir + "/bitmap/bitmap_n" + format_rows(n)
                     + "_c" + std::to_string(c) + "_z" + std::to_string(z);
    } else if (algo_lower == "combit") {
        comp_dir = compressed_dir_path(base_dir, n, c, algorithm) + combit_suffix();
    } else {
        comp_dir = compressed_dir_path(base_dir, n, c, algorithm);
    }
    // Run-length mode appends "_run" so it doesn't collide with the
    // existing uniform-random data at the same cardinality.
    if (run_length) comp_dir += "_run";

    // Tile mode: small_n = n / tile.  Generate small dataset only,
    // compress small bitmaps once, then byte-concat T copies.
    int gen_n = (tile > 1) ? (n / tile) : n;

    std::cout << "[gen_bitmap] n=" << n << " c=" << c
              << " algorithm=" << algorithm
              << (tile > 1 ? (" tile=" + std::to_string(tile)
                              + " small_n=" + std::to_string(gen_n))
                           : "")
              << " base_dir=" << base_dir << std::endl;

    // Run-length mode: every value v ∈ [1..c] owns rows [(v-1)·N/c .. v·N/c),
    // modelling a column store sorted by the indexed attribute (TPC-H Q6
    // shipdate after sort).  Skips dataset read, builds buckets directly.
    if (run_length) {
        if (compression_exists(comp_dir)) {
            std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
            return 0;
        }
        std::cout << "[gen_bitmap] Run-length mode — generating " << c
                  << " per-value bitmaps, each a contiguous run of "
                  << (n / c) << " bits." << std::endl;
        fs::create_directories(comp_dir);

        auto buckets = make_runlength_buckets(static_cast<uint64_t>(n), c);
        if (buckets.empty()) return 1;

        if (!generate_compressed(algorithm, buckets, comp_dir,
                                 static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
            std::cerr << "[gen_bitmap] Error: run-length compression failed." << std::endl;
            return 1;
        }
        std::ofstream done(compressed_done_path(comp_dir));
        done << "algorithm=" << algorithm << "\n"
             << "n=" << n << "\n"
             << "c=" << c << "\n"
             << "mode=run_length\n";
        std::cout << "[gen_bitmap] Run-length compression completed: " << comp_dir << std::endl;
        return 0;
    }

    // Transition mode short-circuits the dataset+raw pipeline: it generates
    // the bucket positions in memory via make_transition_buckets (independent
    // random sampling per bitmap), then runs the standard generate_compressed
    // dispatcher.  All 7 backend gen_X functions are reused unchanged.
    if (transition_mode) {
        if (compression_exists(comp_dir)) {
            std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
            return 0;
        }
        std::cout << "[gen_bitmap] Transition mode — generating " << c
                  << " independent random bitmaps with count_a=" << t
                  << " per 65 536-segment." << std::endl;
        fs::create_directories(comp_dir);

        auto buckets = make_transition_buckets(
            static_cast<uint64_t>(n), c, t,
            /*seed=*/0xCAFE0000u + static_cast<uint64_t>(t));
        if (buckets.empty()) return 1;

        if (!generate_compressed(algorithm, buckets, comp_dir,
                                 static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
            std::cerr << "[gen_bitmap] Error: transition compression failed." << std::endl;
            return 1;
        }
        std::ofstream done(compressed_done_path(comp_dir));
        done << "algorithm=" << algorithm << "\n"
             << "n=" << n << "\n"
             << "c=" << c << "\n"
             << "t=" << t << "\n"
             << "mode=transition\n";
        std::cout << "[gen_bitmap] Transition compression completed: " << comp_dir << std::endl;
        return 0;
    }

    // Overlap mode mirrors transition mode but uses make_overlap_buckets
    // (B carries 95% of A's bits + 5% fresh disjoint) to expose CR's OR
    // array→bitset→array worst path.
    if (overlap_mode) {
        if (compression_exists(comp_dir)) {
            std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
            return 0;
        }
        constexpr double overlap_ratio = 0.95;
        std::cout << "[gen_bitmap] Overlap mode — generating 2 bitmaps with count_a="
                  << o << " per 65 536-segment, " << (overlap_ratio * 100)
                  << "% overlap." << std::endl;
        fs::create_directories(comp_dir);

        auto buckets = make_overlap_buckets(
            static_cast<uint64_t>(n), o, overlap_ratio,
            /*seed=*/0xBEEF0000u + static_cast<uint64_t>(o));
        if (buckets.empty()) return 1;

        if (!generate_compressed(algorithm, buckets, comp_dir,
                                 static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
            std::cerr << "[gen_bitmap] Error: overlap compression failed." << std::endl;
            return 1;
        }
        std::ofstream done(compressed_done_path(comp_dir));
        done << "algorithm=" << algorithm << "\n"
             << "n=" << n << "\n"
             << "c=" << c << "\n"
             << "o=" << o << "\n"
             << "overlap_ratio=" << overlap_ratio << "\n"
             << "mode=overlap\n";
        std::cout << "[gen_bitmap] Overlap compression completed: " << comp_dir << std::endl;
        return 0;
    }

    // Asymmetric mode: A and B have different counts (count_a, count_b),
    // forced disjoint.  Used as a CR "fast normal" reference point in OR.
    if (asym_mode) {
        if (compression_exists(comp_dir)) {
            std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
            return 0;
        }
        std::cout << "[gen_bitmap] Asymmetric mode — A=" << asym_a
                  << " bits/seg, B=" << asym_b << " bits/seg, disjoint." << std::endl;
        fs::create_directories(comp_dir);

        auto buckets = make_asymmetric_disjoint_buckets(
            static_cast<uint64_t>(n), asym_a, asym_b,
            /*seed=*/0xCAFE0000u + static_cast<uint64_t>(asym_a) * 1000
                                + static_cast<uint64_t>(asym_b));
        if (buckets.empty()) return 1;

        if (!generate_compressed(algorithm, buckets, comp_dir,
                                 static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
            std::cerr << "[gen_bitmap] Error: asymmetric compression failed." << std::endl;
            return 1;
        }
        std::ofstream done(compressed_done_path(comp_dir));
        done << "algorithm=" << algorithm << "\n"
             << "n=" << n << "\n"
             << "c=" << c << "\n"
             << "count_a=" << asym_a << "\n"
             << "count_b=" << asym_b << "\n"
             << "mode=asymmetric_disjoint\n";
        std::cout << "[gen_bitmap] Asymmetric compression completed: " << comp_dir << std::endl;
        return 0;
    }

    // Step 1: Ensure dataset (small or full) exists
    std::string ds_file = dataset_path(base_dir, gen_n, c);
    if (fs::exists(ds_file)) {
        std::cout << "[gen_bitmap] Dataset file exists: " << ds_file << std::endl;
    } else if (!dataset_exists_in_index(base_dir, gen_n, c)) {
        std::cout << "[gen_bitmap] Dataset not found. Generating..." << std::endl;
        if (!call_gen_dataset(base_dir, gen_n, c)) {
            std::cerr << "[gen_bitmap] Error: gen_dataset.sh failed." << std::endl;
            return 1;
        }
        std::cout << "[gen_bitmap] Dataset generated successfully." << std::endl;
    } else {
        std::cout << "[gen_bitmap] Dataset already exists." << std::endl;
    }

    // Step 2: Check if compression already done
    if (compression_exists(comp_dir)) {
        std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
        return 0;
    }

    std::cout << "[gen_bitmap] Generating compressed bitmaps..." << std::endl;
    fs::create_directories(comp_dir);

    // Read dataset into buckets
    if (!fs::exists(ds_file)) {
        std::cerr << "[gen_bitmap] Error: dataset not found: " << ds_file << std::endl;
        return 1;
    }

    std::cout << "[gen_bitmap] Reading dataset into buckets..." << std::endl;
    auto buckets = read_dataset_buckets(ds_file, static_cast<uint64_t>(gen_n), c);
    if (buckets.empty()) return 1;

    // Tile mode bypasses the raw-bitmap step (raw at full n would defeat the
    // purpose).  For combit: byte-level segment concat (gen_combit_tile).
    // For other algos: expand the small buckets to full-n positions, then
    // call the existing gen_X — works for all formats whose compress is
    // O(set_bits) (Roaring/EWAH/Concise/Bitset) and acceptable for WAH
    // when each per-value bitmap stays sparse.
    if (tile > 1) {
        bool ok;
        if (algo_lower == "combit") {
            if (L_depth > 0)
                ok = gen_combit_n_tile(buckets, comp_dir,
                                       static_cast<uint64_t>(gen_n), c,
                                       tile, L_depth);
            else
                ok = gen_combit_tile(buckets, comp_dir,
                                     static_cast<uint64_t>(gen_n), c, tile);
        } else {
            std::cout << "[gen_bitmap] Tile mode: expanding small buckets to "
                      << "full n (" << n << ") for "
                      << algorithm << "..." << std::endl;
            std::vector<std::vector<uint32_t>> tiled_buckets(c + 1);
            for (int v = 1; v <= c; v++) {
                tiled_buckets[v].reserve(buckets[v].size() * tile);
                for (int t = 0; t < tile; t++) {
                    uint32_t off = static_cast<uint32_t>(gen_n) * t;
                    for (uint32_t pos : buckets[v])
                        tiled_buckets[v].push_back(off + pos);
                }
            }
            ok = generate_compressed(algorithm, tiled_buckets, comp_dir,
                                     static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val);
        }
        if (!ok) {
            std::cerr << "[gen_bitmap] Error: tile compression failed." << std::endl;
            return 1;
        }
        std::ofstream done(compressed_done_path(comp_dir));
        done << "algorithm=" << algorithm << "\n"
             << "n=" << n << "\n"
             << "c=" << c << "\n"
             << "tile=" << tile << "\n"
             << "small_n=" << gen_n << "\n";
        std::cout << "[gen_bitmap] Tile compression completed: " << comp_dir << std::endl;
        return 0;
    }

    // Generate raw bitmaps if not already present (vanilla mode only)
    std::string raw_dir = raw_dir_path(base_dir, n, c);
    std::string raw_done = raw_dir + "/done.txt";
    if (!fs::exists(raw_done)) {
        std::cout << "[gen_bitmap] Generating raw bitmaps → " << raw_dir << std::endl;
        if (!gen_raw(buckets, raw_dir, static_cast<uint64_t>(n), c)) return 1;
        std::ofstream(raw_done) << "n=" << n << "\nc=" << c << "\n";
    } else {
        std::cout << "[gen_bitmap] Raw bitmaps already exist: " << raw_dir << std::endl;
    }

    // Generate compressed bitmaps
    if (!generate_compressed(algorithm, buckets, comp_dir, static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
        std::cerr << "[gen_bitmap] Error: compression failed." << std::endl;
        return 1;
    }

    // Write done.txt marker
    std::ofstream done(compressed_done_path(comp_dir));
    done << "algorithm=" << algorithm << "\n"
         << "n=" << n << "\n"
         << "c=" << c << "\n";

    std::cout << "[gen_bitmap] Compression completed: " << comp_dir << std::endl;
    return 0;
}
