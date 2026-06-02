

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

#include "fastbit/bitvector.h"
#include "utils/util.h"

#include "croaring/roaring.hh"

#include "ewah/ewah.h"

#include "Concise/concise.h"

#include <ddc.h>
#include <ddc_n.h>

namespace fs = std::filesystem;

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

    std::vector<uint32_t> ix(SEG_BITS);
    for (int v = 1; v <= num_bitmaps; v++) {
        buckets[v].reserve(num_segs * static_cast<size_t>(count_a));
        for (size_t s = 0; s < num_segs; s++) {
            std::iota(ix.begin(), ix.end(), 0);

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

            for (uint32_t p : seg_pos) {
                if (static_cast<uint64_t>(p) < num_rows)
                    buckets[v].push_back(p);
            }
        }
    }
    return buckets;
}

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

    std::vector<std::vector<uint32_t>> buckets(3);
    buckets[1].reserve(num_segs * static_cast<size_t>(count_a));
    buckets[2].reserve(num_segs * static_cast<size_t>(count_b));

    std::vector<uint32_t> ix(SEG_BITS);
    const int prefix_len = count_a + count_b;
    for (size_t s = 0; s < num_segs; s++) {

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

    std::vector<std::vector<uint32_t>> buckets(3);
    buckets[1].reserve(num_segs * static_cast<size_t>(count_a));
    buckets[2].reserve(num_segs * static_cast<size_t>(count_a));

    std::vector<uint32_t> ix(SEG_BITS);
    std::vector<uint8_t>  in_a(SEG_BITS);
    for (size_t s = 0; s < num_segs; s++) {

        std::iota(ix.begin(), ix.end(), 0);
        std::mt19937 rng_a(seed + 2 * s + 0);
        for (int i = 0; i < count_a; i++) {
            std::uniform_int_distribution<uint32_t> pick(i, SEG_BITS - 1);
            std::swap(ix[i], ix[pick(rng_a)]);
        }
        const uint32_t base = static_cast<uint32_t>(s * SEG_BITS);
        std::fill(in_a.begin(), in_a.end(), 0);
        for (int i = 0; i < count_a; i++) in_a[ix[i]] = 1;

        std::vector<uint32_t> b_pos;
        b_pos.reserve(count_a);
        for (uint32_t i = 0; i < carry; i++) b_pos.push_back(base + ix[i]);
        std::mt19937 rng_b(seed + 2 * s + 1);
        uint32_t fresh_added = 0;
        while (fresh_added < fresh) {
            uint32_t p = std::uniform_int_distribution<uint32_t>(0, SEG_BITS - 1)(rng_b);
            if (!in_a[p]) { in_a[p] = 2; b_pos.push_back(base + p); fresh_added++; }
        }

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

bool gen_ewah(const std::vector<std::vector<uint32_t>>& buckets,
              const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ewah::EWAHBoolArray<uint64_t> btv;
        for (uint32_t idx : buckets[v]) {
            btv.set(idx);
        }

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

bool gen_concise(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        ConciseSet<false> btv;
        for (uint32_t idx : buckets[v]) {
            btv.add(idx);
        }

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

bool gen_ddc(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality,
                int word_size, size_t segment_bits = DDC::default_segment_bits)
{
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {

        const auto& raw = buckets[v];
        std::vector<bool> bits(rows, false);
        for (size_t i = 0; i < raw.size() && raw[i] < rows; i++)
            bits[raw[i]] = true;

        DDC cb = DDC::compress(bits,  false, segment_bits);
        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        cb.serialize(out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_ddc] Written " << v << "/" << cardinality << " bitmaps\n";
        }
    }
    std::cout << "[gen_ddc] All " << cardinality << " DDC (w" << word_size
              << ", S=" << segment_bits << ") bitmaps written to " << output_dir << "\n";
    return true;
}

bool gen_ddc_n(const std::vector<std::vector<uint32_t>>& buckets,
                  const std::string& output_dir, uint64_t rows, int cardinality,
                  int depth)
{
    if (depth < 2 || depth > 5) {
        std::cerr << "[gen_ddc_n] Error: depth must be in {2,3,4,5}, got "
                  << depth << "\n";
        return false;
    }
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        const auto& raw = buckets[v];
        std::vector<bool> bits(rows, false);
        for (size_t i = 0; i < raw.size() && raw[i] < rows; i++)
            bits[raw[i]] = true;

        DDCN cb = ddc_n_compress(bits, depth);
        std::string out_path = output_dir + "/" + std::to_string(v) + ".bm";
        std::ofstream out(out_path, std::ios::binary);
        ddc_n_serialize(cb, out);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_ddc_n] Written " << v << "/" << cardinality
                      << " bitmaps (depth=" << depth << ")\n";
        }
    }
    std::cout << "[gen_ddc_n] All " << cardinality << " DDCN (L"
              << depth << ") bitmaps written to " << output_dir << "\n";
    return true;
}

bool gen_ddc_n_tile(const std::vector<std::vector<uint32_t>>& buckets,
                       const std::string& output_dir, uint64_t small_n,
                       int cardinality, int tile_factor, int depth) {
    if (depth < 2 || depth > 5) {
        std::cerr << "[gen_ddc_n_tile] Error: depth must be in {2,3,4,5}, got "
                  << depth << "\n";
        return false;
    }
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {
        std::vector<bool> bits(small_n, false);
        for (uint32_t idx : buckets[v])
            if (idx < small_n) bits[idx] = true;

        DDCN small_cb = ddc_n_compress(bits, depth);

        std::stringstream ss;
        ddc_n_serialize(small_cb, ss);
        std::string buf = ss.str();

        constexpr size_t HEADER_BYTES = 26;
        if (buf.size() < HEADER_BYTES) {
            std::cerr << "[gen_ddc_n_tile] Error: bad serialize size\n";
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
        out.write(buf.data(), 2);
        out.write(reinterpret_cast<const char*>(&new_bit_count), 8);
        out.write(reinterpret_cast<const char*>(&seg_bits),      8);
        out.write(reinterpret_cast<const char*>(&new_num_segs),  8);

        const char* payload = buf.data() + HEADER_BYTES;
        size_t payload_size = buf.size() - HEADER_BYTES;
        for (int i = 0; i < tile_factor; i++)
            out.write(payload, payload_size);

        if (v % 100 == 0 || v == cardinality) {
            std::cout << "[gen_ddc_n_tile] Written " << v << "/" << cardinality
                      << " (depth=" << depth << ", small_n=" << small_n
                      << ", tile=" << tile_factor
                      << " → n=" << new_bit_count << ")\n";
        }
    }
    std::cout << "[gen_ddc_n_tile] All " << cardinality
              << " DDCN (L" << depth << ") bitmaps written (tile mode) to "
              << output_dir << "\n";
    return true;
}

bool gen_ddc_tile(const std::vector<std::vector<uint32_t>>& buckets,
                     const std::string& output_dir,
                     uint64_t small_n, int cardinality, int tile_factor) {
    fs::create_directories(output_dir);

    for (int v = 1; v <= cardinality; v++) {

        std::vector<bool> bits(small_n, false);
        for (uint32_t idx : buckets[v])
            if (idx < small_n) bits[idx] = true;

        DDC small_cb = DDC::compress(bits);

        std::stringstream ss;
        small_cb.serialize(ss);
        std::string buf = ss.str();
        if (buf.size() < 24) {
            std::cerr << "[gen_ddc_tile] Error: bad serialize size\n";
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
            std::cout << "[gen_ddc_tile] Written " << v << "/" << cardinality
                      << " (small_n=" << small_n << ", tile=" << tile_factor
                      << " → n=" << new_bit_count << ")\n";
        }
    }
    std::cout << "[gen_ddc_tile] All " << cardinality
              << " DDC bitmaps written (tile mode) to " << output_dir << "\n";
    return true;
}

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

static std::string raw_dir_path(const std::string& base_dir, int n, int c) {
    return base_dir + "/bitmap/bitmaps_" + format_rows(n)
         + "_c" + std::to_string(c);
}

static std::string compressed_dir_path(const std::string& base_dir, int n, int c,
                                       const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_c" + std::to_string(c) + "_" + algo_lower;
}

static std::string transition_dir_path(const std::string& base_dir, int n,
                                       int count_a, const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_t" + std::to_string(count_a) + "_" + algo_lower;
}

static std::string overlap_dir_path(const std::string& base_dir, int n,
                                    int count_a, const std::string& algorithm) {
    std::string algo_lower = algorithm;
    for (auto& ch : algo_lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return base_dir + "/bitmap/bm_" + format_rows(n)
         + "_o" + std::to_string(count_a) + "_" + algo_lower;
}

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

static bool dataset_exists_in_index(const std::string& base_dir, int n, int c) {
    std::string index_path = base_dir + "/index.txt";
    std::ifstream f(index_path);
    if (!f.is_open()) return false;

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

static bool generate_compressed(const std::string& algorithm,
                                const std::vector<std::vector<uint32_t>>& buckets,
                                const std::string& output_dir,
                                uint64_t rows, int cardinality, int z,
                                int word_size, int depth_n,
                                size_t segment_bits = DDC::default_segment_bits) {
    std::string algo = algorithm;
    for (auto& ch : algo)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (algo == "bitset")  return gen_bitset(buckets, output_dir, rows, cardinality, z);
    if (algo == "wah")     return gen_wah(buckets, output_dir, rows, cardinality);
    if (algo == "roaring") return gen_roaring(buckets, output_dir, rows, cardinality);
    if (algo == "ewah")    return gen_ewah(buckets, output_dir, rows, cardinality);
    if (algo == "concise") return gen_concise(buckets, output_dir, rows, cardinality);
    if (algo == "ddc") {
        if (depth_n > 0)
            return gen_ddc_n(buckets, output_dir, rows, cardinality, depth_n);
        return gen_ddc(buckets, output_dir, rows, cardinality, word_size, segment_bits);
    }

    std::cerr << "[gen_bitmap] Error: unsupported algorithm '" << algorithm << "'\n"
              << "[gen_bitmap] Supported: bitset, wah, roaring, ewah, concise, ddc\n";
    return false;
}

int main(int argc, char* argv[]) {
    int n = -1;
    int c = -1;
    int t = -1;
    int o = -1;
    int asym_a = -1, asym_b = -1;
    int z = 1;
    int w = 8;
    int tile = 1;
    int L_depth = -1;
    long long seg_bits_arg = -1;
    bool run_length = false;
    std::string algorithm;
    std::string base_dir = ".";

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
                  << "  <algorithm>   : compression algorithm (bitset, wah, roaring, ewah, concise, ddc)\n"
                  << "  -d <base_dir> : base directory (default: .)\n"
                  << "  -z <z>        : zip length for bitset mode (default: 1)\n"
                  << "  -w <w>        : DDC word size: 8, 16, 32, or 64 (default: 8)\n"
                  << "  -L <depth>    : DDCN depth (ddc only): 2, 3, 4, or 5.\n"
                  << "                  Overrides -w.  Produces bitmap/bm_<n>_c<c>_ddc_L<depth>/\n"
                  << "                  files in the on-disk DDCN format.  Used by the\n"
                  << "                  bench_L_ops fairness study.  Mutually exclusive with -T.\n"
                  << "  -S <bits>     : DDC segment_bits (ddc only, L4 only): power of 2 in\n"
                  << "                  [64, 2^24].  Default 65536 (2^16).  Output dir gets the\n"
                  << "                  suffix _S<bits> when non-default.  Mutually exclusive with -L\n"
                  << "                  and -T.  Used by the segment-size ablation study.\n"
                  << "  -T <T>        : tile factor (ddc only).  Generate small dataset of\n"
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
        if (algo_lower != "ddc") {
            std::cerr << "Error: -L is only valid with ddc algorithm" << std::endl;
            return 1;
        }
        if (L_depth < 2 || L_depth > 5) {
            std::cerr << "Error: -L must be 2, 3, 4, or 5" << std::endl;
            return 1;
        }
    }

    size_t segment_bits_val = DDC::default_segment_bits;
    if (seg_bits_arg > 0) {
        if (algo_lower != "ddc") {
            std::cerr << "Error: -S is only valid with ddc algorithm" << std::endl;
            return 1;
        }
        if (L_depth > 0) {
            std::cerr << "Error: -S currently only supported with native DDC (L4); not with -L" << std::endl;
            return 1;
        }
        if (tile > 1) {
            std::cerr << "Error: -S and -T are mutually exclusive (tile mode hard-codes seg_bits=n/T)" << std::endl;
            return 1;
        }

        if (seg_bits_arg < 64 || (seg_bits_arg & (seg_bits_arg - 1)) != 0
            || seg_bits_arg > (1LL << 24)) {
            std::cerr << "Error: -S must be a power of 2 in [64, 2^24], got "
                      << seg_bits_arg << std::endl;
            return 1;
        }
        segment_bits_val = static_cast<size_t>(seg_bits_arg);
    }

    auto ddc_suffix = [&]() {
        std::string base = (L_depth > 0)
            ? std::string("_L") + std::to_string(L_depth)
            : std::string("_w") + std::to_string(w);
        if (seg_bits_arg > 0 && static_cast<size_t>(seg_bits_arg) != DDC::default_segment_bits)
            base += "_S" + std::to_string(seg_bits_arg);
        return base;
    };

    std::string comp_dir;
    if (transition_mode) {

        comp_dir = transition_dir_path(base_dir, n, t, algorithm);
        if (algo_lower == "ddc") comp_dir += ddc_suffix();
    } else if (overlap_mode) {

        comp_dir = overlap_dir_path(base_dir, n, o, algorithm);
        if (algo_lower == "ddc") comp_dir += ddc_suffix();
    } else if (asym_mode) {

        comp_dir = asym_dir_path(base_dir, n, asym_a, asym_b, algorithm);
        if (algo_lower == "ddc") comp_dir += ddc_suffix();
    } else if (algo_lower == "bitset") {
        if (z == 1)
            comp_dir = compressed_dir_path(base_dir, n, c, "bitset");
        else
            comp_dir = base_dir + "/bitmap/bitmap_n" + format_rows(n)
                     + "_c" + std::to_string(c) + "_z" + std::to_string(z);
    } else if (algo_lower == "ddc") {
        comp_dir = compressed_dir_path(base_dir, n, c, algorithm) + ddc_suffix();
    } else {
        comp_dir = compressed_dir_path(base_dir, n, c, algorithm);
    }

    if (run_length) comp_dir += "_run";

    int gen_n = (tile > 1) ? (n / tile) : n;

    std::cout << "[gen_bitmap] n=" << n << " c=" << c
              << " algorithm=" << algorithm
              << (tile > 1 ? (" tile=" + std::to_string(tile)
                              + " small_n=" + std::to_string(gen_n))
                           : "")
              << " base_dir=" << base_dir << std::endl;

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
             0xCAFE0000u + static_cast<uint64_t>(t));
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
             0xBEEF0000u + static_cast<uint64_t>(o));
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
             0xCAFE0000u + static_cast<uint64_t>(asym_a) * 1000
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

    if (compression_exists(comp_dir)) {
        std::cout << "[gen_bitmap] Compressed data already exists: " << comp_dir << std::endl;
        return 0;
    }

    std::cout << "[gen_bitmap] Generating compressed bitmaps..." << std::endl;
    fs::create_directories(comp_dir);

    if (!fs::exists(ds_file)) {
        std::cerr << "[gen_bitmap] Error: dataset not found: " << ds_file << std::endl;
        return 1;
    }

    std::cout << "[gen_bitmap] Reading dataset into buckets..." << std::endl;
    auto buckets = read_dataset_buckets(ds_file, static_cast<uint64_t>(gen_n), c);
    if (buckets.empty()) return 1;

    if (tile > 1) {
        bool ok;
        if (algo_lower == "ddc") {
            if (L_depth > 0)
                ok = gen_ddc_n_tile(buckets, comp_dir,
                                       static_cast<uint64_t>(gen_n), c,
                                       tile, L_depth);
            else
                ok = gen_ddc_tile(buckets, comp_dir,
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

    std::string raw_dir = raw_dir_path(base_dir, n, c);
    std::string raw_done = raw_dir + "/done.txt";
    if (!fs::exists(raw_done)) {
        std::cout << "[gen_bitmap] Generating raw bitmaps → " << raw_dir << std::endl;
        if (!gen_raw(buckets, raw_dir, static_cast<uint64_t>(n), c)) return 1;
        std::ofstream(raw_done) << "n=" << n << "\nc=" << c << "\n";
    } else {
        std::cout << "[gen_bitmap] Raw bitmaps already exist: " << raw_dir << std::endl;
    }

    if (!generate_compressed(algorithm, buckets, comp_dir, static_cast<uint64_t>(n), c, z, w, L_depth, segment_bits_val)) {
        std::cerr << "[gen_bitmap] Error: compression failed." << std::endl;
        return 1;
    }

    std::ofstream done(compressed_done_path(comp_dir));
    done << "algorithm=" << algorithm << "\n"
         << "n=" << n << "\n"
         << "c=" << c << "\n";

    std::cout << "[gen_bitmap] Compression completed: " << comp_dir << std::endl;
    return 0;
}
