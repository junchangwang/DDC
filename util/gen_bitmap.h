#pragma once

#include <string>
#include <cstdint>
#include <vector>

/// Read dataset and bucket row indices by value.
/// Returns buckets[0..cardinality], where buckets[v] = list of row indices where data[i]==v.
/// buckets[0] is unused. Returns empty vector on failure.
std::vector<std::vector<uint32_t>> read_dataset_buckets(
    const std::string& dataset_path, uint64_t rows, int cardinality);

/// Generate raw uncompressed bitmaps (little-endian packed bits) into output_dir.
/// One file per value: {output_dir}/1.bm, 2.bm, ...
bool gen_raw(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality);

/// Generate bitset bitmaps with zip_length z.
/// z=1: individual .bm files.  z>1: merged .bmz files (z bitmaps per file).
bool gen_bitset(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality, int z);

/// Generate WAH-compressed bitmaps using FastBit.
bool gen_wah(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality);

/// Generate CRoaring-compressed bitmaps.
bool gen_roaring(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality);

/// Generate EWAH-compressed bitmaps.
bool gen_ewah(const std::vector<std::vector<uint32_t>>& buckets,
              const std::string& output_dir, uint64_t rows, int cardinality);

/// Generate Concise-compressed bitmaps.
bool gen_concise(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality);

/// Generate ComBit-compressed bitmaps with specified word size (8/16/32/64).
bool gen_combit(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality,
                int word_size);

/// Generate ComBitN-compressed bitmaps at a fixed depth (2/3/4/5).
/// Each .bm file is the result of combit_n_compress(bits, depth) followed
/// by combit_n_serialize(...).  Used by the bench_L_ops fairness study so
/// each depth gets a real on-disk artefact that the bench can deserialize
/// directly (no in-memory recompression in the timed window).
bool gen_combit_n(const std::vector<std::vector<uint32_t>>& buckets,
                  const std::string& output_dir, uint64_t rows, int cardinality,
                  int depth);

/// Tile-mode ComBitN generator — mirrors gen_combit_tile.  Compresses a
/// small_n-bit ComBitN once, then byte-concatenates the per-segment payload
/// tile_factor× and rewrites the top-level header.  Used to match the
/// existing L4 .bm files at c=2000 (which were generated tile=100 small_n=1M).
bool gen_combit_n_tile(const std::vector<std::vector<uint32_t>>& buckets,
                       const std::string& output_dir, uint64_t small_n,
                       int cardinality, int tile_factor, int depth);
