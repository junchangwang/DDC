#pragma once

#include <string>
#include <cstdint>
#include <vector>

std::vector<std::vector<uint32_t>> read_dataset_buckets(
    const std::string& dataset_path, uint64_t rows, int cardinality);

bool gen_raw(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality);

bool gen_bitset(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality, int z);

bool gen_wah(const std::vector<std::vector<uint32_t>>& buckets,
             const std::string& output_dir, uint64_t rows, int cardinality);

bool gen_roaring(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality);

bool gen_ewah(const std::vector<std::vector<uint32_t>>& buckets,
              const std::string& output_dir, uint64_t rows, int cardinality);

bool gen_concise(const std::vector<std::vector<uint32_t>>& buckets,
                 const std::string& output_dir, uint64_t rows, int cardinality);

bool gen_ddc(const std::vector<std::vector<uint32_t>>& buckets,
                const std::string& output_dir, uint64_t rows, int cardinality,
                int word_size);

bool gen_ddc_n(const std::vector<std::vector<uint32_t>>& buckets,
                  const std::string& output_dir, uint64_t rows, int cardinality,
                  int depth);

bool gen_ddc_n_tile(const std::vector<std::vector<uint32_t>>& buckets,
                       const std::string& output_dir, uint64_t small_n,
                       int cardinality, int tile_factor, int depth);
