

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>
#include <iosfwd>

// per-segment levels
struct DDCNSeg {
    int   depth;
    size_t bit_count;

    std::vector<uint8_t> l1_lits;
    bool   l1_fill_ones;

    std::vector<uint8_t> l2_lits;
    bool   l2_fill_ones;

    std::vector<uint8_t> l3_lits;
    bool   l3_fill_ones;

    std::vector<uint8_t> l4_lits;
    bool   l4_fill_ones;

    std::vector<uint8_t> top_raw;

    size_t l2_count;
    size_t l3_count;
    size_t l4_count;
    size_t l5_count;
};

struct DDCN {
    int    depth;
    size_t bit_count;
    size_t segment_bits;
    std::vector<DDCNSeg> segments;
};

// compress
DDCN ddc_n_compress(const std::vector<bool>& bits,
                          int depth,
                          size_t segment_bits = (1u << 16),
                          bool l1_fill_ones = false);

std::vector<bool> ddc_n_decompress(const DDCN& cb);

size_t ddc_n_popcount(const DDCN& cb);

// decode to bytes
std::vector<uint8_t> ddc_n_and_dec(const DDCN& a, const DDCN& b);
std::vector<uint8_t> ddc_n_or_dec (const DDCN& a, const DDCN& b);
std::vector<uint8_t> ddc_n_xor_dec(const DDCN& a, const DDCN& b);

std::vector<bool> bytes_to_bits(const std::vector<uint8_t>& bytes, size_t bit_count);

// SIMD decode
std::vector<uint8_t> ddc_n_and_dec_avx(const DDCN& a, const DDCN& b);
std::vector<uint8_t> ddc_n_or_dec_avx (const DDCN& a, const DDCN& b);
std::vector<uint8_t> ddc_n_xor_dec_avx(const DDCN& a, const DDCN& b);

// compressed-result ops
DDCN ddc_n_or (const DDCN& a, const DDCN& b);
DDCN ddc_n_and(const DDCN& a, const DDCN& b);

// bypass config
enum class BypassCfg { BP_ALL, BP_NO_L4, BP_NO_L3, BP_NONE, BP_BRANCHLESS, BP_BL_ALL };

std::vector<uint8_t> ddc_n_or_dec_l4_cfg (const DDCN& a, const DDCN& b, BypassCfg cfg);
std::vector<uint8_t> ddc_n_and_dec_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg cfg);

DDCN ddc_n_or_l4_cfg (const DDCN& a, const DDCN& b, BypassCfg cfg);
DDCN ddc_n_and_l4_cfg(const DDCN& a, const DDCN& b, BypassCfg cfg);

void ddc_n_not_inplace(DDCN& a);

// bypass path stats
struct DDCNPathCounts {
    size_t zz = 0;
    size_t az = 0;
    size_t bz = 0;
    size_t full = 0;
};
const DDCNPathCounts& ddc_n_path_counts();
void ddc_n_reset_path_counts();

size_t ddc_n_seg_bytes(const DDCNSeg& seg);

size_t ddc_n_total_bytes(const DDCN& cb);

// serialize
void ddc_n_serialize(const DDCN& cb, std::ostream& os);
DDCN ddc_n_deserialize(std::istream& is);
