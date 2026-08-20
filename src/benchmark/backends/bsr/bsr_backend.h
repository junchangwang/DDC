#pragma once

#include "../../bitmap_backend.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

// QFilter/BSR backend (SIGMOD'18 reference implementation, vendored at
// src/core/bsr/).  BSR = two parallel arrays: bases[] (non-empty 32-bit
// chunk indices) + states[] (chunk payloads).
//
// Semantics of this adapter:
//   - AND  : upstream kernels, adaptive like the paper (size ratio > 32
//            -> intersect_simdgalloping_bsr, else intersect_qfilter_bsr_b4_v2).
//   - OR/XOR: NOT provided upstream (paper implements intersection only).
//            We supply a faithful scalar base-merge (state OR / state XOR,
//            all-zero results dropped) so multi-way OR benchmarks can run.
//            These two rows must be labelled as our addition in any report.
//   - NOT  : not representable compactly in BSR (omitted all-zero chunks
//            become all-one); not implemented.
struct BsrHandle : public BitmapHandle {
    int*      bases  = nullptr;   // 16B-aligned, size + BSR_PAD ints
    int*      states = nullptr;
    int       size   = 0;         // number of non-empty chunks
    uint64_t  num_rows = 0;

    BsrHandle() = default;
    BsrHandle(const BsrHandle&) = delete;
    BsrHandle& operator=(const BsrHandle&) = delete;
    ~BsrHandle() override;

    void alloc(int capacity);     // aligned alloc with tail padding
    uint64_t storage_bytes() const {
        return static_cast<uint64_t>(size) * (sizeof(int) + sizeof(int));
    }
};

class BsrBackend : public IBitmapBackend {
public:
    std::unique_ptr<BitmapHandle> Create() override;
    void Append(BitmapHandle& btv, bool bit) override;
    std::unique_ptr<BitmapHandle> bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range = 2) override;
    std::unique_ptr<BitmapHandle> bitAnd(const BitmapHandle& a, const BitmapHandle& b) override;
    std::unique_ptr<BitmapHandle> bitXor(const BitmapHandle& a, const BitmapHandle& b) override;
    void Serialize(const BitmapHandle& btv, const std::string& path) override;
    std::unique_ptr<BitmapHandle> Load(const std::string& path) override;
    uint64_t Cardinality(const BitmapHandle& btv) override;
    std::vector<uint32_t> Decode(const BitmapHandle& btv) override;
};

// Shared helpers (also used by convert_bsr and the debit integration).
namespace bsr_util {

constexpr int BSR_PAD = 8;            // tail padding (ints) past `size`
constexpr int GALLOP_RATIO = 32;      // paper's adaptive cross point

void alloc_arrays(int capacity, int** bases, int** states);

// sorted positions -> handle (uses upstream offline_uint_trans_bsr)
void encode_from_positions(const std::vector<uint32_t>& sorted_pos,
                           uint64_t num_rows, BsrHandle& out);

// adaptive intersection per the paper; result arrays must have
// min(size_a,size_b)+BSR_PAD capacity.  Returns result size.
int intersect_adaptive(const int* ba, const int* sa, int size_a,
                       const int* bb, const int* sb, int size_b,
                       int* bc, int* sc);

// --- our additions (upstream has no union/xor); plain scalar merge ---
int union_scalar(const int* ba, const int* sa, int size_a,
                 const int* bb, const int* sb, int size_b,
                 int* bc, int* sc);
int xor_scalar(const int* ba, const int* sa, int size_a,
               const int* bb, const int* sb, int size_b,
               int* bc, int* sc);

uint64_t popcount(const int* states, int size);

// Semantically correct complement OVER THE UNIVERSE [0, num_rows) — OUR
// addition (upstream QFilter/BSR is intersection-only and has no NOT).
// Omitted chunks (all-zero) become all-one; all-one chunks vanish from the
// output; the final chunk is masked to the universe (no padding bits).
// Output arrays need ceil(num_rows/32)+BSR_PAD capacity.  The result is a
// fully materialized BSR (no lazy negation flag), so the measured cost is
// the true representational cost: O(num_rows/32) time, ~2N bits space for
// sparse inputs.  Returns result size.
int not_scalar(const int* ba, const int* sa, int size_a,
               uint64_t num_rows, int* bc, int* sc);

// decode BSR into a dense bitset buffer (u64 words, LSB-first within word);
// used to charge BSR the same "result must drive a bitmap consumer"
// conversion cost that CRoaring pays in this benchmark.
void to_dense_bitset(const int* bases, const int* states, int size,
                     uint64_t* words, size_t num_words);

}
