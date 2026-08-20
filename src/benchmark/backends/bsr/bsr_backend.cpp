#include "bsr_backend.h"

#include "intersection_algos.hpp"   // vendored upstream (src/core/bsr)

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace bsr_util {

void alloc_arrays(int capacity, int** bases, int** states) {
    size_t cap = static_cast<size_t>(std::max(capacity, 0));
    size_t n = cap + BSR_PAD;
    if (posix_memalign(reinterpret_cast<void**>(bases), 64, n * sizeof(int)) != 0
        || posix_memalign(reinterpret_cast<void**>(states), 64, n * sizeof(int)) != 0)
        throw std::bad_alloc();
    // zero only the tail padding: kernels may load (never use) up to 3 ints
    // past the logical end; full-array memset would distort op timings
    std::memset(*bases  + cap, 0, BSR_PAD * sizeof(int));
    std::memset(*states + cap, 0, BSR_PAD * sizeof(int));
}

void encode_from_positions(const std::vector<uint32_t>& sorted_pos,
                           uint64_t num_rows, BsrHandle& out) {
    out.num_rows = num_rows;
    out.alloc(static_cast<int>(sorted_pos.size()));
    if (sorted_pos.empty()) { out.size = 0; return; }
    // upstream expects int*; positions < 2^31 in all our workloads
    std::vector<int> tmp(sorted_pos.begin(), sorted_pos.end());
    out.size = offline_uint_trans_bsr(tmp.data(), static_cast<int>(tmp.size()),
                                      out.bases, out.states);
}

int intersect_adaptive(const int* ba, const int* sa, int size_a,
                       const int* bb, const int* sb, int size_b,
                       int* bc, int* sc) {
    if (size_a == 0 || size_b == 0) return 0;
    const long long big = std::max(size_a, size_b);
    const long long small = std::max(1, std::min(size_a, size_b));
    if (big / small > GALLOP_RATIO)
        return intersect_simdgalloping_bsr(const_cast<int*>(ba), const_cast<int*>(sa), size_a,
                                           const_cast<int*>(bb), const_cast<int*>(sb), size_b,
                                           bc, sc);
    return intersect_qfilter_bsr_b4_v2(const_cast<int*>(ba), const_cast<int*>(sa), size_a,
                                       const_cast<int*>(bb), const_cast<int*>(sb), size_b,
                                       bc, sc);
}

// ---- our additions: upstream ships intersection/subtract only ----
int union_scalar(const int* ba, const int* sa, int size_a,
                 const int* bb, const int* sb, int size_b,
                 int* bc, int* sc) {
    int i = 0, j = 0, n = 0;
    while (i < size_a && j < size_b) {
        if (ba[i] == bb[j]) {
            bc[n] = ba[i]; sc[n] = sa[i] | sb[j]; n++; i++; j++;
        } else if (ba[i] < bb[j]) {
            bc[n] = ba[i]; sc[n] = sa[i]; n++; i++;
        } else {
            bc[n] = bb[j]; sc[n] = sb[j]; n++; j++;
        }
    }
    while (i < size_a) { bc[n] = ba[i]; sc[n] = sa[i]; n++; i++; }
    while (j < size_b) { bc[n] = bb[j]; sc[n] = sb[j]; n++; j++; }
    return n;
}

int xor_scalar(const int* ba, const int* sa, int size_a,
               const int* bb, const int* sb, int size_b,
               int* bc, int* sc) {
    int i = 0, j = 0, n = 0;
    while (i < size_a && j < size_b) {
        if (ba[i] == bb[j]) {
            int st = sa[i] ^ sb[j];
            if (st != 0) { bc[n] = ba[i]; sc[n] = st; n++; }
            i++; j++;
        } else if (ba[i] < bb[j]) {
            bc[n] = ba[i]; sc[n] = sa[i]; n++; i++;
        } else {
            bc[n] = bb[j]; sc[n] = sb[j]; n++; j++;
        }
    }
    while (i < size_a) { bc[n] = ba[i]; sc[n] = sa[i]; n++; i++; }
    while (j < size_b) { bc[n] = bb[j]; sc[n] = sb[j]; n++; j++; }
    return n;
}

int not_scalar(const int* ba, const int* sa, int size_a,
               uint64_t num_rows, int* bc, int* sc) {
    const int total_chunks = static_cast<int>((num_rows + 31) / 32);
    const uint32_t rem = static_cast<uint32_t>(num_rows % 32);
    int i = 0, out = 0;
    for (int chunk = 0; chunk < total_chunks; ++chunk) {
        uint32_t input_state = 0;
        if (i < size_a && ba[i] == chunk) {
            input_state = static_cast<uint32_t>(sa[i]);
            ++i;
        }
        // never shift by 32: build the tail mask only when rem != 0
        uint32_t valid_mask = 0xFFFFFFFFu;
        if (chunk == total_chunks - 1 && rem != 0)
            valid_mask = (1u << rem) - 1u;
        uint32_t output_state = (~input_state) & valid_mask;
        if (output_state != 0) {
            bc[out] = chunk;
            sc[out] = static_cast<int>(output_state);
            ++out;
        }
    }
    return out;
}

uint64_t popcount(const int* states, int size) {
    uint64_t total = 0;
    for (int i = 0; i < size; i++)
        total += static_cast<uint64_t>(__builtin_popcount(static_cast<unsigned>(states[i])));
    return total;
}

void to_dense_bitset(const int* bases, const int* states, int size,
                     uint64_t* words, size_t num_words) {
    std::memset(words, 0, num_words * sizeof(uint64_t));
    for (int i = 0; i < size; i++) {
        // chunk k covers bits [k*32, k*32+31] -> u64 word k/2, half k%2
        uint32_t st = static_cast<uint32_t>(states[i]);
        size_t w = static_cast<size_t>(bases[i]) >> 1;
        if (w >= num_words) continue;
        int shift = (bases[i] & 1) ? 32 : 0;
        words[w] |= (static_cast<uint64_t>(st) << shift);
    }
}

}

using namespace bsr_util;

BsrHandle::~BsrHandle() {
    free(bases);
    free(states);
}

void BsrHandle::alloc(int capacity) {
    free(bases); free(states);
    alloc_arrays(capacity, &bases, &states);
}

// append-mode handle: buffers positions, encodes lazily
struct BsrAppendHandle : public BsrHandle {
    std::vector<uint32_t> raw_positions;
    uint64_t appended_rows = 0;
    bool needs_encode = true;

    void ensure_encoded() {
        if (!needs_encode) return;
        encode_from_positions(raw_positions, appended_rows, *this);
        needs_encode = false;
    }
};

static const BsrHandle& H(const BitmapHandle& h) {
    auto* ah = dynamic_cast<BsrAppendHandle*>(const_cast<BitmapHandle*>(&h));
    if (ah) ah->ensure_encoded();
    return static_cast<const BsrHandle&>(h);
}

std::unique_ptr<BitmapHandle> BsrBackend::Create() {
    return std::make_unique<BsrAppendHandle>();
}

void BsrBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = static_cast<BsrAppendHandle&>(handle);
    if (bit) h.raw_positions.push_back(static_cast<uint32_t>(h.appended_rows));
    h.appended_rows++;
    h.needs_encode = true;
}

std::unique_ptr<BitmapHandle> BsrBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    const BsrHandle& ha = H(a); const BsrHandle& hb = H(b);
    auto res = std::make_unique<BsrHandle>();
    res->num_rows = std::max(ha.num_rows, hb.num_rows);
    res->alloc(std::min(ha.size, hb.size));
    res->size = intersect_adaptive(ha.bases, ha.states, ha.size,
                                   hb.bases, hb.states, hb.size,
                                   res->bases, res->states);
    return res;
}

std::unique_ptr<BitmapHandle> BsrBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t) {
    const BsrHandle& ha = H(a); const BsrHandle& hb = H(b);
    auto res = std::make_unique<BsrHandle>();
    res->num_rows = std::max(ha.num_rows, hb.num_rows);
    res->alloc(ha.size + hb.size);
    res->size = union_scalar(ha.bases, ha.states, ha.size,
                             hb.bases, hb.states, hb.size,
                             res->bases, res->states);
    return res;
}

std::unique_ptr<BitmapHandle> BsrBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    const BsrHandle& ha = H(a); const BsrHandle& hb = H(b);
    auto res = std::make_unique<BsrHandle>();
    res->num_rows = std::max(ha.num_rows, hb.num_rows);
    res->alloc(ha.size + hb.size);
    res->size = xor_scalar(ha.bases, ha.states, ha.size,
                           hb.bases, hb.states, hb.size,
                           res->bases, res->states);
    return res;
}

uint64_t BsrBackend::Cardinality(const BitmapHandle& handle) {
    const BsrHandle& h = H(handle);
    return popcount(h.states, h.size);
}

std::vector<uint32_t> BsrBackend::Decode(const BitmapHandle& handle) {
    const BsrHandle& h = H(handle);
    std::vector<int> out(popcount(h.states, h.size) + BSR_PAD);
    int n = offline_bsr_trans_uint(const_cast<int*>(h.bases), const_cast<int*>(h.states),
                                   h.size, out.data());
    return std::vector<uint32_t>(out.begin(), out.begin() + n);
}

// on-disk format: u64 num_rows | i32 size | bases[size] | states[size]
void BsrBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    const BsrHandle& h = H(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("BsrBackend::Serialize: cannot open " + path);
    out.write(reinterpret_cast<const char*>(&h.num_rows), sizeof(h.num_rows));
    out.write(reinterpret_cast<const char*>(&h.size), sizeof(h.size));
    if (h.size > 0) {
        out.write(reinterpret_cast<const char*>(h.bases),  sizeof(int) * h.size);
        out.write(reinterpret_cast<const char*>(h.states), sizeof(int) * h.size);
    }
}

std::unique_ptr<BitmapHandle> BsrBackend::Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("BsrBackend::Load: cannot open " + path);
    auto h = std::make_unique<BsrHandle>();
    in.read(reinterpret_cast<char*>(&h->num_rows), sizeof(h->num_rows));
    int size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    h->alloc(size);
    h->size = size;
    if (size > 0) {
        in.read(reinterpret_cast<char*>(h->bases),  sizeof(int) * size);
        in.read(reinterpret_cast<char*>(h->states), sizeof(int) * size);
    }
    if (!in) throw std::runtime_error("BsrBackend::Load: truncated file " + path);
    return h;
}
