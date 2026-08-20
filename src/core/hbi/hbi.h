// HBI-Pointer: node-oriented hierarchical bitmap (Morzy et al., ADBIS 2003)
// extended with a compressed-to-compressed Boolean algebra, per the teacher's
// spec for the VLDB revision (R1.W1 / R3-W2 rebuttal experiments).
//
// Faithful-but-fair baseline rules (teacher, meeting 2026-07-22):
//   - real pointer tree: subtrees connected by pointers; the non-empty
//     children of one parent live in ONE contiguous child block (a single
//     arena allocation), navigated by mask + rank + pointer;
//   - NO global packed array, NO offset navigation across subtrees;
//   - NOT materializes the complemented all-zero regions as real nodes —
//     no dual-polarity fill, no lazy/complement flags (that is DDC's design
//     difference, and exactly what the experiment must expose);
//   - AND/OR/XOR/NOT/COMP emit fully independent results (deep-copied
//     subtrees, own arena) — no aliasing into the inputs;
//   - only all-zero nodes are elided (occupancy semantics, as in the paper).
//
// Mask convention: logical child j (0-based, covering lower source positions
// first) maps to mask bit (l-1-j), i.e. MSB-first inside the low l bits.
// Printing the low l bits MSB-first reproduces the paper's node strings
// (Example 1: <1010,1011,0100,0110,1001,1100,0101>).
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace hbi {

struct HbiNode {
    HbiNode* children;      // contiguous block of the non-empty children
    uint32_t mask;          // low node_bits used; bit (l-1-j) = child j non-empty
    uint32_t child_count;   // popcount(mask) for internal nodes, 0 for leaves
};
static_assert(sizeof(HbiNode) == 16, "HbiNode must stay 16 bytes");

// Bump allocator handing out contiguous per-parent child blocks.  This is an
// ALLOCATOR only: no global layout, no index-based navigation.
class Arena {
public:
    static constexpr size_t kChunkNodes = 1u << 16;   // 1 MiB chunks

    HbiNode* alloc(uint32_t n);                        // n contiguous nodes
    void reset();

    uint64_t alloc_calls    = 0;   // number of child-block allocations
    uint64_t nodes_allocated = 0;  // total nodes handed out
    uint64_t reserved_bytes = 0;   // bytes actually reserved from the OS

private:
    std::vector<std::unique_ptr<HbiNode[]>> chunks_;
    size_t used_in_last_ = 0;      // nodes used in chunks_.back()
    size_t last_capacity_ = 0;
};

class HbiBitmap {
public:
    HbiBitmap() = default;
    HbiBitmap(HbiBitmap&&) = default;
    HbiBitmap& operator=(HbiBitmap&&) = default;
    HbiBitmap(const HbiBitmap&) = delete;
    HbiBitmap& operator=(const HbiBitmap&) = delete;

    // --- construction ---------------------------------------------------
    // Build from a dense bitvector (words[i] bit b = source bit 64*i+b).
    static HbiBitmap build_from_dense(const uint64_t* words, uint64_t nbits,
                                      uint32_t node_bits);

    // --- Boolean algebra (compressed -> compressed, independent result) --
    static HbiBitmap and_(const HbiBitmap& a, const HbiBitmap& b);
    static HbiBitmap or_ (const HbiBitmap& a, const HbiBitmap& b);
    static HbiBitmap xor_(const HbiBitmap& a, const HbiBitmap& b);
    static HbiBitmap not_(const HbiBitmap& a);
    // COMP = ~((b1|b2) & (b2|b3)); intermediates stay HBI.
    static HbiBitmap comp(const HbiBitmap& b1, const HbiBitmap& b2,
                          const HbiBitmap& b3);

    // --- delivery / verification ----------------------------------------
    void to_dense(uint64_t* words, size_t num_words) const;
    uint64_t popcount() const;
    uint64_t checksum_dense() const;    // FNV-1a over the dense image

    // --- metrics ---------------------------------------------------------
    uint64_t node_count() const { return node_count_; }
    uint64_t serialized_bytes() const;  // header + node_count * l / 8 (packed)
    uint64_t allocated_bytes() const { return arena_.reserved_bytes; }
    uint64_t alloc_calls() const { return arena_.alloc_calls; }
    uint32_t depth() const { return depth_; }
    uint32_t node_bits() const { return node_bits_; }
    uint64_t nbits() const { return nbits_; }
    bool empty() const { return root_ == nullptr; }

    // --- serialization (BFS level-order masks, packed to l bits/node) ----
    void serialize(const std::string& path) const;
    static HbiBitmap deserialize(const std::string& path);

    // Paper-order node strings (BFS), for the Example 1 unit test.
    std::vector<std::string> node_strings() const;

private:
    uint64_t nbits_ = 0;
    uint32_t node_bits_ = 32;
    uint32_t depth_ = 0;
    uint64_t node_count_ = 0;
    HbiNode* root_ = nullptr;
    Arena arena_;

    friend struct Builder;
};

// Smallest depth d with l^d >= nbits.
uint32_t compute_depth(uint64_t nbits, uint32_t node_bits);

} // namespace hbi
