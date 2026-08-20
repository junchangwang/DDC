#include "hbi.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace hbi {

// ---------------------------------------------------------------- Arena ----
HbiNode* Arena::alloc(uint32_t n) {
    if (n == 0) return nullptr;
    if (chunks_.empty() || used_in_last_ + n > last_capacity_) {
        size_t cap = n > kChunkNodes ? n : kChunkNodes;
        chunks_.emplace_back(new HbiNode[cap]);
        last_capacity_ = cap;
        used_in_last_ = 0;
        reserved_bytes += cap * sizeof(HbiNode);
    }
    HbiNode* p = chunks_.back().get() + used_in_last_;
    used_in_last_ += n;
    alloc_calls++;
    nodes_allocated += n;
    return p;
}

void Arena::reset() {
    chunks_.clear();
    used_in_last_ = last_capacity_ = 0;
    alloc_calls = nodes_allocated = 0;
    reserved_bytes = 0;
}

// ------------------------------------------------------------- helpers ----
uint32_t compute_depth(uint64_t nbits, uint32_t node_bits) {
    uint32_t d = 1;
    // capacity = node_bits^d, grown until it covers nbits
    unsigned __int128 cap = node_bits;
    while (cap < nbits) { cap *= node_bits; d++; }
    return d;
}

static inline uint32_t low_mask(uint32_t l) {
    return l >= 32 ? 0xFFFFFFFFu : ((1u << l) - 1u);
}

static inline int rank_before(uint32_t mask, uint32_t l, uint32_t j) {
    // number of set bits for logical children < j  (bits above bit l-1-j)
    if (j == 0) return 0;
    return __builtin_popcount(mask >> (l - j));
}

// span of ONE child of a node at `level` (levels: 0 = root .. depth-1 = leaf)
static inline uint64_t child_span(uint32_t node_bits, uint32_t depth,
                                  uint32_t level) {
    // node at `level` covers node_bits^(depth-level); child covers /node_bits
    uint64_t s = 1;
    for (uint32_t k = 0; k < depth - level - 1; k++) s *= node_bits;
    return s;
}

// -------------------------------------------------------------- Builder ----
struct Builder {
    const uint64_t* words;
    uint64_t nbits;
    uint32_t l;
    uint32_t depth;
    Arena* arena;
    uint64_t node_count = 0;

    static inline bool bit(const uint64_t* w, uint64_t b) {
        return (w[b >> 6] >> (b & 63)) & 1ull;
    }

    // Returns true if non-empty; fills *out (children allocated in arena).
    bool build(uint32_t level, uint64_t start, HbiNode* out) {
        if (level == depth - 1) {                      // leaf: l source bits
            uint32_t mask = 0;
            uint64_t end = start + l < nbits ? start + l : nbits;
            for (uint64_t p = start; p < end; p++)
                if (bit(words, p)) mask |= 1u << (l - 1 - (uint32_t)(p - start));
            if (!mask) return false;
            out->children = nullptr;
            out->mask = mask;
            out->child_count = 0;
            node_count++;
            return true;
        }
        uint64_t cs = child_span(l, depth, level);
        HbiNode tmp[32];
        uint32_t mask = 0, cnt = 0;
        for (uint32_t j = 0; j < l; j++) {
            uint64_t s = start + (uint64_t)j * cs;
            if (s >= nbits) break;
            if (build(level + 1, s, &tmp[cnt])) {
                mask |= 1u << (l - 1 - j);
                cnt++;
            }
        }
        if (!mask) return false;
        HbiNode* block = arena->alloc(cnt);
        std::memcpy(block, tmp, cnt * sizeof(HbiNode));
        out->children = block;
        out->mask = mask;
        out->child_count = cnt;
        node_count++;
        return true;
    }
};

HbiBitmap HbiBitmap::build_from_dense(const uint64_t* words, uint64_t nbits,
                                      uint32_t node_bits) {
    if (node_bits != 4 && node_bits != 8 && node_bits != 16 && node_bits != 32)
        throw std::runtime_error("hbi: node_bits must be 4/8/16/32");
    HbiBitmap bm;
    bm.nbits_ = nbits;
    bm.node_bits_ = node_bits;
    bm.depth_ = compute_depth(nbits, node_bits);
    Builder b{words, nbits, node_bits, bm.depth_, &bm.arena_, 0};
    HbiNode root;
    if (b.build(0, 0, &root)) {
        HbiNode* r = bm.arena_.alloc(1);
        *r = root;
        bm.root_ = r;
    }
    bm.node_count_ = b.node_count;
    return bm;
}

// ---------------------------------------------------- recursive operators --
// All operator recursions build the RESULT into `arena` and count nodes in
// `nc`; inputs are never aliased into the output (deep_copy materializes).

namespace {

struct OpCtx {
    uint32_t l;
    uint32_t depth;
    uint64_t nbits;
    Arena* arena;
    uint64_t nc = 0;
};

bool deep_copy(OpCtx& c, uint32_t level, const HbiNode* src, HbiNode* out) {
    out->mask = src->mask;
    out->child_count = src->child_count;
    if (level == c.depth - 1 || src->child_count == 0) {
        out->children = nullptr;
        c.nc++;
        return true;
    }
    HbiNode tmp[32];
    for (uint32_t i = 0; i < src->child_count; i++)
        deep_copy(c, level + 1, &src->children[i], &tmp[i]);
    HbiNode* block = c.arena->alloc(src->child_count);
    std::memcpy(block, tmp, src->child_count * sizeof(HbiNode));
    out->children = block;
    c.nc++;
    return true;
}

bool op_and(OpCtx& c, uint32_t level, const HbiNode* a, const HbiNode* b,
            HbiNode* out) {
    if (level == c.depth - 1) {
        uint32_t m = a->mask & b->mask;
        if (!m) return false;
        out->children = nullptr; out->mask = m; out->child_count = 0;
        c.nc++;
        return true;
    }
    uint32_t common = a->mask & b->mask;
    if (!common) return false;
    HbiNode tmp[32];
    uint32_t mask = 0, cnt = 0;
    for (uint32_t j = 0; j < c.l; j++) {
        uint32_t bitj = 1u << (c.l - 1 - j);
        if (!(common & bitj)) continue;
        const HbiNode* ca = &a->children[rank_before(a->mask, c.l, j)];
        const HbiNode* cb = &b->children[rank_before(b->mask, c.l, j)];
        if (op_and(c, level + 1, ca, cb, &tmp[cnt])) {
            mask |= bitj;
            cnt++;
        }
    }
    if (!mask) return false;
    HbiNode* block = c.arena->alloc(cnt);
    std::memcpy(block, tmp, cnt * sizeof(HbiNode));
    out->children = block; out->mask = mask; out->child_count = cnt;
    c.nc++;
    return true;
}

bool op_or(OpCtx& c, uint32_t level, const HbiNode* a, const HbiNode* b,
           HbiNode* out) {
    if (level == c.depth - 1) {
        out->children = nullptr;
        out->mask = a->mask | b->mask;
        out->child_count = 0;
        c.nc++;
        return true;
    }
    uint32_t both = a->mask | b->mask;
    HbiNode tmp[32];
    uint32_t cnt = 0;
    for (uint32_t j = 0; j < c.l; j++) {
        uint32_t bitj = 1u << (c.l - 1 - j);
        if (!(both & bitj)) continue;
        bool ina = a->mask & bitj, inb = b->mask & bitj;
        if (ina && inb) {
            op_or(c, level + 1,
                  &a->children[rank_before(a->mask, c.l, j)],
                  &b->children[rank_before(b->mask, c.l, j)], &tmp[cnt]);
        } else if (ina) {
            deep_copy(c, level + 1,
                      &a->children[rank_before(a->mask, c.l, j)], &tmp[cnt]);
        } else {
            deep_copy(c, level + 1,
                      &b->children[rank_before(b->mask, c.l, j)], &tmp[cnt]);
        }
        cnt++;
    }
    HbiNode* block = c.arena->alloc(cnt);
    std::memcpy(block, tmp, cnt * sizeof(HbiNode));
    out->children = block; out->mask = both; out->child_count = cnt;
    c.nc++;
    return true;
}

bool op_xor(OpCtx& c, uint32_t level, const HbiNode* a, const HbiNode* b,
            HbiNode* out) {
    if (level == c.depth - 1) {
        uint32_t m = a->mask ^ b->mask;
        if (!m) return false;
        out->children = nullptr; out->mask = m; out->child_count = 0;
        c.nc++;
        return true;
    }
    uint32_t any = a->mask | b->mask;
    HbiNode tmp[32];
    uint32_t mask = 0, cnt = 0;
    for (uint32_t j = 0; j < c.l; j++) {
        uint32_t bitj = 1u << (c.l - 1 - j);
        if (!(any & bitj)) continue;
        bool ina = a->mask & bitj, inb = b->mask & bitj;
        bool ok;
        if (ina && inb) {
            ok = op_xor(c, level + 1,
                        &a->children[rank_before(a->mask, c.l, j)],
                        &b->children[rank_before(b->mask, c.l, j)], &tmp[cnt]);
        } else if (ina) {
            ok = deep_copy(c, level + 1,
                           &a->children[rank_before(a->mask, c.l, j)], &tmp[cnt]);
        } else {
            ok = deep_copy(c, level + 1,
                           &b->children[rank_before(b->mask, c.l, j)], &tmp[cnt]);
        }
        if (ok) { mask |= bitj; cnt++; }
    }
    if (!mask) return false;
    HbiNode* block = c.arena->alloc(cnt);
    std::memcpy(block, tmp, cnt * sizeof(HbiNode));
    out->children = block; out->mask = mask; out->child_count = cnt;
    c.nc++;
    return true;
}

// Materialize the complement of an ABSENT (all-zero) region: a genuinely
// full tree over [start, min(start+span, nbits)).  This is the cost the
// occupancy-only representation must pay on NOT — no fill shortcuts.
bool build_full(OpCtx& c, uint32_t level, uint64_t start, HbiNode* out) {
    if (start >= c.nbits) return false;
    if (level == c.depth - 1) {
        uint64_t valid = c.nbits - start < c.l ? c.nbits - start : c.l;
        // MSB-first: the valid source positions are the TOP `valid` bits
        // (closed form; the per-bit loop here inflated NOT by ~18%)
        uint32_t m = low_mask((uint32_t)valid) << (c.l - (uint32_t)valid);
        out->children = nullptr; out->mask = m; out->child_count = 0;
        c.nc++;
        return true;
    }
    uint64_t cs = child_span(c.l, c.depth, level);
    HbiNode tmp[32];
    uint32_t mask = 0, cnt = 0;
    for (uint32_t j = 0; j < c.l; j++) {
        uint64_t s = start + (uint64_t)j * cs;
        if (s >= c.nbits) break;
        if (build_full(c, level + 1, s, &tmp[cnt])) {
            mask |= 1u << (c.l - 1 - j);
            cnt++;
        }
    }
    if (!mask) return false;
    HbiNode* block = c.arena->alloc(cnt);
    std::memcpy(block, tmp, cnt * sizeof(HbiNode));
    out->children = block; out->mask = mask; out->child_count = cnt;
    c.nc++;
    return true;
}

// NOT over the full domain: present leaves are complemented under the valid
// tail mask; absent regions materialize as full subtrees.
bool op_not(OpCtx& c, uint32_t level, uint64_t start, const HbiNode* a,
            HbiNode* out) {
    if (a == nullptr) return build_full(c, level, start, out);
    if (level == c.depth - 1) {
        uint64_t valid = c.nbits - start < c.l ? c.nbits - start : c.l;
        uint32_t vm = low_mask((uint32_t)valid) << (c.l - (uint32_t)valid);
        uint32_t m = (~a->mask) & vm;
        if (!m) return false;
        out->children = nullptr; out->mask = m; out->child_count = 0;
        c.nc++;
        return true;
    }
    uint64_t cs = child_span(c.l, c.depth, level);
    HbiNode tmp[32];
    uint32_t mask = 0, cnt = 0;
    for (uint32_t j = 0; j < c.l; j++) {
        uint64_t s = start + (uint64_t)j * cs;
        if (s >= c.nbits) break;
        uint32_t bitj = 1u << (c.l - 1 - j);
        const HbiNode* ca = (a->mask & bitj)
            ? &a->children[rank_before(a->mask, c.l, j)] : nullptr;
        if (op_not(c, level + 1, s, ca, &tmp[cnt])) {
            mask |= bitj;
            cnt++;
        }
    }
    if (!mask) return false;
    HbiNode* block = c.arena->alloc(cnt);
    std::memcpy(block, tmp, cnt * sizeof(HbiNode));
    out->children = block; out->mask = mask; out->child_count = cnt;
    c.nc++;
    return true;
}

void check_compatible(const HbiBitmap& a, const HbiBitmap& b) {
    if (a.nbits() != b.nbits() || a.node_bits() != b.node_bits())
        throw std::runtime_error("hbi: incompatible operands");
}

} // anonymous namespace

#define HBI_BINOP_BODY(OPFN)                                                  \
    check_compatible(a, b);                                                   \
    HbiBitmap r;                                                              \
    r.nbits_ = a.nbits_; r.node_bits_ = a.node_bits_; r.depth_ = a.depth_;    \
    OpCtx c{a.node_bits_, a.depth_, a.nbits_, &r.arena_, 0};                  \
    if (a.root_ && b.root_) {                                                 \
        HbiNode root;                                                         \
        if (OPFN(c, 0, a.root_, b.root_, &root)) {                            \
            HbiNode* rp = r.arena_.alloc(1);                                  \
            *rp = root;                                                       \
            r.root_ = rp;                                                     \
        }                                                                     \
    }

HbiBitmap HbiBitmap::and_(const HbiBitmap& a, const HbiBitmap& b) {
    HBI_BINOP_BODY(op_and)
    r.node_count_ = c.nc;
    return r;
}

HbiBitmap HbiBitmap::or_(const HbiBitmap& a, const HbiBitmap& b) {
    check_compatible(a, b);
    HbiBitmap r;
    r.nbits_ = a.nbits_; r.node_bits_ = a.node_bits_; r.depth_ = a.depth_;
    OpCtx c{a.node_bits_, a.depth_, a.nbits_, &r.arena_, 0};
    HbiNode root;
    bool ok = false;
    if (a.root_ && b.root_)      ok = op_or(c, 0, a.root_, b.root_, &root);
    else if (a.root_)            ok = deep_copy(c, 0, a.root_, &root);
    else if (b.root_)            ok = deep_copy(c, 0, b.root_, &root);
    if (ok) {
        HbiNode* rp = r.arena_.alloc(1);
        *rp = root;
        r.root_ = rp;
    }
    r.node_count_ = c.nc;
    return r;
}

HbiBitmap HbiBitmap::xor_(const HbiBitmap& a, const HbiBitmap& b) {
    check_compatible(a, b);
    HbiBitmap r;
    r.nbits_ = a.nbits_; r.node_bits_ = a.node_bits_; r.depth_ = a.depth_;
    OpCtx c{a.node_bits_, a.depth_, a.nbits_, &r.arena_, 0};
    HbiNode root;
    bool ok = false;
    if (a.root_ && b.root_)      ok = op_xor(c, 0, a.root_, b.root_, &root);
    else if (a.root_)            ok = deep_copy(c, 0, a.root_, &root);
    else if (b.root_)            ok = deep_copy(c, 0, b.root_, &root);
    if (ok) {
        HbiNode* rp = r.arena_.alloc(1);
        *rp = root;
        r.root_ = rp;
    }
    r.node_count_ = c.nc;
    return r;
}

HbiBitmap HbiBitmap::not_(const HbiBitmap& a) {
    HbiBitmap r;
    r.nbits_ = a.nbits_; r.node_bits_ = a.node_bits_; r.depth_ = a.depth_;
    OpCtx c{a.node_bits_, a.depth_, a.nbits_, &r.arena_, 0};
    HbiNode root;
    if (op_not(c, 0, 0, a.root_, &root)) {
        HbiNode* rp = r.arena_.alloc(1);
        *rp = root;
        r.root_ = rp;
    }
    r.node_count_ = c.nc;
    return r;
}

HbiBitmap HbiBitmap::comp(const HbiBitmap& b1, const HbiBitmap& b2,
                          const HbiBitmap& b3) {
    HbiBitmap t1 = or_(b1, b2);      // stays HBI
    HbiBitmap t2 = or_(b2, b3);      // stays HBI
    HbiBitmap t3 = and_(t1, t2);     // stays HBI
    return not_(t3);
}

// ------------------------------------------------------------- delivery ----
namespace {
// Reverse the low `l` bits (mask is MSB-first per the paper; the dense image
// is LSB-first).  Leaves are l-aligned, so a leaf lands in one aligned l-bit
// lane of the dense image and can be written word-level instead of bit by bit.
static inline uint32_t rev_bits(uint32_t v, uint32_t l) {
    v = ((v & 0x55555555u) << 1) | ((v >> 1) & 0x55555555u);
    v = ((v & 0x33333333u) << 2) | ((v >> 2) & 0x33333333u);
    v = ((v & 0x0F0F0F0Fu) << 4) | ((v >> 4) & 0x0F0F0F0Fu);
    v = ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    v = (v << 16) | (v >> 16);
    return v >> (32 - l);
}

void scatter(const HbiNode* n, uint32_t level, uint64_t start, uint32_t l,
             uint32_t depth, uint64_t nbits, uint64_t* words) {
    if (level == depth - 1) {
        uint64_t lane = rev_bits(n->mask, l);      // LSB-first l-bit payload
        words[start >> 6] |= lane << (start & 63); // start is l-aligned, l<=32
        return;
    }
    uint64_t cs = child_span(l, depth, level);
    uint32_t r = 0;
    for (uint32_t j = 0; j < l; j++) {
        if (n->mask & (1u << (l - 1 - j))) {
            scatter(&n->children[r], level + 1, start + (uint64_t)j * cs,
                    l, depth, nbits, words);
            r++;
        }
    }
}

uint64_t pop_rec(const HbiNode* n, uint32_t level, uint32_t depth) {
    if (level == depth - 1) return __builtin_popcount(n->mask);
    uint64_t t = 0;
    for (uint32_t i = 0; i < n->child_count; i++)
        t += pop_rec(&n->children[i], level + 1, depth);
    return t;
}
} // anonymous namespace

void HbiBitmap::to_dense(uint64_t* words, size_t num_words) const {
    std::memset(words, 0, num_words * sizeof(uint64_t));
    if (root_) scatter(root_, 0, 0, node_bits_, depth_, nbits_, words);
}

uint64_t HbiBitmap::popcount() const {
    return root_ ? pop_rec(root_, 0, depth_) : 0;
}

uint64_t HbiBitmap::checksum_dense() const {
    size_t nw = (nbits_ + 63) / 64;
    std::vector<uint64_t> w(nw);
    to_dense(w.data(), nw);
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : w) { h ^= x; h *= 1099511628211ull; }
    return h;
}

uint64_t HbiBitmap::serialized_bytes() const {
    return 28 + (node_count_ * node_bits_ + 7) / 8;   // header(4+8+4+4+8) + packed masks
}

// -------------------------------------------------------- serialization ----
// Layout: magic "HBI1", nbits u64, node_bits u32, depth u32, node_count u64,
// then BFS level-order masks packed to node_bits each.
namespace {
struct BitWriter {
    std::vector<uint8_t> buf;
    uint64_t acc = 0; int nacc = 0;
    void put(uint32_t v, int bits) {
        acc |= (uint64_t)v << nacc; nacc += bits;
        while (nacc >= 8) { buf.push_back(acc & 0xFF); acc >>= 8; nacc -= 8; }
    }
    void flush() { if (nacc) { buf.push_back(acc & 0xFF); acc = 0; nacc = 0; } }
};
struct BitReader {
    const uint8_t* p; uint64_t acc = 0; int nacc = 0;
    explicit BitReader(const uint8_t* q) : p(q) {}
    uint32_t get(int bits) {
        while (nacc < bits) { acc |= (uint64_t)(*p++) << nacc; nacc += 8; }
        uint32_t v = acc & ((bits == 32) ? 0xFFFFFFFFull : ((1ull << bits) - 1));
        acc >>= bits; nacc -= bits;
        return v;
    }
};
} // anonymous namespace

void HbiBitmap::serialize(const std::string& path) const {
    BitWriter w;
    // BFS
    std::vector<const HbiNode*> cur, nxt;
    if (root_) cur.push_back(root_);
    while (!cur.empty()) {
        for (const HbiNode* n : cur) {
            w.put(n->mask, (int)node_bits_);
            for (uint32_t i = 0; i < n->child_count; i++)
                nxt.push_back(&n->children[i]);
        }
        cur.swap(nxt);
        nxt.clear();
    }
    w.flush();
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("hbi: cannot write " + path);
    out.write("HBI1", 4);
    uint32_t nb = node_bits_, dp = depth_;
    out.write(reinterpret_cast<const char*>(&nbits_), 8);
    out.write(reinterpret_cast<const char*>(&nb), 4);
    out.write(reinterpret_cast<const char*>(&dp), 4);
    out.write(reinterpret_cast<const char*>(&node_count_), 8);
    out.write(reinterpret_cast<const char*>(w.buf.data()), w.buf.size());
}

HbiBitmap HbiBitmap::deserialize(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("hbi: cannot open " + path);
    char magic[4];
    in.read(magic, 4);
    if (std::memcmp(magic, "HBI1", 4) != 0)
        throw std::runtime_error("hbi: bad magic in " + path);
    HbiBitmap bm;
    uint32_t nb = 0, dp = 0;
    in.read(reinterpret_cast<char*>(&bm.nbits_), 8);
    in.read(reinterpret_cast<char*>(&nb), 4);
    in.read(reinterpret_cast<char*>(&dp), 4);
    in.read(reinterpret_cast<char*>(&bm.node_count_), 8);
    bm.node_bits_ = nb;
    bm.depth_ = dp;
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
    if (bm.node_count_ == 0) return bm;
    BitReader r(raw.data());
    // BFS reconstruction: read masks level by level; a node at level < depth-1
    // owns popcount(mask) children in the NEXT level's stream.
    std::vector<HbiNode*> cur, nxt;
    bm.root_ = bm.arena_.alloc(1);
    bm.root_->mask = r.get((int)nb);
    cur.push_back(bm.root_);
    for (uint32_t level = 0; level + 1 < dp && !cur.empty(); level++) {
        // allocate each parent's child block, then fill masks in BFS order
        for (HbiNode* n : cur) {
            uint32_t cc = __builtin_popcount(n->mask);
            n->child_count = cc;
            n->children = bm.arena_.alloc(cc);
            for (uint32_t i = 0; i < cc; i++) nxt.push_back(&n->children[i]);
        }
        for (HbiNode* child : nxt) child->mask = r.get((int)nb);
        cur.swap(nxt);
        nxt.clear();
    }
    for (HbiNode* leaf : cur) { leaf->children = nullptr; leaf->child_count = 0; }
    return bm;
}

std::vector<std::string> HbiBitmap::node_strings() const {
    std::vector<std::string> out;
    std::vector<const HbiNode*> cur, nxt;
    if (root_) cur.push_back(root_);
    while (!cur.empty()) {
        for (const HbiNode* n : cur) {
            std::string s;
            for (uint32_t k = 0; k < node_bits_; k++)
                s += (n->mask & (1u << (node_bits_ - 1 - k))) ? '1' : '0';
            out.push_back(s);
            for (uint32_t i = 0; i < n->child_count; i++)
                nxt.push_back(&n->children[i]);
        }
        cur.swap(nxt);
        nxt.clear();
    }
    return out;
}

} // namespace hbi
