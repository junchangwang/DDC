#include "combit_backend.h"
#include <combit.h>
#include <fstream>
#include <algorithm>

// Handle wrapping ComBit (segmented compressed bitvector).
// Bits are accumulated during Append, then lazily compressed on first use.
struct CombitHandle : public BitmapHandle {
    std::vector<bool> raw_bits;       // accumulated during Append
    ComBit compressed;                // lazily built from raw_bits
    bool is_compressed = false;       // true once compressed from raw_bits
    bool from_operation = false;      // true when result of bitwise op

    void ensure_compressed() {
        if (!is_compressed && !from_operation) {
            compressed = ComBit::compress<8>(raw_bits);
            is_compressed = true;
        }
    }
};

inline CombitHandle& getHandle(BitmapHandle& h) {
    return static_cast<CombitHandle&>(h);
}
inline const CombitHandle& getHandle(const BitmapHandle& h) {
    return static_cast<const CombitHandle&>(h);
}

std::unique_ptr<BitmapHandle> CombitBackend::Create() {
    return std::make_unique<CombitHandle>();
}

void CombitBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    h.raw_bits.push_back(bit);
}

uint64_t CombitBackend::Cardinality(const BitmapHandle& handle) {
    auto& h = const_cast<CombitHandle&>(getHandle(handle));
    h.ensure_compressed();
    return h.compressed.popcount();
}

std::vector<uint32_t> CombitBackend::Decode(const BitmapHandle& handle) {
    auto& h = const_cast<CombitHandle&>(getHandle(handle));
    h.ensure_compressed();
    auto positions = h.compressed.set_bit_positions();
    std::vector<uint32_t> result;
    result.reserve(positions.size());
    for (auto p : positions)
        result.push_back(static_cast<uint32_t>(p));
    return result;
}

// ==============================================================
// Logical Operations using ComBit operator overloads
// ==============================================================

std::unique_ptr<BitmapHandle> CombitBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = const_cast<CombitHandle&>(getHandle(a));
    auto& hb = const_cast<CombitHandle&>(getHandle(b));
    ha.ensure_compressed();
    hb.ensure_compressed();

    auto res = std::make_unique<CombitHandle>();
    res->compressed = ha.compressed | hb.compressed;
    res->from_operation = true;
    res->is_compressed = true;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = const_cast<CombitHandle&>(getHandle(a));
    auto& hb = const_cast<CombitHandle&>(getHandle(b));
    ha.ensure_compressed();
    hb.ensure_compressed();

    auto res = std::make_unique<CombitHandle>();
    res->compressed = ha.compressed & hb.compressed;
    res->from_operation = true;
    res->is_compressed = true;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = const_cast<CombitHandle&>(getHandle(a));
    auto& hb = const_cast<CombitHandle&>(getHandle(b));
    ha.ensure_compressed();
    hb.ensure_compressed();

    auto res = std::make_unique<CombitHandle>();
    res->compressed = ha.compressed ^ hb.compressed;
    res->from_operation = true;
    res->is_compressed = true;
    return res;
}

// ==============================================================
// Serialization: store total bits + set-bit positions
// ==============================================================

void CombitBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = const_cast<CombitHandle&>(getHandle(handle));
    h.ensure_compressed();
    std::ofstream out(path, std::ios::binary);
    if (!out) return;

    uint64_t total_bits = h.compressed.bit_count();
    out.write(reinterpret_cast<const char*>(&total_bits), sizeof(total_bits));

    auto indices = Decode(handle);
    size_t size = indices.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(indices.data()), size * sizeof(uint32_t));
    }
}

std::unique_ptr<BitmapHandle> CombitBackend::Load(const std::string& path) {
    auto res = std::make_unique<CombitHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;

    uint64_t total_bits = 0;
    in.read(reinterpret_cast<char*>(&total_bits), sizeof(total_bits));

    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Reconstruct via raw_bits, then compress
    res->raw_bits.resize(total_bits, false);
    if (size > 0) {
        std::vector<uint32_t> indices(size);
        in.read(reinterpret_cast<char*>(indices.data()), size * sizeof(uint32_t));
        for (auto idx : indices) {
            if (idx < total_bits)
                res->raw_bits[idx] = true;
        }
    }
    res->ensure_compressed();
    return res;
}