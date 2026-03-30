#include "combit_backend.h"
#include <fstream>
#include <algorithm>

// Append-mode handle: accumulates bits, compresses lazily on first use.
struct CombitAppendHandle : public CombitHandle {
    std::vector<bool> raw_bits;
    bool needs_compress = true;

    void ensure_compressed() {
        if (needs_compress) {
            compressed = ComBit::compress<8>(raw_bits);
            needs_compress = false;
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
    return std::make_unique<CombitAppendHandle>();
}

void CombitBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = static_cast<CombitAppendHandle&>(handle);
    h.raw_bits.push_back(bit);
}

uint64_t CombitBackend::Cardinality(const BitmapHandle& handle) {
    // If it's an append handle, ensure compressed first
    if (auto* ah = dynamic_cast<CombitAppendHandle*>(const_cast<BitmapHandle*>(&handle))) {
        ah->ensure_compressed();
    }
    return getHandle(handle).compressed.popcount();
}

std::vector<uint32_t> CombitBackend::Decode(const BitmapHandle& handle) {
    if (auto* ah = dynamic_cast<CombitAppendHandle*>(const_cast<BitmapHandle*>(&handle))) {
        ah->ensure_compressed();
    }
    auto positions = getHandle(handle).compressed.set_bit_positions();
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
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed | getHandle(b).compressed;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed & getHandle(b).compressed;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed ^ getHandle(b).compressed;
    return res;
}

// ==============================================================
// Serialization: store total bits + set-bit positions
// ==============================================================

void CombitBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    if (auto* ah = dynamic_cast<CombitAppendHandle*>(const_cast<BitmapHandle*>(&handle))) {
        ah->ensure_compressed();
    }
    const auto& h = getHandle(handle);
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

    std::vector<uint32_t> indices;
    if (size > 0) {
        indices.resize(size);
        in.read(reinterpret_cast<char*>(indices.data()), size * sizeof(uint32_t));
    }

    // Build vector<bool> and compress
    std::vector<bool> bits(total_bits, false);
    for (auto idx : indices) {
        if (idx < total_bits)
            bits[idx] = true;
    }
    res->compressed = ComBit::compress<8>(bits);
    return res;
}