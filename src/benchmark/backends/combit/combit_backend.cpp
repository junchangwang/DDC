#include "combit_backend.h"
#include <fstream>
#include <algorithm>

// Append-mode handle: accumulates bits, compresses lazily on first use.
struct CombitAppendHandle : public CombitHandle {
    std::vector<bool> raw_bits;
    bool needs_compress = true;

    void ensure_compressed() {
        if (needs_compress) {
            compressed = ComBit::compress(raw_bits);
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

// Ensure compression is done before accessing .compressed field
static void ensureCompressed(const BitmapHandle& h) {
    if (auto* ah = dynamic_cast<CombitAppendHandle*>(const_cast<BitmapHandle*>(&h))) {
        ah->ensure_compressed();
    }
}

std::unique_ptr<BitmapHandle> CombitBackend::Create() {
    return std::make_unique<CombitAppendHandle>();
}

void CombitBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = static_cast<CombitAppendHandle&>(handle);
    h.raw_bits.push_back(bit);
}

uint64_t CombitBackend::Cardinality(const BitmapHandle& handle) {
    ensureCompressed(handle);
    return getHandle(handle).compressed.popcount();
}

std::vector<uint32_t> CombitBackend::Decode(const BitmapHandle& handle) {
    ensureCompressed(handle);
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
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed | getHandle(b).compressed;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed & getHandle(b).compressed;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<CombitHandle>();
    res->compressed = getHandle(a).compressed ^ getHandle(b).compressed;
    return res;
}

// ==============================================================
// Serialization: store total bits + set-bit positions
// ==============================================================

void CombitBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    ensureCompressed(handle);
    const auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    h.compressed.serialize(out);
}

std::unique_ptr<BitmapHandle> CombitBackend::Load(const std::string& path) {
    auto res = std::make_unique<CombitHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;
    res->compressed = ComBit::deserialize(in);
    return res;
}