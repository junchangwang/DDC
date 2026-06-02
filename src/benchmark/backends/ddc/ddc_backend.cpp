#include "ddc_backend.h"
#include <fstream>
#include <algorithm>

struct DDCAppendHandle : public DDCHandle {
    std::vector<bool> raw_bits;
    bool needs_compress = true;

    void ensure_compressed() {  // lazy compress
        if (needs_compress) {
            compressed = DDC::compress(raw_bits);
            needs_compress = false;
        }
    }
};

inline DDCHandle& getHandle(BitmapHandle& h) {
    return static_cast<DDCHandle&>(h);
}
inline const DDCHandle& getHandle(const BitmapHandle& h) {
    return static_cast<const DDCHandle&>(h);
}

static void ensureCompressed(const BitmapHandle& h) {
    if (auto* ah = dynamic_cast<DDCAppendHandle*>(const_cast<BitmapHandle*>(&h))) {
        ah->ensure_compressed();
    }
}

std::unique_ptr<BitmapHandle> DDCBackend::Create() {
    return std::make_unique<DDCAppendHandle>();
}

void DDCBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = static_cast<DDCAppendHandle&>(handle);
    h.raw_bits.push_back(bit);  // buffer raw
}

uint64_t DDCBackend::Cardinality(const BitmapHandle& handle) {
    ensureCompressed(handle);
    return getHandle(handle).compressed.popcount();
}

std::vector<uint32_t> DDCBackend::Decode(const BitmapHandle& handle) {
    ensureCompressed(handle);
    // expand positions
    auto positions = getHandle(handle).compressed.set_bit_positions();
    std::vector<uint32_t> result;
    result.reserve(positions.size());
    for (auto p : positions)
        result.push_back(static_cast<uint32_t>(p));
    return result;
}

std::unique_ptr<BitmapHandle> DDCBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<DDCHandle>();
    res->compressed = getHandle(a).compressed | getHandle(b).compressed;  // OR kernel
    return res;
}

std::unique_ptr<BitmapHandle> DDCBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<DDCHandle>();
    res->compressed = getHandle(a).compressed & getHandle(b).compressed;  // AND kernel
    return res;
}

std::unique_ptr<BitmapHandle> DDCBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    ensureCompressed(a);
    ensureCompressed(b);
    auto res = std::make_unique<DDCHandle>();
    res->compressed = getHandle(a).compressed ^ getHandle(b).compressed;
    return res;
}

void DDCBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    ensureCompressed(handle);
    const auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    h.compressed.serialize(out);
}

// deserialize from disk
std::unique_ptr<BitmapHandle> DDCBackend::Load(const std::string& path) {
    auto res = std::make_unique<DDCHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;
    res->compressed = DDC::deserialize(in);
    return res;
}