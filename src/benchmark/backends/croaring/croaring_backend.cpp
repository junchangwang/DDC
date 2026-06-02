#include "croaring_backend.h"
#include <fstream>
#include <algorithm>
#include <vector>

static CroaringHandle& getHandle(BitmapHandle& handle) {
    return static_cast<CroaringHandle&>(handle);
}

static const CroaringHandle& getHandle(const BitmapHandle& handle) {
    return static_cast<const CroaringHandle&>(handle);
}

std::unique_ptr<BitmapHandle> CroaringBackend::Create() {
    return std::make_unique<CroaringHandle>();
}

void CroaringBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    // if bit is 1, we add the current logical size as a new element in the roaring bitmap
    if (bit) {
        h.bitmap.add(h.current_size);
    }
    // whatever the bit is, we need to increase the logical size by 1, because we have appended one more bit (even if it's 0)
    h.current_size++;
}

// OR kernel
std::unique_ptr<BitmapHandle> CroaringBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto result = std::make_unique<CroaringHandle>();
    const auto& ha = getHandle(a);
    const auto& hb = getHandle(b);
    
    result->bitmap = ha.bitmap | hb.bitmap; // CRoaring
    result->current_size = std::max(ha.current_size, hb.current_size);
    return result;
}

// AND kernel
std::unique_ptr<BitmapHandle> CroaringBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto result = std::make_unique<CroaringHandle>();
    const auto& ha = getHandle(a);
    const auto& hb = getHandle(b);
    
    result->bitmap = ha.bitmap & hb.bitmap;
    result->current_size = std::max(ha.current_size, hb.current_size);
    return result;
}

// XOR kernel
std::unique_ptr<BitmapHandle> CroaringBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto result = std::make_unique<CroaringHandle>();
    const auto& ha = getHandle(a);
    const auto& hb = getHandle(b);
    
    result->bitmap = ha.bitmap ^ hb.bitmap;
    result->current_size = std::max(ha.current_size, hb.current_size);
    return result;
}

void CroaringBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    const auto& h = getHandle(handle);
    
    // serialize bitmap
    size_t expected_size = h.bitmap.getSizeInBytes();
    std::vector<char> buffer(expected_size);
    h.bitmap.write(buffer.data());

    // add current_size at the beginning of the file, followed by the bitmap data
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&h.current_size), sizeof(h.current_size));
    out.write(buffer.data(), expected_size);
}

std::unique_ptr<BitmapHandle> CroaringBackend::Load(const std::string& path) {
    auto result = std::make_unique<CroaringHandle>();
    
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) throw std::runtime_error("Failed to open file for loading CRoaring.");
    
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    
    // read size header then bitmap
    in.read(reinterpret_cast<char*>(&result->current_size), sizeof(result->current_size));
    
    std::streamsize bitmap_size = size - sizeof(result->current_size);
    if (bitmap_size > 0) {
        std::vector<char> buffer(bitmap_size);
        in.read(buffer.data(), bitmap_size);
        result->bitmap = roaring::Roaring::readSafe(buffer.data(), bitmap_size);
    }
    
    return result;
}

uint64_t CroaringBackend::Cardinality(const BitmapHandle& handle) {
    return getHandle(handle).bitmap.cardinality();
}

std::vector<uint32_t> CroaringBackend::Decode(const BitmapHandle& handle) {
    const auto& h = getHandle(handle);
    // decode to positions
    std::vector<uint32_t> result(h.bitmap.cardinality());
    h.bitmap.toUint32Array(result.data());
    return result;
}