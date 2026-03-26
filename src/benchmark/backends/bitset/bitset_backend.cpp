#include "bitset_backend.h"
#include <bitset_vector.hpp>
#include <fstream>
#include <algorithm>

struct BitsetHandle : public BitmapHandle {
    bitset::BitsetVector btv;
    uint64_t current_bits = 0;
};

inline BitsetHandle& getHandle(BitmapHandle& h) {
    return static_cast<BitsetHandle&>(h);
}
inline const BitsetHandle& getHandle(const BitmapHandle& h) {
    return static_cast<const BitsetHandle&>(h);
}

std::unique_ptr<BitmapHandle> BitsetBackend::Create() {
    return std::make_unique<BitsetHandle>();
}

void BitsetBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    if (bit) {
        h.btv.set_bit(h.current_bits);
    }
    h.current_bits++;
}

uint64_t BitsetBackend::Cardinality(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    return h.btv.popcount(false); // plain scalar
}

std::vector<uint32_t> BitsetBackend::Decode(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    std::vector<uint32_t> result = h.btv.decode_positions();
    while (!result.empty() && result.back() >= h.current_bits) {
        result.pop_back();
    }
    return result;
}

std::unique_ptr<BitmapHandle> BitsetBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetHandle>();
    res->btv = bitset::BitsetVector::word_or(ha.btv, hb.btv, false);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> BitsetBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetHandle>();
    res->btv = bitset::BitsetVector::word_and(ha.btv, hb.btv, false);
    res->current_bits = std::min(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> BitsetBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetHandle>();
    res->btv = bitset::BitsetVector::word_xor(ha.btv, hb.btv, false);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

void BitsetBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    size_t packed_bytes = (h.current_bits + 7) / 8;
    // Write raw packed bytes (same format as gen_bitmap)
    out.write(reinterpret_cast<const char*>(h.btv.words().data()), packed_bytes);
}

std::unique_ptr<BitmapHandle> BitsetBackend::Load(const std::string& path) {
    auto res = std::make_unique<BitsetHandle>();
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return res;
    size_t file_size = in.tellg();
    in.seekg(0);
    res->current_bits = file_size * 8;
    res->btv.set_num_bits(res->current_bits);
    size_t num_words = (file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    res->btv.words_mut().resize(num_words, 0);
    in.read(reinterpret_cast<char*>(res->btv.words_mut().data()), file_size);
    return res;
}
