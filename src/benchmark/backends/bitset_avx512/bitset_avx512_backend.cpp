#include "bitset_avx512_backend.h"
#include <fstream>
#include <algorithm>

inline BitsetAVX512Handle& getHandle(BitmapHandle& h) {
    return static_cast<BitsetAVX512Handle&>(h);
}
inline const BitsetAVX512Handle& getHandle(const BitmapHandle& h) {
    return static_cast<const BitsetAVX512Handle&>(h);
}

std::unique_ptr<BitmapHandle> BitsetAVX512Backend::Create() {
    return std::make_unique<BitsetAVX512Handle>();
}

void BitsetAVX512Backend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    if (bit) {
        h.btv.set_bit(h.current_bits);
    }
    h.current_bits++;
}

uint64_t BitsetAVX512Backend::Cardinality(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    return h.btv.popcount(true);
}

std::vector<uint32_t> BitsetAVX512Backend::Decode(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    std::vector<uint32_t> result = h.btv.decode_positions();
    // trim padding
    while (!result.empty() && result.back() >= h.current_bits) {
        result.pop_back();
    }
    return result;
}

std::unique_ptr<BitmapHandle> BitsetAVX512Backend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetAVX512Handle>();
    // OR kernel
    res->btv = bitset::BitsetVector::word_or(ha.btv, hb.btv, true);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> BitsetAVX512Backend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetAVX512Handle>();
    // AND kernel
    res->btv = bitset::BitsetVector::word_and(ha.btv, hb.btv, true);
    res->current_bits = std::min(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> BitsetAVX512Backend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<BitsetAVX512Handle>();
    // XOR kernel
    res->btv = bitset::BitsetVector::word_xor(ha.btv, hb.btv, true);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

void BitsetAVX512Backend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    size_t packed_bytes = (h.current_bits + 7) / 8;
    // dump raw words
    out.write(reinterpret_cast<const char*>(h.btv.words()), packed_bytes);
}

std::unique_ptr<BitmapHandle> BitsetAVX512Backend::Load(const std::string& path) {
    auto res = std::make_unique<BitsetAVX512Handle>();
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return res;
    size_t file_size = in.tellg();
    in.seekg(0);
    res->current_bits = file_size * 8;
    res->btv.set_num_bits(res->current_bits);
    size_t num_words = (file_size + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    res->btv.allocate(num_words);
    in.read(reinterpret_cast<char*>(res->btv.words_mut()), file_size);
    return res;
}
