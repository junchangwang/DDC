#include "concise_backend.h"
#include <fstream>
#include <algorithm>

static ConciseHandle& getHandle(BitmapHandle& handle) {
    return static_cast<ConciseHandle&>(handle);
}

static const ConciseHandle& getHandle(const BitmapHandle& handle) {
    return static_cast<const ConciseHandle&>(handle);
}

std::unique_ptr<BitmapHandle> ConciseBackend::Create() {
    return std::make_unique<ConciseHandle>();
}

void ConciseBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    if (bit) {
        h.btv.add(static_cast<uint32_t>(h.current_bits));
    }
    h.current_bits++;
}

uint64_t ConciseBackend::Cardinality(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    return h.btv.size();
}

std::vector<uint32_t> ConciseBackend::Decode(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    std::vector<uint32_t> result;
    for (auto it = h.btv.begin(); it != h.btv.end(); ++it) {
        result.push_back(*it);
    }
    return result;
}

std::unique_ptr<BitmapHandle> ConciseBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<ConciseHandle>();
    ha.btv.logicalorToContainer(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> ConciseBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<ConciseHandle>();
    ha.btv.logicalandToContainer(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> ConciseBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<ConciseHandle>();
    ha.btv.logicalxorToContainer(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

void ConciseBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    // Write current_bits, then last, lastWordIndex, then the words vector
    out.write(reinterpret_cast<const char*>(&h.current_bits), sizeof(h.current_bits));
    out.write(reinterpret_cast<const char*>(&h.btv.last), sizeof(h.btv.last));
    out.write(reinterpret_cast<const char*>(&h.btv.lastWordIndex), sizeof(h.btv.lastWordIndex));
    if (h.btv.lastWordIndex >= 0) {
        uint32_t count = static_cast<uint32_t>(h.btv.lastWordIndex + 1);
        out.write(reinterpret_cast<const char*>(h.btv.words.data()), count * sizeof(uint32_t));
    }
}

std::unique_ptr<BitmapHandle> ConciseBackend::Load(const std::string& path) {
    auto res = std::make_unique<ConciseHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;
    in.read(reinterpret_cast<char*>(&res->current_bits), sizeof(res->current_bits));
    in.read(reinterpret_cast<char*>(&res->btv.last), sizeof(res->btv.last));
    in.read(reinterpret_cast<char*>(&res->btv.lastWordIndex), sizeof(res->btv.lastWordIndex));
    if (res->btv.lastWordIndex >= 0) {
        uint32_t count = static_cast<uint32_t>(res->btv.lastWordIndex + 1);
        res->btv.words.resize(count);
        in.read(reinterpret_cast<char*>(res->btv.words.data()), count * sizeof(uint32_t));
    }
    return res;
}
