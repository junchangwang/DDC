#include "ewah_backend.h"
#include <fstream>
#include <algorithm>

static EwahHandle& getHandle(BitmapHandle& handle) {
    return static_cast<EwahHandle&>(handle);
}

static const EwahHandle& getHandle(const BitmapHandle& handle) {
    return static_cast<const EwahHandle&>(handle);
}

std::unique_ptr<BitmapHandle> EwahBackend::Create() {
    return std::make_unique<EwahHandle>();
}

void EwahBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    if (bit) {
        h.btv.set(h.current_bits);
    }
    h.current_bits++;
}

uint64_t EwahBackend::Cardinality(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    return h.btv.numberOfOnes();
}

std::vector<uint32_t> EwahBackend::Decode(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    auto positions = h.btv.toArray();
    std::vector<uint32_t> result;
    result.reserve(positions.size());
    for (auto pos : positions) {
        result.push_back(static_cast<uint32_t>(pos));
    }
    return result;
}

std::unique_ptr<BitmapHandle> EwahBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<EwahHandle>();
    ha.btv.logicalor(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> EwahBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<EwahHandle>();
    ha.btv.logicaland(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

std::unique_ptr<BitmapHandle> EwahBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<EwahHandle>();
    ha.btv.logicalxor(hb.btv, res->btv);
    res->current_bits = std::max(ha.current_bits, hb.current_bits);
    return res;
}

void EwahBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    // Write current_bits first, then EWAH compressed data
    out.write(reinterpret_cast<const char*>(&h.current_bits), sizeof(h.current_bits));
    h.btv.write(out);
}

std::unique_ptr<BitmapHandle> EwahBackend::Load(const std::string& path) {
    auto res = std::make_unique<EwahHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;
    in.read(reinterpret_cast<char*>(&res->current_bits), sizeof(res->current_bits));
    res->btv.read(in);
    return res;
}
