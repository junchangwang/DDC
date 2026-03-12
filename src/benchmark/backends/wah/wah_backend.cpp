#include "wah_backend.h"
#include <stdexcept>
#include <vector>
// assist functions to convert between BitmapHandle and FastBit's bitvector
static ibis::bitvector& getBtv(BitmapHandle& handle) {
    return static_cast<WahHandle&>(handle).btv;
}

static const ibis::bitvector& getBtv(const BitmapHandle& handle) {
    return static_cast<const WahHandle&>(handle).btv;
}

std::unique_ptr<BitmapHandle> WahBackend::Create() {
    return std::make_unique<WahHandle>(); // build a new WahHandle, which contains an empty FastBit bitvector
}

void WahBackend::Append(BitmapHandle& handle, bool bit) {
    // FastBit ibis::bitvector
    getBtv(handle) += (bit ? 1 : 0);
}

std::unique_ptr<BitmapHandle> WahBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto result = std::make_unique<WahHandle>();
    result->btv = getBtv(a);
    result->btv |= getBtv(b);
    return result;
}

std::unique_ptr<BitmapHandle> WahBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto result = std::make_unique<WahHandle>();
    result->btv = getBtv(a);
    result->btv &= getBtv(b);
    return result;
}

std::unique_ptr<BitmapHandle> WahBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto result = std::make_unique<WahHandle>();
    result->btv = getBtv(a);
    result->btv ^= getBtv(b);
    return result;
}

void WahBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    // FastBit
    getBtv(handle).write(path.c_str());
}

std::unique_ptr<BitmapHandle> WahBackend::Load(const std::string& path) {
    auto result = std::make_unique<WahHandle>();
    result->btv.read(path.c_str());
    return result;
}
uint64_t WahBackend::Cardinality(const BitmapHandle& handle) {
    return getBtv(handle).cnt();
}

std::vector<uint32_t> WahBackend::Decode(const BitmapHandle& handle) {
    std::vector<uint32_t> result;
    const auto& btv = getBtv(handle);
    ibis::bitvector::pit it(btv);
    while (*it != 0xFFFFFFFFU) {
        result.push_back(*it);
        it.next();
    }
    return result;
}