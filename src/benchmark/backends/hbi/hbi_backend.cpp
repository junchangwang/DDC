#include "hbi_backend.h"

#include <cstring>
#include <stdexcept>

void HbiHandle::ensure_built() {
    if (!needs_build) return;
    const uint64_t nbits = appended_rows;
    std::vector<uint64_t> dense((nbits + 63) / 64, 0);
    for (uint32_t p : raw_positions)
        dense[p >> 6] |= 1ull << (p & 63);
    bm = hbi::HbiBitmap::build_from_dense(dense.data(), nbits, build_node_bits);
    needs_build = false;
}

static const HbiHandle& H(const BitmapHandle& h) {
    auto& hh = const_cast<HbiHandle&>(static_cast<const HbiHandle&>(h));
    hh.ensure_built();
    return static_cast<const HbiHandle&>(h);
}

std::unique_ptr<BitmapHandle> HbiBackend::Create() {
    auto h = std::make_unique<HbiHandle>();
    h->needs_build = true;
    h->build_node_bits = node_bits_;
    return h;
}

void HbiBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = static_cast<HbiHandle&>(handle);
    if (bit) h.raw_positions.push_back(static_cast<uint32_t>(h.appended_rows));
    h.appended_rows++;
    h.needs_build = true;
}

std::unique_ptr<BitmapHandle> HbiBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto res = std::make_unique<HbiHandle>();
    res->bm = hbi::HbiBitmap::and_(H(a).bm, H(b).bm);
    return res;
}

std::unique_ptr<BitmapHandle> HbiBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t) {
    auto res = std::make_unique<HbiHandle>();
    res->bm = hbi::HbiBitmap::or_(H(a).bm, H(b).bm);
    return res;
}

std::unique_ptr<BitmapHandle> HbiBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto res = std::make_unique<HbiHandle>();
    res->bm = hbi::HbiBitmap::xor_(H(a).bm, H(b).bm);
    return res;
}

void HbiBackend::Serialize(const BitmapHandle& btv, const std::string& path) {
    H(btv).bm.serialize(path);
}

std::unique_ptr<BitmapHandle> HbiBackend::Load(const std::string& path) {
    auto h = std::make_unique<HbiHandle>();
    h->bm = hbi::HbiBitmap::deserialize(path);
    return h;
}

uint64_t HbiBackend::Cardinality(const BitmapHandle& btv) {
    return H(btv).bm.popcount();
}

std::vector<uint32_t> HbiBackend::Decode(const BitmapHandle& btv) {
    const hbi::HbiBitmap& bm = H(btv).bm;
    size_t nw = (bm.nbits() + 63) / 64;
    std::vector<uint64_t> dense(nw);
    bm.to_dense(dense.data(), nw);
    std::vector<uint32_t> out;
    out.reserve(bm.popcount());
    for (size_t w = 0; w < nw; w++) {
        uint64_t x = dense[w];
        while (x) {
            out.push_back(static_cast<uint32_t>((w << 6) + __builtin_ctzll(x)));
            x &= x - 1;
        }
    }
    return out;
}
