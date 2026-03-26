#pragma once

#include "../../bitmap_backend.h"

class BitsetAVX512Backend : public IBitmapBackend {
public:
    std::unique_ptr<BitmapHandle> Create() override;

    void Append(BitmapHandle& btv, bool bit) override;

    std::unique_ptr<BitmapHandle> bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range = 2) override;
    std::unique_ptr<BitmapHandle> bitAnd(const BitmapHandle& a, const BitmapHandle& b) override;
    std::unique_ptr<BitmapHandle> bitXor(const BitmapHandle& a, const BitmapHandle& b) override;

    void Serialize(const BitmapHandle& btv, const std::string& path) override;
    std::unique_ptr<BitmapHandle> Load(const std::string& path) override;

    uint64_t Cardinality(const BitmapHandle& btv) override;
    std::vector<uint32_t> Decode(const BitmapHandle& btv) override;
};
