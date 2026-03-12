#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

enum class BackendKind { WAH, CROARING, COMBIT };

struct BitmapHandle { virtual ~BitmapHandle() = default; };

class IBitmapBackend {
public:
    virtual ~IBitmapBackend() = default;
    virtual std::unique_ptr<BitmapHandle> Create() = 0;
    virtual void Append(BitmapHandle& btv, bool bit) = 0;
    virtual std::unique_ptr<BitmapHandle> bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range = 2) = 0;
    virtual std::unique_ptr<BitmapHandle> bitAnd(const BitmapHandle& a, const BitmapHandle& b) = 0;
    virtual std::unique_ptr<BitmapHandle> bitXor(const BitmapHandle& a, const BitmapHandle& b) = 0;
    virtual void Serialize(const BitmapHandle& btv, const std::string& path) = 0;
    virtual std::unique_ptr<BitmapHandle> Load(const std::string& path) = 0;
    
    // how many bits are set to 1
    virtual uint64_t Cardinality(const BitmapHandle& btv) = 0;
    // let's say the bitmap logically represents bits at positions 0, 2, 3 are 1, then this should return a vector {0, 2, 3}
    virtual std::vector<uint32_t> Decode(const BitmapHandle& btv) = 0;
};