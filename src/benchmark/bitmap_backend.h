#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

// backend kinds
enum class BackendKind { WAH, CROARING, DDC, CONCISE };

struct BitmapHandle { virtual ~BitmapHandle() = default; };

class IBitmapBackend {
public:
    virtual ~IBitmapBackend() = default;
    virtual std::unique_ptr<BitmapHandle> Create() = 0;
    virtual void Append(BitmapHandle& btv, bool bit) = 0;
    // boolean kernels
    virtual std::unique_ptr<BitmapHandle> bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range = 2) = 0;
    virtual std::unique_ptr<BitmapHandle> bitAnd(const BitmapHandle& a, const BitmapHandle& b) = 0;
    virtual std::unique_ptr<BitmapHandle> bitXor(const BitmapHandle& a, const BitmapHandle& b) = 0;
    // disk I/O
    virtual void Serialize(const BitmapHandle& btv, const std::string& path) = 0;
    virtual std::unique_ptr<BitmapHandle> Load(const std::string& path) = 0;

    virtual uint64_t Cardinality(const BitmapHandle& btv) = 0;

    virtual std::vector<uint32_t> Decode(const BitmapHandle& btv) = 0;
};