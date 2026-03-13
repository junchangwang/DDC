#include "combit_backend.h"
#include <bitmap_vector.hpp>
#include <fstream>
#include <algorithm>

// Inherit from the unified handle, wrapping the custom BitmapVector
struct CombitHandle : public BitmapHandle {
    combit::BitmapVector btv;
    uint64_t current_bits = 0; // Tracks the current number of appended bits
};

// Helper functions to safely cast the base handle to CombitHandle
inline CombitHandle& getHandle(BitmapHandle& h) {
    return static_cast<CombitHandle&>(h);
}
inline const CombitHandle& getHandle(const BitmapHandle& h) {
    return static_cast<const CombitHandle&>(h);
}

std::unique_ptr<BitmapHandle> CombitBackend::Create() {
    return std::make_unique<CombitHandle>();
}

void CombitBackend::Append(BitmapHandle& handle, bool bit) {
    auto& h = getHandle(handle);
    // The algorithm only records the positions of 1s. 
    // If the bit is 1, set it; otherwise, just increment the counter.
    if (bit) {
        h.btv.set_bit(h.current_bits);
    }
    h.current_bits++;
}

uint64_t CombitBackend::Cardinality(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    // Directly delegates to the native popcount method
    return h.btv.popcount(); 
}

std::vector<uint32_t> CombitBackend::Decode(const BitmapHandle& handle) {
    auto& h = getHandle(handle);
    std::vector<uint32_t> result;
    // Iterate and extract all indices where the bit is 1
    for (uint64_t i = 0; i < h.current_bits; ++i) {
        if (h.btv.get_bit(i)) {
            result.push_back(i);
        }
    }
    return result;
}

// ==============================================================
// Logical Operations (Temporary Simulation):
// Note: Since native OR/AND/XOR operations are not provided yet, 
// we simulate them via bit-by-bit traversal to pass the benchmark.
// Replace these with native bitwise operators when implemented.
// ==============================================================

std::unique_ptr<BitmapHandle> CombitBackend::bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<CombitHandle>();
    
    uint64_t max_bits = std::max(ha.current_bits, hb.current_bits);
    for (uint64_t i = 0; i < max_bits; ++i) {
        bool bit_a = (i < ha.current_bits) ? ha.btv.get_bit(i) : false;
        bool bit_b = (i < hb.current_bits) ? hb.btv.get_bit(i) : false;
        
        if (bit_a || bit_b) {
            res->btv.set_bit(i);
        }
    }
    res->current_bits = max_bits;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitAnd(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<CombitHandle>();
    
    uint64_t min_bits = std::min(ha.current_bits, hb.current_bits);
    for (uint64_t i = 0; i < min_bits; ++i) {
        if (ha.btv.get_bit(i) && hb.btv.get_bit(i)) {
            res->btv.set_bit(i);
        }
    }
    res->current_bits = min_bits;
    return res;
}

std::unique_ptr<BitmapHandle> CombitBackend::bitXor(const BitmapHandle& a, const BitmapHandle& b) {
    auto& ha = getHandle(a);
    auto& hb = getHandle(b);
    auto res = std::make_unique<CombitHandle>();
    
    uint64_t max_bits = std::max(ha.current_bits, hb.current_bits);
    for (uint64_t i = 0; i < max_bits; ++i) {
        bool bit_a = (i < ha.current_bits) ? ha.btv.get_bit(i) : false;
        bool bit_b = (i < hb.current_bits) ? hb.btv.get_bit(i) : false;
        
        if (bit_a != bit_b) {
            res->btv.set_bit(i);
        }
    }
    res->current_bits = max_bits;
    return res;
}

// ==============================================================
// Serialization & Deserialization:
// We store the total length + the array of indices for 1s 
// to ensure general compatibility and correct reconstruction.
// ==============================================================

void CombitBackend::Serialize(const BitmapHandle& handle, const std::string& path) {
    auto& h = getHandle(handle);
    std::ofstream out(path, std::ios::binary);
    if (!out) return;
    
    // Write the total number of bits
    out.write(reinterpret_cast<const char*>(&h.current_bits), sizeof(h.current_bits));
    
    // Reuse Decode to get all indices of 1s and write them
    auto indices = Decode(handle);
    size_t size = indices.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
        out.write(reinterpret_cast<const char*>(indices.data()), size * sizeof(uint32_t));
    }
}

std::unique_ptr<BitmapHandle> CombitBackend::Load(const std::string& path) {
    auto res = std::make_unique<CombitHandle>();
    std::ifstream in(path, std::ios::binary);
    if (!in) return res;

    // Read the total number of bits
    in.read(reinterpret_cast<char*>(&res->current_bits), sizeof(res->current_bits));
    
    // Read the number of indices
    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    // Read the specific indices and reconstruct the bitmap
    if (size > 0) {
        std::vector<uint32_t> indices(size);
        in.read(reinterpret_cast<char*>(indices.data()), size * sizeof(uint32_t));
        for (auto idx : indices) {
            res->btv.set_bit(idx);
        }
    }
    return res;
}