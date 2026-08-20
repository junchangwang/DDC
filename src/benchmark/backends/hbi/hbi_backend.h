#pragma once

#include "../../bitmap_backend.h"
#include "hbi.h"

#include <cstdint>
#include <vector>

// HBI-Pointer backend (hbi branch): node-oriented hierarchical bitmap after
// Morzy et al. (ADBIS 2003), extended with a compressed-to-compressed Boolean
// algebra for the R1.W1/R3-W2 rebuttal experiments.  See src/core/hbi/hbi.h
// for the faithful-but-fair baseline rules (real pointer tree, per-parent
// contiguous child blocks, mask+rank+pointer navigation, NOT materializes
// elided zero regions, results never alias inputs).
struct HbiHandle : public BitmapHandle {
    hbi::HbiBitmap bm;

    // append-mode buffer (raw .bm build path)
    std::vector<uint32_t> raw_positions;
    uint64_t appended_rows = 0;
    bool needs_build = false;
    uint32_t build_node_bits = 32;

    void ensure_built();
    uint64_t storage_bytes() const { return bm.serialized_bytes(); }
};

class HbiBackend : public IBitmapBackend {
public:
    explicit HbiBackend(uint32_t node_bits = 32) : node_bits_(node_bits) {}
    std::unique_ptr<BitmapHandle> Create() override;
    void Append(BitmapHandle& btv, bool bit) override;
    std::unique_ptr<BitmapHandle> bitOr(const BitmapHandle& a, const BitmapHandle& b, uint32_t range = 2) override;
    std::unique_ptr<BitmapHandle> bitAnd(const BitmapHandle& a, const BitmapHandle& b) override;
    std::unique_ptr<BitmapHandle> bitXor(const BitmapHandle& a, const BitmapHandle& b) override;
    void Serialize(const BitmapHandle& btv, const std::string& path) override;
    std::unique_ptr<BitmapHandle> Load(const std::string& path) override;
    uint64_t Cardinality(const BitmapHandle& btv) override;
    std::vector<uint32_t> Decode(const BitmapHandle& btv) override;

private:
    uint32_t node_bits_;   // width used by the raw Append/build path
};
