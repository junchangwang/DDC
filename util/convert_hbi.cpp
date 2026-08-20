// convert_hbi: bit-exact conversion of the frozen synthetic datasets into
// HBI-Pointer serialized files.  Reads the existing BSR .bm files (themselves
// bit-exact conversions of the roaring dirs; the raw datasets are unseeded
// and can never be regenerated), expands to dense, builds the HBI tree and
// serializes it.  Output goes to a SEPARATE data root (bitmap_hbi/) so the
// frozen bitmap/ tree is never touched.
//
// Per-file validation (all four required by the 2026-07-22 spec):
//   1. total bit length matches num_rows;
//   2. set-bit count: HBI popcount == dense popcount;
//   3. dense checksum: FNV over HBI's dense image == FNV over source dense;
//   4. tail validity: no set bit at or beyond num_rows after round-trip.
//
// Usage: convert_hbi <bsr_dir> <out_dir> <num_rows> <node_bits>
#include "../src/core/hbi/hbi.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct BsrFile {
    uint64_t num_rows = 0;
    std::vector<int> bases, states;
};

static BsrFile load_bsr(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    BsrFile f;
    int size = 0;
    in.read(reinterpret_cast<char*>(&f.num_rows), 8);
    in.read(reinterpret_cast<char*>(&size), 4);
    f.bases.resize(size);
    f.states.resize(size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(f.bases.data()), 4l * size);
        in.read(reinterpret_cast<char*>(f.states.data()), 4l * size);
    }
    if (!in) throw std::runtime_error("truncated " + path);
    return f;
}

static uint64_t fnv(const std::vector<uint64_t>& w) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t x : w) { h ^= x; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
            "usage: %s <bsr_dir> <out_dir> <num_rows> <node_bits>\n", argv[0]);
        return 1;
    }
    const std::string src_dir = argv[1], out_dir = argv[2];
    const uint64_t num_rows = std::stoull(argv[3]);
    const uint32_t node_bits = (uint32_t)std::stoul(argv[4]);
    fs::create_directories(out_dir);

    const size_t nw = (num_rows + 63) / 64;
    std::vector<uint64_t> dense(nw), back(nw);
    uint64_t files = 0, total_nodes = 0, total_ser = 0;

    std::vector<fs::path> inputs;
    for (auto& e : fs::directory_iterator(src_dir))
        if (e.path().extension() == ".bm") inputs.push_back(e.path());
    std::sort(inputs.begin(), inputs.end());

    for (const auto& p : inputs) {
        BsrFile f = load_bsr(p.string());
        if (f.num_rows != num_rows)
            throw std::runtime_error("num_rows mismatch in " + p.string());
        std::memset(dense.data(), 0, nw * 8);
        uint64_t src_pop = 0;
        for (size_t i = 0; i < f.bases.size(); i++) {
            uint32_t st = (uint32_t)f.states[i];
            uint64_t base_bit = (uint64_t)f.bases[i] * 32ull;
            src_pop += __builtin_popcount(st);
            size_t w = base_bit >> 6;
            int sh = (base_bit & 63);
            dense[w] |= (uint64_t)st << sh;   // 32-bit chunk, sh is 0 or 32
        }
        const uint64_t src_sum = fnv(dense);

        hbi::HbiBitmap bm =
            hbi::HbiBitmap::build_from_dense(dense.data(), num_rows, node_bits);

        // validations 1-4
        if (bm.nbits() != num_rows)
            throw std::runtime_error("V1 bit-length failed: " + p.string());
        if (bm.popcount() != src_pop)
            throw std::runtime_error("V2 popcount failed: " + p.string());
        bm.to_dense(back.data(), nw);
        if (fnv(back) != src_sum)
            throw std::runtime_error("V3 checksum failed: " + p.string());
        uint32_t tail = num_rows & 63;
        if (tail && (back[nw - 1] >> tail) != 0)
            throw std::runtime_error("V4 tail failed: " + p.string());

        const std::string out = (fs::path(out_dir) / p.filename()).string();
        bm.serialize(out);
        files++;
        total_nodes += bm.node_count();
        total_ser += bm.serialized_bytes();
    }
    std::printf("%s -> %s: %llu files, node_bits=%u, depth=%u, "
                "total_nodes=%llu, total_serialized=%.4f MiB (all 4 checks passed)\n",
                src_dir.c_str(), out_dir.c_str(), (unsigned long long)files,
                node_bits, hbi::compute_depth(num_rows, node_bits),
                (unsigned long long)total_nodes, total_ser / 1048576.0);
    return 0;
}
