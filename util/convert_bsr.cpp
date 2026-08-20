// Convert an existing bm_<n>_c<c>_roaring/ directory into a matching
// bm_<n>_c<c>_bsr/ directory so the QFilter/BSR backend benchmarks the
// exact same sets as every other scheme (datasets are not reproducible:
// gen_dataset.sh is unseeded, so re-generation would change the data).
//
// Usage: convert_bsr <roaring_dir> <bsr_dir> [num_rows]

#include "croaring/roaring.hh"
#include "../src/benchmark/backends/bsr/bsr_backend.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: convert_bsr <roaring_dir> <bsr_dir> [num_rows]\n";
        return 1;
    }
    fs::path src_dir = argv[1], dst_dir = argv[2];
    uint64_t num_rows_arg = (argc >= 4) ? std::stoull(argv[3]) : 0;

    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(src_dir))
        if (e.path().extension() == ".bm") files.push_back(e.path());
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });
    if (files.empty()) { std::cerr << "no .bm files in " << src_dir << "\n"; return 1; }
    fs::create_directories(dst_dir);

    BsrBackend backend;
    uint64_t total_bytes = 0, total_chunks = 0;
    for (auto& f : files) {
        // roaring .bm layout: u32 current_size | roaring bytes
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        std::streamsize fsz = in.tellg();
        in.seekg(0);
        uint32_t current_size = 0;
        in.read(reinterpret_cast<char*>(&current_size), sizeof(current_size));
        std::vector<char> buf(static_cast<size_t>(fsz) - sizeof(current_size));
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        roaring::Roaring r = roaring::Roaring::readSafe(buf.data(), buf.size());

        std::vector<uint32_t> pos(r.cardinality());
        r.toUint32Array(pos.data());   // sorted ascending

        BsrHandle h;
        uint64_t rows = num_rows_arg ? num_rows_arg : current_size;
        bsr_util::encode_from_positions(pos, rows, h);

        fs::path out = dst_dir / f.filename();
        backend.Serialize(h, out.string());
        total_bytes += h.storage_bytes();
        total_chunks += static_cast<uint64_t>(h.size);
    }
    std::cout << "[convert_bsr] " << files.size() << " bitmaps -> " << dst_dir
              << "  chunks=" << total_chunks
              << "  payload=" << total_bytes / (1024.0*1024.0) << " MiB\n";
    return 0;
}
