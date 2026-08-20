// Rebuild a bm_*_ddc_w8 dir from the matching roaring dir so the DDC sets
// are bit-identical to roaring/wah/ewah/concise/bsr (fixes the c=2000
// data-identity break found in the 2026-07-16 audit: that ddc dir was
// regenerated from a different unseeded random draw on May 25).
// Usage: convert_ddc_from_roaring <roaring_dir> <ddc_dir> <num_rows>
#include "croaring/roaring.hh"
#include <ddc.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
namespace fs = std::filesystem;
int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: <roaring_dir> <ddc_dir> <num_rows>\n"; return 1; }
    fs::path src = argv[1], dst = argv[2];
    uint64_t num_rows = std::stoull(argv[3]);
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(src))
        if (e.path().extension() == ".bm") files.push_back(e.path());
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string()); });
    fs::create_directories(dst);
    int done = 0;
    for (auto& f : files) {
        std::ifstream in(f, std::ios::binary | std::ios::ate);
        std::streamsize fsz = in.tellg(); in.seekg(0);
        uint32_t cs = 0; in.read(reinterpret_cast<char*>(&cs), 4);
        std::vector<char> buf(static_cast<size_t>(fsz) - 4);
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        roaring::Roaring r = roaring::Roaring::readSafe(buf.data(), buf.size());
        std::vector<uint32_t> pos(r.cardinality());
        r.toUint32Array(pos.data());
        std::vector<bool> bits(num_rows, false);
        for (uint32_t p : pos) if (p < num_rows) bits[p] = true;
        DDC cb = DDC::compress(bits, false);
        std::ofstream out(dst / f.filename(), std::ios::binary);
        cb.serialize(out);
        if (++done % 200 == 0) std::cerr << done << "/" << files.size() << "\n";
    }
    std::cout << "[convert_ddc] " << files.size() << " bitmaps -> " << dst << "\n";
    return 0;
}
