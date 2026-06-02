

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ddc.h"
#include "fastbit/bitvector.h"
#include "roaring.hh"
#include "ewah/ewah.h"

namespace fs = std::filesystem;

static std::vector<uint8_t> read_raw(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    auto sz = in.tellg();
    in.seekg(0);
    std::vector<uint8_t> buf(sz);
    in.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static std::vector<uint8_t> read_from_croaring(const std::string& path,
                                               size_t num_rows) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return {};
    std::streamsize fsize = in.tellg();
    in.seekg(0, std::ios::beg);
    uint32_t logical = 0;
    in.read(reinterpret_cast<char*>(&logical), sizeof(logical));
    std::streamsize body = fsize - static_cast<std::streamsize>(sizeof(logical));
    std::vector<uint8_t> bits((num_rows + 7) / 8, 0);
    if (body <= 0) return bits;
    std::vector<char> buf(body);
    in.read(buf.data(), body);
    roaring::Roaring r = roaring::Roaring::readSafe(buf.data(), body);
    std::vector<uint32_t> pos(r.cardinality());
    r.toUint32Array(pos.data());
    for (uint32_t p : pos) {
        if (p < num_rows) bits[p / 8] |= uint8_t(1) << (p % 8);
    }
    return bits;
}

static void write_ddc(const std::string& out, const std::vector<uint8_t>& raw,
                         size_t n) {
    std::vector<bool> bits(n, false);
    for (size_t i = 0; i < n; ++i) bits[i] = (raw[i / 8] >> (i % 8)) & 1;
    auto cb = DDC::compress(bits);
    fs::create_directories(fs::path(out).parent_path());
    std::ofstream o(out, std::ios::binary);
    cb.serialize(o);
}

static void write_wah(const std::string& out, const std::vector<uint8_t>& raw,
                      size_t n) {
    ibis::bitvector bv;

    for (size_t i = 0; i < n; ++i)
        bv += int((raw[i / 8] >> (i % 8)) & 1);
    fs::create_directories(fs::path(out).parent_path());
    bv.write(out.c_str());
}

static void write_croaring(const std::string& out, const std::vector<uint8_t>& raw,
                           size_t n) {
    roaring::Roaring r;
    for (size_t i = 0; i < n; ++i)
        if ((raw[i / 8] >> (i % 8)) & 1) r.add(static_cast<uint32_t>(i));
    r.runOptimize();
    r.shrinkToFit();
    fs::create_directories(fs::path(out).parent_path());
    std::ofstream o(out, std::ios::binary);
    uint32_t logical = static_cast<uint32_t>(n);
    o.write(reinterpret_cast<const char*>(&logical), sizeof(logical));
    size_t need = r.getSizeInBytes();
    std::vector<char> buf(need);
    r.write(buf.data());
    o.write(buf.data(), need);
}

static void write_ewah(const std::string& out, const std::vector<uint8_t>& raw,
                       size_t n) {
    ewah::EWAHBoolArray<uint64_t> bv;
    for (size_t i = 0; i < n; ++i)
        if ((raw[i / 8] >> (i % 8)) & 1) bv.set(i);
    fs::create_directories(fs::path(out).parent_path());
    std::ofstream o(out, std::ios::binary);
    uint64_t header = n;
    o.write(reinterpret_cast<const char*>(&header), sizeof(header));
    bv.write(o);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: compress_any <input_dir> <output_base> <num_rows>\n";
        return 1;
    }
    std::string in_dir   = argv[1];
    std::string out_base = argv[2];
    size_t num_rows      = std::stoull(argv[3]);
    bool ddc_only     = false;
    bool from_croaring   = false;
    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--ddc-only")  ddc_only   = true;
        else if (a == "--from-croaring") from_croaring = true;
    }

    std::string cb_dir   = out_base + "_ddc";
    std::string wah_dir  = out_base + "_wah";
    std::string cr_dir   = out_base + "_croaring";
    std::string ew_dir   = out_base + "_ewah";

    auto t0 = std::chrono::high_resolution_clock::now();
    int count = 0;
    for (auto& col : fs::directory_iterator(in_dir)) {
        if (!col.is_directory()) continue;
        std::string col_name = col.path().filename().string();
        for (auto& f : fs::directory_iterator(col.path())) {
            if (f.path().extension() != ".bm") continue;
            std::string fname = f.path().filename().string();
            std::string rel   = col_name + "/" + fname;
            std::vector<uint8_t> raw = from_croaring
                ? read_from_croaring(f.path().string(), num_rows)
                : read_raw(f.path().string());
            if (raw.empty()) continue;
            write_ddc  (cb_dir  + "/" + rel, raw, num_rows);
            if (!ddc_only) {
                write_wah     (wah_dir + "/" + rel, raw, num_rows);
                write_croaring(cr_dir  + "/" + rel, raw, num_rows);
                write_ewah    (ew_dir  + "/" + rel, raw, num_rows);
            }
            ++count;
            std::cout << "[" << count << "] " << rel << std::endl;
        }
    }

    std::ofstream(cb_dir + "/done.txt") << "num_rows=" << num_rows << "\n";

    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::high_resolution_clock::now() - t0).count();
    std::cout << "\n[done] " << count << " bitmaps in " << dt << " ms\n";
    return 0;
}
