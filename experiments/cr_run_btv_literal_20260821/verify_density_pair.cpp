#include "benchmark/bitmap_backend.h"
#include "benchmark/backends/Concise/concise_backend.h"
#include "benchmark/backends/bitset_avx512/bitset_avx512_backend.h"
#include "benchmark/backends/croaring/croaring_backend.h"
#include "benchmark/backends/ddc/ddc_backend.h"
#include "benchmark/backends/ewah/ewah_backend.h"
#include "benchmark/backends/wah/wah_backend.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

struct BackendSpec {
    std::string name;
    std::string suffix;
    std::function<std::unique_ptr<IBitmapBackend>()> make;
};

struct Snapshot {
    uint64_t cardinality = 0;
    uint64_t hash = 0;
    std::vector<uint32_t> positions;
};

uint64_t fnv1a_positions(const std::vector<uint32_t>& positions) {
    uint64_t hash = kFnvOffset;
    for (uint32_t position : positions) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(position >> shift);
            hash *= kFnvPrime;
        }
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

void validate_positions(const std::vector<uint32_t>& positions,
                        uint64_t rows,
                        const std::string& label) {
    if (!std::is_sorted(positions.begin(), positions.end())) {
        throw std::runtime_error(label + ": decoded positions are not sorted");
    }
    auto duplicate = std::adjacent_find(positions.begin(), positions.end());
    if (duplicate != positions.end()) {
        throw std::runtime_error(label + ": duplicate decoded position " +
                                 std::to_string(*duplicate));
    }
    if (!positions.empty() && positions.back() >= rows) {
        throw std::runtime_error(label + ": decoded position outside row range: " +
                                 std::to_string(positions.back()));
    }
}

Snapshot load_snapshot(const BackendSpec& spec,
                       const fs::path& root,
                       const std::string& case_name,
                       const std::string& artifact,
                       uint64_t rows) {
    const fs::path dir = root / ("bm_100m_" + case_name + "_" + spec.suffix);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("missing backend directory: " + dir.string());
    }

    auto backend = spec.make();
    std::unique_ptr<BitmapHandle> result;
    if (artifact == "A") {
        result = backend->Load((dir / "1.bm").string());
    } else if (artifact == "B") {
        result = backend->Load((dir / "2.bm").string());
    } else if (artifact == "OR") {
        auto a = backend->Load((dir / "1.bm").string());
        auto b = backend->Load((dir / "2.bm").string());
        result = backend->bitOr(*a, *b);
    } else {
        throw std::runtime_error("unknown artifact: " + artifact);
    }

    Snapshot snapshot;
    snapshot.cardinality = backend->Cardinality(*result);
    snapshot.positions = backend->Decode(*result);
    validate_positions(snapshot.positions, rows, spec.name + "/" + artifact);
    if (snapshot.cardinality != snapshot.positions.size()) {
        throw std::runtime_error(spec.name + "/" + artifact +
                                 ": cardinality differs from decoded count");
    }
    snapshot.hash = fnv1a_positions(snapshot.positions);
    return snapshot;
}

std::string first_difference(const std::vector<uint32_t>& expected,
                             const std::vector<uint32_t>& actual) {
    const size_t shared = std::min(expected.size(), actual.size());
    for (size_t i = 0; i < shared; ++i) {
        if (expected[i] != actual[i]) {
            return "index " + std::to_string(i) + ": expected " +
                   std::to_string(expected[i]) + ", got " +
                   std::to_string(actual[i]);
        }
    }
    if (expected.size() != actual.size()) {
        return "decoded lengths differ: expected " +
               std::to_string(expected.size()) + ", got " +
               std::to_string(actual.size());
    }
    return "none";
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --root GRIDINDEP_BITMAP_DIR --case A66_B66"
                 " [--rows 100000000]\n";
}

}  // namespace

int main(int argc, char** argv) {
    fs::path root;
    std::string case_name;
    uint64_t rows = 100000000;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--root" && i + 1 < argc) {
            root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            case_name = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            rows = std::stoull(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    if (root.empty() || case_name.empty() || !fs::is_directory(root)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::vector<BackendSpec> backends = {
        {"BitsetAVX", "bitset", [] { return std::make_unique<BitsetAVX512Backend>(); }},
        {"DDC", "ddc_w8", [] { return std::make_unique<DDCBackend>(); }},
        {"CRoaring", "roaring", [] { return std::make_unique<CroaringBackend>(); }},
        {"WAH", "wah", [] { return std::make_unique<WahBackend>(); }},
        {"EWAH", "ewah", [] { return std::make_unique<EwahBackend>(); }},
        {"Concise", "concise", [] { return std::make_unique<ConciseBackend>(); }},
    };

    try {
        std::cout << "case=" << case_name << " rows=" << rows
                  << " reference=BitsetAVX\n";
        std::cout << "artifact,backend,cardinality,fnv1a64,exact\n";

        for (const std::string artifact : {"A", "B", "OR"}) {
            Snapshot reference =
                load_snapshot(backends.front(), root, case_name, artifact, rows);
            std::cout << artifact << ',' << backends.front().name << ','
                      << reference.cardinality << ',' << hex64(reference.hash)
                      << ",REFERENCE\n";

            for (size_t i = 1; i < backends.size(); ++i) {
                Snapshot candidate =
                    load_snapshot(backends[i], root, case_name, artifact, rows);
                const bool exact = candidate.cardinality == reference.cardinality &&
                                   candidate.hash == reference.hash &&
                                   candidate.positions == reference.positions;
                std::cout << artifact << ',' << backends[i].name << ','
                          << candidate.cardinality << ',' << hex64(candidate.hash)
                          << ',' << (exact ? "PASS" : "FAIL") << '\n';
                if (!exact) {
                    std::cerr << "Mismatch " << backends[i].name << '/' << artifact
                              << ": "
                              << first_difference(reference.positions,
                                                  candidate.positions)
                              << '\n';
                    return 1;
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "Verifier error: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ALL_EXACT_MATCH\n";
    return 0;
}
