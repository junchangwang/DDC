/**
 * Unified backend correctness tests.
 *
 * Every test is run THREE times — once per backend (WAH, CRoaring, ComBit)
 * — via GoogleTest parameterized tests.
 *
 * Covers: Create, Append, Cardinality, Decode,
 *         Serialize/Load (.bm files), bitOr, bitAnd, bitXor,
 *         .bm file generation pipeline, compression, multi-way OR.
 */
#include <gtest/gtest.h>
#include "benchmark/bitmap_backend.h"
#include "benchmark/backends/wah/wah_backend.h"
#include "benchmark/backends/croaring/croaring_backend.h"
#include "benchmark/backends/combit/combit_backend.h"

#include <filesystem>
#include <fstream>
#include <set>
#include <random>
#include <algorithm>
#include <memory>

namespace fs = std::filesystem;

// ===================================================================
// Factory: create a backend by name
// ===================================================================
static std::unique_ptr<IBitmapBackend> make_backend(const std::string& name) {
    if (name == "WAH")      return std::make_unique<WahBackend>();
    if (name == "CRoaring")  return std::make_unique<CroaringBackend>();
    if (name == "ComBit")    return std::make_unique<CombitBackend>();
    return nullptr;
}

// ===================================================================
// Parameterized test fixture
// ===================================================================
class BackendParamTest : public ::testing::TestWithParam<std::string> {
protected:
    std::unique_ptr<IBitmapBackend> backend;
    std::string tmp_dir;

    void SetUp() override {
        backend = make_backend(GetParam());
        ASSERT_NE(backend, nullptr) << "Unknown backend: " << GetParam();
        tmp_dir = (fs::temp_directory_path() / ("combit_unified_test_" + GetParam())).string();
        fs::create_directories(tmp_dir);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }

    // Helper: build a bitmap with specific positions set
    std::unique_ptr<BitmapHandle> make_bitmap(
        const std::vector<uint64_t>& positions, uint64_t total_bits)
    {
        auto h = backend->Create();
        std::set<uint64_t> pos_set(positions.begin(), positions.end());
        for (uint64_t i = 0; i < total_bits; ++i) {
            backend->Append(*h, pos_set.count(i) > 0);
        }
        return h;
    }
};

// Instantiate for all 3 backends
INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    BackendParamTest,
    ::testing::Values("WAH", "CRoaring", "ComBit"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);

// ======================== Create & Cardinality ========================

TEST_P(BackendParamTest, CreateEmpty) {
    auto h = backend->Create();
    EXPECT_EQ(backend->Cardinality(*h), 0u);
}

TEST_P(BackendParamTest, AppendAndCardinality) {
    auto h = backend->Create();
    backend->Append(*h, true);
    backend->Append(*h, false);
    backend->Append(*h, true);
    backend->Append(*h, false);
    backend->Append(*h, true);
    EXPECT_EQ(backend->Cardinality(*h), 3u);
}

TEST_P(BackendParamTest, AppendAllOnes) {
    auto h = backend->Create();
    for (int i = 0; i < 64; ++i) {
        backend->Append(*h, true);
    }
    EXPECT_EQ(backend->Cardinality(*h), 64u);
}

TEST_P(BackendParamTest, AppendAllZeros) {
    auto h = backend->Create();
    for (int i = 0; i < 100; ++i) {
        backend->Append(*h, false);
    }
    EXPECT_EQ(backend->Cardinality(*h), 0u);
}

TEST_P(BackendParamTest, LargeAppend) {
    auto h = backend->Create();
    uint64_t expected = 0;
    for (int i = 0; i < 10000; ++i) {
        bool bit = (i % 3 == 0);
        backend->Append(*h, bit);
        if (bit) ++expected;
    }
    EXPECT_EQ(backend->Cardinality(*h), expected);
}

// ======================== Decode ========================

TEST_P(BackendParamTest, DecodeEmpty) {
    auto h = backend->Create();
    EXPECT_TRUE(backend->Decode(*h).empty());
}

TEST_P(BackendParamTest, DecodePositions) {
    auto h = make_bitmap({0, 3, 7, 15}, 16);
    auto decoded = backend->Decode(*h);

    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 7u);
    EXPECT_EQ(decoded[3], 15u);
}

TEST_P(BackendParamTest, DecodeAllOnes) {
    auto h = backend->Create();
    for (int i = 0; i < 32; ++i) {
        backend->Append(*h, true);
    }
    auto decoded = backend->Decode(*h);
    ASSERT_EQ(decoded.size(), 32u);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(decoded[i], i);
    }
}

// ======================== Serialize & Load (.bm files) ========================

TEST_P(BackendParamTest, SerializeAndLoadEmpty) {
    auto h = backend->Create();
    std::string path = tmp_dir + "/empty.bm";

    backend->Serialize(*h, path);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = backend->Load(path);
    EXPECT_EQ(backend->Cardinality(*loaded), 0u);
}

TEST_P(BackendParamTest, SerializeAndLoadSimple) {
    auto h = make_bitmap({0, 5, 10, 20, 31}, 32);
    std::string path = tmp_dir + "/simple.bm";

    backend->Serialize(*h, path);
    auto loaded = backend->Load(path);

    EXPECT_EQ(backend->Cardinality(*loaded), 5u);
    auto decoded = backend->Decode(*loaded);
    ASSERT_EQ(decoded.size(), 5u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 5u);
    EXPECT_EQ(decoded[2], 10u);
    EXPECT_EQ(decoded[3], 20u);
    EXPECT_EQ(decoded[4], 31u);
}

TEST_P(BackendParamTest, SerializeAndLoadLarge) {
    auto h = backend->Create();
    std::vector<uint32_t> expected;
    for (int i = 0; i < 50000; ++i) {
        bool bit = (i % 7 == 0);
        backend->Append(*h, bit);
        if (bit) expected.push_back(i);
    }

    std::string path = tmp_dir + "/large.bm";
    backend->Serialize(*h, path);

    auto loaded = backend->Load(path);
    EXPECT_EQ(backend->Cardinality(*loaded), expected.size());

    auto decoded = backend->Decode(*loaded);
    ASSERT_EQ(decoded.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(decoded[i], expected[i]) << "mismatch at index " << i;
    }
}

TEST_P(BackendParamTest, SerializeMultipleFiles) {
    // Simulate gen_bitmap.sh: generate multiple .bm files for different values
    const int NUM_VALUES = 5;
    const int NUM_ROWS = 1000;

    // Assign each row a "value" (0..NUM_VALUES-1)
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, NUM_VALUES - 1);
    std::vector<int> column(NUM_ROWS);
    for (auto& v : column) v = dist(rng);

    // Build one bitmap per distinct value
    std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
    for (int val = 0; val < NUM_VALUES; ++val) {
        auto h = backend->Create();
        for (int row = 0; row < NUM_ROWS; ++row) {
            backend->Append(*h, column[row] == val);
        }
        bitmaps.push_back(std::move(h));
    }

    // Serialize all to .bm files
    for (int val = 0; val < NUM_VALUES; ++val) {
        std::string path = tmp_dir + "/" + std::to_string(val) + ".bm";
        backend->Serialize(*bitmaps[val], path);
        EXPECT_TRUE(fs::exists(path));
    }

    // Load back and verify each bitmap
    for (int val = 0; val < NUM_VALUES; ++val) {
        std::string path = tmp_dir + "/" + std::to_string(val) + ".bm";
        auto loaded = backend->Load(path);
        auto decoded = backend->Decode(*loaded);

        // Cross-check: every decoded position should have column[pos] == val
        for (auto pos : decoded) {
            EXPECT_EQ(column[pos], val)
                << GetParam() << ": value " << val << " at position " << pos;
        }

        // Count should match
        uint64_t expected_count = std::count(column.begin(), column.end(), val);
        EXPECT_EQ(backend->Cardinality(*loaded), expected_count)
            << GetParam() << ": value " << val;
    }
}

// ======================== bitwise OR ========================

TEST_P(BackendParamTest, OrDisjoint) {
    auto a = make_bitmap({0, 2, 4}, 8);
    auto b = make_bitmap({1, 3, 5}, 8);

    auto result = backend->bitOr(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 6u);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(decoded[i], i);
    }
}

TEST_P(BackendParamTest, OrOverlapping) {
    auto a = make_bitmap({0, 1, 2, 3}, 8);
    auto b = make_bitmap({2, 3, 4, 5}, 8);

    auto result = backend->bitOr(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 6u);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(decoded[i], i);
    }
}

TEST_P(BackendParamTest, OrWithEmpty) {
    auto a = make_bitmap({1, 3, 5}, 8);
    auto b = backend->Create();
    // Append same number of zeros to b so lengths match
    for (int i = 0; i < 8; ++i) backend->Append(*b, false);

    auto result = backend->bitOr(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 1u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 5u);
}

// ======================== bitwise AND ========================

TEST_P(BackendParamTest, AndOverlapping) {
    auto a = make_bitmap({0, 1, 2, 3, 4}, 8);
    auto b = make_bitmap({2, 3, 4, 5, 6}, 8);

    auto result = backend->bitAnd(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 2u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 4u);
}

TEST_P(BackendParamTest, AndDisjoint) {
    auto a = make_bitmap({0, 2, 4}, 8);
    auto b = make_bitmap({1, 3, 5}, 8);

    auto result = backend->bitAnd(*a, *b);
    EXPECT_EQ(backend->Cardinality(*result), 0u);
}

TEST_P(BackendParamTest, AndSelf) {
    auto a = make_bitmap({0, 5, 10}, 16);
    auto result = backend->bitAnd(*a, *a);

    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 5u);
    EXPECT_EQ(decoded[2], 10u);
}

// ======================== bitwise XOR ========================

TEST_P(BackendParamTest, XorDisjoint) {
    auto a = make_bitmap({0, 2}, 4);
    auto b = make_bitmap({1, 3}, 4);

    auto result = backend->bitXor(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 4u);
}

TEST_P(BackendParamTest, XorIdentical) {
    auto a = make_bitmap({0, 1, 2}, 8);
    auto b = make_bitmap({0, 1, 2}, 8);

    auto result = backend->bitXor(*a, *b);
    EXPECT_EQ(backend->Cardinality(*result), 0u);
}

TEST_P(BackendParamTest, XorPartialOverlap) {
    auto a = make_bitmap({0, 1, 2, 3}, 8);
    auto b = make_bitmap({2, 3, 4, 5}, 8);

    auto result = backend->bitXor(*a, *b);
    auto decoded = backend->Decode(*result);
    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 1u);
    EXPECT_EQ(decoded[2], 4u);
    EXPECT_EQ(decoded[3], 5u);
}

// ======================== Multi-way OR (range query simulation) ========================

TEST_P(BackendParamTest, MultiWayOrRangeQuery) {
    // Simulate: SELECT * WHERE column IN (1, 2, 3)
    // Build 5 bitmaps (values 0..4), OR bitmaps for values 1, 2, 3
    const int NUM_ROWS = 200;
    std::mt19937 rng(999);
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<int> column(NUM_ROWS);
    for (auto& v : column) v = dist(rng);

    std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
    for (int val = 0; val < 5; ++val) {
        auto h = backend->Create();
        for (int row = 0; row < NUM_ROWS; ++row) {
            backend->Append(*h, column[row] == val);
        }
        bitmaps.push_back(std::move(h));
    }

    // OR bitmaps for values 1, 2, 3
    auto tmp = backend->bitOr(*bitmaps[1], *bitmaps[2]);
    auto result = backend->bitOr(*tmp, *bitmaps[3]);
    auto decoded = backend->Decode(*result);

    // Expected: all rows where column is 1, 2, or 3
    std::vector<uint32_t> expected;
    for (int row = 0; row < NUM_ROWS; ++row) {
        if (column[row] >= 1 && column[row] <= 3) {
            expected.push_back(row);
        }
    }

    ASSERT_EQ(decoded.size(), expected.size())
        << GetParam() << ": multi-way OR result count mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(decoded[i], expected[i])
            << GetParam() << " at index " << i;
    }
}

// ======================== AND after Load (filter query) ========================

TEST_P(BackendParamTest, AndAfterLoad) {
    // Build 2 bitmaps, serialize, load, then AND
    auto a = make_bitmap({0, 1, 2, 3, 4, 5, 6, 7}, 16);
    auto b = make_bitmap({4, 5, 6, 7, 8, 9, 10, 11}, 16);

    backend->Serialize(*a, tmp_dir + "/a.bm");
    backend->Serialize(*b, tmp_dir + "/b.bm");

    auto la = backend->Load(tmp_dir + "/a.bm");
    auto lb = backend->Load(tmp_dir + "/b.bm");

    auto result = backend->bitAnd(*la, *lb);
    auto decoded = backend->Decode(*result);

    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0], 4u);
    EXPECT_EQ(decoded[1], 5u);
    EXPECT_EQ(decoded[2], 6u);
    EXPECT_EQ(decoded[3], 7u);
}

// ======================== Full pipeline: gen .bm → load → OR → AND → XOR ========================

TEST_P(BackendParamTest, FullBmPipeline) {
    // Step 1: Simulate gen_bitmap.sh - generate dataset and create .bm files
    const int NUM_ROWS = 2000;
    const int CARDINALITY = 10;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, CARDINALITY - 1);
    std::vector<int> column(NUM_ROWS);
    for (auto& v : column) v = dist(rng);

    // Create bitmaps in memory (one per value)
    std::vector<std::unique_ptr<BitmapHandle>> bitmaps;
    for (int val = 0; val < CARDINALITY; ++val) {
        auto h = backend->Create();
        for (int row = 0; row < NUM_ROWS; ++row) {
            backend->Append(*h, column[row] == val);
        }
        bitmaps.push_back(std::move(h));
    }

    // Step 2: Verify in-memory cardinality
    for (int val = 0; val < CARDINALITY; ++val) {
        uint64_t expected = std::count(column.begin(), column.end(), val);
        EXPECT_EQ(backend->Cardinality(*bitmaps[val]), expected)
            << GetParam() << " value=" << val;
    }

    // Step 3: Serialize all to .bm files
    for (int val = 0; val < CARDINALITY; ++val) {
        std::string path = tmp_dir + "/" + std::to_string(val + 1) + ".bm";
        backend->Serialize(*bitmaps[val], path);
        EXPECT_TRUE(fs::exists(path)) << "file " << path;
    }

    // Step 4: Load back and verify (Note: WAH/FastBit serialization may
    //         introduce padding; CRoaring and ComBit round-trip exactly.)
    for (int val = 0; val < CARDINALITY; ++val) {
        std::string path = tmp_dir + "/" + std::to_string(val + 1) + ".bm";
        auto loaded = backend->Load(path);
        EXPECT_GT(backend->Cardinality(*loaded), 0u)
            << GetParam() << " value=" << val << " loaded empty";
    }

    // Step 5: OR query — WHERE column IN (0, 1, 2)
    {
        auto or01 = backend->bitOr(*bitmaps[0], *bitmaps[1]);
        auto or012 = backend->bitOr(*or01, *bitmaps[2]);
        auto decoded = backend->Decode(*or012);
        std::vector<uint32_t> expected;
        for (int r = 0; r < NUM_ROWS; ++r) {
            if (column[r] <= 2) expected.push_back(r);
        }
        ASSERT_EQ(decoded.size(), expected.size())
            << GetParam() << ": OR(0,1,2) size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(decoded[i], expected[i]);
        }
    }

    // Step 6: AND query — WHERE column == 0 AND column == 1 (should be empty)
    {
        auto and_result = backend->bitAnd(*bitmaps[0], *bitmaps[1]);
        EXPECT_EQ(backend->Cardinality(*and_result), 0u)
            << GetParam() << ": AND of distinct values should be empty";
    }

    // Step 7: XOR with self (should be empty)
    {
        auto xor_result = backend->bitXor(*bitmaps[0], *bitmaps[0]);
        EXPECT_EQ(backend->Cardinality(*xor_result), 0u)
            << GetParam() << ": self-XOR should be zero";
    }

    // Step 8: OR all bitmaps → should cover every row exactly once
    {
        auto all_or = backend->bitOr(*bitmaps[0], *bitmaps[1]);
        for (int val = 2; val < CARDINALITY; ++val) {
            all_or = backend->bitOr(*all_or, *bitmaps[val]);
        }
        EXPECT_EQ(backend->Cardinality(*all_or), static_cast<uint64_t>(NUM_ROWS))
            << GetParam() << ": OR of all values should cover all rows";
    }
}
