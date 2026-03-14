#include <gtest/gtest.h>
#include "benchmark/backends/combit/combit_backend.h"
#include <bitmap_vector.hpp>
#include <cstdio>
#include <filesystem>
#include <set>

namespace fs = std::filesystem;

// ===================================================================
// Backend operations tests: Serialize/Load, OR/AND/XOR, Cardinality
// ===================================================================

class BackendTest : public ::testing::Test {
protected:
    CombitBackend backend;
    std::string tmp_dir;

    void SetUp() override {
        tmp_dir = fs::temp_directory_path() / "combit_test";
        fs::create_directories(tmp_dir);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }

    // Helper: create a bitmap with specific bit positions set
    std::unique_ptr<BitmapHandle> make_bitmap(const std::vector<uint64_t>& positions, uint64_t total_bits) {
        auto h = backend.Create();
        // Append bit by bit up to total_bits
        std::set<uint64_t> pos_set(positions.begin(), positions.end());
        for (uint64_t i = 0; i < total_bits; ++i) {
            backend.Append(*h, pos_set.count(i) > 0);
        }
        return h;
    }
};

// ======================== Cardinality ========================

TEST_F(BackendTest, CardinalityEmpty) {
    auto h = backend.Create();
    EXPECT_EQ(backend.Cardinality(*h), 0u);
}

TEST_F(BackendTest, CardinalityAfterAppend) {
    auto h = backend.Create();
    backend.Append(*h, true);
    backend.Append(*h, false);
    backend.Append(*h, true);
    backend.Append(*h, false);
    backend.Append(*h, true);

    EXPECT_EQ(backend.Cardinality(*h), 3u);
}

TEST_F(BackendTest, CardinalityLarge) {
    auto h = backend.Create();
    for (int i = 0; i < 10000; ++i) {
        backend.Append(*h, i % 3 == 0);  // every 3rd bit
    }
    // ceil(10000/3) = 3334
    EXPECT_EQ(backend.Cardinality(*h), 3334u);
}

// ======================== Decode ========================

TEST_F(BackendTest, DecodeEmpty) {
    auto h = backend.Create();
    auto result = backend.Decode(*h);
    EXPECT_TRUE(result.empty());
}

TEST_F(BackendTest, DecodeSimple) {
    auto h = make_bitmap({0, 2, 5, 9}, 10);
    auto result = backend.Decode(*h);

    ASSERT_EQ(result.size(), 4u);
    EXPECT_EQ(result[0], 0u);
    EXPECT_EQ(result[1], 2u);
    EXPECT_EQ(result[2], 5u);
    EXPECT_EQ(result[3], 9u);
}

TEST_F(BackendTest, DecodeAllOnes) {
    auto h = backend.Create();
    for (int i = 0; i < 32; ++i) {
        backend.Append(*h, true);
    }
    auto result = backend.Decode(*h);
    ASSERT_EQ(result.size(), 32u);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(result[i], i);
    }
}

// ======================== Serialize / Load ========================

TEST_F(BackendTest, SerializeAndLoadEmpty) {
    auto h = backend.Create();
    std::string path = tmp_dir + "/empty.bm";

    backend.Serialize(*h, path);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = backend.Load(path);
    EXPECT_EQ(backend.Cardinality(*loaded), 0u);
    EXPECT_TRUE(backend.Decode(*loaded).empty());
}

TEST_F(BackendTest, SerializeAndLoadSimple) {
    auto h = make_bitmap({0, 3, 7, 15, 31}, 32);
    std::string path = tmp_dir + "/simple.bm";

    backend.Serialize(*h, path);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = backend.Load(path);
    EXPECT_EQ(backend.Cardinality(*loaded), 5u);

    auto decoded = backend.Decode(*loaded);
    ASSERT_EQ(decoded.size(), 5u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 7u);
    EXPECT_EQ(decoded[3], 15u);
    EXPECT_EQ(decoded[4], 31u);
}

TEST_F(BackendTest, SerializeAndLoadLarge) {
    auto h = backend.Create();
    std::vector<uint32_t> expected;
    for (int i = 0; i < 10000; ++i) {
        bool bit = (i % 7 == 0);
        backend.Append(*h, bit);
        if (bit) expected.push_back(i);
    }

    std::string path = tmp_dir + "/large.bm";
    backend.Serialize(*h, path);

    auto loaded = backend.Load(path);
    EXPECT_EQ(backend.Cardinality(*loaded), expected.size());

    auto decoded = backend.Decode(*loaded);
    ASSERT_EQ(decoded.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(decoded[i], expected[i]) << "index " << i;
    }
}

TEST_F(BackendTest, SerializeFileSize) {
    // Sparse bitmap: file should be small
    auto h = make_bitmap({0, 9999}, 10000);
    std::string path = tmp_dir + "/sparse.bm";

    backend.Serialize(*h, path);
    auto file_size = fs::file_size(path);

    // 2 indices * 4 bytes + 8 bytes (current_bits) + 8 bytes (size) = ~24 bytes
    EXPECT_LT(file_size, 100u);
}

TEST_F(BackendTest, LoadNonexistentFile) {
    auto loaded = backend.Load(tmp_dir + "/does_not_exist.bm");
    // Should return empty bitmap, not crash
    EXPECT_EQ(backend.Cardinality(*loaded), 0u);
}

// ======================== Bitwise OR ========================

TEST_F(BackendTest, OrDisjoint) {
    auto a = make_bitmap({0, 2, 4}, 8);
    auto b = make_bitmap({1, 3, 5}, 8);

    auto result = backend.bitOr(*a, *b);
    auto decoded = backend.Decode(*result);

    ASSERT_EQ(decoded.size(), 6u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 1u);
    EXPECT_EQ(decoded[2], 2u);
    EXPECT_EQ(decoded[3], 3u);
    EXPECT_EQ(decoded[4], 4u);
    EXPECT_EQ(decoded[5], 5u);
}

TEST_F(BackendTest, OrOverlapping) {
    auto a = make_bitmap({0, 1, 2, 3}, 8);
    auto b = make_bitmap({2, 3, 4, 5}, 8);

    auto result = backend.bitOr(*a, *b);
    auto decoded = backend.Decode(*result);

    ASSERT_EQ(decoded.size(), 6u);
    for (uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(decoded[i], i);
    }
}

TEST_F(BackendTest, OrWithEmpty) {
    auto a = make_bitmap({1, 3, 5}, 8);
    auto b = backend.Create();

    auto result = backend.bitOr(*a, *b);
    auto decoded = backend.Decode(*result);

    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 1u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 5u);
}

TEST_F(BackendTest, OrDifferentLengths) {
    auto a = make_bitmap({0, 1}, 4);
    auto b = make_bitmap({10, 15}, 16);

    auto result = backend.bitOr(*a, *b);
    EXPECT_EQ(backend.Cardinality(*result), 4u);

    auto decoded = backend.Decode(*result);
    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 1u);
    EXPECT_EQ(decoded[2], 10u);
    EXPECT_EQ(decoded[3], 15u);
}

TEST_F(BackendTest, OrSelf) {
    auto a = make_bitmap({0, 5, 10}, 16);
    auto result = backend.bitOr(*a, *a);

    auto decoded = backend.Decode(*result);
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 5u);
    EXPECT_EQ(decoded[2], 10u);
}

// ======================== Bitwise AND ========================

TEST_F(BackendTest, AndOverlapping) {
    auto a = make_bitmap({0, 1, 2, 3, 4}, 8);
    auto b = make_bitmap({2, 3, 4, 5, 6}, 8);

    auto result = backend.bitAnd(*a, *b);
    auto decoded = backend.Decode(*result);

    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 2u);
    EXPECT_EQ(decoded[1], 3u);
    EXPECT_EQ(decoded[2], 4u);
}

TEST_F(BackendTest, AndDisjoint) {
    auto a = make_bitmap({0, 2, 4}, 8);
    auto b = make_bitmap({1, 3, 5}, 8);

    auto result = backend.bitAnd(*a, *b);
    auto decoded = backend.Decode(*result);

    EXPECT_TRUE(decoded.empty());
}

TEST_F(BackendTest, AndWithEmpty) {
    auto a = make_bitmap({0, 1, 2}, 8);
    auto b = backend.Create();

    auto result = backend.bitAnd(*a, *b);
    EXPECT_EQ(backend.Cardinality(*result), 0u);
}

TEST_F(BackendTest, AndSelf) {
    auto a = make_bitmap({0, 5, 10}, 16);
    auto result = backend.bitAnd(*a, *a);

    auto decoded = backend.Decode(*result);
    ASSERT_EQ(decoded.size(), 3u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 5u);
    EXPECT_EQ(decoded[2], 10u);
}

// ======================== Bitwise XOR ========================

TEST_F(BackendTest, XorDisjoint) {
    auto a = make_bitmap({0, 2}, 4);
    auto b = make_bitmap({1, 3}, 4);

    auto result = backend.bitXor(*a, *b);
    auto decoded = backend.Decode(*result);

    ASSERT_EQ(decoded.size(), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(decoded[i], i);
    }
}

TEST_F(BackendTest, XorIdentical) {
    auto a = make_bitmap({0, 1, 2}, 4);
    auto b = make_bitmap({0, 1, 2}, 4);

    auto result = backend.bitXor(*a, *b);
    EXPECT_EQ(backend.Cardinality(*result), 0u);
}

TEST_F(BackendTest, XorPartialOverlap) {
    auto a = make_bitmap({0, 1, 2, 3}, 8);
    auto b = make_bitmap({2, 3, 4, 5}, 8);

    auto result = backend.bitXor(*a, *b);
    auto decoded = backend.Decode(*result);

    // XOR: {0,1} union {4,5}
    ASSERT_EQ(decoded.size(), 4u);
    EXPECT_EQ(decoded[0], 0u);
    EXPECT_EQ(decoded[1], 1u);
    EXPECT_EQ(decoded[2], 4u);
    EXPECT_EQ(decoded[3], 5u);
}

// ======================== Multi-way OR ========================

TEST_F(BackendTest, MultiWayOr) {
    // Simulate range query: OR of multiple bitmaps
    auto b1 = make_bitmap({0, 10, 20}, 32);
    auto b2 = make_bitmap({5, 15, 25}, 32);
    auto b3 = make_bitmap({3, 13, 23}, 32);

    auto tmp = backend.bitOr(*b1, *b2);
    auto result = backend.bitOr(*tmp, *b3);

    auto decoded = backend.Decode(*result);
    ASSERT_EQ(decoded.size(), 9u);

    std::vector<uint32_t> expected = {0, 3, 5, 10, 13, 15, 20, 23, 25};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(decoded[i], expected[i]) << "index " << i;
    }
}

// ======================== Full pipeline: Create → Append → Serialize → Load → Query ========================

TEST_F(BackendTest, FullPipeline) {
    // 1. Create and populate
    auto h = backend.Create();
    std::vector<uint32_t> expected_positions;
    for (int i = 0; i < 1000; ++i) {
        bool bit = (i % 5 == 0) || (i % 7 == 0);
        backend.Append(*h, bit);
        if (bit) expected_positions.push_back(i);
    }

    // 2. Check cardinality
    EXPECT_EQ(backend.Cardinality(*h), expected_positions.size());

    // 3. Serialize to .bm file
    std::string path = tmp_dir + "/pipeline.bm";
    backend.Serialize(*h, path);
    EXPECT_TRUE(fs::exists(path));

    // 4. Load from .bm file
    auto loaded = backend.Load(path);
    EXPECT_EQ(backend.Cardinality(*loaded), expected_positions.size());

    // 5. Decode and verify all positions
    auto decoded = backend.Decode(*loaded);
    ASSERT_EQ(decoded.size(), expected_positions.size());
    for (size_t i = 0; i < expected_positions.size(); ++i) {
        EXPECT_EQ(decoded[i], expected_positions[i]) << "index " << i;
    }

    // 6. OR with itself should be identical
    auto or_result = backend.bitOr(*loaded, *loaded);
    EXPECT_EQ(backend.Cardinality(*or_result), expected_positions.size());

    // 7. AND with itself should be identical
    auto and_result = backend.bitAnd(*loaded, *loaded);
    EXPECT_EQ(backend.Cardinality(*and_result), expected_positions.size());

    // 8. XOR with itself should be empty
    auto xor_result = backend.bitXor(*loaded, *loaded);
    EXPECT_EQ(backend.Cardinality(*xor_result), 0u);
}

// ======================== Generate .bm file and verify ========================

TEST_F(BackendTest, GenerateBmFile) {
    // Simulate what gen_bitmap.sh does: create bitmap for a column value
    // Column with 1000 rows, value "42" appears at positions divisible by 10
    auto h = backend.Create();
    for (int row = 0; row < 1000; ++row) {
        backend.Append(*h, row % 10 == 0);
    }

    // Save as .bm file
    std::string bm_path = tmp_dir + "/42.bm";
    backend.Serialize(*h, bm_path);

    // Check file was created
    EXPECT_TRUE(fs::exists(bm_path));
    EXPECT_GT(fs::file_size(bm_path), 0u);

    // Reload and verify
    auto loaded = backend.Load(bm_path);
    EXPECT_EQ(backend.Cardinality(*loaded), 100u);

    auto positions = backend.Decode(*loaded);
    ASSERT_EQ(positions.size(), 100u);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_EQ(positions[i], static_cast<uint32_t>(i * 10));
    }
}
