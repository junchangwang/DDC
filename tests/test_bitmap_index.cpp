#include <gtest/gtest.h>
#include <bitmap_index.hpp>
#include <string>

using namespace combit;

TEST(BitmapIndex, EmptyIndex) {
    BitmapIndex<int> idx;
    EXPECT_EQ(idx.cardinality(), 0u);
    EXPECT_EQ(idx.lookup(42), nullptr);
    EXPECT_TRUE(idx.get_positions(42).empty());
}

TEST(BitmapIndex, SingleInsert) {
    BitmapIndex<int> idx;
    idx.insert(5, 0);

    EXPECT_EQ(idx.cardinality(), 1u);
    EXPECT_NE(idx.lookup(5), nullptr);
    EXPECT_TRUE(idx.lookup(5)->get_bit(0));

    auto pos = idx.get_positions(5);
    ASSERT_EQ(pos.size(), 1u);
    EXPECT_EQ(pos[0], 0u);
}

TEST(BitmapIndex, MultipleKeys) {
    BitmapIndex<std::string> idx;
    idx.insert("red",   0);
    idx.insert("blue",  1);
    idx.insert("red",   2);
    idx.insert("green", 3);
    idx.insert("blue",  4);
    idx.insert("red",   5);

    EXPECT_EQ(idx.cardinality(), 3u);

    auto red_pos = idx.get_positions("red");
    ASSERT_EQ(red_pos.size(), 3u);
    EXPECT_EQ(red_pos[0], 0u);
    EXPECT_EQ(red_pos[1], 2u);
    EXPECT_EQ(red_pos[2], 5u);

    auto blue_pos = idx.get_positions("blue");
    ASSERT_EQ(blue_pos.size(), 2u);
    EXPECT_EQ(blue_pos[0], 1u);
    EXPECT_EQ(blue_pos[1], 4u);

    auto green_pos = idx.get_positions("green");
    ASSERT_EQ(green_pos.size(), 1u);
    EXPECT_EQ(green_pos[0], 3u);
}

TEST(BitmapIndex, LookupMissing) {
    BitmapIndex<int> idx;
    idx.insert(1, 0);
    idx.insert(2, 1);

    EXPECT_EQ(idx.lookup(999), nullptr);
    EXPECT_TRUE(idx.get_positions(999).empty());
}

TEST(BitmapIndex, GetEncodingRoundTrip) {
    BitmapIndex<int> idx;
    // Insert a value at many positions to create runs
    for (uint64_t i = 0; i < 100; ++i) {
        idx.insert(42, i);
    }

    ComBitEncoding enc = idx.get_encoding(42);
    // Should be compressed (all-one run)
    EXPECT_FALSE(enc.content.empty());
    EXPECT_GT(enc.decompressed_bit_count(), 0u);

    // Verify round-trip: decode and check all bits are set
    auto words = ComBitDecoder::decode_words(enc);
    BitmapVector bv;
    bv.load(enc);
    for (uint64_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(bv.get_bit(i)) << "bit " << i;
    }
}

TEST(BitmapIndex, GetEncodingMissingKey) {
    BitmapIndex<int> idx;
    idx.insert(1, 0);

    ComBitEncoding enc = idx.get_encoding(999);
    EXPECT_TRUE(enc.header.empty());
    EXPECT_TRUE(enc.content.empty());
}

TEST(BitmapIndex, HighCardinality) {
    BitmapIndex<int> idx;
    // 1000 distinct keys, each at one position
    for (int k = 0; k < 1000; ++k) {
        idx.insert(k, static_cast<uint64_t>(k));
    }

    EXPECT_EQ(idx.cardinality(), 1000u);

    for (int k = 0; k < 1000; ++k) {
        auto pos = idx.get_positions(k);
        ASSERT_EQ(pos.size(), 1u) << "key " << k;
        EXPECT_EQ(pos[0], static_cast<uint64_t>(k));
    }
}

TEST(BitmapIndex, SparsePositions) {
    BitmapIndex<int> idx;
    // Key at positions far apart
    idx.insert(1, 0);
    idx.insert(1, 10000);
    idx.insert(1, 50000);

    auto pos = idx.get_positions(1);
    ASSERT_EQ(pos.size(), 3u);
    EXPECT_EQ(pos[0], 0u);
    EXPECT_EQ(pos[1], 10000u);
    EXPECT_EQ(pos[2], 50000u);

    // Encoding should compress the zero gaps efficiently
    ComBitEncoding enc = idx.get_encoding(1);
    EXPECT_LT(enc.content.size(), 200u);
}
