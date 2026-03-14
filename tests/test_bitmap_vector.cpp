#include <gtest/gtest.h>
#include <bitmap_vector.hpp>

using namespace combit;

TEST(BitmapVector, EmptyVector) {
    BitmapVector bv;
    EXPECT_EQ(bv.size(), 0u);
    EXPECT_EQ(bv.popcount(), 0u);
    EXPECT_FALSE(bv.get_bit(0));
    EXPECT_FALSE(bv.get_bit(100));
}

TEST(BitmapVector, SetAndGetBit) {
    BitmapVector bv;
    bv.set_bit(0);
    bv.set_bit(5);
    bv.set_bit(15);

    EXPECT_TRUE(bv.get_bit(0));
    EXPECT_TRUE(bv.get_bit(5));
    EXPECT_TRUE(bv.get_bit(15));
    EXPECT_FALSE(bv.get_bit(1));
    EXPECT_FALSE(bv.get_bit(14));
    EXPECT_EQ(bv.popcount(), 3u);
}

TEST(BitmapVector, SetBitExpandsVector) {
    BitmapVector bv;
    bv.set_bit(100);

    EXPECT_TRUE(bv.get_bit(100));
    EXPECT_FALSE(bv.get_bit(99));
    EXPECT_GE(bv.size(), 101u);
}

TEST(BitmapVector, EncodeAndLoad) {
    BitmapVector bv;
    bv.set_bit(3);
    bv.set_bit(20);
    bv.set_bit(35);

    ComBitEncoding enc = bv.encode();

    BitmapVector bv2;
    bv2.load(enc);

    EXPECT_EQ(bv.raw_words(), bv2.raw_words());
    EXPECT_TRUE(bv2.get_bit(3));
    EXPECT_TRUE(bv2.get_bit(20));
    EXPECT_TRUE(bv2.get_bit(35));
    EXPECT_FALSE(bv2.get_bit(0));
}

TEST(BitmapVector, AllBitsInWord) {
    BitmapVector bv;
    // Set all 16 bits in the first word
    for (uint64_t i = 0; i < WORD_SIZE; ++i) {
        bv.set_bit(i);
    }
    EXPECT_EQ(bv.popcount(), WORD_SIZE);
    EXPECT_EQ(bv.raw_words()[0], LITERAL_ALL_ONE);

    // Encoding should produce a single fill word
    ComBitEncoding enc = bv.encode();
    ASSERT_EQ(enc.content.size(), 1u);
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_bit(enc.content[0]), 1);
    EXPECT_EQ(fill_length(enc.content[0]), 1);
}

TEST(BitmapVector, IdempotentSet) {
    BitmapVector bv;
    bv.set_bit(7);
    bv.set_bit(7);
    bv.set_bit(7);
    EXPECT_EQ(bv.popcount(), 1u);
}

TEST(BitmapVector, LargeSparseBitmap) {
    BitmapVector bv;
    bv.set_bit(0);
    bv.set_bit(10000);

    EXPECT_TRUE(bv.get_bit(0));
    EXPECT_TRUE(bv.get_bit(10000));
    EXPECT_FALSE(bv.get_bit(5000));

    // Encoding should compress the large zero gap efficiently
    ComBitEncoding enc = bv.encode();
    // The compressed form should be much smaller than 10001/16 ≈ 626 words
    EXPECT_LT(enc.content.size(), 100u);
}