#include <gtest/gtest.h>
#include <combit.h>
#include <vector>
#include <random>

// Helper: generate random bitvector
static std::vector<bool> gen_bits(size_t n, double density, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(density);
    std::vector<bool> bits(n);
    for (size_t i = 0; i < n; i++) bits[i] = dist(rng);
    return bits;
}

// ===================================================================
// ComBitBtv round-trip tests
// ===================================================================

TEST(ComBitBtvTest, RoundtripWS8) {
    auto b = gen_bits(10000, 0.10);
    auto cb = ComBitBtv<8>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_EQ(bool(b[i]), bool(dec[i])) << "mismatch at " << i;
}

TEST(ComBitBtvTest, RoundtripWS16) {
    auto b = gen_bits(10000, 0.10);
    auto cb = ComBitBtv<16>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_EQ(bool(b[i]), bool(dec[i]));
}

TEST(ComBitBtvTest, RoundtripWS32) {
    auto b = gen_bits(10000, 0.10);
    auto cb = ComBitBtv<32>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_EQ(bool(b[i]), bool(dec[i]));
}

TEST(ComBitBtvTest, RoundtripWS64) {
    auto b = gen_bits(10000, 0.10);
    auto cb = ComBitBtv<64>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_EQ(bool(b[i]), bool(dec[i]));
}

TEST(ComBitBtvTest, RoundtripEmpty) {
    std::vector<bool> b;
    auto cb = ComBitBtv<8>::compress(b);
    EXPECT_EQ(cb.decompress().size(), 0u);
}

TEST(ComBitBtvTest, RoundtripAllZeros) {
    auto b = gen_bits(1000, 0.0);
    auto cb = ComBitBtv<8>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_FALSE(bool(dec[i]));
}

TEST(ComBitBtvTest, RoundtripAllOnes) {
    auto b = gen_bits(1000, 1.0);
    auto cb = ComBitBtv<8>::compress(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_TRUE(bool(dec[i]));
}

// ===================================================================
// ComBitBtv bitwise operations
// ===================================================================

TEST(ComBitBtvTest, BitwiseAND) {
    auto a = gen_bits(1000, 0.3, 1);
    auto b = gen_bits(1000, 0.3, 2);
    auto ca = ComBitBtv<8>::compress(a);
    auto cb = ComBitBtv<8>::compress(b);
    auto result = (ca & cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) && bool(b[i]));
}

TEST(ComBitBtvTest, BitwiseOR) {
    auto a = gen_bits(1000, 0.3, 1);
    auto b = gen_bits(1000, 0.3, 2);
    auto ca = ComBitBtv<8>::compress(a);
    auto cb = ComBitBtv<8>::compress(b);
    auto result = (ca | cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) || bool(b[i]));
}

TEST(ComBitBtvTest, BitwiseXOR) {
    auto a = gen_bits(1000, 0.3, 1);
    auto b = gen_bits(1000, 0.3, 2);
    auto ca = ComBitBtv<8>::compress(a);
    auto cb = ComBitBtv<8>::compress(b);
    auto result = (ca ^ cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) != bool(b[i]));
}

TEST(ComBitBtvTest, BitwiseNOT) {
    auto a = gen_bits(1000, 0.3, 1);
    auto ca = ComBitBtv<8>::compress(a);
    auto result = (~ca).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), !bool(a[i]));
}

// ===================================================================
// ComBitBtv popcount
// ===================================================================

TEST(ComBitBtvTest, Popcount) {
    auto b = gen_bits(10000, 0.25, 123);
    size_t expected = 0;
    for (size_t i = 0; i < b.size(); i++)
        if (b[i]) expected++;
    EXPECT_EQ(ComBitBtv<8>::compress(b).popcount(), expected);
    EXPECT_EQ(ComBitBtv<16>::compress(b).popcount(), expected);
    EXPECT_EQ(ComBitBtv<32>::compress(b).popcount(), expected);
}

// ===================================================================
// Segmented ComBit tests
// ===================================================================

TEST(ComBitTest, SegmentedRoundtrip) {
    auto b = gen_bits(200000, 0.10);
    auto cb = ComBit::compress<8>(b);
    auto dec = cb.decompress();
    ASSERT_EQ(b.size(), dec.size());
    for (size_t i = 0; i < b.size(); i++)
        EXPECT_EQ(bool(b[i]), bool(dec[i])) << "mismatch at " << i;
}

TEST(ComBitTest, SegmentCount) {
    auto b = gen_bits(200000, 0.10);
    auto cb = ComBit::compress<8>(b, false, 65536);
    EXPECT_EQ(cb.num_segments(), 4u); // 200000 / 65536 = 3.05, ceil = 4
}

TEST(ComBitTest, SegmentedAND) {
    auto a = gen_bits(200000, 0.3, 1);
    auto b = gen_bits(200000, 0.3, 2);
    auto ca = ComBit::compress<8>(a);
    auto cb = ComBit::compress<8>(b);
    auto result = (ca & cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) && bool(b[i]));
}

TEST(ComBitTest, SegmentedOR) {
    auto a = gen_bits(200000, 0.3, 1);
    auto b = gen_bits(200000, 0.3, 2);
    auto ca = ComBit::compress<8>(a);
    auto cb = ComBit::compress<8>(b);
    auto result = (ca | cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) || bool(b[i]));
}

TEST(ComBitTest, SegmentedPopcount) {
    auto b = gen_bits(200000, 0.25, 123);
    size_t expected = 0;
    for (size_t i = 0; i < b.size(); i++)
        if (b[i]) expected++;
    EXPECT_EQ(ComBit::compress<8>(b).popcount(), expected);
}

TEST(ComBitTest, CrossWordSizeAND) {
    auto a = gen_bits(200000, 0.3, 1);
    auto b = gen_bits(200000, 0.3, 2);
    auto ca = ComBit::compress<8>(a);
    auto cb = ComBit::compress<16>(b);
    auto result = (ca & cb).decompress();
    ASSERT_EQ(result.size(), a.size());
    for (size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(bool(result[i]), bool(a[i]) && bool(b[i]));
}
