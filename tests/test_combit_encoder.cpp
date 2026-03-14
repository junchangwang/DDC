#include <gtest/gtest.h>
#include <types.hpp>
#include <combit_encoder.hpp>
#include <combit_decoder.hpp>

using namespace combit;

TEST(ComBitEncoder, EmptyInput) {
    std::vector<bool> bits;
    ComBitEncoding enc = ComBitEncoder::encode(bits);
    EXPECT_TRUE(enc.header.empty());
    EXPECT_TRUE(enc.content.empty());
    EXPECT_EQ(enc.decompressed_bit_count(), 0u);
}

TEST(ComBitEncoder, AllZerosSingleWord) {
    // One word of all zeros → compressed fill word
    std::vector<Word> words = { LITERAL_ALL_ZERO };
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 1u);
    EXPECT_TRUE(enc.header[0]);               // compressed
    EXPECT_EQ(fill_bit(enc.content[0]), 0);   // zero fill
    EXPECT_EQ(fill_length(enc.content[0]), 1); // 1 word
}

TEST(ComBitEncoder, AllOnesSingleWord) {
    std::vector<Word> words = { LITERAL_ALL_ONE };
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 1u);
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_bit(enc.content[0]), 1);
    EXPECT_EQ(fill_length(enc.content[0]), 1);
}

TEST(ComBitEncoder, SingleLiteralWord) {
    std::vector<Word> words = { 0x4000 };  // 0100 0000 0000 0000
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 1u);
    EXPECT_FALSE(enc.header[0]);           // literal
    EXPECT_EQ(enc.content[0], 0x4000);
}

TEST(ComBitEncoder, ZeroRunThenLiteralThenOneRun) {
    // Mimics the README example pattern:
    // 2 all-zero words, 1 literal, 3 all-one words
    // (but with 16-bit words)
    std::vector<Word> words = {
        LITERAL_ALL_ZERO,   // word 0
        LITERAL_ALL_ZERO,   // word 1
        Word(0x40FF),        // word 2: literal (0100 0000 1111 1111)
        LITERAL_ALL_ONE,    // word 3
        LITERAL_ALL_ONE,    // word 4
        LITERAL_ALL_ONE     // word 5
    };

    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    // Expect: header = 1 0 1, content = [fill-zero(2), 0x40FF, fill-one(3)]
    ASSERT_EQ(enc.content.size(), 3u);

    // First: compressed zero fill, length 2
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_bit(enc.content[0]), 0);
    EXPECT_EQ(fill_length(enc.content[0]), 2);

    // Second: literal
    EXPECT_FALSE(enc.header[1]);
    EXPECT_EQ(enc.content[1], Word(0x40FF));

    // Third: compressed one fill, length 3
    EXPECT_TRUE(enc.header[2]);
    EXPECT_EQ(fill_bit(enc.content[2]), 1);
    EXPECT_EQ(fill_length(enc.content[2]), 3);
}

TEST(ComBitEncoder, LargeRunExceedsMaxFillLength) {
    // Create a run longer than MAX_FILL_LENGTH to verify splitting
    size_t big_run = static_cast<size_t>(MAX_FILL_LENGTH) + 10;
    std::vector<Word> words(big_run, LITERAL_ALL_ZERO);

    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    // Should produce exactly 2 fill words
    ASSERT_EQ(enc.content.size(), 2u);
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_length(enc.content[0]), MAX_FILL_LENGTH);
    EXPECT_TRUE(enc.header[1]);
    EXPECT_EQ(fill_length(enc.content[1]), 10);
}

TEST(ComBitEncoder, AlternatingLiterals) {
    // No compression possible: all mixed words
    std::vector<Word> words = { Word(0x0001), Word(0x8000), Word(0x5555) };
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 3u);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FALSE(enc.header[i]);
        EXPECT_EQ(enc.content[i], words[i]);
    }
}

TEST(ComBitEncoder, MixedRunsAndLiterals) {
    std::vector<Word> words = {
        LITERAL_ALL_ONE,    // run of 1
        Word(0x1234),        // literal
        LITERAL_ALL_ZERO,   // run of 2 zeros
        LITERAL_ALL_ZERO,
        Word(0xABCD),        // literal
    };

    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 4u);
    // fill-one(1), literal(0x1234), fill-zero(2), literal(0xABCD)
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_bit(enc.content[0]), 1);
    EXPECT_EQ(fill_length(enc.content[0]), 1);

    EXPECT_FALSE(enc.header[1]);
    EXPECT_EQ(enc.content[1], Word(0x1234));

    EXPECT_TRUE(enc.header[2]);
    EXPECT_EQ(fill_bit(enc.content[2]), 0);
    EXPECT_EQ(fill_length(enc.content[2]), 2);

    EXPECT_FALSE(enc.header[3]);
    EXPECT_EQ(enc.content[3], Word(0xABCD));
}

TEST(ComBitEncoder, RoundTripFromBits) {
    std::vector<bool> bits;
    // 48 bits: 16 zeros, 16 mixed, 16 ones
    for (int i = 0; i < 16; ++i) bits.push_back(false);
    // mixed: 0100 0000 1111 1111
    bits.push_back(false); bits.push_back(true);
    for (int i = 0; i < 6; ++i) bits.push_back(false);
    for (int i = 0; i < 8; ++i) bits.push_back(true);
    // all ones
    for (int i = 0; i < 16; ++i) bits.push_back(true);

    ComBitEncoding enc = ComBitEncoder::encode(bits);
    std::vector<bool> decoded = ComBitDecoder::decode(enc);

    ASSERT_EQ(decoded.size(), bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        EXPECT_EQ(decoded[i], bits[i]) << "Mismatch at bit " << i;
    }
}