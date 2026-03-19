#include <gtest/gtest.h>
#include <bitmap_vector.hpp>
#include <combit_encoder.hpp>
#include <combit_decoder.hpp>
#include <types.hpp>

using namespace combit;

// ===================================================================
// Compression ratio tests: verify that compressed size < uncompressed
// ===================================================================

TEST(Compression, AllZeroCompression) {
    // 1000 words of all zeros → should compress to very few fill words
    std::vector<Word> words(1000, LITERAL_ALL_ZERO);
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    // Uncompressed: 1000 words; compressed should be <= 2 fill words
    EXPECT_LE(enc.content.size(), 2u);
    double ratio = static_cast<double>(enc.content.size()) / words.size();
    EXPECT_LT(ratio, 0.01);  // > 100x compression
}

TEST(Compression, AllOneCompression) {
    std::vector<Word> words(1000, LITERAL_ALL_ONE);
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    EXPECT_LE(enc.content.size(), 2u);
    double ratio = static_cast<double>(enc.content.size()) / words.size();
    EXPECT_LT(ratio, 0.01);
}

TEST(Compression, SparseOneInMillion) {
    // Simulate sparse bitmap: 1 bit set out of ~1 million bits
    BitmapVector bv;
    bv.set_bit(500000);

    ComBitEncoding enc = bv.encode();
    size_t uncompressed_words = bv.raw_words().size();

    // Should be massively compressed
    EXPECT_GT(uncompressed_words, 30000u);
    EXPECT_LT(enc.content.size(), 100u);

    // Verify correctness after round-trip
    BitmapVector bv2;
    bv2.load(enc);
    EXPECT_TRUE(bv2.get_bit(500000));
    EXPECT_FALSE(bv2.get_bit(0));
    EXPECT_FALSE(bv2.get_bit(499999));
}

TEST(Compression, DenseNearlyFull) {
    // Dense bitmap: all bits set except a few
    BitmapVector bv;
    // Set first 1600 bits (100 words, all ones)
    for (uint64_t i = 0; i < 1600; ++i) {
        bv.set_bit(i);
    }
    // Unset 3 bits by creating a gap at word boundaries
    // (We can't directly unset, but the 100 all-one words should compress well)

    ComBitEncoding enc = bv.encode();
    // 100 all-one words → should compress to a small number of fill words
    EXPECT_LE(enc.content.size(), 5u);
}

TEST(Compression, AlternatingPattern) {
    // Worst case: every word is a different literal → no compression
    std::vector<Word> words;
    for (int i = 0; i < 100; ++i) {
        words.push_back(Word(i * 17 + 1));  // distinct non-zero, non-all-one values
    }

    ComBitEncoding enc = ComBitEncoder::encode_words(words);
    // No compression possible: every word is literal
    EXPECT_EQ(enc.content.size(), 100u);
    for (size_t i = 0; i < enc.header.size(); ++i) {
        EXPECT_FALSE(enc.header[i]);  // all literal
    }
}

TEST(Compression, MixedRunsAndLiterals) {
    // Pattern: 50 zeros, 1 literal, 50 ones, 1 literal, 50 zeros
    std::vector<Word> words;
    for (int i = 0; i < 50; ++i) words.push_back(LITERAL_ALL_ZERO);
    words.push_back(Word(0x1234));
    for (int i = 0; i < 50; ++i) words.push_back(LITERAL_ALL_ONE);
    words.push_back(Word(0xABCD));
    for (int i = 0; i < 50; ++i) words.push_back(LITERAL_ALL_ZERO);

    ComBitEncoding enc = ComBitEncoder::encode_words(words);
    // 152 input words → 5 output words (fill-zero, literal, fill-one, literal, fill-zero)
    EXPECT_EQ(enc.content.size(), 5u);
    double ratio = static_cast<double>(enc.content.size()) / words.size();
    EXPECT_LT(ratio, 0.05);
}

TEST(Compression, CompressionRatioReport) {
    // Test various densities and report compression ratio
    struct TestCase {
        std::string name;
        double density;  // fraction of bits set to 1
        size_t num_bits;
    };

    std::vector<TestCase> cases = {
        {"very_sparse_0.1%",  0.001, 100000},
        {"sparse_1%",         0.01,  100000},
        {"moderate_10%",      0.10,  100000},
        {"dense_50%",         0.50,  10000},
        {"very_dense_99%",    0.99,  10000},
    };

    for (const auto& tc : cases) {
        BitmapVector bv;
        // Deterministic pseudo-random bit setting
        size_t step = static_cast<size_t>(1.0 / tc.density);
        if (step == 0) step = 1;
        for (size_t i = 0; i < tc.num_bits; i += step) {
            bv.set_bit(i);
        }
        // Ensure the vector extends to full size
        if (bv.size() < tc.num_bits) {
            bv.set_bit(tc.num_bits - 1);
        }

        ComBitEncoding enc = bv.encode();
        size_t uncompressed_words = bv.raw_words().size();
        size_t compressed_words = enc.content.size();

        // Verify round-trip
        BitmapVector bv2;
        bv2.load(enc);
        EXPECT_EQ(bv.raw_words(), bv2.raw_words())
            << "Round-trip failed for " << tc.name;

        // Very sparse and very dense should compress well
        if (tc.density <= 0.01 || tc.density >= 0.99) {
            EXPECT_LT(compressed_words, uncompressed_words / 2)
                << tc.name << ": expected >2x compression";
        }
    }
}

TEST(Compression, ExactMaxFillLength) {
    // Exactly MAX_FILL_LENGTH words → single fill word
    std::vector<Word> words(MAX_FILL_LENGTH, LITERAL_ALL_ZERO);
    ComBitEncoding enc = ComBitEncoder::encode_words(words);

    ASSERT_EQ(enc.content.size(), 1u);
    EXPECT_TRUE(enc.header[0]);
    EXPECT_EQ(fill_length(enc.content[0]), MAX_FILL_LENGTH);
}

TEST(Compression, DecompressedBitCount) {
    std::vector<Word> words = {
        LITERAL_ALL_ZERO,
        LITERAL_ALL_ZERO,
        Word(0x1234),
        LITERAL_ALL_ONE,
    };

    ComBitEncoding enc = ComBitEncoder::encode_words(words);
    EXPECT_EQ(enc.decompressed_bit_count(), 4 * WORD_SIZE);

    // Empty encoding
    ComBitEncoding empty;
    EXPECT_EQ(empty.decompressed_bit_count(), 0u);
}
