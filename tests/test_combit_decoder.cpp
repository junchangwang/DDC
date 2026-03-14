#include <gtest/gtest.h>
#include <types.hpp>
#include <combit_encoder.hpp>
#include <combit_decoder.hpp>

using namespace combit;

TEST(ComBitDecoder, EmptyEncoding) {
    ComBitEncoding enc;
    auto words = ComBitDecoder::decode_words(enc);
    EXPECT_TRUE(words.empty());
}

TEST(ComBitDecoder, SingleFillZero) {
    ComBitEncoding enc;
    enc.header.push_back(true);
    enc.content.push_back(make_fill_word(0, 5));

    auto words = ComBitDecoder::decode_words(enc);
    ASSERT_EQ(words.size(), 5u);
    for (auto w : words) {
        EXPECT_EQ(w, LITERAL_ALL_ZERO);
    }
}

TEST(ComBitDecoder, SingleFillOne) {
    ComBitEncoding enc;
    enc.header.push_back(true);
    enc.content.push_back(make_fill_word(1, 3));

    auto words = ComBitDecoder::decode_words(enc);
    ASSERT_EQ(words.size(), 3u);
    for (auto w : words) {
        EXPECT_EQ(w, LITERAL_ALL_ONE);
    }
}

TEST(ComBitDecoder, LiteralWord) {
    ComBitEncoding enc;
    enc.header.push_back(false);
    enc.content.push_back(Word(0x1234));

    auto words = ComBitDecoder::decode_words(enc);
    ASSERT_EQ(words.size(), 1u);
    EXPECT_EQ(words[0], Word(0x1234));
}

TEST(ComBitDecoder, DecodeToBits) {
    ComBitEncoding enc;
    enc.header = { true, false, true };
    enc.content = {
        make_fill_word(0, 1),       // 16 zeros
        Word(0x4000),               // literal: 0100 0000 0000 0000
        make_fill_word(1, 1)        // 16 ones
    };

    auto bits = ComBitDecoder::decode(enc);
    ASSERT_EQ(bits.size(), 48u);

    // First 16: all zero
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FALSE(bits[i]) << "bit " << i;
    }
    // Next 16: 0100 0000 0000 0000
    EXPECT_FALSE(bits[16]);
    EXPECT_TRUE(bits[17]);
    for (size_t i = 18; i < 32; ++i) {
        EXPECT_FALSE(bits[i]) << "bit " << i;
    }
    // Last 16: all one
    for (size_t i = 32; i < 48; ++i) {
        EXPECT_TRUE(bits[i]) << "bit " << i;
    }
}

TEST(ComBitDecoder, RoundTripManyPatterns) {
    // Generate various patterns and verify encode → decode round-trip
    for (int pattern = 0; pattern < 8; ++pattern) {
        std::vector<Word> original;
        for (int w = 0; w < 20; ++w) {
            switch (pattern) {
                case 0: original.push_back(LITERAL_ALL_ZERO); break;
                case 1: original.push_back(LITERAL_ALL_ONE); break;
                case 2: original.push_back(Word(w * 0x111)); break;
                case 3: original.push_back(w % 2 == 0 ? LITERAL_ALL_ZERO : Word(0x5555)); break;
                case 4: original.push_back(w < 10 ? LITERAL_ALL_ZERO : LITERAL_ALL_ONE); break;
                case 5: original.push_back(w % 3 == 0 ? LITERAL_ALL_ONE : Word(w)); break;
                case 6: original.push_back(w == 10 ? Word(0xBEEF) : LITERAL_ALL_ZERO); break;
                case 7: original.push_back(w % 4 < 2 ? LITERAL_ALL_ONE : Word(0x0F0F)); break;
            }
        }

        ComBitEncoding enc = ComBitEncoder::encode_words(original);
        auto decoded = ComBitDecoder::decode_words(enc);

        ASSERT_EQ(decoded.size(), original.size())
            << "Pattern " << pattern;
        for (size_t i = 0; i < original.size(); ++i) {
            EXPECT_EQ(decoded[i], original[i])
                << "Pattern " << pattern << " word " << i;
        }
    }
}