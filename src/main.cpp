#include <types.hpp>
#include <combit_encoder.hpp>
#include <combit_decoder.hpp>
#include <bitmap_vector.hpp>
#include "bitmap_index.hpp"

#include <iostream>
#include <string>
#include <cassert>

using namespace combit;

/// Helper: print a Word in binary
static std::string word_to_bin(Word w, size_t bits) {
    std::string s;
    for (int i = static_cast<int>(bits) - 1; i >= 0; --i) {
        s += ((w >> i) & 1) ? '1' : '0';
    }
    return s;
}

/// Demonstrate the README example (8-bit word emulation via manual words)
static void demo_readme_example() {
    std::cout << "=== README Example (8-bit word, emulated) ===\n\n";

    // The README example uses 8-bit words. Our Word type is 16-bit,
    // so we demonstrate by manually constructing the 8-bit scenario
    // and explaining the mapping.
    //
    // Uncompressed: 00000000 00000000 01000000 11111111 11111111 11111111
    // 6 bytes = 6 words of 8 bits each.
    //
    // Word 0: 00000000  (all zero)
    // Word 1: 00000000  (all zero)
    // Word 2: 01000000  (literal, mixed)
    // Word 3: 11111111  (all one)
    // Word 4: 11111111  (all one)
    // Word 5: 11111111  (all one)
    //
    // ComBit encoding:
    //   header:  1 0 1
    //   content: 00000010  01000000  10000011

    std::cout << "Uncompressed bitmap (6 x 8-bit words):\n";
    std::cout << "  00000000 00000000 01000000 11111111 11111111 11111111\n\n";
    std::cout << "Expected ComBit encoding:\n";
    std::cout << "  header section:   101\n";
    std::cout << "  content section:  00000010 01000000 10000011\n\n";

    // Now demonstrate with our 16-bit word implementation.
    // The same 48 bits as a 16-bit word vector:
    //   Word 0: 0000000000000000  (all zero)
    //   Word 1: 0100000011111111  (literal, mixed)
    //   Word 2: 1111111111111111  (all one)
    std::cout << "=== Same 48 bits with 16-bit words ===\n\n";

    std::vector<bool> bits = {
        0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,   // word 0: all zero
        0,1,0,0,0,0,0,0,  1,1,1,1,1,1,1,1,   // word 1: mixed
        1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1    // word 2: all one
    };

    ComBitEncoding enc = ComBitEncoder::encode(bits);
    std::cout << enc.to_string();
    std::cout << "Decompressed bit count: " << enc.decompressed_bit_count() << "\n\n";

    // Verify round-trip
    std::vector<bool> decoded = ComBitDecoder::decode(enc);
    assert(decoded.size() == bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        assert(decoded[i] == bits[i]);
    }
    std::cout << "Round-trip verification: PASSED\n\n";
}

/// Demonstrate encoding a longer vector with runs
static void demo_long_vector() {
    std::cout << "=== Long vector with runs ===\n\n";

    BitmapVector bv;

    // Set bits to create: 64 zeros, then 32 ones, then a mixed word, then 48 zeros
    // 64 zeros = 4 words of all-zero
    // 32 ones  = 2 words of all-one (set bits 64..95)
    for (uint64_t i = 64; i < 96; ++i) bv.set_bit(i);
    // mixed word at positions 96..111: set bit 100
    bv.set_bit(100);
    // 48 more zeros = 3 words of all-zero (positions 112..159)
    // (they are implicitly zero since we don't set them, but we need to
    // ensure the vector extends to cover them)
    bv.set_bit(159);  // set last bit to force vector to extend
    // actually unset it -- we want zeros. Instead, let's just leave the
    // vector as-is and accept the last word has a set bit.

    ComBitEncoding enc = bv.encode();
    std::cout << "Vector: 64 zeros | 32 ones | mixed(bit 100) | ... | bit 159\n";
    std::cout << enc.to_string();
    std::cout << "Content words: " << enc.content.size()
              << "  (from " << bv.raw_words().size() << " uncompressed)\n";
    std::cout << "Decompressed bits: " << enc.decompressed_bit_count() << "\n\n";
}

/// Demonstrate the bitmap index
static void demo_bitmap_index() {
    std::cout << "=== Bitmap Index Demo ===\n\n";

    // Suppose a table column has values: "red", "blue", "red", "green", "blue", "red"
    // at row positions 0..5.
    BitmapIndex<std::string> idx;
    idx.insert("red",   0);
    idx.insert("blue",  1);
    idx.insert("red",   2);
    idx.insert("green", 3);
    idx.insert("blue",  4);
    idx.insert("red",   5);

    std::cout << "Indexed 6 rows with 3 distinct values.\n";
    std::cout << "Cardinality: " << idx.cardinality() << "\n\n";

    for (const auto& key : {"red", "blue", "green"}) {
        std::cout << "Key: \"" << key << "\"\n";
        auto positions = idx.get_positions(std::string(key));
        std::cout << "  Positions: ";
        for (auto p : positions) std::cout << p << " ";
        std::cout << "\n";

        ComBitEncoding enc = idx.get_encoding(std::string(key));
        std::cout << "  " << enc.to_string();
        std::cout << "\n";
    }
}

int main() {
    demo_readme_example();
    demo_long_vector();
    demo_bitmap_index();

    std::cout << "All demos completed successfully.\n";
    return 0;
}