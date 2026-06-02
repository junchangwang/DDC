#include <ddc.h>

#include <iostream>
#include <string>
#include <vector>
#include <random>

int main() {
    std::cout << "=== DDC Demo ===\n\n";

    // spec example
    auto cb = DDCBtv::from_string(
        "00000000 00000000 00001000 00000000 00000000 00000001");
    std::cout << "Specification example:\n";
    cb.print();
    std::cout << "\n";

    // random bits
    std::mt19937_64 rng(42);
    std::bernoulli_distribution dist(0.05);
    std::vector<bool> bits(200000);
    for (size_t i = 0; i < bits.size(); i++)
        bits[i] = dist(rng);

    // compress
    auto ddc = DDC::compress(bits);
    std::cout << "Segmented DDC (200K bits, density=0.05):\n";
    ddc.print();
    std::cout << "\n";

    std::vector<bool> bits2(200000);
    std::bernoulli_distribution dist2(0.10);
    for (size_t i = 0; i < bits2.size(); i++)
        bits2[i] = dist2(rng);

    auto ddc2 = DDC::compress(bits2);
    auto result = ddc & ddc2;  // AND

    std::cout << "AND result:\n";
    std::cout << "  Popcount A:      " << ddc.popcount() << "\n";
    std::cout << "  Popcount B:      " << ddc2.popcount() << "\n";
    std::cout << "  Popcount A & B:  " << result.popcount() << "\n";
    std::cout << "  Compression ratio: " << result.compression_ratio() << "x\n\n";

    std::cout << "Demo completed.\n";
    return 0;
}