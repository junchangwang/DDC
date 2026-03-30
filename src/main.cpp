#include <combit.h>

#include <iostream>
#include <string>
#include <vector>
#include <random>

int main() {
    std::cout << "=== ComBit Demo ===\n\n";

    // Demonstrate the specification example from the paper
    auto cb = ComBitBtv<8>::from_string(
        "00000000 00000000 00001000 00000000 00000000 00000001");
    std::cout << "Specification example:\n";
    cb.print();
    std::cout << "\n";

    // Demonstrate segmented compression
    std::mt19937_64 rng(42);
    std::bernoulli_distribution dist(0.05);
    std::vector<bool> bits(200000);
    for (size_t i = 0; i < bits.size(); i++)
        bits[i] = dist(rng);

    auto combit = ComBit::compress<8>(bits);
    std::cout << "Segmented ComBit (200K bits, density=0.05):\n";
    combit.print();
    std::cout << "\n";

    // Demonstrate AND operation
    std::vector<bool> bits2(200000);
    std::bernoulli_distribution dist2(0.10);
    for (size_t i = 0; i < bits2.size(); i++)
        bits2[i] = dist2(rng);

    auto combit2 = ComBit::compress<8>(bits2);
    auto result = combit & combit2;

    std::cout << "AND result:\n";
    std::cout << "  Popcount A:      " << combit.popcount() << "\n";
    std::cout << "  Popcount B:      " << combit2.popcount() << "\n";
    std::cout << "  Popcount A & B:  " << result.popcount() << "\n";
    std::cout << "  Compression ratio: " << result.compression_ratio() << "x\n\n";

    std::cout << "Demo completed.\n";
    return 0;
}