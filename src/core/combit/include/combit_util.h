#ifndef COMBIT_UTIL_H
#define COMBIT_UTIL_H

#include <vector>
#include <cstdint>
#include <random>
#include <chrono>

/// Prevent compiler from optimizing away benchmark results.
static volatile size_t g_combit_sink = 0;

/// Generate a bitvector where each bit is independently set with probability
/// `density`.
static std::vector<bool> generate_uniform(size_t num_bits, double density,
                                          uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(density);
    std::vector<bool> bits(num_bits);
    for (size_t i = 0; i < num_bits; i++)
        bits[i] = dist(rng);
    return bits;
}

/// Generate a bitvector where 1-bits are grouped in `num_clusters` clusters
/// of `cluster_size` bits each.
static std::vector<bool> generate_clustered(size_t num_bits,
                                            size_t num_clusters,
                                            size_t cluster_size,
                                            uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<bool> bits(num_bits, false);
    if (num_bits == 0 || cluster_size == 0) return bits;
    std::uniform_int_distribution<size_t> pos_dist(
        0, num_bits > cluster_size ? num_bits - cluster_size : 0);
    for (size_t c = 0; c < num_clusters; c++) {
        size_t start = pos_dist(rng);
        for (size_t i = start; i < start + cluster_size && i < num_bits; i++)
            bits[i] = true;
    }
    return bits;
}

/// Convert vector<bool> to packed uint64_t words (for uncompressed baseline).
static std::vector<uint64_t> bools_to_words(const std::vector<bool>& bits) {
    size_t nw = (bits.size() + 63) / 64;
    std::vector<uint64_t> words(nw, 0);
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i])
            words[i / 64] |= uint64_t(1) << (i % 64);
    }
    return words;
}

/// Uncompressed AND on packed words.
static std::vector<uint64_t> raw_and(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] & b[i];
    return r;
}

/// Uncompressed OR on packed words.
static std::vector<uint64_t> raw_or(const std::vector<uint64_t>& a,
                                    const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] | b[i];
    return r;
}

/// Uncompressed XOR on packed words.
static std::vector<uint64_t> raw_xor(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] ^ b[i];
    return r;
}

/// Measure average wall-clock time (ms) of `f()` over `iterations` runs,
/// after one warm-up call.
template<typename Func>
static double time_ms(Func&& f, int iterations = 3) {
    f();  // warm-up
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
        f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
           / iterations;
}

#endif // COMBIT_UTIL_H
