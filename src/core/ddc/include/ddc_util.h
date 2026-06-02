#ifndef DDC_UTIL_H
#define DDC_UTIL_H

#include <vector>
#include <cstdint>
#include <random>
#include <chrono>

static volatile size_t g_ddc_sink = 0;

// uniform gen
static std::vector<bool> generate_uniform(size_t num_bits, double density,
                                          uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(density);
    std::vector<bool> bits(num_bits);
    for (size_t i = 0; i < num_bits; i++)
        bits[i] = dist(rng);
    return bits;
}

// clustered shipdate
static std::vector<bool> generate_q6_shipdate_like(size_t N,
                                                   size_t day_idx,
                                                   size_t total_days,
                                                   double internal_density,
                                                   uint64_t seed = 42) {
    std::vector<bool> bits(N, false);
    if (N == 0 || total_days == 0 || internal_density <= 0.0) return bits;

    const size_t expected_set = static_cast<size_t>(
        static_cast<double>(N) / static_cast<double>(total_days));

    size_t cluster_w = static_cast<size_t>(
        static_cast<double>(expected_set) / internal_density);
    if (cluster_w > N) cluster_w = N;

    const size_t center = static_cast<size_t>(
        static_cast<double>(day_idx) * N / total_days);
    const size_t half_w = cluster_w / 2;
    size_t start = (center > half_w) ? center - half_w : 0;
    size_t end   = start + cluster_w;
    if (end > N) end = N;

    // fill cluster
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(internal_density);
    for (size_t i = start; i < end; i++) {
        bits[i] = dist(rng);
    }
    return bits;
}

// run-based shipdate
static std::vector<bool> generate_q6_shipdate_runs(size_t N,
                                                   size_t day_idx,
                                                   size_t total_days,
                                                   double cluster_density,
                                                   size_t run_length,
                                                   uint64_t seed = 42) {
    std::vector<bool> bits(N, false);
    if (N == 0 || total_days == 0 || run_length == 0) return bits;

    const size_t expected_set = static_cast<size_t>(
        static_cast<double>(N) / static_cast<double>(total_days));

    size_t num_runs = expected_set / run_length;
    if (num_runs == 0) num_runs = 1;

    size_t cluster_w = static_cast<size_t>(
        static_cast<double>(N) * cluster_density);
    if (cluster_w < run_length) cluster_w = run_length;
    if (cluster_w > N)          cluster_w = N;

    const size_t center = static_cast<size_t>(
        static_cast<double>(day_idx) * N / total_days);
    const size_t half_w = cluster_w / 2;
    size_t start = (center > half_w) ? center - half_w : 0;
    size_t end   = start + cluster_w;
    if (end > N) {
        end   = N;
        start = (end > cluster_w) ? end - cluster_w : 0;
    }
    if (end - start < run_length) return bits;

    // scatter runs
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pos_dist(
        start, end - run_length);
    for (size_t k = 0; k < num_runs; k++) {
        size_t p = pos_dist(rng);
        for (size_t j = 0; j < run_length; j++) {
            bits[p + j] = true;
        }
    }
    return bits;
}

// random clusters
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

// pack to words
static std::vector<uint64_t> bools_to_words(const std::vector<bool>& bits) {
    size_t nw = (bits.size() + 63) / 64;
    std::vector<uint64_t> words(nw, 0);
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i])
            words[i / 64] |= uint64_t(1) << (i % 64);
    }
    return words;
}

static std::vector<uint64_t> raw_and(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] & b[i];
    return r;
}

static std::vector<uint64_t> raw_or(const std::vector<uint64_t>& a,
                                    const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] | b[i];
    return r;
}

static std::vector<uint64_t> raw_xor(const std::vector<uint64_t>& a,
                                     const std::vector<uint64_t>& b) {
    std::vector<uint64_t> r(a.size());
    for (size_t i = 0; i < a.size(); i++)
        r[i] = a[i] ^ b[i];
    return r;
}

// bench timer
template<typename Func>
static double time_ms(Func&& f, int iterations = 3) {
    f(); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
        f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count()
           / iterations;
}

#endif
