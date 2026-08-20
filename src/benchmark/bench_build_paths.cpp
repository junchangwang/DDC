// bench_build_paths — isolate the two phases behind the TPC-H Q6 numbers:
//   (1) union BUILD from a sorted position list (what or_many/or_range does)
//   (2) the AND kernel proper on the built sets
// Same synthetic profile as SF10 Q6: N = 59,986,052 rows,
// densities shipdate-year 1/7, discount 3/11, quantity 24/50.
//
// Paths compared per phase:
//   build: DDC scatter (seg=4096, the debit path) | DDC scatter (seg=65536)
//          | BSR dense-stamp + compact | CRoaring addMany
//   AND:   DDC decompressed dense &= | BSR qfilter kernel | CR operator&

#include "ddc.h"
#include "intersection_algos.hpp"
#include "croaring/roaring.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using clk = std::chrono::high_resolution_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
static double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static std::vector<uint32_t> gen_positions(size_t n_rows, double density, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint32_t> pos;
    pos.reserve(static_cast<size_t>(n_rows * density * 1.01));
    // geometric gap sampling for uniform bernoulli positions
    std::geometric_distribution<uint32_t> gap(density);
    uint64_t p = gap(rng);
    while (p < n_rows) {
        pos.push_back(static_cast<uint32_t>(p));
        p += 1 + gap(rng);
    }
    return pos;
}

struct BsrBuf {
    int* bases = nullptr; int* states = nullptr; int size = 0;
    void alloc(size_t cap) {
        free(bases); free(states);
        posix_memalign(reinterpret_cast<void**>(&bases), 64, (cap + 8) * sizeof(int));
        posix_memalign(reinterpret_cast<void**>(&states), 64, (cap + 8) * sizeof(int));
        memset(bases + cap, 0, 8 * sizeof(int));
        memset(states + cap, 0, 8 * sizeof(int));
    }
    ~BsrBuf() { free(bases); free(states); }
};

// replicate IndexedBSR::build_union dense-stamp path exactly
static void bsr_stamp_build(const std::vector<uint32_t>& pos, size_t n_rows, BsrBuf& out) {
    const size_t num_chunks = (n_rows + 31) / 32;
    std::vector<uint32_t> dense(num_chunks, 0);
    for (uint32_t p : pos) dense[p >> 5] |= (1u << (p & 31));
    size_t nz = 0;
    for (size_t i = 0; i < num_chunks; i++) nz += (dense[i] != 0);
    out.alloc(nz);
    int m = 0;
    for (size_t i = 0; i < num_chunks; i++)
        if (dense[i]) { out.bases[m] = static_cast<int>(i); out.states[m] = static_cast<int>(dense[i]); m++; }
    out.size = m;
}

int main() {
    const size_t N = 59986052;
    const int ITER = 5;
    struct Case { const char* name; double density; };
    const Case cases[] = {
        {"ship 1/7  (14.3%)", 1.0 / 7.0},
        {"disc 3/11 (27.3%)", 3.0 / 11.0},
        {"qty 24/50 (48.0%)", 24.0 / 50.0},
    };

    std::vector<std::vector<uint32_t>> pos_sets;
    for (auto& c : cases) pos_sets.push_back(gen_positions(N, c.density, 42 + (&c - cases)));

    std::cout << "=== BUILD phase (median of " << ITER << ", ms) ===\n";
    for (size_t ci = 0; ci < 3; ci++) {
        const auto& pos = pos_sets[ci];
        std::vector<double> t_ddc4k, t_ddc64k, t_bsr, t_cr, t_ddc_shift, t_ddc_flat;
        for (int it = 0; it < ITER + 1; it++) {  // first iter = warm-up
            auto t0 = clk::now();
            DDC d4 = DDC::from_sparse_positions({}, N, 4096);
            d4.scatter_or_decompressed(pos.data(), pos.size());
            auto t1 = clk::now();
            DDC d64 = DDC::from_sparse_positions({}, N, 65536);
            d64.scatter_or_decompressed(pos.data(), pos.size());
            auto t2 = clk::now();
            BsrBuf b; bsr_stamp_build(pos, N, b);
            auto t3 = clk::now();
            // variant: shift/mask scatter with cached segment L1 pointers
            DDC d4s = DDC::from_sparse_positions({}, N, 4096);
            {
                auto& segs = d4s.segments();
                std::vector<uint8_t*> l1p(segs.size());
                for (size_t s2 = 0; s2 < segs.size(); s2++)
                    l1p[s2] = const_cast<uint8_t*>(segs[s2].l1_lit_data());
                for (uint32_t pp : pos) {
                    uint32_t seg = pp >> 12, in = pp & 4095;
                    l1p[seg][in >> 3] |= (uint8_t)(0x80 >> (in & 7));
                }
            }
            auto t3b = clk::now();
            // variant: flat MSB-first byte stamp + memcpy into segments
            DDC d4f = DDC::from_sparse_positions({}, N, 4096);
            {
                static std::vector<uint8_t> flat;
                flat.assign((N + 7) / 8, 0);
                for (uint32_t pp : pos)
                    flat[pp >> 3] |= (uint8_t)(0x80 >> (pp & 7));
                auto& segs = d4f.segments();
                size_t off = 0;
                for (auto& sg : segs) {
                    size_t nb = sg.l2_count();
                    std::memcpy(const_cast<uint8_t*>(sg.l1_lit_data()), flat.data() + off, nb);
                    off += nb;
                }
            }
            auto t3c = clk::now();
            roaring::Roaring r; r.addMany(pos.size(), pos.data());
            auto t4 = clk::now();
            if (it == 0) continue;
            t_ddc4k.push_back(ms(t0, t1)); t_ddc64k.push_back(ms(t1, t2));
            t_bsr.push_back(ms(t2, t3));   t_cr.push_back(ms(t3c, t4));
            t_ddc_shift.push_back(ms(t3, t3b));
            t_ddc_flat.push_back(ms(t3b, t3c));
        }
        std::cout << cases[ci].name
                  << "  n=" << pos.size()
                  << "  DDC-scatter(4096)=" << med(t_ddc4k)
                  << "  DDC-scatter(65536)=" << med(t_ddc64k)
                  << "  BSR-stamp=" << med(t_bsr)
                  << "  CR-addMany=" << med(t_cr)
                  << "\n            DDC-scatter(shift+ptrcache)=" << med(t_ddc_shift)
                  << "  DDC-flatstamp+copy=" << med(t_ddc_flat) << "\n";
    }

    std::cout << "\n=== AND phase (median of " << ITER << ", ms) ===\n";
    {
        // DDC decompressed dense AND (what Q6 does: disc &= qty; disc &= ship)
        DDC ship = DDC::from_sparse_positions({}, N, 4096);
        ship.scatter_or_decompressed(pos_sets[0].data(), pos_sets[0].size());
        DDC disc = DDC::from_sparse_positions({}, N, 4096);
        disc.scatter_or_decompressed(pos_sets[1].data(), pos_sets[1].size());
        DDC qty = DDC::from_sparse_positions({}, N, 4096);
        qty.scatter_or_decompressed(pos_sets[2].data(), pos_sets[2].size());

        std::vector<double> t_ddc;
        for (int it = 0; it < ITER + 1; it++) {
            DDC a = disc;                       // copy outside timing
            auto t0 = clk::now();
            a &= qty;
            a &= ship;
            auto t1 = clk::now();
            if (it) t_ddc.push_back(ms(t0, t1));
        }
        std::cout << "DDC dense &= x2:  " << med(t_ddc) << "  (Q6 observed: 2.1)\n";

        // BSR qfilter AND (what Q6 BSR does: and(ship,disc) then and(r1,qty))
        BsrBuf bship, bdisc, bqty;
        bsr_stamp_build(pos_sets[0], N, bship);
        bsr_stamp_build(pos_sets[1], N, bdisc);
        bsr_stamp_build(pos_sets[2], N, bqty);
        std::cout << "BSR chunks: ship=" << bship.size << " disc=" << bdisc.size
                  << " qty=" << bqty.size << "\n";
        BsrBuf r1, r2;
        r1.alloc(std::min(bship.size, bdisc.size));
        r2.alloc(bqty.size);
        std::vector<double> t_bsr;
        for (int it = 0; it < ITER + 1; it++) {
            auto t0 = clk::now();
            int n1 = intersect_qfilter_bsr_b4_v2(bship.bases, bship.states, bship.size,
                                                 bdisc.bases, bdisc.states, bdisc.size,
                                                 r1.bases, r1.states);
            int n2 = intersect_qfilter_bsr_b4_v2(r1.bases, r1.states, n1,
                                                 bqty.bases, bqty.states, bqty.size,
                                                 r2.bases, r2.states);
            auto t1 = clk::now();
            (void)n2;
            if (it) t_bsr.push_back(ms(t0, t1));
        }
        std::cout << "BSR qfilter x2:   " << med(t_bsr) << "  (Q6 observed: 12.8)\n";

        // CR AND
        roaring::Roaring rship, rdisc, rqty;
        rship.addMany(pos_sets[0].size(), pos_sets[0].data());
        rdisc.addMany(pos_sets[1].size(), pos_sets[1].data());
        rqty.addMany(pos_sets[2].size(), pos_sets[2].data());
        std::vector<double> t_cr;
        for (int it = 0; it < ITER + 1; it++) {
            auto t0 = clk::now();
            roaring::Roaring x = rship & rdisc;
            roaring::Roaring y = x & rqty;
            auto t1 = clk::now();
            (void)y;
            if (it) t_cr.push_back(ms(t0, t1));
        }
        std::cout << "CR & x2:          " << med(t_cr) << "  (Q6 observed: 12.9)\n";
    }
    return 0;
}
