// Density x density OR grid for DDC (teacher, 2026-08-01).
// 100M bits per bitvector; densities 0.1..50%; D2 caliber = tight-loop
// N_ITER median per pair; harness default kernel mode
// (ddc_compress_results=false); v3 run-bypass at its default (ON — a no-op
// here: uniform scatter leaves no empty/full segment at any grid density).
// Internal consistency per pair: |A|B| == |A|+|B|-|A&B| and |A| == generated
// set count.  Emits CSV with OR/AND/XOR medians.
#include "ddc.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

static double med(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    const size_t N = 100000000;
    const int N_ITER = 100;
    const std::vector<double> dens = {0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50};
    ddc_compress_results = false;                 // harness default mode

    // TWO independent families per density (X = row operand, Y = column
    // operand) so the diagonal is a genuine same-density OR of independent
    // bitvectors, not a cache-hot self-OR (v1 artifact, caught by review).
    std::vector<DDC> X, Y;
    std::vector<size_t> popX, popY;
    for (int fam = 0; fam < 2; fam++) {
        for (size_t i = 0; i < dens.size(); i++) {
            std::mt19937_64 rng((fam ? 730260801ull : 920260801ull) + i);
            std::bernoulli_distribution coin(dens[i] / 100.0);
            std::vector<bool> bits(N);
            size_t p = 0;
            for (size_t j = 0; j < N; j++) { bool b = coin(rng); bits[j] = b; p += b; }
            (fam ? Y : X).push_back(DDC::compress(bits));
            (fam ? popY : popX).push_back(p);
            if ((fam ? Y : X).back().popcount() != p) { std::printf("GEN pop mismatch d=%g\n", dens[i]); return 1; }
            std::printf("gen %c d=%.1f%% pop=%zu (%.4f%%)\n", fam ? 'Y' : 'X', dens[i], p, 100.0 * p / N);
        }
    }

    FILE* f = fopen("realgrid_or_100m.csv", "w");
    fprintf(f, "dA_pct,dB_pct,or_ms,and_ms,xor_ms,popA,popB,popOR,identity_ok\n");
    int bad = 0;
    for (size_t i = 0; i < dens.size(); i++) {
        for (size_t j = 0; j < dens.size(); j++) {
            const DDC &A = X[i], &B = Y[j];
            { DDC w = A | B; (void)w; DDC x = A & B; (void)x; DDC y = A ^ B; (void)y; }
            std::vector<double> to, ta, tx;
            for (int it = 0; it < N_ITER; it++) {
                auto t0 = std::chrono::steady_clock::now();
                DDC r = A | B;
                to.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count());
                asm volatile("" :: "r"(&r) : "memory");
            }
            for (int it = 0; it < N_ITER; it++) {
                auto t0 = std::chrono::steady_clock::now();
                DDC r = A & B;
                ta.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count());
                asm volatile("" :: "r"(&r) : "memory");
            }
            for (int it = 0; it < N_ITER; it++) {
                auto t0 = std::chrono::steady_clock::now();
                DDC r = A ^ B;
                tx.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count());
                asm volatile("" :: "r"(&r) : "memory");
            }
            DDC ror = A | B, rand_ = A & B;
            const size_t po = ror.popcount(), pa = rand_.popcount();
            const bool ok = (po == popX[i] + popY[j] - pa);
            if (!ok) bad++;
            fprintf(f, "%g,%g,%.6f,%.6f,%.6f,%zu,%zu,%zu,%d\n",
                    dens[i], dens[j], med(to), med(ta), med(tx), popX[i], popY[j], po, (int)ok);
            std::printf("d %4.1f|%4.1f  OR %8.3fms  AND %8.3fms  XOR %8.3fms  %s\n",
                        dens[i], dens[j], med(to), med(ta), med(tx), ok ? "ok" : "IDENTITY FAIL");
            fflush(f);
        }
    }
    fclose(f);
    std::printf("==== density grid done, identity failures: %d ====\n", bad);
    return bad ? 1 : 0;
}
