# DDC: Decoupling Data and Control for Bitmap Compression

## Motivation

Bitmap indexing accelerates analytical workloads by replacing storage scans
with bitwise operations over precomputed bitvectors. To control storage, the
bitvectors are compressed with RLE-based (WAH, EWAH, CONCISE) or hybrid
(Roaring) mechanisms. However, at the *moderate-to-dense* bit densities
(0.1%–50%) where bitmap indexes for analytical workloads actually operate —
e.g., the mid-range bitvectors produced by range and group encoding — bitwise
operations on these compressed forms become extremely slow, often slower than
merging two uncompressed bitvectors.

**DDC** is a bitvector compression mechanism that **D**ecouples **D**ata and
**C**ontrol bits to deliver high compression while sustaining efficient bitwise
operations across the full density range. It (1) separates control flags from
data payloads so bitwise operations map directly onto AVX-512 expand-load /
compress-store, (2) recursively compresses the control flags into a multi-layer
hierarchy adapted to the bit density, and (3) reuses that hierarchy as a
coarse-to-fine skip index that bypasses fill regions with simple mask tests.
DDC eliminates the throughput-degradation window that WAH, EWAH, CONCISE, and
Roaring all exhibit, while remaining the smallest at moderate densities.

This repository holds the standalone DDC library and the synthetic
micro-benchmarks. The end-to-end DuckDB integration lives in a separate
repository (see *DuckDB integration* below).

## How is this project organized?

- `src/core/ddc/` — the DDC compression library (a git submodule; see its own
  README). `DDCBtv` is one compressed segment (L1 literal bytes / L2 word-marker
  bits / L3 byte-marker bits / L4 top-marker bits); `DDC` is the segmented
  bitvector; `DDCN` is the depth-parameterised variant (L2–L5) used by the
  hierarchy-depth study.
- `src/benchmark/` — the micro-benchmark harness and the baseline backends
  (WAH, EWAH, CONCISE, Roaring, bitset) under `backends/`. Each experiment in
  the paper has a dedicated driver:
  - `benchmark_main.cpp` (`benchmark_app`) — OR/AND/NOT/COMP throughput vs. bit
    density, and compressed size (Fig. 8–9).
  - `bench_L_ops.cpp` / `bench_L_sizes.cpp` — hierarchy depth L2–L5: throughput
    and size (Fig. 10).
  - `bench_seg.cpp` — segment-size ablation (Fig. 11).
  - `bench_bypass.cpp` — skip-index bypass ablation (Fig. 12).
- `util/` — dataset and bitmap generation (`gen_bitmap`, `gen_dataset.sh`,
  `gen_test_bitmaps.sh`).
- `src/core/{croaring,fastbit,ewah,Concise}/` — unmodified third-party baseline
  libraries.

## How to build and run the experiments?

DDC requires C++17, CMake, the Boost library, and a CPU/compiler with AVX-512
(AVX-512F, BW, VBMI2). Build everything with:

```sh
git submodule update --init --recursive   # fetch the DDC core (src/core/ddc)
./build.sh                                 # CMake Release -> build/
```

Run the synthetic experiments on a generated bitmap root (see *Data
preparation*). The depth / segment / bypass drivers take `<bitmap_root> <out_csv>`:

```sh
# OR/AND/NOT/COMP throughput + size, all backends (Fig. 8-9)
./build/benchmark_app --backend all --bm-dir bitmap --num-rows 100000000 --csv results.csv

# Hierarchy depth L2-L5 (Fig. 10): throughput + size
./build/bench_L_ops   bitmap  bench_L_ops.csv
./build/bench_L_sizes bitmap  bench_L_sizes.csv

# Segment-size ablation (Fig. 11)
./build/bench_seg     bitmap  bench_seg.csv

# Skip-index bypass ablation (Fig. 12)
./build/bench_bypass  bitmap  bench_bypass.csv
```

## Data preparation

The experiments run over 100M-row synthetic bitmaps spanning the full density
range (cardinality 2–2000, uniform distribution — the worst case for
compression). Generate the per-scheme bitmap directories with `gen_bitmap`:

```sh
# one bitmap set per cardinality, per scheme, written under ./bitmap/
for c in 2 5 10 20 50 100 200 500 1000 2000; do
  ./build/gen_bitmap -n 100000000 -c $c ddc     -d .   # -> bitmap/bm_100m_c<c>_ddc_w8/
  ./build/gen_bitmap -n 100000000 -c $c wah     -d .
  ./build/gen_bitmap -n 100000000 -c $c roaring -d .
  ./build/gen_bitmap -n 100000000 -c $c ewah    -d .
  ./build/gen_bitmap -n 100000000 -c $c concise -d .
done
```

`gen_bitmap` options: `-c` cardinality (column-style disjoint bitmaps), `-t`/`-o`
the transition / overlap stress sets, `-w` DDC word size, `-L` DDCN depth (2–5),
`-S` DDC segment size. Generating the full 100M set takes a while; a small
correctness set can be produced with `util/gen_test_bitmaps.sh`.

## DuckDB integration

The end-to-end TPC-H evaluation inside DuckDB is in the companion DuckDB
repository: the `extension/debit` extension swaps the underlying compression
mechanism under a shared bitmap-operator harness. See that repository's README.

## Reference

DDC: Decoupling Data and Control for Bitmap Compression. *PVLDB* (under
submission).
