## DDC: Decoupling Data and Control for Bitmap Compression

### Motivation

Bitmap indexing is increasingly pushed into the core of analytical DBMSs [1,2,3], where a query is evaluated by merging precomputed bitvectors with bitwise operations (AND/OR/NOT).
This line of work shows that bitmap indexing is promising for analytical workloads, but it also reveals one critical performance bottleneck: bitvectors are usually compressed to save storage, but bitwise operations on the compressed form are very slow.

The root cause is that existing bitvector compression schemes (WAH, EWAH, CONCISE, CRoaring, and others) target sparse bitvectors.
WAH, for example, was motivated by physical-energy use cases where the indexed attribute has very high cardinality (millions of distinct values), producing sparse bitvectors.
Analytical workloads such as TPC-H instead produce bitvectors with moderate-to-dense bit densities for two reasons.
First, attributes typically have small cardinality (fewer than a thousand distinct values).
Second, encoding schemes such as Range Encoding and Group Encoding [3] generate and rely on dense bitvectors to accelerate queries.
On these dense bitvectors, traditional compression schemes are very slow and become a severe bottleneck for bitmap indexing.
A bitmap-indexed plan can end up slower than a native columnar scan.

To address this critical performance issue in using bitmap indexing for analytical DBMS, we propose **DDC**, a bitvector compression mechanism that **D**ecouples **D**ata and **C**ontrol bits to deliver high compression while sustaining efficient bitwise operations across the full density range.
It (1) separates control flags from data payloads so bitwise operations map directly onto AVX-512 expand-load/compress-store, (2) recursively compresses the control flags into a multi-layer hierarchy adapted to the bit density, and (3) reuses that hierarchy as a coarse-to-fine skip index that bypasses fill regions with simple mask tests.
DDC eliminates the throughput-degradation window that WAH, EWAH, CONCISE, and Roaring all exhibit, while remaining the smallest at moderate densities.


- [1] UpBit: Scalable In-Memory Updatable Bitmap Indexing. In SIGMOD'16
- [2] CUBIT: Concurrent Updatable Bitmap Indexing. In VLDB'25
- [3] RABIT: Efficient Range Queries with Bitmap Indexing. In SIGMOD'26


### How is this project organized?

This repository contains our implementation of DDC and state-of-the-art baseline compression algorithms (including WAH, EWAH, CONCISE, and CRoaring), along with a microbenchmark framework that evaluates these algorithms.
The bitmap index-powered DuckDB implementation using these compression mechanisms is in another repository (see *DuckDB integration* below).

- `src/core/ddc/`: the DDC compression library (a git submodule; see its own README). `DDCBtv` is one compressed segment (L1 literal bytes / L2 word-control bits / L3 byte-control bits / L4 top-control bits). `DDC` is the whole bitvector. `DDCN` is the depth-parameterised variant (L2-L5) used by the hierarchy-depth study.
- `src/core/{croaring,fastbit,ewah,Concise}/`: unmodified third-party baseline libraries.
- `src/benchmark/`: the micro-benchmark framework and the baseline backends (WAH, EWAH, CONCISE, Roaring, bitset) under `backends/`. Each experiment in the paper has a dedicated driver.
- `util/`: dataset and bitmap generation (`gen_bitmap`, `gen_dataset.sh`, `gen_test_bitmaps.sh`).

### How to build and run the experiments?

DDC requires C++17, CMake, the Boost library, and a CPU/compiler with AVX-512
(AVX-512F, BW, VBMI2). Build everything with:

```sh
./build.sh                                 # CMake Release -> build/
```

Run the synthetic experiments on a generated bitmap directory (see *Data preparation*).
The hierarchy-depth, segment-size, and bypass evaluations accept parameters in the form of `<bitmap_root> <out_csv>`:

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

### Data preparation

By default, our evaluation uses bitmaps with 100M rows, varying bit density by setting different cardinalities in a 100M-integer array.
The generated bitmaps are placed under `bitmap/`. For example:

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

### DuckDB integration

The end-to-end TPC-H evaluation inside DuckDB using bitmap indexing can be found at https://github.com/junchangwang/duckdb-DDC .

### Reference

DDC: Decoupling Data and Control for Bitmap Compression. (under
submission).
