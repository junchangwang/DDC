## DDC: Decoupling Data and Control for Bitmap Compression

### Motivation

Bitmap indexing is increasingly pushed into the core of analytical DBMSs [1,2,3], where a query is evaluated by merging precomputed bitvectors with bitwise operations (AND/OR/NOT).
This line of work shows that bitmap indexing is promising for analytical workloads, but it also reveals one critical performance bottleneck: **bitvectors are usually compressed to save storage, but bitwise operations on the compressed form are very slow**.

The root cause is that existing bitvector compression schemes (WAH, EWAH, CONCISE, CRoaring, and others) target sparse bitvectors.
WAH, for example, was motivated by physical-energy use cases where the indexed attribute has very high cardinality (millions of distinct values), producing sparse bitvectors.
Analytical workloads such as TPC-H instead produce bitvectors with moderate-to-dense bit densities for two reasons.
First, attributes typically have small cardinality (fewer than a thousand distinct values).
Second, encoding schemes such as Range Encoding and Group Encoding [3] generate and rely on dense bitvectors to accelerate queries.
On these bitvectors, traditional compression schemes are very slow and become a severe bottleneck for bitmap indexing.
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

- `src/core/ddc/`: the DDC library (see `src/core/ddc/README.md`). `DDCBtv` is one segment (L1 literal bytes / L2 word flags / L3 byte flags / L4 top flags); `DDC` is the segmented bitvector; `DDCN` is the depth-parameterised variant (L2–L5) used by the hierarchy-depth study.
- `src/core/{croaring,fastbit,ewah,Concise}/`: third-party baseline libraries.
- `src/benchmark/`: the benchmark drivers and baseline backends (under `backends/`).
- `util/`: dataset and bitmap generation (`gen_bitmap`, `gen_dataset.sh`, `gen_test_bitmaps.sh`).

### Build

Requires C++17, CMake, Boost, and a CPU/compiler with AVX-512 (F, BW, VBMI2).

```sh
./build.sh        # CMake Release; binaries in build/
```

### Data preparation

Experiments use 100M-row bitmaps; bit density is set by the cardinality. All bitmaps are written under `bitmap/`.

```sh
# Fig. 8-9: every scheme, cardinalities 2..2000
for c in 2 5 10 20 50 100 200 500 1000 2000; do
  for s in ddc wah roaring ewah concise; do
    ./build/gen_bitmap -n 100000000 -c $c $s -d .
  done
done
# Fig. 10: DDC at hierarchy depths L2-L5 (the size sweep extends to higher cardinalities)
for c in 2 5 10 20 50 100 200 500 1000 2000 3000 5000 10000 20000 50000; do
  for L in 2 3 4 5; do ./build/gen_bitmap -n 100000000 -c $c ddc -L $L -d .; done
done
# Fig. 11: DDC at segment sizes 2^12,2^14,2^18 (2^16 = the default _ddc_w8 above)
for c in 10 100 1000; do
  for S in 4096 16384 262144; do ./build/gen_bitmap -n 100000000 -c $c ddc -S $S -d .; done
done
```

Directory names: `bm_100m_c<c>_<scheme>` (DDC is `bm_100m_c<c>_ddc_w8`), depth variants `…_ddc_L<2-5>`, segment variants `…_ddc_w8_S<bits>`. Generation also writes the raw uncompressed bitmaps under `bitmap/bitmaps_100m_c*/` (tens of GB); delete them once the compressed sets exist with `util/del_bitmap.sh`.

### Running the experiments

`benchmark_app` takes one scheme and one bitmap directory per run (the scheme must match the directory format).

```sh
# Fig. 8-9: size + OR/AND/NOT/COMP throughput (one run per scheme directory; backend auto-detected)
for d in bitmap/bm_100m_c*_ddc_w8 bitmap/bm_100m_c*_{wah,roaring,ewah,concise}; do
  ./build/benchmark_app --backend all --compressed-dir "$d" --num-rows 100000000 --csv results.csv
done

# Fig. 10: hierarchy depth L2-L5
./build/bench_L_ops   bitmap bench_L_ops.csv     # throughput
./build/bench_L_sizes bitmap bench_L_sizes.csv   # size

# Fig. 11: segment size
./build/bench_seg     bitmap bench_seg.csv

# Fig. 12: skip-index bypass
./build/bench_bypass  bitmap bench_bypass.csv
```

The `bench_*` drivers scan `bitmap/` for the variants generated above and write one CSV each.

### DuckDB integration

End-to-end TPC-H with bitmap indexing: https://github.com/junchangwang/duckdb-DDC

### Reference

DDC: Decoupling Data and Control for Bitmap Compression (under submission).
