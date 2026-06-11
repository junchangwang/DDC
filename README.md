## DDC: Decoupling Data and Control for Bitmap Compression

### Motivation

Bitmap indexing is increasingly pushed into the core of analytical DBMSs [1,2,3], where a query is evaluated by merging precomputed bitvectors with bitwise operations (AND/OR/NOT).
This line of work shows that bitmap indexing is promising for analytical workloads, but it also reveals one critical performance bottleneck: **bitvectors are usually compressed to save storage, but bitwise operations on the compressed form are very slow**.

The root cause is that existing bitvector compression schemes (WAH, EWAH, CONCISE, CRoaring, and others) target sparse bitvectors.
WAH, for example, was motivated by physical-energy use cases where the indexed attribute has very high cardinality (millions of distinct values), producing sparse bitvectors.
Analytical workloads such as TPC-H instead produce bitvectors with moderate-to-dense bit densities for two reasons.
First, attributes typically have small cardinality (fewer than a thousand distinct values).
Second, encoding schemes such as Range Encoding and Group Encoding [2] generate and rely on dense bitvectors to accelerate queries.
On these bitvectors, traditional compression schemes are very slow and become a severe bottleneck for bitmap indexing.
A bitmap-indexed plan can end up slower than a native columnar scan.

To address this critical performance issue in using bitmap indexing for analytical DBMS, we propose **DDC**, a bitvector compression mechanism that **D**ecouples **D**ata and **C**ontrol bits to deliver high compression while sustaining efficient bitwise operations across the full density range.
It (1) separates control flags from data payloads so bitwise operations map directly onto AVX-512 expand-load/compress-store, (2) recursively compresses the control flags into a multi-layer hierarchy adapted to the bit density, and (3) reuses that hierarchy as a coarse-to-fine skip index that bypasses fill regions with simple mask tests.
DDC eliminates the throughput-degradation window that WAH, EWAH, CONCISE, and Roaring all exhibit, while remaining the smallest at moderate densities.


- [1] CUBIT: Concurrent Updatable Bitmap Indexing. In VLDB'25
- [2] RABIT: Efficient Range Queries with Bitmap Indexing. In SIGMOD'26
- [3] BitEngine: Query Engines Using Bitmap Indexing. [Preprint](https://github.com/junchangwang/DDC/blob/main/doc/BitEngine.pdf)


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

The experiments run on pre-built bitmaps in the format of `.bm`. 
`gen_bitmap` builds them automatically, and generating the full set for every encoding scheme and cardinality.
To build the complete bitmap set (every scheme, cardinalities 2 to 2000), run the following command.
All bitmaps will be written to `bitmap/`.

```sh
for c in 2 5 10 20 50 100 200 500 1000 2000; do
  for s in ddc wah roaring ewah concise; do
    ./build/gen_bitmap -n 100000000 -c $c $s -d .
  done
done
```

Notet that building the whole bitmap set would take a long time, so we also provide a pre-generated set for evaluating DDC. Please check it at https://1drv.ms/u/c/4f1a15e54aa03b95/IQAF6kOEX0QuSozMCcHo5N67AbvZOqjlNJprn80V2F74mvw?e=nKIVsl


### Running the experiments

`benchmark_app` takes one scheme and one bitmap directory per run, and the scheme must match the directory format.

```sh
# Fig. 8-9 size and OR/AND/NOT/COMP throughput (one run per scheme directory, backend auto-detected)
for d in bitmap/bm_100m_c*_ddc_w8 bitmap/bm_100m_c*_{wah,roaring,ewah,concise}; do
  ./build/benchmark_app --backend all --compressed-dir "$d" --num-rows 100000000 --csv results.csv
done

# Fig. 10 hierarchy depth L2-L5 (throughput and size)
./build/bench_L_ops   bitmap bench_L_ops.csv
./build/bench_L_sizes bitmap bench_L_sizes.csv

# Fig. 11 segment size
./build/bench_seg     bitmap bench_seg.csv

# Fig. 12 skip-index bypass
./build/bench_bypass  bitmap bench_bypass.csv
```

The `bench_*` drivers scan `bitmap/` for the variants generated above and write one CSV each.

### DuckDB integration

End-to-end TPC-H with bitmap indexing: https://github.com/junchangwang/duckdb-DDC

### Reference

DDC: Decoupling Data and Control for Bitmap Compression (under submission).
