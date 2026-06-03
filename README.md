## DDC: Decoupling Data and Control for Bitmap Compression

### Motivation

Bitmap indexing accelerates analytical workloads by replacing storage scans with bitwise operations over precomputed bitvectors. The bitvectors are usually compressed with RLE-based (WAH, EWAH, CONCISE) or hybrid (Roaring) schemes. At the moderate-to-dense bit densities (0.1%–50%) where analytical bitmap indexes operate, bitwise operations on these compressed forms become very slow — often slower than merging two uncompressed bitvectors.

**DDC** is a bitvector compression scheme that **D**ecouples **D**ata and **C**ontrol bits. It (1) stores control flags separately from data payloads so bitwise operations map onto AVX-512 expand-load/compress-store, (2) recursively compresses the control flags into a multi-layer hierarchy adapted to the bit density, and (3) reuses that hierarchy as a coarse-to-fine skip index over fill regions. DDC stays the smallest at moderate densities and avoids the throughput-degradation window that WAH, EWAH, CONCISE, and Roaring exhibit.

This repository contains the DDC implementation, the baseline schemes (WAH, EWAH, CONCISE, CRoaring, bitset), and the microbenchmark framework. The DuckDB integration is in a separate repository (see *DuckDB integration*).

### Repository layout

- `src/core/ddc/` — the DDC library (see `src/core/ddc/README.md`). `DDCBtv` is one segment (L1 literal bytes / L2 word flags / L3 byte flags / L4 top flags); `DDC` is the segmented bitvector; `DDCN` is the depth-parameterised variant (L2–L5) used by the hierarchy-depth study.
- `src/core/{croaring,fastbit,ewah,Concise}/` — third-party baseline libraries.
- `src/benchmark/` — the benchmark drivers and baseline backends (under `backends/`).
- `util/` — dataset and bitmap generation (`gen_bitmap`, `gen_dataset.sh`, `gen_test_bitmaps.sh`).

### Build

Requires C++17, CMake, Boost, and a CPU/compiler with AVX-512 (F, BW, VBMI2).

```sh
./build.sh        # CMake Release; binaries in build/
```

### Data preparation

Experiments use 100M-row bitmaps; bit density is set by the cardinality. All bitmaps are written under `bitmap/`.

```sh
for c in 2 5 10 20 50 100 200 500 1000 2000; do
  for s in ddc wah roaring ewah concise; do            # Fig. 8-9 (size + throughput)
    ./build/gen_bitmap -n 100000000 -c $c $s -d .
  done
  for L in 2 3 4 5; do                                 # Fig. 10 (hierarchy depth L2-L5)
    ./build/gen_bitmap -n 100000000 -c $c ddc -L $L -d .
  done
done
for c in 10 100 1000; do                               # Fig. 11 (segment size; 2^16 = default above)
  for S in 4096 16384 262144; do
    ./build/gen_bitmap -n 100000000 -c $c ddc -S $S -d .
  done
done
```

Directory names: `bm_100m_c<c>_<scheme>` (DDC is `bm_100m_c<c>_ddc_w8`), depth variants `…_ddc_L<2-5>`, segment variants `…_ddc_w8_S<bits>`.

### Running the experiments

`benchmark_app` takes one scheme and one bitmap directory per run (the scheme must match the directory format).

```sh
# Fig. 8-9: size + OR/AND/NOT/COMP throughput, all schemes, per cardinality
for c in 2 5 10 20 50 100 200 500 1000 2000; do
  ./build/benchmark_app --backend ddc      --compressed-dir bitmap/bm_100m_c${c}_ddc_w8 --num-rows 100000000 --csv results.csv
  ./build/benchmark_app --backend wah      --compressed-dir bitmap/bm_100m_c${c}_wah     --num-rows 100000000 --csv results.csv
  ./build/benchmark_app --backend croaring --compressed-dir bitmap/bm_100m_c${c}_roaring --num-rows 100000000 --csv results.csv
  ./build/benchmark_app --backend ewah     --compressed-dir bitmap/bm_100m_c${c}_ewah    --num-rows 100000000 --csv results.csv
  ./build/benchmark_app --backend concise  --compressed-dir bitmap/bm_100m_c${c}_concise --num-rows 100000000 --csv results.csv
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
