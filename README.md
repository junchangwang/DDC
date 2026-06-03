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
