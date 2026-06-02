# DDC-core: Decoupling Data and Control for Bitmap Compression

## Motivation

DDC-core is the standalone bitvector-compression library behind **DDC**. Prior
schemes interleave control flags with data words in a single stream, forcing a
branch-heavy dual-cursor merge that collapses at the moderate-to-dense bit
densities where analytical bitmap indexes operate. DDC instead **D**ecouples
**D**ata and **C**ontrol: it stores literal payloads and control flags in
separate arrays, recursively compresses the control flags into a multi-layer
hierarchy, and reuses that hierarchy as a skip index. The decoupled layout maps
each bitwise operation onto a branch-free AVX-512 expand-load / SIMD-op /
compress-store kernel that runs at density-independent throughput.

This is the core library only; it is consumed as a submodule by the DDC project
(synthetic micro-benchmarks) and by the DuckDB integration.

## How is this project organized?

The library is four-level. `DDCBtv` is one compressed segment — **L1** literal
bytes (8-bit words), **L2** one marker bit per word, **L3** one marker bit per
L2 byte, **L4** one marker bit per L3 byte; fill words and fill marker bytes are
virtual (never stored) and each layer picks the fill polarity that minimises
literals. `DDC` is a segmented bitvector of independently-compressed `DDCBtv`
segments (default 2^16 bits each). `DDCN` is the depth-parameterised variant
(L2–L5) used by the hierarchy-depth study.

```
include/ddc.h          DDCBtv, DDC, SparseDDC  (four-level format + public API)
include/ddc_n.h        DDCN — depth-parameterised (L2-L5) variant + bypass cfg
include/ddc_util.h     synthetic bitmap generators + benchmark helpers
src/ddc.cpp            compress / decompress / popcount / serialization
src/and.cpp            AVX-512 AND kernel + skip-index bypass
src/or.cpp / xor.cpp   AVX-512 OR / XOR kernels
src/not.cpp            NOT (metadata flip + SIMD XOR over L1 literals)
src/ddc_n.cpp          DDCN depth-2..5 kernels (hierarchy-depth study)
src/ddc_eval.cpp       standalone correctness + micro-benchmark driver
CMakeLists.txt         standalone build (also builds when used as a submodule)
```

## How to build and run?

Requires C++17 and a CPU/compiler with AVX-512 VBMI2 (GCC 8+/Clang 7+); the
CMake config probes for it and falls back to scalar paths if absent.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/ddc_eval
```

`ddc_eval` runs (1) `DDCBtv` correctness (round-trip, AND/OR/XOR/NOT, popcount),
(2) segmented `DDC` correctness, (3) a compression-ratio analysis with an
L1/L2/L3/L4 size breakdown, and (4) AND/OR/XOR timing vs. an uncompressed
baseline.

## Data preparation

The standalone `ddc_eval` needs no external data — it generates the synthetic
bitvectors it tests/benchmarks internally (via `ddc_util.h`), varying density
and clustering. The full paper micro-benchmarks (throughput/size vs. density,
hierarchy depth, segment size, bypass — Fig. 8–12) live in the parent **DDC**
project, which links this library and reads 100M-row bitmap sets produced by its
`gen_bitmap` tool.

## Reference

DDC: Decoupling Data and Control for Bitmap Compression. *PVLDB* (under
submission).
