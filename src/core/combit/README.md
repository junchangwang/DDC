# ComBit — Compressed Bitvector with AVX-512 Acceleration

## Overview

ComBit is a two-level compressed bitvector library:

- **`ComBitBtv<WordSize>`** — a fixed-length bitvector segment compressed with
  separated leading bits and literal words.
- **`ComBit`** — a segmented bitvector composed of independently compressed
  `ComBitBtv` segments (default 2^16 bits each).

### Compression scheme

Each `ComBitBtv` segment stores two arrays:

1. **Leading bitstring** — one bit per word; `1` = fill (all-zeros or all-ones),
   `0` = literal.
2. **Literal data array** — the literal word values, stored sequentially.

Fill words are **virtual** — they are never stored.  Only one fill value
(all-0 or all-1) is used per segment, chosen by the `fill_ones` parameter.

### Example (WordSize = 8)

```
Original:  00000000 00000000 00001000 00000000 00000000 00000001

Leading bits:    1 1 0 1 1 0
Literal words:   [00001000, 00000001]
```

### Key features

- **Tunable word size** — 8, 16, 32, or 64 bits (template parameter `WordSize`).
- **Segmented design** — `ComBit` partitions data into fixed-size segments; each
  segment can use a different word size and fill_ones setting.
- **AVX-512 accelerated bitwise ops** — AND, OR, and XOR use
  `_mm512_mask_expandloadu` + SIMD intrinsics (`_mm512_and_si512`,
  `_mm512_or_si512`, `_mm512_xor_si512`) to operate directly on compressed data
  without full decompression.
- **Cross-word-size operations** — `cross_and`, `cross_or`, `cross_xor` combine
  segments with different word sizes, always producing `ComBitBtv<64>` output.
- **Single fill type** — eliminates one bit of overhead per fill entry.
- **Separated arrays** — leading bits and literal data are stored in independent
  arrays for cache-friendly SIMD access.

## Build

Requires a compiler with C++17 and AVX-512 VBMI2 support (GCC 8+, Clang 7+).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Two targets are produced:

| Target | Description |
|--------|-------------|
| `combit_eval` | Release build for benchmarks |
| `combit_debug` | Same code with `COMBIT_DEBUG` timing output per operation |

## Run

```bash
./combit_eval
```

The evaluation program runs:

1. **Correctness tests** — verifies round-trip encoding for `ComBitBtv` and
   segmented `ComBit`, bitwise operations (AND, OR, XOR, NOT), and popcount.
2. **Segmented correctness tests** — validates `ComBit` segment-wise operations
   across word sizes.
3. **Compression ratio analysis** — measures compression across different bit
   densities, distributions (uniform / clustered), and word sizes.
4. **AND performance benchmarks** — same-WS, cross-WS, and segmented AND timing
   vs. uncompressed baseline.
5. **OR performance benchmarks** — same-WS, cross-WS, and segmented OR timing.
6. **XOR performance benchmarks** — same-WS, cross-WS, and segmented XOR timing.

## Project Structure

```
combit_new/
├── CMakeLists.txt          Build configuration (combit_eval + combit_debug)
├── README.md               This file
├── main.cpp                Evaluation driver (tests + benchmarks)
├── include/
│   └── combit.h            ComBitBtv<WordSize> template + ComBit segmented class
└── src/
    ├── combit.cpp           Core methods (compress, decompress, NOT, popcount, …)
    ├── and.cpp              AVX-512 AND: operator&, cross_and, ComBit::operator&
    ├── or.cpp               AVX-512 OR:  operator|, cross_or,  ComBit::operator|
    └── xor.cpp              AVX-512 XOR: operator^, cross_xor, ComBit::operator^
```

## API Quick Reference

### ComBitBtv (single segment)

```cpp
#include "combit.h"

// Compress from raw bits (WordSize = 8, 16, 32, or 64)
auto seg = ComBitBtv<8>::compress(bitvector);   // std::vector<bool>
auto seg = ComBitBtv<8>::from_string("00001000 11111111 00000000");

// Decompress
std::vector<bool> bits = seg.decompress();
std::string s = seg.to_string();

// Bitwise operations (all return ComBitBtv<64>)
auto r = seg1 & seg2;   // AND
auto r = seg1 | seg2;   // OR
auto r = seg1 ^ seg2;   // XOR
auto r = ~seg1;          // NOT (returns same WordSize)

// Cross-word-size operations (return ComBitBtv<64>)
auto r = cross_and(seg8, seg32);
auto r = cross_or(seg16, seg64);
auto r = cross_xor(seg8, seg64);

// Queries
size_t n = seg.popcount();
auto positions = seg.set_bit_positions();

// Statistics
double ratio = seg.compression_ratio();
auto breakdown = seg.size_breakdown();
seg.print();
```

### ComBit (segmented)

```cpp
// Compress with uniform word size across all segments
auto cb = ComBit::compress<8>(bitvector);
auto cb = ComBit::compress<32>(bitvector, /*fill_ones=*/false, /*segment_bits=*/65536);

// Decompress
std::vector<bool> bits = cb.decompress();

// Bitwise operations (segment-wise, via cross_and/cross_or/cross_xor)
auto r = cb1 & cb2;
auto r = cb1 | cb2;
auto r = cb1 ^ cb2;
auto r = ~cb1;

// Queries
size_t n = cb.popcount();
size_t segs = cb.num_segments();
```
