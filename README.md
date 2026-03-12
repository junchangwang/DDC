<!-- filepath: /home/junchang/repos/projects/combit-bitmap/combit-bitmap-index/README.md -->
# ComBit Bitmap Index

A standalone C++ implementation of the **Compressed Bitvector (ComBit) bitmap index design** to be used as secondary indexes in databases.

## Overview

ComBit compresses bitmap vectors into two sections:

- **Header section**: One bit per content word. `1` = compressed (fill) word, `0` = literal word.
- **Content section**: The actual words. A compressed word's MSB indicates whether it compresses `0`s or `1`s; the remaining bits store the run length (in words).

### Example (8-bit words)

Uncompressed:
```
00000000 00000000 01000000 11111111 11111111 11111111
```

ComBit compressed:
```
header:   101
content:  00000010 01000000 10000011
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

```bash
./combit_demo       # Run the demo
ctest            # Run the tests
```

## Components

| File | Description |
|---|---|
| `types.hpp/cpp` | Core types: `Word`, `ComBitEncoding`, helper functions |
| `combit_encoder.hpp/cpp` | Compresses uncompressed bitmaps into ComBit format |
| `combit_decoder.hpp/cpp` | Decompresses ComBit back to uncompressed bitmaps |
| `bitmap_vector.hpp/cpp` | A single bitmap vector with set/get/encode/load |
| `bitmap_index.hpp` | Template bitmap index mapping keys → bitmap vectors |

## Word Size

The default word size is **16 bits** (`uint16_t`), matching the PostgreSQL reference. Change the `Word` typedef in `types.hpp` to adjust.# combit
