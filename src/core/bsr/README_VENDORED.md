# Vendored: QFilter / BSR reference implementation (UNMODIFIED)

Source: https://github.com/Caesar11/GraphSetIntersection
Commit: 05262ba27ca486049205fd9df16519fdf2e2a9cc
Paper:  Han, Zou, Yu. "Speeding Up Set Intersections in Graph Algorithms
        using SIMD Instructions." SIGMOD 2018.

Files copied verbatim from src/: intersection_algos.{hpp,cpp}, util.{hpp,cpp}.
Do NOT edit these files. All adaptation lives in
src/benchmark/backends/bsr/ (combit) and extension/debit/bsr_index.hpp (duckdb).

Notes on the upstream API:
- BSR = two parallel arrays (int* bases, int* states), PACK_WIDTH=32.
- Arrays must be 16-byte aligned with >=4 ints of tail padding
  (kernels use _mm_load_si128 at 4-aligned offsets past qs boundaries and
  _mm_storeu_si128 up to size_c+3 on outputs).
- Upstream provides INTERSECTION (+ subtract) only. No union / NOT.
