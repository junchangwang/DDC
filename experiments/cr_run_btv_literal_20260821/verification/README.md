# Density-pair cross-backend verifier

`verify_density_pair` loads the two serialized bitmaps for one `gridindep`
density pair, computes OR through each backend, and checks all decoded set-bit
positions against BitsetAVX. It reports the backend cardinality and a canonical
FNV-1a hash for `1.bm`, `2.bm`, and the OR result. A passing row also requires
full vector equality; cardinality and hash equality alone are not accepted.

Backends checked:

- DDC
- CRoaring
- WAH
- EWAH
- Concise
- BitsetAVX (reference)

## Build and run

From the repository root:

```bash
cmake -S . -B build-native -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target verify_density_pair -j2
./build-native/verify_density_pair \
  --root ../combit/gridindep/bitmap \
  --case A66_B66
```

The process exits nonzero on a missing directory, malformed decoding,
cardinality mismatch, hash mismatch, or exact position mismatch.

## Smoke test

`A66_B66` passed on 2026-08-21. The captured output is in
`A66_B66_smoke.txt`.
