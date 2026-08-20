# Native CRoaring sensitivity experiment (2026-08-20)

This branch is an isolated, reversible sensitivity study. It does not modify
the paper or overwrite any frozen CSV, bitmap, workbook, or figure.

## One changed variable

The paper-facing measurements charge CRoaring dense-bitvector delivery in
`OR_op_conv`, `AND_op_conv`, `NOT_op_conv`, and the historical `COMP_op`.
This experiment instead selects CRoaring's native `OR_op`, `AND_op`, and
`NOT_op`, and adds a new `COMP_op_native` row. Existing delivered rows are
preserved unchanged.

The result is a backend-native sensitivity, not a common output contract:
CRoaring, WAH, EWAH, and DDC keep their own result representations.

## Reproducibility boundary

- All input bitmap directories are read-only and live in the parent workspace.
- `run_native_cr.py` remeasures CRoaring in fresh processes and records both
  native and delivered anchors from the same binary.
- Earth, JOB, and clustering use three process replicates. The 55-point density
  grid uses one fresh process per point, matching its historical process-level
  replication.
- `run_table4_cr_sizes.py` remeasures both plain and run-optimized CRoaring
  native payload sizes from the ten frozen JOB membership files.
- Raw CSV and logs remain local and are hashed by the analysis step; compact
  result CSVs and figures are retained on this branch.

## Important limitations

- JOB and Earth historical DDC COMP used an equivalent shorter execution plan;
  native CR results must not be described as an equal-plan kernel comparison
  unless all backends are remeasured with the same literal plan.
- Clustering retains its historical self-AND convention so this sensitivity
  changes only the CR delivery boundary.
- Density inputs call `runOptimize()`, but their random containers normally
  remain array/bitset rather than run containers.

## Commands

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build build-native --target benchmark_app -j16

python3 experiments/cr_native_no_btv_20260820/run_table4_cr_sizes.py
python3 experiments/cr_native_no_btv_20260820/run_native_cr.py \
  --groups earth job cluster density --reps 3 --density-reps 1 --cpu 2
```

## Measured results

- Table 4 CRoaring run payload: 218.628477 MiB, 25% above DDC; the
  non-run CRoaring total was 291.233639 MiB.
- Earth OR+COMP CRoaring/DDC geometric mean: 1.1375x after removing dense
  delivery, versus 2.0635x in the selected-delivery figure. CRoaring wins
  5 of the 10 OR+COMP cells under the native boundary.
- JOB COMP CRoaring/DDC geometric mean: 7.4602x, versus 23.6225x with dense
  delivery. This remains a sensitivity result because the frozen DDC baseline
  used its shorter equivalent COMP plan.
- Clustering all-operation CRoaring/DDC geometric mean: 0.8137x across the
  complete 11-point sweep.
- Density-grid OR CRoaring/DDC geometric mean: 1.3730x.

Compact results are in `results/`; the 12 PDF and 12 PNG visualizations are in
`figures/`. Formal raw CSVs and logs are retained on this branch. The aborted
outer-iteration mismatch is kept locally outside Git tracking and is never read
by the analyzer.
