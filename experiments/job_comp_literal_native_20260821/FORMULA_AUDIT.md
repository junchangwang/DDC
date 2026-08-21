# Cross-experiment COMP formula audit

## Cross-backend plan mismatch

The following frozen figures compare a DDC-only optimized plan
`~(B | (A & C))` with the other backends' literal plan
`~((A | B) & (B | C))`:

- Main-paper Figure 9d and the revision's BSR/HBI copies of its COMP curves.
- The frozen Earth COMP figure and the later native-CR sensitivity that reused
  the frozen non-CR baselines.
- The frozen JOB COMP figure and the native-CR sensitivity that reused the
  frozen non-CR baselines.

The old DDC path also forced compressed intermediates, so its effect cannot be
attributed to the rewrite alone. These figures are not equal-plan kernel
comparisons and require a fresh all-backend rerun before making such a claim.

## Unified literal plan

- Current `src/benchmark/benchmark_main.cpp` executes the literal four-call
  expression for DDC, CRoaring, WAH, and EWAH.
- The current paper-facing CF/clustering operation sweep uses that literal
  plan for all four backends.
- This experiment (`job_comp_literal_native_20260821`) uses that literal plan
  and fresh measurements for all four backends.
- Density heatmaps measure only OR and are not affected by COMP rewriting.

## DDC-only ablations

The segment-size and bypass COMP ablations still use the optimized three-call
plan. They compare only DDC configurations under the same plan, so the
within-DDC trend is not a cross-backend advantage. Their captions should not
describe the measured operation as a literal four-call COMP.

## End-to-end queries

The templated JOB query implementation uses one high-level `or_many`,
`or_single`, and `and_inplace` plan for every backend. TPC-H has no synthetic
COMP benchmark in its end-to-end query path. No equivalent DDC-only Boolean
formula shortening was found in these real-query paths.

## Separate CF data issue

This is unrelated to formula rewriting: the paper-facing
`eva/CF/clustering_ops_crrun.dat` records the CF1 approximately 1 WAH NOT point
as `299.6715 op/s`, while its declared source
`operation_sweep_crrun.csv` records `1495.6715 op/s`, an exact fivefold
difference. The frozen paper data point requires separate correction.
