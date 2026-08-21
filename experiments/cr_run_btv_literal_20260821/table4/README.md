# Table 4: run-optimized CRoaring payload sizes

This directory contains a standalone rendering of Table 4. It does not modify
the paper source.

## Data provenance

The CRoaring column is copied from the authoritative display file:

```text
../../cr_native_no_btv_20260820/results/table4_run_display.csv
SHA-256: 0d3ace35cfad208030e8c0837024f5c81d83af976f66687b35beaf8e163df8a3
```

That file reports `getSizeInBytes(false)` after `runOptimize()`. The DDC, WAH,
EWAH, attribute-statistics, and display-rounding columns are retained from the
existing Table 4. `Inc.` is `(baseline / DDC - 1)`; values above 100% are shown
as a multiplier, matching the paper's convention.

Exact totals:

- DDC: 183,360,177.875 bytes = 174.865892 MiB, displayed as 174.9 MiB.
- CRoaring run: 229,248,574 bytes = 218.628477 MiB, displayed as 218.6 MiB.
- CRoaring run relative increase from DDC: 25.026%, displayed as +25%.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error table4_cr_run.tex
```

The output is `table4_cr_run.pdf`.
