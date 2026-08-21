# JOB same-literal native COMP rerun (2026-08-21)

This isolated experiment does not modify the paper or any earlier result.

All four backends execute the same literal Boolean DAG:

```text
t1 = A | B
t2 = B | C
t3 = t1 & t2
result = ~t3
```

No backend applies the distributive rewrite `~(B | (A & C))`. CRoaring uses
`COMP_op_native`; DDC, WAH, and EWAH use `COMP_op`. All retain their
backend-native result representation and exclude dense-bitvector delivery.
DDC keeps the benchmark's default result mode (`ddc_compress_results=false`).
Concretely, that default leaves DDC mixed result segments decompressed, while
CRoaring, WAH, and EWAH retain compressed results. This intentionally matches
the existing microbenchmark path requested for this rerun; it is not a common
serialized-output contract.

The operands remain distinct equality bitmaps from one field. Therefore, for
cardinality at least three, the literal four-call execution still has the
data-level identity `result = ~B`; for the two-value company-type field where
the harness reuses `C=A`, the result is `~(A|B)` (empty for this dataset). This
experiment controls the executed formula but is not a general overlapping-COMP
workload.

The current `benchmark_app` performs one untimed COMP warm-up and 100 timed
COMP evaluations per process, then reports their median. The runner uses five
fresh processes per field/backend, pins every process to one CPU, normalizes
the allocator environment, rotates case/backend order, and reports the median
of the five process medians. `--iterations=1` is held constant because that
option controls unrelated outer benchmark work, not the hard-coded 100-sample
COMP loop. Before timing, the runner performs exact decoded-position/hash
verification of all four backends and refuses to run unless the host is at
least 85% idle overall and both SMT threads of the selected core are at least
95% idle.

Run after confirming the host is idle:

```bash
python3 run_job_literal.py --reps 5 --cpu 14
python3 analyze_results.py
python3 plot_results.py
```

The timer stops before final-result destruction in all backends. DDC's local
`t2` intermediate is destroyed inside its helper before the timer stops,
whereas CRoaring, WAH, and EWAH destroy their intermediates afterward. This is
a small DDC-disadvantaging lifecycle asymmetry inherited from the existing
microbenchmark and is disclosed rather than hidden.

## Measured result

The formal run completed 200 fresh processes and 20 exact verification runs.
All decoded OR/AND/NOT/COMP results passed. Geometric-mean latency relative to
DDC is `1.295x` for CRoaring, `0.559x` for WAH, and `0.292x` for EWAH. Winners
are DDC on three fields, CRoaring on one, WAH on one, and EWAH on five.
Against the preceding mixed-plan sensitivity, fresh literal-plan times change
by a geometric mean of `5.755x` for DDC, `0.999x` for CRoaring, `0.942x` for
WAH, and `0.980x` for EWAH. The unchanged CRoaring anchor and large DDC shift
confirm that the old DDC-only path (the rewrite plus forced compressed-result
mode), rather than a CRoaring rerun drift, caused most of the earlier ratio
inflation. This comparison does not isolate the rewrite from that historical
result-mode change.

| Field | DDC (us) | CRoaring (us) | CR/DDC | WAH (us) | EWAH (us) | Winner |
|---|---:|---:|---:|---:|---:|---|
| Company type | 2.980 | 5.270 | 1.768 | 0.777 | 0.241 | EWAH |
| Title kind | 42.987 | 34.179 | 0.795 | 383.655 | 218.348 | CRoaring |
| Cast role | 43.819 | 66.276 | 1.512 | 1.176 | 0.323 | EWAH |
| Person-info type | 84.755 | 178.393 | 2.105 | 1069.300 | 896.248 | DDC |
| Movie-info type | 21.904 | 19.831 | 0.905 | 1.215 | 0.350 | EWAH |
| Production year | 76.980 | 476.434 | 6.189 | 787.322 | 626.418 | DDC |
| Company country | 7.166 | 29.923 | 4.176 | 42.242 | 57.319 | DDC |
| Keyword movie ID | 5.100 | 2.585 | 0.507 | 1.232 | 0.346 | EWAH |
| Cast movie ID | 398.970 | 133.991 | 0.336 | 42.761 | 48.810 | WAH |
| Cast person ID | 49.110 | 36.427 | 0.742 | 2.162 | 0.812 | EWAH |

Compact results are in `results/`; the new figure is
`figures/job_comp_literal_native_relative.pdf`. No earlier figure or paper file
was changed.
