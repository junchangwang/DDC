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
