# JOB WAH/EWAH BTV-delivery sensitivity (2026-08-21)

This isolated experiment does not overwrite the paper or any earlier result.

## Question

How does the JOB COMP figure change when WAH and EWAH also pay for adaptation
to the current dense `bitset_t` interface?

## Protocol

- All four backends execute literal COMP:
  `t1=A|B; t2=B|C; t3=t1&t2; result=~t3`.
- DDC selects native `COMP_op`.
- CRoaring selects its existing delivered `COMP_op`.
- WAH and EWAH select the new delivered `COMP_op_conv` rows.
- WAH uses a direct 31-bit compressed-word exporter; it does not enumerate set
  positions.
- EWAH uses its compressed `raw_iterator()` and copies runs/dirty words.
- CR, WAH, and EWAH adapter timings include allocation, population, and free.
- Five fresh processes are run for every field/backend cell; each process
  reports the median of 100 inner measurements.

This is an adapter sensitivity, not a claim that all four backends return the
same physical layout. DDC retains its native segmented bitvector result. The
existing CR adapter also retains its historical behavior of shrinking the
logical `arraysize` to the highest set word after initially allocating capacity
for `N` bits.

The operands are same-column equality bitmaps. For fields with at least three
values the logical result is `~B`; for the two-value field it is `~(A|B)` and is
empty. This remains a specialized, NOT-dominated workload rather than a
general overlapping COMP query.

## Reproduction

```bash
cmake -S . -B build-job-btv -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-job-btv --target benchmark_app verify_job_btv_delivery -j2
python3 experiments/job_wah_ewah_btv_20260821/run_job.py
python3 experiments/job_wah_ewah_btv_20260821/analyze_job.py
python3 experiments/job_wah_ewah_btv_20260821/plot_job.py
```

Formal results and audit notes are added only after a complete clean run.
