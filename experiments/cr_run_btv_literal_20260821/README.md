# CR runOptimize + BTV delivery + literal COMP rerun (2026-08-21)

This isolated experiment does not modify the paper or any earlier result.

## Protocol

- Every CRoaring input was generated after an explicit `runOptimize()` call.
  Container selection remains automatic; run containers are not forced.
  The generator call is `combit/util/gen_bitmap.cpp:317` (introduced by commit
  `a3accb0a`; source SHA-256
  `f2d13301746164b97d069679dd73a7e5da0ed6259c664713ea9f3d09f1c366d3`).
- CRoaring OR/AND/NOT use the delivered `*_op_conv` rows. CRoaring COMP uses
  `COMP_op`, which executes literal COMP and converts the final result to a
  dense bitvector once.
- DDC, WAH, and EWAH use their existing `*_op`/`COMP_op` paths. This is the
  paper's selected-delivery boundary, not a common dense-output contract.
- All backends execute literal COMP:

```text
t1 = A | B
t2 = B | C
t3 = t1 & t2
result = ~t3
```

- DDC's timer stops before intermediate destruction, matching the other
  backends. DDC keeps the existing default `ddc_compress_results=false` path.
- Clustering AND uses the requested ordinary pair `A&B`, not self-AND.
- Every timed process uses `--iterations 1`; each pure operation internally
  performs 100 measurements and reports its median.

Earth, JOB, and clustering use five fresh processes per field/backend. The 55
independent density pairs use three fresh processes per backend, then the upper
triangle is mirrored for plotting. All processes are pinned to one idle CPU,
use normalized allocator thresholds, and rotate field/backend order.

The equality-bitmap COMP workloads remain data-level degeneracies (`~B` for
three distinct values and `~(A|B)` for the two-value case), even though every
backend executes all four literal calls. They are not general overlapping-COMP
workloads.

Table 4 is delivered separately in `table4/` and changes only the CRoaring
column to the payload measured after `runOptimize()`; the existing fixed-zero
DDC and native WAH/EWAH payload protocols remain unchanged.

## Measured results

The formal run completed 1,510 fresh timing processes after 81 exact
cross-backend verification cases. CRoaring inputs contain at least one run
container in 3/5 Earth cases, 8/10 JOB cases, and 10/11 clustering cases. All
55 random density pairs contain zero run containers after `runOptimize()` chose
array/bitset containers instead.

| Group | Cells | CR/DDC GM | WAH/DDC GM | EWAH/DDC GM | Winners |
|---|---:|---:|---:|---:|---|
| Earth OR+COMP | 10 | 2.176x | 6.829x | 5.109x | DDC 10 |
| JOB COMP | 10 | 4.091x | 0.565x | 0.307x | DDC 4, EWAH 5, WAH 1 |
| Clustering, all 11 points and 4 ops | 44 | 3.622x | 0.657x | 1.077x | DDC 16, CR 5, WAH 13, EWAH 10 |
| Density OR, 55 independent cells | 55 | 2.301x | 4.687x | 5.280x | DDC 53, WAH 2 |

Density additionally gives Bitset-AVX512/DDC `1.181x` and Concise/DDC
`19.735x`. The full symmetric heatmap contains mirrored values; only 55 upper
triangle pairs are independent measurements.

Compact CSV/JSON results are in `results/`; the 12 PDF and 12 PNG figures are
in `figures/`. Table 4 TeX/PDF files are in `table4/`.

Independent audit found no result or figure blocker. The only clear
process-level outlier is density `A32768_B32768` EWAH
(`12.6859/5.60929/5.79421 ms`); the reported three-process median is
`5.79421 ms`, and min/max remain in the raw aggregate. Sub-microsecond WAH/EWAH
points at the fully sorted clustering endpoint should be interpreted with the
timer-resolution caveat of the inherited microbenchmark.
