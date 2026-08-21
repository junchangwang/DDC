# CR runOptimize + BTV delivery + literal COMP rerun (2026-08-21)

This isolated experiment does not modify the paper or any earlier result.

## Protocol

- Every CRoaring input was generated after an explicit `runOptimize()` call.
  Container selection remains automatic; run containers are not forced.
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
