#!/usr/bin/env python3
"""plot_and_density.py — standard cardinality-sweep PDF chart.

Reads build/unified_results.csv (output of benchmark_app on the
bm_100m_c<N>_<algo>/ directories — disjoint bitmaps, OR result density depends on overlap).
Emits a clean PDF showing OR op-only performance vs bitmap density
across all 7 backends.

Convention: solid lines, distinct colours per backend, distinct point
markers per backend.  No dashed lines.

x-axis: density d = 1/cardinality (log scale, reversed: dense left,
        sparse right; densest is c=2 at d=0.5).
y-axis: performance (Gbit/s) = 100 Mbits ÷ median OR time.

Usage:  plot_and_density.py [csv] [out_pdf]
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

NUM_BITS = 100_000_000               # 100 M-row bitmaps
GBITS_PER_MS = NUM_BITS / 1e6        # = 100.0

D_MAX = 0.5      # leftmost (densest, c = 2)
D_MIN = 0.00002     # rightmost (c = 50) — within this window ComBit sits above
                 # CR across the entire sweep.  Past d < 0.02, CR catches up
                 # because disjoint OR on dense inputs is essentially free
                 # for CR's array-array intersection.

# One extra "transition worst-case" point on the density axis — CR's
# array→bitset→array OR worst path.  Each input has count_a=3500 bits per
# 65 536-segment (array container), B carries 95% of A's bits + 5% fresh
# disjoint, so |A|+|B|=7000 > 4096 triggers CR to allocate a transient
# bitset, but |A∪B|≈3675 ≤ 4096 forces CR to convert that bitset back to
# an array (regret path).  All 7 backends' numbers come from
# bm_100m_o3500_<algo>/.
TRANSITION_COUNT_A = 3500     # count_a in the .bm dir name
TRANSITION_DENSITY = TRANSITION_COUNT_A / 65536.0   # ≈ 0.053

# Backend rendering (solid lines, distinct colours, distinct markers).
# gnuplot point types: 5=square, 7=filled circle, 9=triangle, 11=down-triangle,
#                      13=diamond, 15=pentagon, 4=hollow square, etc.
BACKENDS = [
    ("ComBIT (New)",     "ComBit",             "#1f4ed8", 7,  3.2),  # vivid blue, slightly thicker
    ("CRoaring",         "CRoaring",           "#dc2626", 5,  2.4),  # red
    ("Bitset (AVX512)",  "Bitset (AVX-512)",   "#0891b2", 9,  2.2),  # teal
    ("Bitset (Plain)",   "Bitset (scalar)",    "#7c3aed", 11, 2.2),  # purple
    ("WAH (FastBit)",    "WAH (FastBit)",      "#16a34a", 13, 2.0),  # green
    ("EWAH",             "EWAH",               "#ea580c", 4,  2.0),  # orange
    ("Concise",          "Concise",            "#92400e", 15, 2.0),  # brown
]


def load(csv_path: Path):
    """Returns two dicts:
       std:  {(backend, cardinality): median_ms} for standard disjoint sweep
       tran: {backend: median_ms} for the single TRANSITION_COUNT_A point
             (independent random bitmaps — CR's bitset→array worst path)
    Note: TRANSITION_COUNT_A (=3500) is NOT in the standard sweep, so any row
    with cardinality==3500 in the CSV is unambiguously from the transition
    dir (bm_100m_o3500_*) regardless of result_card."""
    std:  dict[tuple[str, int], float] = {}
    tran: dict[str, float] = {}
    standard_cs = {2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000, 20000, 50000}   # density ∈ {0.5, 0.2, 0.1, 0.05, 0.02}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["num_rows"]) != NUM_BITS:
                continue
            if row["operation"] != "OR_op":
                continue
            c = int(row["cardinality"])
            backend = row["backend"]
            time_ms = float(row["time_ms"])
            # TRANSITION_COUNT_A is never in standard_cs → unambiguous overlap.
            if c == TRANSITION_COUNT_A:
                tran[backend] = time_ms
                continue
            if c not in standard_cs:
                continue
            # OR-specific note: for AND we filtered transition rows by
            # result_cardinality>0 (since disjoint AND = 0), but OR of
            # disjoint bitmaps is non-zero, so res_card can't disambiguate.
            # Use FIRST-WINS instead: the benchmark dirlist is sorted -V
            # which puts `_c<N>_` standard dirs BEFORE `_t<N>_` transition
            # dirs, so the first row in the CSV for each (backend, c) is
            # always the standard-sweep entry.
            key = (backend, c)
            if key in std:
                continue
            std[key] = time_ms
    return std, tran


def write_dat(std_samples, tran_samples, dat_path: Path):
    """Write rows sorted by density DESCENDING (so the line goes left→right
    from dense to sparse).  Standard rows have density = 1/c.  The transition
    row sits at TRANSITION_DENSITY (≈ 0.053, between c=10 and c=20 on the axis)."""
    rows = []
    for (_, c), _ in std_samples.items():
        rows.append((1.0 / c, c, "std"))
    if tran_samples:
        rows.append((TRANSITION_DENSITY, TRANSITION_COUNT_A, "tran"))
    rows = sorted(set(rows), key=lambda r: -r[0])  # densest first

    with dat_path.open("w") as f:
        f.write("# density\tcardinality\tkind\t" + "\t".join(b[0] for b in BACKENDS) + "\n")
        for d, c, kind in rows:
            row = [f"{d:.6g}", str(c), kind]
            for be, *_ in BACKENDS:
                ms = tran_samples.get(be) if kind == "tran" else std_samples.get((be, c))
                row.append(f"{GBITS_PER_MS / ms:.4f}" if ms and ms > 0 else "?")
            f.write("\t".join(row) + "\n")


def write_gp(gp_path: Path, dat_path: Path, pdf_path: Path):
    plots = []
    # Columns: 1=density, 2=cardinality, 3=kind, 4..N=perf per backend.
    for col, (_, label, colour, pt, lw) in enumerate(BACKENDS, start=4):
        plots.append(
            f"  '{dat_path.name}' using 1:{col} with linespoints "
            f"lw {lw} pt {pt} ps 1.1 lc rgb '{colour}' title '{label}'"
        )
    plot_clause = ", \\\n".join(plots)

    # Pure density values on the x-axis — full sweep c=2..50000 (d=0.5..2e-5).
    xtics = (
        f"('0.5' 0.5, "
        f"'{TRANSITION_DENSITY:.3f}' {TRANSITION_DENSITY}, "
        f"'0.1' 0.1, '0.02' 0.02, "
        f"'5·10^{{-3}}' 0.005, '10^{{-3}}' 0.001, "
        f"'2·10^{{-4}}' 0.0002, '10^{{-4}}' 0.0001, "
        f"'2·10^{{-5}}' 0.00002)"
    )

    script = f"""# auto-generated by plot_and_density.py
set terminal pdfcairo size 10in,6in enhanced font 'DejaVu Sans,11'
set output '{pdf_path.name}'

set title "OR op-only: performance vs bitmap density\\n{{/*0.85 (100 M rows; disjoint bitmaps from bm\\\\_100m\\\\_c<c>\\\\_<algo>/; ComBit default = decompressed; CR includes pairwise + to-bitset)}}" font ',13'
set xlabel "bitmap density  d = 1 / cardinality   (left = dense, right = sparse)"
set ylabel "performance  (Gbit/s, 100 Mbits ÷ median OR time)"

set logscale x 10
set xrange [{D_MAX * 1.18}:{D_MIN * 0.85}]
set xtics {xtics}
set xtics offset 0,-0.3 font ',8'
set format x ''

# Log y so the RLE backends' sparse-extreme spikes (thousands of Gbit/s on
# disjoint OR ≈ full bitset) don't crush the dense-end story.
set logscale y 10
set yrange [1:*]
set ytics ('1' 1, '2' 2, '5' 5, '10' 10, '20' 20, '50' 50, '100' 100, \\
           '200' 200, '500' 500, '1000' 1000, '2000' 2000, '5000' 5000) \\
           font ',9'

set grid back lc rgb '#cbd5e1' lw 0.6
set border lw 1.2 lc rgb '#0f172a'
set key outside right top vertical Left reverse samplen 2.6 spacing 1.2 font ',10'

# Annotation: highlight the density (≈ 0.053) where CR's bitset→array
# OR-result worst path sits.  Light vertical guide only — no text
# label since the axis tick at 0.244 is already visible.
set arrow from {TRANSITION_DENSITY},graph 0 to {TRANSITION_DENSITY},graph 1 nohead \\
    lw 0.8 dt 3 lc rgb '#9ca3af'

plot \\
{plot_clause}
"""
    gp_path.write_text(script)


def main():
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/unified_results.csv")
    out_pdf  = Path(sys.argv[2] if len(sys.argv) > 2 else "or_density.pdf")

    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found")
    std_samples, tran_samples = load(csv_path)
    if not std_samples:
        sys.exit("error: no standard-sweep AND_op rows found")

    cs = sorted({c for (_, c) in std_samples.keys()})
    print(f"[load] standard:   {len(std_samples)} (backend, c) cells; cardinalities: {cs}")
    print(f"[load] transition: t={TRANSITION_COUNT_A} → {len(tran_samples)} backends "
          f"({sorted(tran_samples.keys())})")

    dat_path = out_pdf.with_suffix('.dat')
    gp_path  = out_pdf.with_suffix('.gp')
    write_dat(std_samples, tran_samples, dat_path)
    write_gp(gp_path, dat_path, out_pdf)
    print(f"[write] {dat_path}\n[write] {gp_path}")

    workdir = out_pdf.parent if out_pdf.parent.as_posix() else Path(".")
    r = subprocess.run(["gnuplot", gp_path.name], cwd=workdir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[gnuplot stderr]", r.stderr, file=sys.stderr)
        sys.exit(r.returncode)
    print(f"[ok]   wrote {out_pdf}")


if __name__ == "__main__":
    main()
