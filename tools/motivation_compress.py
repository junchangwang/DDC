#!/usr/bin/env python3
"""motivation_compress.py — OR / AND density chart for DDC's COMPRESS
result path (operator returns a Compressed segment instead of the default
Decompressed form).

Differences from motivation.py (which charts DDC's Decompressed path):
  * DDC row in CSV is "DDC (compress)" not "DDC (New)".
  * NO Y_OVERRIDES for DDC — the raw measurements are plotted directly
    because the storyboard here is "compress is X× slower than decompress",
    not "DDC is a stable baseline".
  * X-axis spans the full standard sweep (c=2..2000), not capped at 10⁻³.
  * Bitset reference dashed line + arrow callout reused from motivation.

Output:
  out_plots/or_density_compress.pdf
  out_plots/and_density_compress.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# Triangle markers with centroid at path origin (matches motivation.py).
TRI_SCALE = 1.5
TRI_DOWN_CENTERED = MplPath([
    [-0.5 * TRI_SCALE,  (1.0/3) * TRI_SCALE],
    [ 0.5 * TRI_SCALE,  (1.0/3) * TRI_SCALE],
    [ 0.0,             (-2.0/3) * TRI_SCALE],
    [-0.5 * TRI_SCALE,  (1.0/3) * TRI_SCALE],
])
TRI_UP_CENTERED = MplPath([
    [-0.5 * TRI_SCALE, (-1.0/3) * TRI_SCALE],
    [ 0.5 * TRI_SCALE, (-1.0/3) * TRI_SCALE],
    [ 0.0,              (2.0/3) * TRI_SCALE],
    [-0.5 * TRI_SCALE, (-1.0/3) * TRI_SCALE],
])
SQUARE_SIZE = 0.75 / 2
SQUARE_CENTERED = MplPath([
    [-SQUARE_SIZE, -SQUARE_SIZE],
    [ SQUARE_SIZE, -SQUARE_SIZE],
    [ SQUARE_SIZE,  SQUARE_SIZE],
    [-SQUARE_SIZE,  SQUARE_SIZE],
    [-SQUARE_SIZE, -SQUARE_SIZE],
])

plt.rcParams.update({
    "font.family": "Linux Libertine O",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "axes.linewidth": 1.0,
    "xtick.major.size": 4,
    "xtick.major.width": 0.9,
    "xtick.labelsize": 15,
    "ytick.major.size": 4,
    "ytick.major.width": 0.9,
    "ytick.labelsize": 15,
    "legend.frameon": True,
    "legend.edgecolor": "#475569",
    "legend.fontsize": 14,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "pdf.fonttype": 42,
})

NUM_BITS = 100_000_000
GBITS_PER_MS = NUM_BITS / 1e6
SEG_BITS = 65_536

D_MAX = 0.5
D_MIN = 5e-4
STANDARD_CS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# Engineered points — same as motivation.py.
OR_POINTS = [
    ("t3500",       3500, 3500 / SEG_BITS, "array→bitset",                       {"CRoaring"}),
    ("A2500_B100",  2500, 2500 / SEG_BITS, "array merge",                         {"CRoaring"}),
    ("o2200",       2200, 2200 / SEG_BITS, "array→bitset→array",                  {"CRoaring"}),
]
AND_POINTS = [
    ("t16000",     16000, 16000 / SEG_BITS, "bitset→array",                       {"CRoaring"}),
]

# Same backend list / colours / markers as motivation.py, except we use
# the "DDC (compress)" CSV row instead of "DDC (New)".
BACKENDS_DECOMPRESS_OR = [
    ("DDC (compress)", "DDC (compress)", "#1f4ed8",  "x",                2.4, 11),
    ("CRoaring",          "CRoaring",          "#16a34a",  TRI_DOWN_CENTERED,  2.2, 15),
    ("WAH (FastBit)",     "WAH (FastBit)",     "#dc2626",  TRI_UP_CENTERED,    2.0, 15),
    ("EWAH",              "EWAH",              "#1e3a8a",  SQUARE_CENTERED,    2.0, 11),
    ("Concise",           "Concise",           "#ca8a04",  "o",                2.0, 11),
]
BACKENDS_DECOMPRESS = BACKENDS_DECOMPRESS_OR        # used for AND as well

SHARP_BACKENDS = {"CRoaring"}

# Bitset reference baseline (flat across the sweep).
FLAT_BASELINES: dict[tuple[str, str], float] = {
    ("Bitset (AVX512)", "OR_op"):  74.5,
    ("Bitset (AVX512)", "AND_op"): 60.0,
}


def load_csv(csv_path: Path, operation: str):
    out: dict[tuple[str, int], float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["num_rows"]) != NUM_BITS:
                continue
            if row["operation"] != operation:
                continue
            c = int(row["cardinality"])
            be = row["backend"]
            key = (be, c)
            if key in out:
                continue
            out[key] = float(row["time_ms"])
    return out


def plot_one(samples, backends, engineered_points, out_pdf: Path,
             operation: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.8, 3.9))

    point_map: dict[int, float] = {c: 1.0 / c for c in STANDARD_CS}
    eng_keys = set()
    eng_only_for: dict[int, set[str] | None] = {}
    for _, key, x_density, _, only_for in engineered_points:
        point_map[key] = x_density
        eng_keys.add(key)
        eng_only_for[key] = only_for

    for csv_label, disp_label, colour, marker, lw, msize in backends:
        baseline_y = FLAT_BASELINES.get((csv_label, operation))
        if baseline_y is not None:
            ax.plot([D_MAX * 1.18, D_MIN * 0.85],
                    [baseline_y, baseline_y],
                    color=colour, label=disp_label, linewidth=lw,
                    linestyle="-")
            continue

        items: list[tuple[float, float]] = []
        for c_key, dens in point_map.items():
            only_for = eng_only_for.get(c_key)
            if only_for is not None and csv_label not in only_for:
                continue
            if (csv_label == "CRoaring" and operation == "OR_op"
                    and c_key == 20):
                continue
            ms = samples.get((csv_label, c_key))
            if ms and ms > 0:
                items.append((dens, GBITS_PER_MS / ms))
        items.sort(key=lambda t: -t[0])
        if len(items) < 2:
            continue

        xs = np.array([t[0] for t in items])
        ys = np.array([t[1] for t in items])
        log_xs_sorted = np.log(xs[::-1])
        ys_sorted     = ys[::-1].astype(float)
        dense_log_x   = np.linspace(log_xs_sorted[0], log_xs_sorted[-1], 400)

        if csv_label in SHARP_BACKENDS:
            ys_smooth = gaussian_filter1d(ys_sorted, sigma=0.5, mode="nearest")
            spline = PchipInterpolator(log_xs_sorted, ys_smooth)
        else:
            ys_smooth = gaussian_filter1d(ys_sorted, sigma=0.85, mode="nearest")
            spline = CubicSpline(log_xs_sorted, ys_smooth, bc_type="natural")
        dense_y = np.maximum(spline(dense_log_x), 0.0)
        dense_x = np.exp(dense_log_x)

        skip_keys: set[int] = set()
        if operation == "OR_op":
            skip_keys.add(20)
        marker_keys = sorted([k for k in point_map.keys()
                              if k not in skip_keys],
                             key=lambda k: -point_map[k])
        marker_log_x = np.log(np.array([point_map[k] for k in marker_keys]))
        marker_indices = [int(np.argmin(np.abs(dense_log_x - mlx)))
                          for mlx in marker_log_x]

        ax.plot(dense_x, dense_y, color=colour, label=disp_label,
                linewidth=lw, linestyle="-",
                marker=marker, markersize=msize,
                markerfacecolor="none", markeredgecolor=colour,
                markeredgewidth=1.7, markevery=marker_indices,
                solid_joinstyle="round", solid_capstyle="round")

    # --- Axes ---
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(D_MAX * 1.18, D_MIN * 0.85)

    ax.set_yscale("linear")
    ymax_data = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=ymax_data * 1.12)

    # Full standard sweep ticks: 2, 5, 10, ..., 2000.
    ax.set_xticks([1.0 / c for c in STANDARD_CS])
    ax.set_xticklabels([str(c) for c in STANDARD_CS])
    ax.minorticks_off()
    ax.set_xlabel("cardinality   (dense ← → sparse)")
    ax.set_ylabel(ylabel)

    # Bitset dashed reference line (only for OR — compress AND uses the
    # real baseline column already loaded as a flat line above).
    if operation == "OR_op":
        BITSET_Y = 70.0
        BITSET_COLOUR = "#b45309"
        ax.plot([D_MAX * 1.18, D_MIN * 0.85], [BITSET_Y, BITSET_Y],
                color=BITSET_COLOUR, linewidth=1.6,
                linestyle=(0, (6, 4)), zorder=2.5)
        ax.text(0.45, 79.0, "Bitset",
                color=BITSET_COLOUR, fontsize=16,
                ha="center", va="center",
                family="Linux Libertine O", zorder=3)
        ax.annotate("",
                    xy=(0.20, BITSET_Y),
                    xytext=(0.32, 77.0),
                    arrowprops=dict(arrowstyle="->",
                                    color=BITSET_COLOUR,
                                    lw=1.6, shrinkA=2, shrinkB=2),
                    zorder=3)

    ax.legend(loc="upper right", bbox_to_anchor=(0.998, 0.998),
              ncol=5, frameon=True,
              edgecolor="#94a3b8", borderpad=0.3,
              columnspacing=0.7, handletextpad=0.3,
              handlelength=1.2)

    for spine in ax.spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#1f2937")

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")


def main() -> None:
    default_csv = Path(__file__).resolve().parent / "plot_results.csv"
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_csv)
    out_dir  = Path(sys.argv[2] if len(sys.argv) > 2 else
                    Path(__file__).resolve().parent.parent / "out_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found")

    or_samples  = load_csv(csv_path, "OR_op")
    and_samples = load_csv(csv_path, "AND_op")
    print(f"[load] OR_op rows: {len(or_samples)}, AND_op rows: {len(and_samples)}")

    plot_one(or_samples, BACKENDS_DECOMPRESS_OR, OR_POINTS,
             out_dir / "or_density_compress.pdf", "OR_op",
             "Bitwise OR Performance (op/s)")
    plot_one(and_samples, BACKENDS_DECOMPRESS, AND_POINTS,
             out_dir / "and_density_compress.pdf", "AND_op",
             "Bitwise AND Performance (op/s)")


if __name__ == "__main__":
    main()
