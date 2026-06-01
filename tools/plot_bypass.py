#!/usr/bin/env python3
"""plot_bypass.py — bypass-ablation line charts (eva style).

Reads bench_bypass.csv (cardinality,config,or_ms,and_ms,comp_ms,...) and
draws one PDF per op (OR / AND / COMP).  Each PDF has three lines — the
three bypass configurations of the default depth-4 walker:

    all      : L4(batch) + L3(region) bypass both ON  (= pre-dip-fix)
    L4_off   : L4(batch) OFF, L3(region) ON
    L4L3_off : L4(batch) + L3(region) BOTH OFF        (= current default)

X axis: bit density (log, dense -> sparse).  Y axis: throughput (op/s).
Visual style mirrors eva/graphs_OR_AND/motivation_eva.py.

Output (next to this script):
    or_bypass.pdf  and_bypass.pdf  comp_bypass.pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- eva style rcParams (mirrored from motivation_eva.py) -------------------
plt.rcParams.update({
    "font.family": "Linux Libertine O",
    "font.serif": ["Linux Libertine O"],
    "font.sans-serif": ["Linux Libertine O"],
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.linewidth": 1.0,
    "xtick.major.size": 4, "xtick.major.width": 0.9, "xtick.labelsize": 13,
    "ytick.major.size": 4, "ytick.major.width": 0.9, "ytick.labelsize": 13,
    "legend.frameon": True, "legend.edgecolor": "#475569",
    "legend.fontsize": 11, "legend.borderpad": 0.25,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.4,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Linux Libertine O",
    "mathtext.it": "Linux Libertine O:italic",
    "pdf.fonttype": 42,
})

# config_key -> (display label, colour, marker)
CONFIGS = [
    ("all",      "All bypass", "#1f4ed8", "x"),
    ("L4_off",   "L4 off",     "#16a34a", "^"),
    ("L4L3_off", "L4+L3 off",  "#dc2626", "o"),
]
# op field -> (axis label, output pdf, y_top, yticks)
OPS = [
    ("or_ms",   "OR Throughput (op/s)",   "or_bypass.pdf",   850.0, [0, 400, 800]),
    ("and_ms",  "AND Throughput (op/s)",  "and_bypass.pdf",  850.0, [0, 400, 800]),
    ("comp_ms", "Comp. Throughput (op/s)", "comp_bypass.pdf", 165.0, [0, 50, 100, 150]),
]


def load(csv_path: Path):
    """{config: [(density, op_ms_dict)]} sorted densest-first."""
    rows = list(csv.DictReader(csv_path.open()))
    out: dict[str, list] = {k: [] for k, *_ in CONFIGS}
    for r in rows:
        c = int(r["cardinality"])
        out[r["config"]].append((1.0 / c, r))
    for k in out:
        out[k].sort(key=lambda t: -t[0])
    return out


def plot_op(data, op_field, ylabel, out_pdf: Path, y_top, yticks):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    dmin = 1e9
    for key, label, colour, marker in CONFIGS:
        pts = data[key]
        xs = [d for d, _ in pts]
        ys = [1000.0 / float(r[op_field]) for _, r in pts]
        dmin = min(dmin, min(xs))
        ax.plot(xs, ys, color=colour, label=label, linewidth=2.0,
                marker=marker, markersize=8, markerfacecolor="none",
                markeredgecolor=colour, markeredgewidth=1.4,
                solid_joinstyle="round", solid_capstyle="round")

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(0.5 * 1.18, dmin * 0.85)
    ax.set_xticks([0.5, 0.1, 0.01, 0.001])
    ax.set_xticklabels(["50%", "10%", "1%", "0.1%"])
    ax.set_xlabel("Bit Density (log scale)")
    ax.xaxis.set_label_coords(0.45, -0.12)

    ax.set_ylim(0, y_top)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.minorticks_off()

    ax.legend(loc="lower center", ncol=3, frameon=True, edgecolor="#94a3b8",
              borderpad=0.25, columnspacing=0.9, handletextpad=0.3,
              handlelength=1.4, fontsize=10.5)

    for sp in ax.spines.values():
        sp.set_color("#1f2937"); sp.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="#1f2937")

    fig.tight_layout(pad=0.0)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")


def main():
    here = Path(__file__).resolve().parent
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "bench_bypass.csv"
    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found (run bench_bypass first)")
    data = load(csv_path)
    for op_field, ylabel, pdf, y_top, yticks in OPS:
        plot_op(data, op_field, ylabel, here / pdf, y_top, yticks)


if __name__ == "__main__":
    main()
