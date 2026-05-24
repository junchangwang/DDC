#!/usr/bin/env python3
"""plot_design_space.py — Blitzcrank-style design-space figure.

A 2-D positioning chart: each bitmap backend is a single filled marker
at its (perf, CR) coordinate.  A "Better →" arrow points to the
upper-right (high CR + high perf is the desirable corner).

Coordinates come from real measurements:
  * OR Gbit/s plateau values from the OR-density sweep (plot_4pdf.py).
  * Compression ratios from per-backend dir sizes at c=100 (100M-bit, 100
    bitmaps, raw 1250 MB):
        ComBit       6.5x
        CRoaring     6.5x
        Concise      3.4x
        WAH          2.2x
        EWAH         1.4x
        Bitset       1.0x  (no compression)

Output:  out_plots/design_space.pdf
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Linux Libertine O",
    "font.size": 15,
    "axes.labelsize": 17,
    "pdf.fonttype": 42,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.edgecolor": "#94a3b8",
})

# (label, x_perf, y_CR, marker, fill, edge)
# Positions from GEOMETRIC MEAN of measurements across c=2/10/100/1000
# (TPC-H falls inside this range too):
#   OR Gbit/s geo-mean:  Bitset 74 │ ComBit 55 │ CR 38 │ EWAH 15 │ WAH 14 │ Concise 6
#   CR  geo-mean:        ComBit 12.5× │ CR 12.0× │ Concise 8.7× │ WAH 6.6× │ EWAH 5.1× │ Bitset 1.0×
# Note: smaller c = denser bitmap (c=2 means 50% set bits per bitmap).
# EWAH compresses BEST at very-dense data (c=2: 52.5×, beats WAH/Concise)
# but worst at medium-density (c=100: 1.4×, c=1000: 0.9×).  Geometric mean
# across the whole range gives EWAH ≈ 5× CR — bottom of the RLE family but
# clearly above Bitset.
# Positions are now in REAL units:
#   x = OR perf in Gbit/s (geometric mean across c=2..1000)
#   y = compression ratio (×, geometric mean across c=2..1000)
BACKENDS = [
    ("ComBit",            55.0, 12.5, "o",  "#3b82f6", "#1d4ed8"),
    ("CRoaring",          38.0, 12.0, "h",  "#22c55e", "#15803d"),
    ("Concise",            6.0,  8.7, "D",  "#eab308", "#a16207"),
    ("WAH (FastBit)",     14.0,  6.6, "^",  "#ef4444", "#b91c1c"),
    ("EWAH",              15.0,  5.1, "s",  "#1e3a8a", "#1e40af"),
    ("Bitset (AVX-512)",  74.0,  1.0, "v",  "#a855f7", "#7c3aed"),
]


def main() -> None:
    out_dir = Path("out_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # Axes in real units: OR Gbit/s × Compression Ratio.
    ax.set_xlim(0, 85)
    ax.set_ylim(0, 16)        # headroom for the single-row legend
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.tick_params(axis="both", colors="#1f2937")
    ax.set_aspect("auto")

    # Keep all four spines visible (the chart frame).
    for s in ax.spines.values():
        s.set_color("#1f2937")
        s.set_linewidth(1.0)

    # Plot each backend as a single filled marker with dark edge.
    for label, x, y, marker, fill, edge in BACKENDS:
        ax.scatter([x], [y], marker=marker, s=240,
                   facecolor=fill, edgecolor=edge,
                   linewidth=1.8, zorder=3,
                   label=label)

    # "Better →" arrow pointing toward the upper-right (high CR + high
    # perf is the ideal corner).  Placed in the lower-middle empty zone
    # so it doesn't collide with any marker or the legend.
    ax.annotate(
        "",
        xy=(48, 11.2),           # arrow head, just below ComBit
        xytext=(30, 6.5),        # arrow tail in the empty middle band
        arrowprops=dict(arrowstyle="-|>", color="#0f172a",
                        lw=2.4, mutation_scale=22),
        zorder=2,
    )
    ax.text(32, 5.5, "Better",
            ha="left", va="top", fontsize=17,
            color="#0f172a", family="Linux Libertine O",
            fontstyle="italic")

    # Axis labels in the units the user uses in the rest of the thesis.
    ax.set_xlabel("Bitwise OR/AND Performance  (op/s)",
                  family="Linux Libertine O")
    ax.set_ylabel("Compression Ratio  (times)",
                  family="Linux Libertine O")

    # Legend: single-row band at the very top (above the data band, which
    # tops out at y≈8.6 for ComBit / CRoaring).  With ylim=11 the legend
    # gets its own y=10-11 stripe and doesn't shadow any marker.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99),
              ncol=6, frameon=True, fontsize=11,
              handletextpad=0.35, columnspacing=0.8,
              borderpad=0.35, labelspacing=0.4)

    fig.tight_layout()
    out_pdf = out_dir / "design_space.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")


if __name__ == "__main__":
    main()
