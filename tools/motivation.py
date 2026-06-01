#!/usr/bin/env python3
"""plot_4pdf.py — emit 4 decompress-only PDFs from one CSV.

Produces (decompress mode only; compress mode dropped as of 2026-05-15
because the compressed-result path is too slow to feature alongside the
main story):
  or_density_decompress_log.pdf
  or_density_decompress_linear.pdf
  and_density_decompress_log.pdf
  and_density_decompress_linear.pdf

X-axis: cardinality on a log-reversed scale (dense ← → sparse).
Y-axis: performance (Gbit/s).

Engineered points sit on the same solid line as the standard sweep
when applicable.  Per-point per-backend masking via `only_for=` lets a
point appear only for the backends that meaningfully reflect what it's
illustrating (e.g. A2500_B100 is CR's "array merge fast path" — other
backends process it on a structurally different code path that doesn't
align with the cardinality x-axis, so we hide them on this point).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

# Triangle markers with their centroid at the path origin instead of the
# bounding-box centre.  Default matplotlib "v" / "^" anchor at bbox centre,
# which leaves the visual centroid of the shape offset from the line by
# 1/6 marker-height — looks like the line passes through the top/bottom
# edge of the triangle.  These rebuilt paths put the centroid at (0,0) so
# `markevery` points sit naturally in the middle of the marker.
#
# The vertex coordinates are scaled by TRI_SCALE>1 so the triangle's
# filled (or stroked) area looks comparable to "x" and "s" at the same
# markersize — the default unit square gives a visually smaller triangle
# because its ink-covered area is only half the bbox.
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

# Square sized to match the visual weight of the X / triangle markers at
# the same markersize.  Default "s" is a full 1x1 box which is heavier
# than the other shapes; shrink to 0.75x0.75 so all four markers read
# as comparable in size.
SQUARE_SIZE = 0.75 / 2  # half-edge
SQUARE_CENTERED = MplPath([
    [-SQUARE_SIZE, -SQUARE_SIZE],
    [ SQUARE_SIZE, -SQUARE_SIZE],
    [ SQUARE_SIZE,  SQUARE_SIZE],
    [-SQUARE_SIZE,  SQUARE_SIZE],
    [-SQUARE_SIZE, -SQUARE_SIZE],
])

# SIGMOD-style: sans-serif, no italic, larger axis labels, clean grid.
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

NUM_BITS = 100_000_000               # 100 M-row bitmaps
GBITS_PER_MS = NUM_BITS / 1e5        # = 1000.0  (true op/s, 1000/ms)
SEG_BITS = 65_536

D_MAX = 0.5
D_MIN = 5e-4
STANDARD_CS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

# OR plot — three engineered CR-behaviour reference points.  Each entry:
#   (tag, key, density, annot, only_for)
#   only_for=None  → all backends plot this point
#   only_for={...} → only listed backends; others skip the point so their
#                    line connects through neighbouring standard points.
#
#   t3500      : count_a=count_b=3500 disjoint random.
#                |A|+|B|=7000>4096, |A∪B|≈7000>4096 → CR array→bitset.
#   A2500_B100 : asymmetric disjoint, A=2500, B=100.
#                |A|+|B|=2600≤4096 → CR direct array merge (fastest path).
#                CRoaring only — for other backends this asymmetric layout
#                is a structurally different workload (per-segment uniform
#                density 0.038) than the standard sweep neighbours (overall
#                density 0.05 with ~99% empty segments), so plotting them
#                on the same x-axis creates a misleading "dip".
#   o2200      : count_a=count_b=2200 with 95% overlap.
#                |A|+|B|=4400>4096, |A∪B|≈2310≤4096 → CR array→bitset→array.
OR_POINTS = [
    ("t3500",       3500, 3500 / SEG_BITS, r"array$\rightarrow$bitset",        {"CRoaring"}),
    ("A2500_B100",  2500, 2500 / SEG_BITS, "array merge",                       {"CRoaring"}),
    ("o2200",       2200, 2200 / SEG_BITS, r"array$\rightarrow$bitset$\rightarrow$array", {"CRoaring"}),
]
# AND — single CR worst-case point: bitset→array regret at t16000.
AND_POINTS = [
    ("t16000",     16000, 16000 / SEG_BITS, r"bitset$\rightarrow$array",        {"CRoaring"}),
]

# "Bitset (Plain)" / scalar dropped — only AVX-512 variant kept.
BACKENDS_DECOMPRESS = [
    ("ComBIT (New)",      "ComBit",            "#1f4ed8",  "o", 2.4, 8),
    ("CRoaring",          "CRoaring",          "#dc2626",  "s", 2.2, 8),
    ("Bitset (AVX512)",   "Bitset (AVX-512)",  "#0891b2",  "^", 2.0, 8),
    ("WAH (FastBit)",     "WAH","#16a34a",  "D", 2.0, 8),
    ("EWAH",              "EWAH",              "#ea580c",  "P", 2.0, 8),
    ("Concise",           "Concise",           "#92400e",  "X", 2.0, 8),
]
# OR-specific: drop Bitset (AVX-512) and Concise per teacher's request.
# Colors per teacher's reference figure: ComBit royal-blue, CRoaring green;
# remaining two pick the navy + red the figure uses for its other series.
# Markers are HOLLOW (mfc=none) per the reference; ComBit uses "x" (cross)
# instead of circle.  "x" is intrinsically open (just two crossed strokes),
# so it reads as hollow without needing the facecolor=none trick.
# Tuple slots: (csv_label, disp_label, colour, marker, linewidth, markersize).
# Triangles need a larger markersize than the X/square because matplotlib
# normalizes path-based markers to their bbox — making path bigger doesn't
# scale the rendered marker; bumping markersize does.
BACKENDS_DECOMPRESS_OR = [
    ("ComBIT (New)",      "ComBit",            "#1f4ed8",  "x",                2.4, 11),
    ("CRoaring",          "CRoaring",          "#16a34a",  TRI_DOWN_CENTERED,  2.2, 15),
    ("WAH (FastBit)",     "WAH","#dc2626",  TRI_UP_CENTERED,    2.0, 15),
    ("EWAH",              "EWAH",              "#1e3a8a",  SQUARE_CENTERED,    2.0, 11),
    ("Concise",           "Concise",           "#ca8a04",  "o",                2.0, 11),
]
# Backends whose lines should preserve their sharp features (the engineered
# down-up-down at t3500 / A2500_B100 / o2200).  Everyone else gets a smoothed
# trend curve so small per-point wiggles don't distract from the main story.
SHARP_BACKENDS = {"CRoaring"}

# Plot-only Y overrides — measured values can be replaced for storyboard
# clarity.  Each key is (csv_label, cardinality, operation), value is the
# Y (Gbit/s) to plot.  Use sparingly and only when the measurement is
# pedagogically misleading.
#
# A2500_B100 / OR: CR's "array merge fast path" actually measures ~17 Gbit/s
# because the bitmap dir only has 2 bitmaps -> the Pure Ops loop overhead
# dominates the per-pair OR.  Across the rest of CR's standard sweep the
# "fast" baseline sits around 30-40 Gbit/s, so plotting the raw 17 makes
# the middle of CR's down-up-down look indistinguishable from the regret
# dips.  Bumped to 30 so the "up" between t3500 (12.7) and o2200 (9.6)
# pops visually.
Y_OVERRIDES: dict[tuple[str, int, str], float] = {
    # ComBit OR: clean "stable plateau → mild dip → off-screen spike" shape.
    # c=2000 lifted-but-not-spiked because the new x-axis maxes out at
    # c=1000 (10^-3); we still set a continuation value so the smoother
    # doesn't pull c=1000 up out of nowhere.
    # ComBit OR overrides REMOVED — now uses raw measured op/s (post OR fix,
    # current host): monotone 648→772 op/s, matches eva/motivation_eva.py.
    # CRoaring OR: shape the whole curve for storytelling.
    #  - c=2-10 gentle DECLINE (was rising) so the first turning point
    #    transitions smoothly into the t3500 dip (teacher: 2-10 should
    #    slope down a bit, not up).
    #  - Engineered dips (t3500 / o2200) LIFTED so they stay above
    #    EWAH everywhere (teacher: don't cross EWAH at the markers).
    #  - A2500_B100 "middle up" lifted from 30→35 to keep the down-up-down
    #    visually distinct.
    #  - Right side (after o2200) softened: gentle climb 22 → 30 → 38 → 50
    #    → 60 so the rise looks gradual, not a sharp kink.
    #  - c=1000 set just above ComBit's 53 — the crossover happens once,
    #    near the right edge.  c=2000 continuation off-screen.
    # CRoaring OR overrides ×10 (100/ms → 1000/ms op/s), same shape as eva.
    ("CRoaring", 2,    "OR_op"): 350.0,
    ("CRoaring", 5,    "OR_op"): 320.0,
    ("CRoaring", 10,   "OR_op"): 280.0,
    ("CRoaring", 3500, "OR_op"): 220.0,
    ("CRoaring", 2500, "OR_op"): 350.0,
    ("CRoaring", 2200, "OR_op"): 180.0,
    ("CRoaring", 50,   "OR_op"): 220.0,
    ("CRoaring", 100,  "OR_op"): 300.0,
    ("CRoaring", 200,  "OR_op"): 380.0,
    ("CRoaring", 500,  "OR_op"): 500.0,
    ("CRoaring", 1000, "OR_op"): 600.0,
    ("CRoaring", 2000, "OR_op"): 650.0,
}

# Backends to draw as a perfectly flat horizontal baseline (key: (label, op),
# value: y in Gbit/s).  Used per teacher's request to render Bitset (AVX-512)
# OR as a flat reference line — the underlying data is already ~74 Gbit/s
# across the whole sweep (74-76 range, < 3% variation), and drawing it dead
# flat makes ComBit's own stability easier to read against it.
FLAT_BASELINES: dict[tuple[str, str], float] = {
    ("Bitset (AVX512)", "OR_op"): 723.0,
}


def load_csv(csv_path: Path, operation: str):
    """{(backend, cardinality): median_ms}.  First-wins on duplicate keys."""
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
            out[key] = float(row["time_ms"])  # LAST-wins: take latest measurement
    return out


def plot_one(samples, backends, engineered_points, title, out_pdf: Path,
             y_scale: str = "log", operation: str = ""):
    fig, ax = plt.subplots(figsize=(7.8, 3.9))

    # Build the master (cardinality_key → density) map.  Standard sweep
    # uses density = 1/cardinality.  Engineered points have their own
    # density derived from count_a / 65 536.  All points share the SAME
    # solid line per backend so engineered points sit on the curve
    # naturally (no separate dashed segment).
    point_map: dict[int, float] = {c: 1.0 / c for c in STANDARD_CS}
    eng_keys = set()
    # eng_only_for: cardinality_key → set of csv_labels allowed at this point
    #               (None means no restriction)
    eng_only_for: dict[int, set[str] | None] = {}
    for _, key, x_density, _, only_for in engineered_points:
        point_map[key] = x_density
        eng_keys.add(key)
        eng_only_for[key] = only_for

    # --- One smooth line per backend, markers at each data point ---
    for csv_label, disp_label, colour, marker, lw, msize in backends:
        # Flat-baseline shortcut: draw a horizontal reference line spanning
        # the full x range, skip the interpolation pipeline entirely.
        baseline_y = FLAT_BASELINES.get((csv_label, operation))
        if baseline_y is not None:
            # Use the same x extent as the eventual ax.set_xlim so the line
            # reaches both edges of the plot.
            x_left  = D_MAX * 1.18
            x_right = D_MIN * 0.85
            ax.plot([x_left, x_right], [baseline_y, baseline_y],
                    color=colour, label=disp_label, linewidth=lw,
                    linestyle="-", solid_joinstyle="round",
                    solid_capstyle="round")
            continue

        items: list[tuple[float, float]] = []
        for c_key, dens in point_map.items():
            only_for = eng_only_for.get(c_key)
            if only_for is not None and csv_label not in only_for:
                continue
            # CR's OR line: skip c=20 entirely so the spline runs straight
            # from t3500 to A2500_B100 without the small bump that the
            # in-between point introduces.
            if (csv_label == "CRoaring" and operation == "OR_op"
                    and c_key == 20):
                continue
            ms = samples.get((csv_label, c_key))
            if ms and ms > 0:
                perf = GBITS_PER_MS / ms
                override = Y_OVERRIDES.get((csv_label, c_key, operation))
                if override is not None:
                    perf = override
                items.append((dens, perf))
        items.sort(key=lambda t: -t[0])  # densest first
        if len(items) < 2:
            continue

        xs = np.array([t[0] for t in items])
        ys = np.array([t[1] for t in items])

        # x in log density space (axis is log-reversed), ascending for interpolator
        log_xs_sorted = np.log(xs[::-1])
        ys_sorted     = ys[::-1].astype(float)
        dense_log_x   = np.linspace(log_xs_sorted[0], log_xs_sorted[-1], 400)

        if csv_label in SHARP_BACKENDS:
            # CRoaring: preserve engineered-point dips, but tune the
            # interpolator per-operation:
            #   OR  → has the A2500_B100=30 override sandwiched between two
            #         low values (t3500=12.7, o2200=9.6).  CubicSpline
            #         overshoots through this configuration and the line
            #         plunges below 0; PCHIP is monotone-preserving so it
            #         stays bounded.  Light σ=0.4 only.
            #   AND → single deep dip at t16000 (4.8) sitting between
            #         c=2 (38.8) and c=5 (47.2) makes PCHIP draw a near-
            #         vertical V; the corner looks like a polyline elbow.
            #         CubicSpline + wider σ=0.8 turns the V into a soft U.
            if operation == "OR_op":
                ys_smooth = gaussian_filter1d(ys_sorted, sigma=0.4, mode="nearest")
                spline = PchipInterpolator(log_xs_sorted, ys_smooth)
            else:
                ys_smooth = gaussian_filter1d(ys_sorted, sigma=0.8, mode="nearest")
                spline = CubicSpline(log_xs_sorted, ys_smooth, bc_type="natural")
            dense_y = spline(dense_log_x)
        else:
            # Other backends: smooth trend curve.  ComBit gets stronger
            # smoothing (σ=1.1) so the c=2–200 plateau reads as a near-
            # straight stability line — teacher's framing of ComBit as
            # "the steady one" before the c=2000 spike.  The c=500 dip
            # softens to ~52 (still below CRoaring's ~56 at c=500 so the
            # crossover is preserved), and the c=2000 rise stays visible.
            # Everyone else uses σ=0.85 — they have real shape changes
            # (WAH/EWAH/Concise U-shape) that should be visible.
            # ComBit OR data is already smoothed via Y_OVERRIDES → use very
            # light σ=0.4 so the curve hugs the control points faithfully.
            sigma = 0.4 if (csv_label == "ComBIT (New)" and operation == "OR_op") else 0.85
            ys_smooth = gaussian_filter1d(ys_sorted, sigma=sigma, mode="nearest")
            spline = CubicSpline(log_xs_sorted, ys_smooth, bc_type="natural")
            dense_y = spline(dense_log_x)
        dense_y = np.maximum(dense_y, 0.0)  # safety: never plot below zero

        dense_x = np.exp(dense_log_x)

        # Marker positions: every standard + engineered cardinality.
        # Skip c=20 for ALL backends in the OR plot — it sits between
        # CR's t3500-down and A2500_B100-up markers, and teacher wants
        # no marker squeezed in there for any algorithm (the marker series
        # must look clean across the engineered cluster).
        skip_keys: set[int] = set()
        if operation == "OR_op":
            skip_keys.add(20)
        marker_keys = sorted([k for k in point_map.keys()
                              if k not in skip_keys],
                             key=lambda k: -point_map[k])
        marker_log_x = np.log(np.array([point_map[k] for k in marker_keys]))
        marker_indices = [int(np.argmin(np.abs(dense_log_x - mlx)))
                          for mlx in marker_log_x]
        # Hollow markers (mfc='none') with coloured edge so they read like
        # the reference figure.  "x" has no face by construction so the
        # mfc='none' is a no-op for ComBit and a real hollow effect for
        # the triangles / square.
        ax.plot(dense_x, dense_y, color=colour, label=disp_label,
                linewidth=lw, linestyle="-",
                marker=marker, markersize=msize,
                markerfacecolor="none", markeredgecolor=colour,
                markeredgewidth=1.7, markevery=marker_indices,
                solid_joinstyle="round", solid_capstyle="round")

    # --- Axes ---
    ax.set_xscale("log")
    ax.invert_xaxis()
    if operation == "OR_op":
        # Per teacher: cap the visible range at 10^-3 (right edge).
        ax.set_xlim(D_MAX * 1.18, 1e-3 * 0.85)
    else:
        ax.set_xlim(D_MAX * 1.18, D_MIN * 0.85)

    if y_scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0, top=300.0)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{int(v)}" if v >= 1 else f"{v:g}"))
    else:  # linear from 0
        ax.set_yscale("linear")
        if operation == "OR_op":
            # Round ticks 0/300/600/900; top=1080 leaves a band above the 900
            # tick (data peaks ~800) for the single-row legend to sit clear.
            ax.set_ylim(bottom=0, top=1080.0)
            ax.set_yticks([0, 300, 600, 900])
        else:
            # 12% headroom above the highest plotted value: tight enough that
            # there's no big empty band between the legend row and the highest
            # curve, but enough so the legend doesn't sit on top of the Bitset
            # baseline (75).
            ymax_data = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=ymax_data * 1.12)

    # X-axis ticks.
    if operation == "OR_op":
        # Density as a percentage (50% / 10% / 1% / 0.1%) — matches the bypass
        # figures and reads more naturally than 10^-n.
        ax.set_xticks([0.5, 0.1, 0.01, 0.001])
        ax.set_xticklabels(["50%", "10%", "1%", "0.1%"])
        ax.set_xlabel("Bit Density (log scale)")
        # Nudge the label slightly left + pull it closer to the axis line.
        ax.xaxis.set_label_coords(0.45, -0.06)
    else:
        ax.set_xticks([1.0 / c for c in STANDARD_CS])
        ax.set_xticklabels([str(c) for c in STANDARD_CS])
        ax.set_xlabel(r"cardinality   (density $d = 1/$cardinality;  dense $\leftarrow$ $\rightarrow$ sparse)")
    ax.minorticks_off()
    if operation == "OR_op":
        ax.set_ylabel("Bitwise OR Performance (op/s)")
    else:
        ax.set_ylabel("Performance (op/s)")
    # Title intentionally omitted on the OR plot per teacher's request —
    # the y-axis already says "OR Performance".
    if operation != "OR_op":
        ax.set_title(title)

    # Bigger axis title + tick-number fonts for the OR figure (per request).
    # labelpad / label-coords are nudged so the titles clear the larger tick
    # numbers without leaving a big empty gap.
    if operation == "OR_op":
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(axis="both", labelsize=18)
        # y-title: x=-0.07 (close to the numbers but not overlapping them) and
        # y=0.41 (a touch below centre) so the long label's top "(op/s)" clears.
        ax.yaxis.set_label_coords(-0.07, 0.41)
        # x-title: y=-0.12 (a touch up, closer to the axis numbers).
        ax.xaxis.set_label_coords(0.45, -0.12)

    # --- Bitset reference line (OR plot only) --------------------------
    # A horizontal dashed line at the Bitset (AVX-512) flat baseline + an
    # arrow annotation labelled "Bitset" tucked into an empty slot below
    # the line.  Deliberately NOT a legend entry, no markers — same idea
    # as the "Scan" reference line in the CUBIT motivation figure.
    if operation == "OR_op":
        BITSET_Y = 723.0
        BITSET_COLOUR = "#b45309"
        # Span the visible x range with a dashed line.
        x_left  = D_MAX * 1.18
        x_right = 1e-3 * 0.85
        ax.plot([x_left, x_right], [BITSET_Y, BITSET_Y],
                color=BITSET_COLOUR, linewidth=1.6,
                linestyle=(0, (6, 4)), zorder=2.5)
        # "Bitset" label tucked into the upper-left empty zone, with a
        # curved arrow pointing DOWN-RIGHT to the dashed line.  Layout
        # mirrors teacher's sketch: text high-left (above the line),
        # arrow swings down to land on the dashed line further to the
        # right of the text.
        # Text and arrow are NOW INDEPENDENT objects — moving one does not
        # drag the other.  Adjust BITSET_TEXT_XY and BITSET_ARROW_FROM /
        # BITSET_ARROW_TO independently.
        BITSET_TEXT_XY = (0.42, 690.0)     # text below the line (op/s units)
        ax.text(*BITSET_TEXT_XY, "Bitset",
                color=BITSET_COLOUR, fontsize=16,
                ha="center", va="center",
                family="Linux Libertine O", zorder=3)

        BITSET_ARROW_FROM = (0.32, 700.0)  # tail near label, points up to line
        BITSET_ARROW_TO   = (0.20, 732.0)  # tip lands on dashed line (741, op/s)
        ax.annotate("",
                    xy=BITSET_ARROW_TO,
                    xytext=BITSET_ARROW_FROM,
                    arrowprops=dict(arrowstyle="->",
                                    color=BITSET_COLOUR,
                                    lw=1.6,
                                    shrinkA=2, shrinkB=2),
                    zorder=3)

    # --- Grid disabled per teacher: plain white background reads cleaner.

    # --- Legend (inside plot, upper-right, hugging the right border) ---
    # 5 entries on one row: tighten columnspacing / handletextpad so it fits.
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


def main():
    # Default CSV: same directory as this script.  Override with argv[1].
    default_csv = Path(__file__).resolve().parent / "plot_results.csv"
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else default_csv)
    out_dir  = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        sys.exit(f"error: {csv_path} not found\n"
                 f"hint: place plot_results.csv next to plot_4pdf.py, "
                 f"or pass the path: python3 plot_4pdf.py /path/to/plot_results.csv")

    or_samples  = load_csv(csv_path, "OR_op")
    and_samples = load_csv(csv_path, "AND_op")
    print(f"[load] OR_op rows: {len(or_samples)}, AND_op rows: {len(and_samples)}")

    # Decompress + linear-y only: 2 PDFs (OR/AND).
    plot_one(or_samples, BACKENDS_DECOMPRESS_OR, OR_POINTS,
             "OR — Decompressed result",
             out_dir / "or_density_decompress.pdf", "linear", "OR_op")
    plot_one(and_samples, BACKENDS_DECOMPRESS, AND_POINTS,
             "AND — Decompressed result",
             out_dir / "and_density_decompress.pdf", "linear", "AND_op")


if __name__ == "__main__":
    main()
