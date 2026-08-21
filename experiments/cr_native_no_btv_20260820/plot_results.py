#!/usr/bin/env python3
"""Render native-CR sensitivity figures from the compact result CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.path import Path as MarkerPath
from matplotlib.transforms import Bbox


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_FIGURES = HERE / "figures"
INPUT_FILES = (
    "earth_native_logical.csv",
    "job_native_logical.csv",
    "cluster_native_logical.csv",
    "density_native_heatmap.csv",
)

BACKENDS = ("DDC", "CRoaring", "WAH", "EWAH")
OPERATIONS = ("OR", "AND", "NOT", "COMP")
EARTH_CASES = ("month", "qflag", "decade", "temp_bin", "year")
JOB_CASES = (
    "mc_ctype",
    "kind",
    "role",
    "pi_itype",
    "mi_itype",
    "year",
    "country",
    "movie_keyword_movie_id",
    "cast_info_movie_id",
    "cast_info_person_id",
)
CLUSTER_CASES = ("iid", "f4", "f16", "f64", "f256")

EARTH_LABELS = {
    "month": "Month",
    "qflag": "QFlag",
    "decade": "Decade",
    "temp_bin": "Temperature bin",
    "year": "Year",
}
JOB_LABELS = {
    "mc_ctype": "Company\ntype",
    "kind": "Title\nkind",
    "role": "Cast\nrole",
    "pi_itype": "Person-info\ntype",
    "mi_itype": "Movie-info\ntype",
    "year": "Production\nyear",
    "country": "Company\ncountry",
    "movie_keyword_movie_id": "Keyword\nmovie ID",
    "cast_info_movie_id": "Cast\nmovie ID",
    "cast_info_person_id": "Cast\nperson ID",
}

DENSITY_LABELS = ("0.1%", "0.5%", "1%", "2%", "5%", "10%", "20%", "30%", "40%", "50%")
DENSITY_BACKENDS = (
    ("DDC", "ddc"),
    ("CRoaring", "croaring"),
    ("WAH", "wah"),
    ("EWAH", "ewah"),
    ("Bitset-AVX512", "bitset_avx"),
    ("Concise", "concise"),
)

LOGICAL_COLUMNS = {
    "group",
    "case",
    "num_rows",
    "operation",
    "backend",
    "time_ms",
    "latency_over_ddc",
    "winner",
    "source_operation",
    "source_file",
    "replicates",
    "min_ms",
    "max_ms",
}
CLUSTER_EXTRA_COLUMNS = {"cf1", "throughput_ops_s"}
DENSITY_COLUMNS = ["backend", "density_A", "density_B", "or_time_ms"]

BAR_STYLES = {
    "CRoaring": ("#00817F", "#005755", "xxxx"),
    "WAH": ("#8B0303", "#5d0202", "/"),
    "EWAH": ("#4E0980", "#330558", r"\\\\"),
}

TRIANGLE_SCALE = 1.5
TRIANGLE_DOWN = MarkerPath(
    [
        [-0.5 * TRIANGLE_SCALE, (1.0 / 3.0) * TRIANGLE_SCALE],
        [0.5 * TRIANGLE_SCALE, (1.0 / 3.0) * TRIANGLE_SCALE],
        [0.0, (-2.0 / 3.0) * TRIANGLE_SCALE],
        [-0.5 * TRIANGLE_SCALE, (1.0 / 3.0) * TRIANGLE_SCALE],
    ]
)
TRIANGLE_UP = MarkerPath(
    [
        [-0.5 * TRIANGLE_SCALE, (-1.0 / 3.0) * TRIANGLE_SCALE],
        [0.5 * TRIANGLE_SCALE, (-1.0 / 3.0) * TRIANGLE_SCALE],
        [0.0, (2.0 / 3.0) * TRIANGLE_SCALE],
        [-0.5 * TRIANGLE_SCALE, (-1.0 / 3.0) * TRIANGLE_SCALE],
    ]
)
SQUARE_HALF = 0.75 / 2.0
SQUARE = MarkerPath(
    [
        [-SQUARE_HALF, -SQUARE_HALF],
        [SQUARE_HALF, -SQUARE_HALF],
        [SQUARE_HALF, SQUARE_HALF],
        [-SQUARE_HALF, SQUARE_HALF],
        [-SQUARE_HALF, -SQUARE_HALF],
    ]
)
LINE_STYLES = (
    ("DDC", "#1f4ed8", "x", 2.2, 9),
    ("CRoaring", "#16a34a", TRIANGLE_DOWN, 2.0, 12),
    ("WAH", "#dc2626", TRIANGLE_UP, 1.8, 12),
    ("EWAH", "#b8860b", SQUARE, 1.8, 9),
)
OP_YLABELS = {
    "OR": "OR Throughput (op/s)",
    "AND": "AND Throughput (op/s)",
    "NOT": "NOT Throughput (op/s)",
    "COMP": "Comp. Throughput (op/s)",
}
CR_NATIVE_SOURCE = {
    "OR": "OR_op",
    "AND": "AND_op",
    "NOT": "NOT_op",
    "COMP": "COMP_op_native",
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "native_gwr", ["#a1d99b", "#ffffff", "#d73027"]
)


def select_font_family() -> str:
    for family in ("Linux Libertine O", "DejaVu Serif"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise RuntimeError("No supported serif font is available")


FONT_FAMILY = select_font_family()


def configure_fonts(base_size: float = 12.0) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.sans-serif": [FONT_FAMILY],
            "font.size": base_size,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.cal": FONT_FAMILY,
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.65,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the isolated native-CR sensitivity figures"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="Wait up to this many seconds for all four compact CSVs",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--png-dpi", type=int, default=200)
    return parser.parse_args()


def wait_for_inputs(results_dir: Path, wait_seconds: float, poll_seconds: float) -> list[Path]:
    if wait_seconds < 0 or poll_seconds <= 0:
        raise ValueError("wait-seconds must be non-negative and poll-seconds positive")
    paths = [results_dir / name for name in INPUT_FILES]
    deadline = time.monotonic() + wait_seconds
    while True:
        missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
        if not missing:
            return paths
        if time.monotonic() >= deadline:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(f"compact result CSVs are not ready: {names}")
        print("waiting for: " + ", ".join(path.name for path in missing), flush=True)
        time.sleep(min(poll_seconds, max(deadline - time.monotonic(), 0.0)))


def positive_float(value: str, location: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{location}: expected a positive finite value, got {value!r}")
    return result


def read_logical(
    path: Path,
    expected_group: str,
    cases: tuple[str, ...],
    operations: tuple[str, ...],
    cluster: bool = False,
) -> dict[tuple[str, str, str], dict[str, float | str]]:
    required = LOGICAL_COLUMNS | (CLUSTER_EXTRA_COLUMNS if cluster else set())
    expected = {
        (case, operation, backend)
        for case in cases
        for operation in operations
        for backend in BACKENDS
    }
    rows: dict[tuple[str, str, str], dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"{path}: missing columns {missing}")
        for line_number, row in enumerate(reader, start=2):
            if row["group"] != expected_group:
                raise ValueError(
                    f"{path}:{line_number}: group {row['group']!r} != {expected_group!r}"
                )
            key = (row["case"], row["operation"], row["backend"])
            if key not in expected:
                continue
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate row {key}")
            time_ms = positive_float(row["time_ms"], f"{path}:{line_number}:time_ms")
            ratio = positive_float(
                row["latency_over_ddc"], f"{path}:{line_number}:latency_over_ddc"
            )
            record: dict[str, float | str] = {
                "time_ms": time_ms,
                "ratio": ratio,
                "source_operation": row["source_operation"],
            }
            if row["backend"] == "CRoaring":
                expected_source = CR_NATIVE_SOURCE[row["operation"]]
                if row["source_operation"] != expected_source:
                    raise ValueError(
                        f"{path}:{line_number}: CRoaring {row['operation']} uses "
                        f"{row['source_operation']!r}, expected native {expected_source!r}"
                    )
            if cluster:
                cf1 = positive_float(row["cf1"], f"{path}:{line_number}:cf1")
                throughput = positive_float(
                    row["throughput_ops_s"],
                    f"{path}:{line_number}:throughput_ops_s",
                )
                expected_throughput = 1000.0 / time_ms
                if not math.isclose(
                    throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"{path}:{line_number}: throughput {throughput} != "
                        f"1000/time_ms {expected_throughput}"
                    )
                record["cf1"] = cf1
                record["throughput"] = throughput
            rows[key] = record

    missing_rows = sorted(expected - rows.keys())
    if missing_rows:
        raise ValueError(f"{path}: missing {len(missing_rows)} rows; first={missing_rows[:4]}")

    for case in cases:
        for operation in operations:
            ddc = float(rows[(case, operation, "DDC")]["time_ms"])
            for backend in BACKENDS:
                record = rows[(case, operation, backend)]
                calculated = float(record["time_ms"]) / ddc
                if not math.isclose(
                    float(record["ratio"]), calculated, rel_tol=2e-9, abs_tol=2e-9
                ):
                    raise ValueError(
                        f"{path}: ratio mismatch for {(case, operation, backend)}: "
                        f"{record['ratio']} != {calculated}"
                    )
            if cluster:
                cf_values = {
                    float(rows[(case, operation, backend)]["cf1"])
                    for backend in BACKENDS
                }
                if len(cf_values) != 1:
                    raise ValueError(f"{path}: inconsistent cf1 for {case}/{operation}")
    return rows


def power_of_two_ticks(max_value: float, min_value: float) -> tuple[float, float, list[float]]:
    lower_exp = min(-3, math.floor(math.log2(min_value)))
    upper_exp = max(1, math.ceil(math.log2(max_value * 1.12)))
    occupied = (math.log2(max_value) - lower_exp) / (upper_exp - lower_exp)
    if occupied > 0.9:
        upper_exp += 1
    lower = 2.0**lower_exp
    upper = 2.0**upper_exp
    tick_start = lower_exp if lower_exp >= -2 else lower_exp + 1
    ticks = [2.0**exponent for exponent in range(tick_start, upper_exp + 1)]
    return lower, upper, ticks


def ratio_tick(value: float, _: int) -> str:
    return f"{value:g}"


def save_fixed(
    fig: plt.Figure, stem: Path, png_dpi: int, tight: bool = False
) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    options = {"bbox_inches": "tight", "pad_inches": 0.05} if tight else {}
    fig.savefig(pdf, format="pdf", **options)
    fig.savefig(png, format="png", dpi=png_dpi, **options)
    plt.close(fig)
    return [pdf, png]


def plot_relative_bars(
    rows: dict[tuple[str, str, str], dict[str, float | str]],
    cases: tuple[str, ...],
    operations: tuple[str, ...],
    case_labels: dict[str, str],
    output_stem: Path,
    png_dpi: int,
) -> list[Path]:
    configure_fonts(14)
    entries = [(case, operation) for case in cases for operation in operations]
    x = np.arange(1, len(entries) + 1, dtype=float)
    all_ratios = [
        float(rows[(case, operation, backend)]["ratio"])
        for case, operation in entries
        for backend in BACKENDS
    ]
    lower, upper, ticks = power_of_two_ticks(max(all_ratios), min(all_ratios))

    fig, axis = plt.subplots(figsize=(12, 3.25))
    width = 0.20
    bar_handles = []
    bar_backends = BACKENDS[1:]
    for offset, backend in zip((-0.25, 0.0, 0.25), bar_backends):
        face, edge, hatch = BAR_STYLES[backend]
        values = [float(rows[(case, operation, backend)]["ratio"]) for case, operation in entries]
        heights = [value - lower for value in values]
        container = axis.bar(
            x + offset,
            heights,
            width=width,
            bottom=lower,
            facecolor=to_rgba(face, 0.12),
            edgecolor=edge,
            linewidth=0.9,
            hatch=hatch,
            label=backend,
            zorder=3,
        )
        bar_handles.append(container)

    ddc_handle = Line2D(
        [0], [0], color="#147014", linewidth=1.6,
        linestyle=(0, (3, 4)), label="DDC (1.0)"
    )
    axis.axhline(1.0, color="#147014", linewidth=1.6, linestyle=(0, (3, 4)), zorder=2)
    axis.set_yscale("log", base=2)
    axis.set_ylim(lower, upper)
    if len(operations) == 1 and len(ticks) > 6:
        ticks = ticks[::2]
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(ratio_tick))
    axis.yaxis.set_minor_locator(mticker.NullLocator())
    axis.set_ylabel("Latency relative to DDC", fontsize=16, labelpad=8)
    axis.yaxis.set_label_coords(-0.075, 0.5)
    axis.tick_params(axis="y", labelsize=13, direction="out")
    axis.tick_params(axis="x", labelsize=14, length=0, pad=4)
    axis.set_xlim(0.45, len(entries) + 0.55)
    axis.set_xticks(x)

    if len(operations) > 1:
        axis.set_xticklabels([operation for _, operation in entries])
        for case_index, case in enumerate(cases):
            center = case_index * len(operations) + (len(operations) + 1) / 2
            axis.text(
                center,
                -0.235,
                case_labels[case],
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=14,
                color="#0f172a",
            )
            if case_index:
                boundary = case_index * len(operations) + 0.5
                axis.axvline(
                    boundary,
                    color="#a0a0a0",
                    linewidth=0.8,
                    linestyle=(0, (1, 3)),
                    zorder=1,
                )
        bottom = 0.29
    else:
        axis.set_xticklabels([case_labels[case] for case, _ in entries])
        axis.tick_params(axis="x", labelsize=12)
        bottom = 0.23

    for spine in axis.spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(1.0)
    fig.legend(
        handles=[*bar_handles, ddc_handle],
        labels=[*bar_backends, "DDC (1.0)"],
        loc="upper center",
        bbox_to_anchor=(0.69, 0.99),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#b7c3d0",
        framealpha=1.0,
        fontsize=13,
        handlelength=1.4,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.83, bottom=bottom)
    return save_fixed(fig, output_stem, png_dpi, tight=True)


def plot_cluster(
    rows: dict[tuple[str, str, str], dict[str, float | str]],
    output_dir: Path,
    png_dpi: int,
) -> list[Path]:
    configure_fonts(12)
    outputs: list[Path] = []
    for operation in OPERATIONS:
        xs = [float(rows[(case, operation, "DDC")]["cf1"]) for case in CLUSTER_CASES]
        all_y = [
            float(rows[(case, operation, backend)]["throughput"])
            for case in CLUSTER_CASES
            for backend in BACKENDS
        ]
        fig, axis = plt.subplots(figsize=(4, 2.5))
        for backend, color, marker, line_width, marker_size in LINE_STYLES:
            ys = [
                float(rows[(case, operation, backend)]["throughput"])
                for case in CLUSTER_CASES
            ]
            axis.plot(
                xs,
                ys,
                color=color,
                label=backend,
                linewidth=line_width,
                marker=marker,
                markersize=marker_size,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.3,
                solid_joinstyle="round",
                solid_capstyle="round",
                clip_on=False,
            )

        axis.set_xscale("log", base=2)
        axis.set_xlim(min(xs) / 1.2, max(xs) * 1.2)
        axis.set_xticks([1.0, 4.0, 16.0, 64.0, 256.0])
        axis.set_xticklabels(["1", "4", "16", "64", "256"], fontsize=13)
        axis.set_xlabel(r"Clustering Factor $\mathrm{CF}_{1}$ (log scale)", fontsize=15)
        axis.xaxis.set_label_coords(0.5, -0.14)

        axis.set_yscale("log")
        log_low = math.log10(min(all_y))
        log_high = math.log10(max(all_y))
        log_span = max(log_high - log_low, 1.0)
        axis.set_ylim(10 ** (log_low - 0.05 * log_span), 10 ** (log_high + 0.28 * log_span))
        axis.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        axis.tick_params(axis="y", labelsize=13)
        axis.set_ylabel(OP_YLABELS[operation], fontsize=15, labelpad=2)
        axis.yaxis.set_label_coords(-0.135, 0.46 if operation == "OR" else 0.42)
        axis.minorticks_off()
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=True,
            edgecolor="#94a3b8",
            borderpad=0.25,
            columnspacing=0.6,
            handletextpad=0.3,
            handlelength=1.2,
            fontsize=11,
        )
        for spine in axis.spines.values():
            spine.set_color("#1f2937")
            spine.set_linewidth(1.0)
        axis.tick_params(axis="both", colors="#1f2937")
        fig.subplots_adjust(left=0.17, right=0.96, top=0.985, bottom=0.22)
        outputs.extend(
            save_fixed(fig, output_dir / f"{operation.lower()}_clustering_native", png_dpi)
        )
    return outputs


def read_density(path: Path) -> dict[str, np.ndarray]:
    grids = {
        backend: np.full((len(DENSITY_LABELS), len(DENSITY_LABELS)), np.nan)
        for backend, _ in DENSITY_BACKENDS
    }
    indices = {label: index for index, label in enumerate(DENSITY_LABELS)}
    seen: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != DENSITY_COLUMNS:
            raise ValueError(
                f"{path}: columns {reader.fieldnames} != expected {DENSITY_COLUMNS}"
            )
        for line_number, row in enumerate(reader, start=2):
            backend = row["backend"]
            density_a = row["density_A"]
            density_b = row["density_B"]
            if backend not in grids:
                raise ValueError(f"{path}:{line_number}: unknown backend {backend!r}")
            if density_a not in indices or density_b not in indices:
                raise ValueError(
                    f"{path}:{line_number}: unknown densities {density_a!r}/{density_b!r}"
                )
            key = (backend, density_a, density_b)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate cell {key}")
            seen.add(key)
            grids[backend][indices[density_a], indices[density_b]] = positive_float(
                row["or_time_ms"], f"{path}:{line_number}:or_time_ms"
            )

    expected_count = len(DENSITY_BACKENDS) * len(DENSITY_LABELS) ** 2
    if len(seen) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} cells, found {len(seen)}")
    for backend, grid in grids.items():
        if np.isnan(grid).any():
            raise ValueError(f"{path}: missing cells for {backend}")
        if not np.allclose(grid, grid.T, rtol=0.0, atol=0.0):
            raise ValueError(f"{path}: non-symmetric grid for {backend}")
    return grids


def format_heatmap_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_density(
    grids: dict[str, np.ndarray], output_dir: Path, png_dpi: int
) -> list[Path]:
    configure_fonts(18)
    plt.rcParams.update(
        {
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )
    global_min = min(float(np.min(grid)) for grid in grids.values())
    global_max = max(float(np.max(grid)) for grid in grids.values())
    norm = LogNorm(vmin=global_min, vmax=global_max)
    figures: list[tuple[plt.Figure, str]] = []
    tight_boxes: list[Bbox] = []
    size = len(DENSITY_LABELS)
    edges = np.arange(size + 1) - 0.5

    for backend, slug in DENSITY_BACKENDS:
        grid = grids[backend]
        fig, axis = plt.subplots(figsize=(7.6, 5.8511))
        mesh = axis.pcolormesh(edges, edges, grid, cmap=HEATMAP_CMAP, norm=norm, shading="flat")
        axis.set_aspect("equal")
        for boundary in edges:
            axis.axhline(boundary, color="#4a4a4a", linewidth=1.1)
            axis.axvline(boundary, color="#4a4a4a", linewidth=1.1)
        for i in range(size):
            for j in range(size):
                axis.text(
                    j,
                    i,
                    format_heatmap_value(float(grid[i, j])),
                    ha="center",
                    va="center",
                    fontsize=18,
                    color="black",
                )
        axis.set_xticks(range(size), DENSITY_LABELS)
        axis.set_yticks(range(size), DENSITY_LABELS)
        axis.set_xlabel("Bit Density of Bitvector $B$")
        axis.set_ylabel("Bit Density of Bitvector $A$")
        axis.set_xlim(-0.5, size - 0.5)
        axis.set_ylim(-0.5, size - 0.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.02)
        colorbar.set_label("OR Latency (ms)", fontsize=18)
        colorbar.set_ticks([1.5, 3, 6, 12, 24, 48])
        colorbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:g}"))
        colorbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
        colorbar.ax.tick_params(labelsize=18)
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        tight_boxes.append(fig.get_tightbbox(fig.canvas.get_renderer()))
        figures.append((fig, slug))

    union = Bbox.union(tight_boxes)
    common_bbox = Bbox.from_extents(
        union.x0 - 0.03,
        union.y0 + 0.015,
        union.x1 + 0.03,
        union.y1 + 0.02,
    )
    outputs: list[Path] = []
    for fig, slug in figures:
        pdf = output_dir / f"density_grid_{slug}_native.pdf"
        png = output_dir / f"density_grid_{slug}_native.png"
        fig.savefig(pdf, bbox_inches=common_bbox, pad_inches=0)
        fig.savefig(png, bbox_inches=common_bbox, pad_inches=0, dpi=png_dpi)
        plt.close(fig)
        outputs.extend((pdf, png))
    return outputs


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        wait_for_inputs(results_dir, args.wait_seconds, args.poll_seconds)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    output_dir.mkdir(parents=True, exist_ok=True)

    earth = read_logical(
        results_dir / "earth_native_logical.csv",
        "earth",
        EARTH_CASES,
        ("OR", "COMP"),
    )
    job = read_logical(
        results_dir / "job_native_logical.csv",
        "job",
        JOB_CASES,
        ("COMP",),
    )
    cluster = read_logical(
        results_dir / "cluster_native_logical.csv",
        "cluster",
        CLUSTER_CASES,
        OPERATIONS,
        cluster=True,
    )
    density = read_density(results_dir / "density_native_heatmap.csv")

    outputs: list[Path] = []
    outputs.extend(
        plot_relative_bars(
            earth,
            EARTH_CASES,
            ("OR", "COMP"),
            EARTH_LABELS,
            output_dir / "earth_or_comp_native_relative",
            args.png_dpi,
        )
    )
    outputs.extend(
        plot_relative_bars(
            job,
            JOB_CASES,
            ("COMP",),
            JOB_LABELS,
            output_dir / "job_comp_native_relative",
            args.png_dpi,
        )
    )
    outputs.extend(plot_cluster(cluster, output_dir, args.png_dpi))
    outputs.extend(plot_density(density, output_dir, args.png_dpi))
    for path in outputs:
        print(f"wrote {path}")
    print(f"rendered {len(outputs) // 2} figures from four validated compact CSVs")


if __name__ == "__main__":
    main()
