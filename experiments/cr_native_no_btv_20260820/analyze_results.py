#!/usr/bin/env python3
"""Build compact native-CR sensitivity tables from isolated raw measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_WORKSPACE = REPO.parent

OPERATIONS = ("OR", "AND", "NOT", "COMP")
BACKENDS = ("DDC", "CRoaring", "WAH", "EWAH")
EARTH_CASES = ("month", "qflag", "decade", "temp_bin", "year")
EARTH_SPECS = {
    "month": (21_844_315, 12),
    "qflag": (21_844_315, 17),
    "decade": (21_844_315, 33),
    "temp_bin": (21_844_315, 232),
    "year": (21_844_315, 325),
}
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
JOB_LOGICAL_FIELD = {
    "mc_ctype": "mc_ctype",
    "kind": "kind",
    "role": "role",
    "pi_itype": "pi_itype",
    "mi_itype": "mi_itype",
    "year": "year",
    "country": "country",
    "movie_keyword_movie_id": "movie_keyword.movie_id",
    "cast_info_movie_id": "cast_info.movie_id",
    "cast_info_person_id": "cast_info.person_id",
}
CLUSTER_CASES = (
    "iid",
    "f2",
    "f4",
    "f8",
    "f16",
    "f32",
    "f64",
    "f128",
    "f256",
    "f1024",
    "sorted",
)
COUNTS = (66, 328, 655, 1311, 3277, 6554, 13107, 19661, 26214, 32768)
DENSITY_LABELS = ("0.1%", "0.5%", "1%", "2%", "5%", "10%", "20%", "30%", "40%", "50%")
DENSITY_BACKENDS = ("DDC", "CRoaring", "WAH", "EWAH", "Bitset-AVX512", "Concise")

NATIVE_OPERATION = {
    "OR": "OR_op",
    "AND": "AND_op",
    "NOT": "NOT_op",
    "COMP": "COMP_op_native",
}
DELIVERED_OPERATION = {
    "OR": "OR_op_conv",
    "AND": "AND_op_conv",
    "NOT": "NOT_op_conv",
    "COMP": "COMP_op",
}
BASELINE_BACKEND = {
    "DDC (New)": "DDC",
    "DDC": "DDC",
    "CRoaring": "CRoaring",
    "WAH (FastBit)": "WAH",
    "WAH": "WAH",
    "EWAH": "EWAH",
}

LONG_FIELDS = (
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
)
CLUSTER_FIELDS = LONG_FIELDS + ("cf1", "throughput_ops_s")
ANCHOR_FIELDS = (
    "group",
    "case",
    "operation",
    "native_cr_ms",
    "same_run_delivered_cr_ms",
    "historical_delivered_cr_ms",
    "native_over_same_run_delivered",
    "native_speedup_from_removing_delivery",
    "delivery_overhead_ms",
    "delivery_share_pct",
    "same_run_delivered_over_historical",
    "replicates",
    "native_source_operation",
    "delivered_source_operation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--raw-root", type=Path, default=HERE / "raw")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--table4-run-csv",
        type=Path,
        default=HERE / "results" / "table4_cr_run_sizes.csv",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"{path}: missing CSV header")
        return list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path, workspace: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(workspace.resolve()))
    except ValueError:
        return str(path)


def positive_float(value: str, context: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise RuntimeError(f"{context}: invalid positive float {value!r}")
    return result


def geometric_mean(values: Iterable[float]) -> float:
    checked = list(values)
    if not checked or any(
        value <= 0.0 or not math.isfinite(value) for value in checked
    ):
        raise RuntimeError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def single_row(rows: Iterable[dict[str, str]], context: str) -> dict[str, str]:
    selected = list(rows)
    if len(selected) != 1:
        raise RuntimeError(f"{context}: expected one row, found {len(selected)}")
    return selected[0]


def run_metadata_paths(raw_root: Path) -> list[Path]:
    paths = sorted(raw_root.parent.glob("run_metadata_*.json"))
    if paths:
        return paths
    legacy = raw_root.parent / "run_metadata.json"
    return [legacy] if legacy.is_file() else []


def infer_expected_reps(raw_root: Path, group: str) -> int | None:
    matches: list[tuple[Path, dict[str, object]]] = []
    metadata_paths = run_metadata_paths(raw_root)
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if group in metadata.get("groups", []):
            matches.append((metadata_path, metadata))
    if not matches:
        if metadata_paths:
            raise RuntimeError(
                f"group {group!r} is absent from run metadata files: "
                f"{[str(path) for path in metadata_paths]}"
            )
        return None
    if len(matches) != 1:
        raise RuntimeError(
            f"group {group!r} occurs in multiple run metadata files: "
            f"{[str(path) for path, _ in matches]}"
        )
    metadata_path, metadata = matches[0]
    key = "density_reps" if group == "density" else "reps"
    reps = int(metadata[key])
    if reps < 1:
        raise RuntimeError(f"{metadata_path}: invalid {key}={reps}")
    return reps


def collect_raw_cr(
    raw_root: Path,
    source_base: Path,
    group: str,
    cases: Sequence[str],
    rows_by_case: dict[str, int],
    operations: Sequence[str],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], list[Path]]:
    expected_reps = infer_expected_reps(raw_root, group)
    result: dict[tuple[str, str, str], dict[str, object]] = {}
    source_paths: list[Path] = []
    counts: set[int] = set()
    for case in cases:
        paths = sorted((raw_root / group).glob(f"r*_{case}.csv"))
        if not paths:
            raise FileNotFoundError(raw_root / group / f"r*_{case}.csv")
        if expected_reps is not None and len(paths) != expected_reps:
            raise RuntimeError(
                f"{group}/{case}: expected {expected_reps} replicates, found {len(paths)}"
            )
        counts.add(len(paths))
        source_paths.extend(paths)
        selected: dict[str, list[float]] = defaultdict(list)
        selected_cardinality: dict[str, list[str]] = defaultdict(list)
        for path in paths:
            rows = read_csv(path)
            wanted = {NATIVE_OPERATION[operation] for operation in operations}
            wanted.update(DELIVERED_OPERATION[operation] for operation in operations)
            matching = [
                row
                for row in rows
                if row.get("backend") == "CRoaring"
                and int(row.get("num_rows", "-1")) == rows_by_case[case]
                and row.get("operation") in wanted
            ]
            by_operation = {row["operation"]: row for row in matching}
            if len(by_operation) != len(matching):
                raise RuntimeError(f"{path}: duplicate CR operation row")
            missing = wanted - set(by_operation)
            if missing:
                raise RuntimeError(f"{path}: missing CR rows {sorted(missing)}")
            for operation in operations:
                for boundary, source_operation in (
                    ("native", NATIVE_OPERATION[operation]),
                    ("delivered", DELIVERED_OPERATION[operation]),
                ):
                    row = by_operation[source_operation]
                    selected[f"{operation}:{boundary}"].append(
                        positive_float(row["time_ms"], f"{path}:{source_operation}")
                    )
                    selected_cardinality[f"{operation}:{boundary}"].append(
                        row["result_cardinality"]
                    )
        for operation in operations:
            for boundary in ("native", "delivered"):
                key = f"{operation}:{boundary}"
                values = selected[key]
                if len(values) != len(paths):
                    raise RuntimeError(f"{group}/{case}/{key}: incomplete replicates")
                if operation != "COMP" and len(set(selected_cardinality[key])) != 1:
                    raise RuntimeError(
                        f"{group}/{case}/{key}: result cardinality changed"
                    )
                result[(case, operation, boundary)] = {
                    "time_ms": statistics.median(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "replicates": len(values),
                    "source_operation": (
                        NATIVE_OPERATION[operation]
                        if boundary == "native"
                        else DELIVERED_OPERATION[operation]
                    ),
                    "source_file": f"{relative(raw_root / group, source_base)}/r*_{case}.csv",
                }
    if expected_reps is None and len(counts) != 1:
        raise RuntimeError(f"{group}: inconsistent replicate counts {sorted(counts)}")
    return result, source_paths


def baseline_cell(
    time_ms: float,
    source_operation: str,
    source_file: Path,
    workspace: Path,
    replicates: int = 1,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, object]:
    if time_ms <= 0.0 or not math.isfinite(time_ms):
        raise RuntimeError(f"{source_file}: invalid time {time_ms}")
    return {
        "time_ms": time_ms,
        "min_ms": time_ms if minimum is None else minimum,
        "max_ms": time_ms if maximum is None else maximum,
        "replicates": replicates,
        "source_operation": source_operation,
        "source_file": relative(source_file, workspace),
    }


def build_long_rows(
    group: str,
    cases: Sequence[str],
    rows_by_case: dict[str, int],
    cells: dict[tuple[str, str, str], dict[str, object]],
    extra_by_case: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for case in cases:
        for operation in OPERATIONS:
            times = {
                backend: float(cells[(case, operation, backend)]["time_ms"])
                for backend in BACKENDS
            }
            ddc = times["DDC"]
            winner = min(
                BACKENDS, key=lambda backend: (times[backend], BACKENDS.index(backend))
            )
            for backend in BACKENDS:
                source = cells[(case, operation, backend)]
                row: dict[str, object] = {
                    "group": group,
                    "case": case,
                    "num_rows": rows_by_case[case],
                    "operation": operation,
                    "backend": backend,
                    "time_ms": times[backend],
                    "latency_over_ddc": times[backend] / ddc,
                    "winner": winner,
                    "source_operation": source["source_operation"],
                    "source_file": source["source_file"],
                    "replicates": source["replicates"],
                    "min_ms": source["min_ms"],
                    "max_ms": source["max_ms"],
                }
                if extra_by_case:
                    row.update(extra_by_case[case])
                output.append(row)
    return output


def load_earth_baseline(
    workspace: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    dict[tuple[str, str], float],
    list[Path],
]:
    adaptive_path = (
        workspace
        / "R3W1_realworld_excel_20260809/earth_adaptive_single_replay_20260809/results_earth_adaptive_single_20260809.csv"
    )
    current_path = (
        workspace
        / "R3W1_realworld_excel_20260809/earth_final5/results_earth_final5_20260809.csv"
    )
    ewah_path = (
        workspace
        / "R3W1_realworld_excel_20260809/ewah_corrected/earth/results_earth_ewah_corrected_20260809.csv"
    )
    adaptive = read_csv(adaptive_path)
    current = read_csv(current_path)
    corrected_ewah = read_csv(ewah_path)
    cells: dict[tuple[str, str, str], dict[str, object]] = {}
    historical_cr: dict[tuple[str, str], float] = {}
    for case in EARTH_CASES:
        rows, cardinality = EARTH_SPECS[case]
        for operation in OPERATIONS:
            source_operation = f"{operation}_op"
            specifications = (
                ("DDC", adaptive, adaptive_path, "DDC (New)"),
                ("WAH", current, current_path, "WAH (FastBit)"),
                ("EWAH", corrected_ewah, ewah_path, "EWAH"),
            )
            for backend, source_rows, source_path, csv_backend in specifications:
                row = single_row(
                    (
                        item
                        for item in source_rows
                        if item["backend"] == csv_backend
                        and int(item["num_rows"]) == rows
                        and int(item["cardinality"]) == cardinality
                        and item["operation"] == source_operation
                    ),
                    f"Earth {case}/{operation}/{backend}",
                )
                cells[(case, operation, backend)] = baseline_cell(
                    positive_float(
                        row["time_ms"], f"Earth {case}/{operation}/{backend}"
                    ),
                    source_operation,
                    source_path,
                    workspace,
                )
            delivered_operation = DELIVERED_OPERATION[operation]
            row = single_row(
                (
                    item
                    for item in current
                    if item["backend"] == "CRoaring"
                    and int(item["num_rows"]) == rows
                    and int(item["cardinality"]) == cardinality
                    and item["operation"] == delivered_operation
                ),
                f"Earth historical CR {case}/{operation}",
            )
            historical_cr[(case, operation)] = positive_float(
                row["time_ms"], f"Earth historical CR {case}/{operation}"
            )
    return cells, historical_cr, [adaptive_path, current_path, ewah_path]


def load_job_baseline(
    workspace: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    dict[tuple[str, str], float],
    dict[str, int],
    list[Path],
]:
    path = (
        workspace
        / "R3W1_realworld_excel_20260809/job_unified10/job-unified10-logical.csv"
    )
    rows = read_csv(path)
    cells: dict[tuple[str, str, str], dict[str, object]] = {}
    historical_cr: dict[tuple[str, str], float] = {}
    rows_by_case: dict[str, int] = {}
    for case in JOB_CASES:
        case_rows = [row for row in rows if row["field"] == JOB_LOGICAL_FIELD[case]]
        row_counts = {int(row["num_rows"]) for row in case_rows}
        if len(row_counts) != 1:
            raise RuntimeError(f"JOB {case}: invalid row counts {sorted(row_counts)}")
        rows_by_case[case] = row_counts.pop()
        for operation in OPERATIONS:
            for backend in ("DDC", "WAH", "EWAH"):
                row = single_row(
                    (
                        item
                        for item in case_rows
                        if item["operation"] == operation
                        and item["algorithm"] == backend
                    ),
                    f"JOB {case}/{operation}/{backend}",
                )
                cells[(case, operation, backend)] = baseline_cell(
                    positive_float(row["time_ms"], f"JOB {case}/{operation}/{backend}"),
                    row["csv_operation"],
                    path,
                    workspace,
                )
            row = single_row(
                (
                    item
                    for item in case_rows
                    if item["operation"] == operation
                    and item["algorithm"] == "CRoaring"
                ),
                f"JOB historical CR {case}/{operation}",
            )
            historical_cr[(case, operation)] = positive_float(
                row["time_ms"], f"JOB historical CR {case}/{operation}"
            )
    return cells, historical_cr, rows_by_case, [path]


def load_cluster_baseline(
    workspace: Path,
) -> tuple[
    dict[tuple[str, str, str], dict[str, object]],
    dict[tuple[str, str], float],
    dict[str, dict[str, object]],
    list[Path],
]:
    path = (
        workspace / "R2W1_clustering_ops_v2_20260813/results/operation_sweep_crrun.csv"
    )
    rows = read_csv(path)
    cells: dict[tuple[str, str, str], dict[str, object]] = {}
    historical_cr: dict[tuple[str, str], float] = {}
    extra: dict[str, dict[str, object]] = {}
    for case in CLUSTER_CASES:
        cf1_values = {float(row["f"]) for row in rows if row["factor"] == case}
        if len(cf1_values) != 1:
            raise RuntimeError(
                f"cluster {case}: invalid CF1 values {sorted(cf1_values)}"
            )
        extra[case] = {"cf1": cf1_values.pop()}
        for operation in OPERATIONS:
            for backend in ("DDC", "WAH", "EWAH"):
                row = single_row(
                    (
                        item
                        for item in rows
                        if item["factor"] == case
                        and item["operation"] == operation
                        and item["backend"] == backend
                    ),
                    f"cluster {case}/{operation}/{backend}",
                )
                cells[(case, operation, backend)] = baseline_cell(
                    positive_float(
                        row["median_ms"], f"cluster {case}/{operation}/{backend}"
                    ),
                    f"{operation}_op",
                    path,
                    workspace,
                    int(row["replicates"]),
                    float(row["min_ms"]),
                    float(row["max_ms"]),
                )
            row = single_row(
                (
                    item
                    for item in rows
                    if item["factor"] == case
                    and item["operation"] == operation
                    and item["backend"] == "CRoaring"
                ),
                f"cluster historical CR {case}/{operation}",
            )
            historical_cr[(case, operation)] = positive_float(
                row["median_ms"], f"cluster historical CR {case}/{operation}"
            )
    return cells, historical_cr, extra, [path]


def load_density_baseline(
    workspace: Path,
) -> tuple[dict[tuple[str, str, str], float], list[Path]]:
    path = workspace / "AAA/heatmap_or_data.csv"
    rows = read_csv(path)
    expected_fields = ["backend", "density_A", "density_B", "or_time_ms"]
    if list(rows[0]) != expected_fields:
        raise RuntimeError(f"{path}: unexpected columns {list(rows[0])}")
    result: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["backend"], row["density_A"], row["density_B"])
        if key in result:
            raise RuntimeError(f"{path}: duplicate density cell {key}")
        result[key] = positive_float(row["or_time_ms"], f"{path}:{key}")
    expected = len(DENSITY_BACKENDS) * len(DENSITY_LABELS) ** 2
    if len(result) != expected:
        raise RuntimeError(f"{path}: expected {expected} cells, found {len(result)}")
    return result, [path]


def add_native_cr_cells(
    cells: dict[tuple[str, str, str], dict[str, object]],
    raw: dict[tuple[str, str, str], dict[str, object]],
    cases: Sequence[str],
) -> None:
    for case in cases:
        for operation in OPERATIONS:
            cells[(case, operation, "CRoaring")] = dict(
                raw[(case, operation, "native")]
            )


def build_density_rows(
    baseline: dict[tuple[str, str, str], float],
    raw: dict[tuple[str, str, str], dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    count_to_label = dict(zip(COUNTS, DENSITY_LABELS, strict=True))
    native: dict[tuple[str, str], float] = {}
    for index_a, count_a in enumerate(COUNTS):
        for count_b in COUNTS[index_a:]:
            case = f"A{count_a}_B{count_b}"
            value = float(raw[(case, "OR", "native")]["time_ms"])
            label_a = count_to_label[count_a]
            label_b = count_to_label[count_b]
            native[(label_a, label_b)] = value
            native[(label_b, label_a)] = value
    for backend in DENSITY_BACKENDS:
        for density_a in DENSITY_LABELS:
            for density_b in DENSITY_LABELS:
                value = (
                    native[(density_a, density_b)]
                    if backend == "CRoaring"
                    else baseline[(backend, density_a, density_b)]
                )
                output.append(
                    {
                        "backend": backend,
                        "density_A": density_a,
                        "density_B": density_b,
                        "or_time_ms": value,
                    }
                )
    return output


def format_size(size_bytes: float) -> str:
    mib = size_bytes / (1 << 20)
    if mib < 0.001:
        return f"{round(size_bytes):d} B"
    if mib < 0.1:
        return f"{mib:.3f}"
    return f"{mib:.1f}"


def format_increase(increase: float) -> str:
    if increase <= -0.995:
        return "approximately -100%"
    if increase >= 1.0:
        return f"+{increase:.1f}x"
    return f"{increase * 100:+.0f}%"


def build_table4_rows(
    workspace: Path, table4_run_path: Path
) -> tuple[list[dict[str, object]], list[Path]]:
    baseline_path = (
        workspace / "R3W1_realworld_excel_20260809/job_unified10/job-unified10-size.csv"
    )
    baseline = {row["field"]: row for row in read_csv(baseline_path)}
    run_rows = read_csv(table4_run_path)
    run_by_slug = {row["slug"]: row for row in run_rows if row["slug"] != "TOTAL"}
    output: list[dict[str, object]] = []
    total_ddc = 0.0
    total_cr = 0
    for slug in JOB_CASES:
        baseline_field = JOB_LOGICAL_FIELD[slug]
        if baseline_field not in baseline or slug not in run_by_slug:
            raise RuntimeError(f"Table 4: missing {slug}")
        source = baseline[baseline_field]
        measured = run_by_slug[slug]
        if int(source["n"]) != int(measured["n"]) or int(source["c"]) != int(
            measured["c"]
        ):
            raise RuntimeError(f"Table 4 {slug}: dimension mismatch")
        if int(source["croaring_payload_bytes"]) != int(
            measured["cr_plain_payload_bytes"]
        ):
            raise RuntimeError(f"Table 4 {slug}: plain CR cross-check failed")
        ddc_bytes = int(source["ddc_payload_bits"]) / 8.0
        cr_bytes = int(measured["cr_run_payload_bytes"])
        increase = cr_bytes / ddc_bytes - 1.0
        winner = "CRoaring" if cr_bytes < ddc_bytes else "DDC"
        output.append(
            {
                "field": source["source"],
                "slug": slug,
                "n": int(source["n"]),
                "cardinality": int(source["c"]),
                "ddc_payload_bytes": ddc_bytes,
                "ddc_mib": ddc_bytes / (1 << 20),
                "ddc_display": format_size(ddc_bytes),
                "croaring_run_payload_bytes": cr_bytes,
                "croaring_run_mib": cr_bytes / (1 << 20),
                "croaring_run_display": format_size(cr_bytes),
                "croaring_inc_fraction": increase,
                "croaring_inc_display": format_increase(increase),
                "winner": winner,
            }
        )
        total_ddc += ddc_bytes
        total_cr += cr_bytes
    total_increase = total_cr / total_ddc - 1.0
    output.append(
        {
            "field": "TOTAL",
            "slug": "TOTAL",
            "n": "",
            "cardinality": "",
            "ddc_payload_bytes": total_ddc,
            "ddc_mib": total_ddc / (1 << 20),
            "ddc_display": format_size(total_ddc),
            "croaring_run_payload_bytes": total_cr,
            "croaring_run_mib": total_cr / (1 << 20),
            "croaring_run_display": format_size(total_cr),
            "croaring_inc_fraction": total_increase,
            "croaring_inc_display": format_increase(total_increase),
            "winner": "CRoaring" if total_cr < total_ddc else "DDC",
        }
    )
    return output, [baseline_path, table4_run_path]


def build_anchor_rows(
    group: str,
    cases: Sequence[str],
    operations: Sequence[str],
    raw: dict[tuple[str, str, str], dict[str, object]],
    historical: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for case in cases:
        for operation in operations:
            native = raw[(case, operation, "native")]
            delivered = raw[(case, operation, "delivered")]
            native_ms = float(native["time_ms"])
            delivered_ms = float(delivered["time_ms"])
            historical_ms = historical[(case, operation)]
            output.append(
                {
                    "group": group,
                    "case": case,
                    "operation": operation,
                    "native_cr_ms": native_ms,
                    "same_run_delivered_cr_ms": delivered_ms,
                    "historical_delivered_cr_ms": historical_ms,
                    "native_over_same_run_delivered": native_ms / delivered_ms,
                    "native_speedup_from_removing_delivery": delivered_ms / native_ms,
                    "delivery_overhead_ms": delivered_ms - native_ms,
                    "delivery_share_pct": 100.0
                    * (delivered_ms - native_ms)
                    / delivered_ms,
                    "same_run_delivered_over_historical": delivered_ms / historical_ms,
                    "replicates": native["replicates"],
                    "native_source_operation": native["source_operation"],
                    "delivered_source_operation": delivered["source_operation"],
                }
            )
    return output


def summarize_long(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    ratios: dict[str, list[float]] = defaultdict(list)
    winner_by_cell: dict[tuple[str, str], str] = {}
    for row in rows:
        ratios[str(row["backend"])].append(float(row["latency_over_ddc"]))
        winner_by_cell[(str(row["case"]), str(row["operation"]))] = str(row["winner"])
    return {
        "rows": len(rows),
        "cells": len(winner_by_cell),
        "geometric_mean_latency_over_ddc": {
            backend: geometric_mean(ratios[backend]) for backend in BACKENDS
        },
        "wins": dict(sorted(Counter(winner_by_cell.values()).items())),
    }


def summarize_density(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_backend = {
        backend: {
            (str(row["density_A"]), str(row["density_B"])): float(row["or_time_ms"])
            for row in rows
            if row["backend"] == backend
        }
        for backend in DENSITY_BACKENDS
    }
    ddc = by_backend["DDC"]
    return {
        "rows": len(rows),
        "cells_per_backend": len(ddc),
        "geometric_mean_or_latency_over_ddc": {
            backend: geometric_mean(
                by_backend[backend][cell] / ddc[cell] for cell in ddc
            )
            for backend in DENSITY_BACKENDS
        },
    }


def summarize_anchors(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for group in ("earth", "job", "cluster", "density"):
        selected = [row for row in rows if row["group"] == group]
        output[group] = {
            "rows": len(selected),
            "native_over_same_run_delivered_geomean": geometric_mean(
                float(row["native_over_same_run_delivered"]) for row in selected
            ),
            "native_speedup_from_removing_delivery_geomean": geometric_mean(
                float(row["native_speedup_from_removing_delivery"]) for row in selected
            ),
            "same_run_delivered_over_historical_geomean": geometric_mean(
                float(row["same_run_delivered_over_historical"]) for row in selected
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(workspace)

    earth_cells, earth_historical, earth_sources = load_earth_baseline(workspace)
    earth_raw, earth_raw_paths = collect_raw_cr(
        raw_root,
        workspace,
        "earth",
        EARTH_CASES,
        {case: EARTH_SPECS[case][0] for case in EARTH_CASES},
        OPERATIONS,
    )
    add_native_cr_cells(earth_cells, earth_raw, EARTH_CASES)
    earth_rows = build_long_rows(
        "earth",
        EARTH_CASES,
        {case: EARTH_SPECS[case][0] for case in EARTH_CASES},
        earth_cells,
    )

    job_cells, job_historical, job_rows_by_case, job_sources = load_job_baseline(
        workspace
    )
    job_raw, job_raw_paths = collect_raw_cr(
        raw_root, workspace, "job", JOB_CASES, job_rows_by_case, OPERATIONS
    )
    add_native_cr_cells(job_cells, job_raw, JOB_CASES)
    job_rows = build_long_rows("job", JOB_CASES, job_rows_by_case, job_cells)

    cluster_cells, cluster_historical, cluster_extra, cluster_sources = (
        load_cluster_baseline(workspace)
    )
    cluster_raw, cluster_raw_paths = collect_raw_cr(
        raw_root,
        workspace,
        "cluster",
        CLUSTER_CASES,
        {case: 100_000_000 for case in CLUSTER_CASES},
        OPERATIONS,
    )
    add_native_cr_cells(cluster_cells, cluster_raw, CLUSTER_CASES)
    cluster_rows = build_long_rows(
        "cluster",
        CLUSTER_CASES,
        {case: 100_000_000 for case in CLUSTER_CASES},
        cluster_cells,
        cluster_extra,
    )
    for row in cluster_rows:
        row["throughput_ops_s"] = 1000.0 / float(row["time_ms"])

    density_baseline, density_sources = load_density_baseline(workspace)
    density_cases = tuple(
        f"A{count_a}_B{count_b}"
        for index_a, count_a in enumerate(COUNTS)
        for count_b in COUNTS[index_a:]
    )
    density_raw, density_raw_paths = collect_raw_cr(
        raw_root,
        workspace,
        "density",
        density_cases,
        {case: 100_000_000 for case in density_cases},
        ("OR",),
    )
    density_rows = build_density_rows(density_baseline, density_raw)

    table4_run_path = args.table4_run_csv.resolve()
    table4_rows, table4_sources = build_table4_rows(workspace, table4_run_path)

    count_to_label = dict(zip(COUNTS, DENSITY_LABELS, strict=True))
    density_historical = {
        (f"A{count_a}_B{count_b}", "OR"): density_baseline[
            ("CRoaring", count_to_label[count_a], count_to_label[count_b])
        ]
        for index_a, count_a in enumerate(COUNTS)
        for count_b in COUNTS[index_a:]
    }
    anchor_rows = []
    anchor_rows.extend(
        build_anchor_rows("earth", EARTH_CASES, OPERATIONS, earth_raw, earth_historical)
    )
    anchor_rows.extend(
        build_anchor_rows("job", JOB_CASES, OPERATIONS, job_raw, job_historical)
    )
    anchor_rows.extend(
        build_anchor_rows(
            "cluster", CLUSTER_CASES, OPERATIONS, cluster_raw, cluster_historical
        )
    )
    anchor_rows.extend(
        build_anchor_rows(
            "density", density_cases, ("OR",), density_raw, density_historical
        )
    )

    expected_counts = {
        "earth": len(EARTH_CASES) * len(OPERATIONS) * len(BACKENDS),
        "job": len(JOB_CASES) * len(OPERATIONS) * len(BACKENDS),
        "cluster": len(CLUSTER_CASES) * len(OPERATIONS) * len(BACKENDS),
        "density": len(DENSITY_BACKENDS) * len(DENSITY_LABELS) ** 2,
        "table4": len(JOB_CASES) + 1,
        "anchors": (
            len(EARTH_CASES) * len(OPERATIONS)
            + len(JOB_CASES) * len(OPERATIONS)
            + len(CLUSTER_CASES) * len(OPERATIONS)
            + len(density_cases)
        ),
    }
    actual_counts = {
        "earth": len(earth_rows),
        "job": len(job_rows),
        "cluster": len(cluster_rows),
        "density": len(density_rows),
        "table4": len(table4_rows),
        "anchors": len(anchor_rows),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"row-count validation failed: {actual_counts} != {expected_counts}"
        )

    write_csv_atomic(output_dir / "earth_native_logical.csv", LONG_FIELDS, earth_rows)
    write_csv_atomic(output_dir / "job_native_logical.csv", LONG_FIELDS, job_rows)
    write_csv_atomic(
        output_dir / "cluster_native_logical.csv", CLUSTER_FIELDS, cluster_rows
    )
    write_csv_atomic(
        output_dir / "density_native_heatmap.csv",
        ("backend", "density_A", "density_B", "or_time_ms"),
        density_rows,
    )
    write_csv_atomic(
        output_dir / "table4_run_display.csv", tuple(table4_rows[0]), table4_rows
    )
    write_csv_atomic(output_dir / "anchor_comparison.csv", ANCHOR_FIELDS, anchor_rows)

    measured_raw_paths = (
        earth_raw_paths + job_raw_paths + cluster_raw_paths + density_raw_paths
    )
    table4_raw_paths = sorted((raw_root / "table4_cr_run").glob("*.csv"))
    if len(table4_raw_paths) != len(JOB_CASES):
        raise RuntimeError(
            f"Table 4: expected {len(JOB_CASES)} raw files, "
            f"found {len(table4_raw_paths)}"
        )
    log_root = raw_root.parent / "logs"
    measured_logs = []
    for raw_path in measured_raw_paths + table4_raw_paths:
        log_path = log_root / raw_path.relative_to(raw_root).with_suffix(".log")
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        measured_logs.append(log_path)
    metadata_paths = run_metadata_paths(raw_root) + [
        raw_root.parent / "table4_run_metadata.json"
    ]
    for metadata_path in metadata_paths:
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
    source_paths = sorted(
        set(
            earth_sources
            + job_sources
            + cluster_sources
            + density_sources
            + table4_sources
            + measured_raw_paths
            + table4_raw_paths
            + measured_logs
            + metadata_paths
        )
    )
    summary = {
        "schema_version": 1,
        "measurement_contract": (
            "Backend-native result sensitivity: CRoaring uses OR_op, AND_op, "
            "NOT_op, and COMP_op_native; frozen non-CR baselines retain their "
            "native result representations."
        ),
        "row_counts": actual_counts,
        "groups": {
            "earth": summarize_long(earth_rows),
            "job": summarize_long(job_rows),
            "cluster": summarize_long(cluster_rows),
            "density": summarize_density(density_rows),
        },
        "anchor_comparison": summarize_anchors(anchor_rows),
        "table4": {
            "ddc_total_mib": table4_rows[-1]["ddc_mib"],
            "croaring_run_total_mib": table4_rows[-1]["croaring_run_mib"],
            "croaring_increase_over_ddc_fraction": table4_rows[-1][
                "croaring_inc_fraction"
            ],
            "croaring_run_wins": sum(
                row["winner"] == "CRoaring" for row in table4_rows[:-1]
            ),
            "ddc_wins": sum(row["winner"] == "DDC" for row in table4_rows[:-1]),
        },
        "sources": [
            {
                "path": relative(path, workspace),
                "sha256": sha256(path),
            }
            for path in source_paths
        ],
    }
    write_json_atomic(output_dir / "summary.json", summary)
    for name, count in actual_counts.items():
        print(f"{name}: {count} rows")
    print(f"wrote compact results to {output_dir}")


if __name__ == "__main__":
    main()
