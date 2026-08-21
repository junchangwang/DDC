#!/usr/bin/env python3
"""Aggregate the fresh selected-delivery rerun without frozen baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import run_all as run


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RESULTS = HERE / "results"
CORE_DISPLAY = tuple(run.DISPLAY[key] for key in run.CORE_BACKENDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def cluster_factors() -> dict[str, float]:
    source = (
        run.WORKSPACE / "R2W1_clustering_micro_100m_20260813" / "results" /
        "size_summary.csv"
    )
    result: dict[str, float] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["factor"]] = float(row["actual_f"])
    missing = {case.label for case in run.cluster_cases(5)} - result.keys()
    if missing:
        raise RuntimeError(f"missing clustering factors {sorted(missing)}")
    return result


def main() -> None:
    metadata_path = HERE / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256(Path(metadata["binary"])) != metadata["binary_sha256"]:
        raise RuntimeError("benchmark binary hash drift")
    cases = [run.Case(**record) for record in metadata["cases"]]
    factors = cluster_factors()
    grouped: dict[str, list[dict[str, object]]] = {
        "earth": [], "job": [], "cluster": []
    }
    density_upper: list[dict[str, object]] = []
    container_rows: list[dict[str, object]] = []
    summary_groups: dict[str, object] = {}

    for group in ("earth", "job", "cluster", "density"):
        group_cases = [case for case in cases if case.group == group]
        if not group_cases:
            continue
        wins: Counter[str] = Counter()
        ratios: dict[str, list[float]] = {}
        cells = 0
        for case in group_cases:
            aggregate: dict[tuple[str, str], tuple[float, float, float, list[str], list[float]]] = {}
            for backend in case.backends:
                samples_by_op: dict[str, list[float]] = {
                    operation: [] for operation in case.operations
                }
                sources: list[str] = []
                for rep in range(1, case.repetitions + 1):
                    path = RAW / f"{group}_r{rep}_{case.label}_{backend}.csv"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    selected = run.selected_rows(path, case, backend)
                    sources.append(str(path.relative_to(HERE)))
                    for operation, row in selected.items():
                        samples_by_op[operation].append(float(row["time_ms"]))
                for operation, samples in samples_by_op.items():
                    aggregate[(backend, operation)] = (
                        statistics.median(samples), min(samples), max(samples),
                        sources, samples,
                    )

            for operation in case.operations:
                cells += 1
                winner_key = min(
                    case.backends, key=lambda key: aggregate[(key, operation)][0]
                )
                winner = run.DISPLAY[winner_key]
                wins[winner] += 1
                ddc_time = aggregate[("ddc", operation)][0]
                for backend in case.backends:
                    display = run.DISPLAY[backend]
                    median, minimum, maximum, sources, samples = aggregate[(backend, operation)]
                    ratio = median / ddc_time
                    ratios.setdefault(display, []).append(ratio)
                    if group == "density":
                        density_upper.append({
                            "backend": display,
                            "case": case.label,
                            "or_time_ms": median,
                            "min_ms": minimum,
                            "max_ms": maximum,
                            "replicates": case.repetitions,
                            "source_operation": run.target_operation(backend, operation),
                            "source_files": ";".join(sources),
                        })
                        continue
                    record: dict[str, object] = {
                        "group": group,
                        "case": case.label,
                        "num_rows": case.rows,
                        "operation": operation,
                        "backend": display,
                        "time_ms": median,
                        "latency_over_ddc": ratio,
                        "winner": winner,
                        "source_operation": run.target_operation(backend, operation),
                        "source_file": ";".join(sources),
                        "replicates": case.repetitions,
                        "min_ms": minimum,
                        "max_ms": maximum,
                    }
                    if group == "cluster":
                        record["cf1"] = factors[case.label]
                        record["throughput_ops_s"] = 1000.0 / median
                    grouped[group].append(record)

        summary_groups[group] = {
            "cells": cells,
            "wins": dict(wins),
            "geometric_mean_latency_over_ddc": {
                backend: geomean(values) for backend, values in ratios.items()
            },
        }

    container_pattern = re.compile(
        r"\[CRoaring Storage\] array=(\d+).*?run=(\d+).*?bitset=(\d+).*?"
        r"total_bytes=(\d+)"
    )
    for case in cases:
        observed = set()
        for rep in range(1, case.repetitions + 1):
            log_path = HERE / "logs" / f"{case.group}_r{rep}_{case.label}_croaring.log"
            match = container_pattern.search(log_path.read_text(encoding="utf-8"))
            if not match:
                raise RuntimeError(f"missing CRoaring storage stats in {log_path}")
            observed.add(tuple(int(value) for value in match.groups()))
        if len(observed) != 1:
            raise RuntimeError(
                f"inconsistent CRoaring storage stats for {case.group}/{case.label}: "
                f"{sorted(observed)}"
            )
        arrays, runs, bitsets, total_bytes = observed.pop()
        container_rows.append({
            "group": case.group,
            "case": case.label,
            "array_containers": arrays,
            "run_containers": runs,
            "bitset_containers": bitsets,
            "container_payload_bytes": total_bytes,
            "has_run_container": int(runs > 0),
            "replicates_checked": case.repetitions,
        })

    RESULTS.mkdir(exist_ok=True)
    for group, rows in grouped.items():
        if not rows:
            continue
        path = RESULTS / f"{group}_selected_delivery.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    density_labels = tuple(run.COUNT_TO_DENSITY[value] for value in run.COUNTS)
    density_lookup: dict[tuple[str, str, str], float] = {}
    for row in density_upper:
        token_a, token_b = str(row["case"]).split("_B", 1)
        count_a = int(token_a[1:])
        count_b = int(token_b)
        label_a = run.COUNT_TO_DENSITY[count_a]
        label_b = run.COUNT_TO_DENSITY[count_b]
        backend = str(row["backend"])
        density_lookup[(backend, label_a, label_b)] = float(row["or_time_ms"])
        density_lookup[(backend, label_b, label_a)] = float(row["or_time_ms"])
    density_rows = []
    for backend in (run.DISPLAY[key] for key in run.DENSITY_BACKENDS):
        for label_a in density_labels:
            for label_b in density_labels:
                key = (backend, label_a, label_b)
                if key not in density_lookup:
                    raise RuntimeError(f"missing density cell {key}")
                density_rows.append({
                    "backend": backend,
                    "density_A": label_a,
                    "density_B": label_b,
                    "or_time_ms": density_lookup[key],
                })
    with (RESULTS / "density_selected_delivery.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(density_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(density_rows)
    with (RESULTS / "density_upper_triangle.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(density_upper[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(density_upper)
    with (RESULTS / "croaring_container_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(container_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(container_rows)

    summary = {
        "schema_version": 1,
        "literal_comp_plan": metadata["literal_comp_plan"],
        "croaring_optimization": metadata["croaring_optimization"],
        "delivery_boundary": metadata["delivery_boundary"],
        "groups": summary_groups,
        "croaring_cases_with_run_containers": {
            group: sum(
                int(row["has_run_container"])
                for row in container_rows if row["group"] == group
            )
            for group in ("earth", "job", "cluster", "density")
        },
        "binary_sha256": metadata["binary_sha256"],
        "metadata_sha256": sha256(metadata_path),
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
