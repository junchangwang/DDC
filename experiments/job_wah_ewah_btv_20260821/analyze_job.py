#!/usr/bin/env python3
"""Aggregate the isolated JOB WAH/EWAH BTV-delivery experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from run_job import BACKENDS, CSV_BACKEND, SOURCE_OPERATION, cases


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RESULTS = HERE / "results"
NATIVE_OPERATION = {
    "ddc": "COMP_op", "croaring": "COMP_op_native",
    "wah": "COMP_op", "ewah": "COMP_op",
}
DISPLAY = {"ddc": "DDC", "croaring": "CRoaring", "wah": "WAH", "ewah": "EWAH"}


def read_value(path: Path, backend: str, operation: str, rows: int) -> tuple[float, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    matches = [
        row for row in records
        if row["backend"] == CSV_BACKEND[backend]
        and row["operation"] == operation
        and int(row["num_rows"]) == rows
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{path}: expected one {backend}/{operation} row")
    value = float(matches[0]["time_ms"])
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{path}: invalid time")
    return value, int(matches[0]["result_cardinality"])


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    metadata = json.loads((HERE / "run_metadata.json").read_text(encoding="utf-8"))
    reps = int(metadata["process_replicates"])
    expected = {key: int(value) for key, value in metadata["expected_result_cardinality"].items()}
    selected: list[dict[str, object]] = []
    breakdown: list[dict[str, object]] = []
    ratios: dict[str, list[float]] = {DISPLAY[backend]: [] for backend in BACKENDS}
    winners: Counter[str] = Counter()

    for case in cases():
        aggregates: dict[str, float] = {}
        per_backend: dict[str, dict[str, object]] = {}
        for backend in BACKENDS:
            selected_values: list[float] = []
            native_values: list[float] = []
            cards: list[int] = []
            sources: list[str] = []
            for rep in range(1, reps + 1):
                path = RAW / f"r{rep}_{case.label}_{backend}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                selected_value, card = read_value(
                    path, backend, SOURCE_OPERATION[backend], case.rows
                )
                native_value, _ = read_value(
                    path, backend, NATIVE_OPERATION[backend], case.rows
                )
                selected_values.append(selected_value)
                native_values.append(native_value)
                cards.append(card); sources.append(str(path.relative_to(HERE)))
            if any(card != expected[case.label] for card in cards):
                raise RuntimeError(f"{case.label}/{backend}: cardinality mismatch")
            delivered = statistics.median(selected_values)
            native = statistics.median(native_values)
            aggregates[backend] = delivered
            per_backend[backend] = {
                "native_ms": native, "delivered_ms": delivered,
                "min_ms": min(selected_values), "max_ms": max(selected_values),
                "sources": ";".join(sources),
            }

        winning_backend = min(BACKENDS, key=aggregates.get)
        winners[DISPLAY[winning_backend]] += 1
        ddc = aggregates["ddc"]
        for backend in BACKENDS:
            data = per_backend[backend]
            ratio = aggregates[backend] / ddc
            ratios[DISPLAY[backend]].append(ratio)
            selected.append({
                "group": "job", "case": case.label, "num_rows": case.rows,
                "operation": "COMP",
                "backend": DISPLAY[backend],
                "time_ms": aggregates[backend], "latency_over_ddc": ratio,
                "winner": DISPLAY[winning_backend],
                "source_operation": SOURCE_OPERATION[backend],
                "replicates": reps, "min_ms": data["min_ms"],
                "max_ms": data["max_ms"], "result_cardinality": expected[case.label],
                "source_file": data["sources"],
            })
            native = float(data["native_ms"])
            delivered = float(data["delivered_ms"])
            breakdown.append({
                "case": case.label, "num_rows": case.rows,
                "backend": DISPLAY[backend], "native_ms": native,
                "selected_delivered_ms": delivered,
                "delivered_minus_native_medians_ms": delivered - native,
                "median_difference_fraction": (
                    0.0 if delivered == native else (delivered - native) / delivered
                ),
                "native_operation": NATIVE_OPERATION[backend],
                "selected_operation": SOURCE_OPERATION[backend],
            })

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "job_wah_ewah_btv.csv", selected)
    write_csv(RESULTS / "job_delivery_breakdown.csv", breakdown)
    summary = {
        "schema_version": 1, "cells": len(cases()),
        "literal_plan": metadata["literal_plan"],
        "delivery_boundary": metadata["delivery_boundary"],
        "wins": dict(winners),
        "geometric_mean_latency_over_ddc": {
            backend: geometric_mean(values) for backend, values in ratios.items()
        },
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
