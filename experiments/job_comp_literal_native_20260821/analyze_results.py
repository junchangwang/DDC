#!/usr/bin/env python3
"""Validate and aggregate the same-literal JOB COMP sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import run_job_literal as run


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
RESULTS = HERE / "results"
BACKEND_DISPLAY = {
    "ddc": "DDC",
    "croaring": "CRoaring",
    "wah": "WAH",
    "ewah": "EWAH",
}
FROZEN_LOGICAL = (
    HERE.parent / "cr_native_no_btv_20260820" / "results" /
    "job_native_logical.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    metadata_path = HERE / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reps = int(metadata["process_replicates"])
    if metadata["literal_plan"] != run.LITERAL_PLAN:
        raise RuntimeError("literal plan metadata mismatch")
    if sha256(Path(metadata["binary"])) != metadata["binary_sha256"]:
        raise RuntimeError("benchmark binary hash drift")

    expected = {key: int(value) for key, value in metadata["expected_result_cardinality"].items()}
    compact_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    wins: Counter[str] = Counter()
    ratios: dict[str, list[float]] = {name: [] for name in BACKEND_DISPLAY.values()}

    for case in run.cases():
        aggregates: dict[str, tuple[float, float, float, list[str]]] = {}
        sample_map: dict[str, list[float]] = {}
        for backend in run.BACKENDS:
            samples: list[float] = []
            sources: list[str] = []
            for rep in range(1, reps + 1):
                path = RAW / f"r{rep}_{case.label}_{backend}.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                row = run.select_row(path, backend, case.rows)
                samples.append(float(row["time_ms"]))
                sources.append(str(path.relative_to(HERE)))
                if backend in {"ddc", "croaring"}:
                    observed = int(row["result_cardinality"])
                    if observed != expected[case.label]:
                        raise RuntimeError(
                            f"{path}: cardinality {observed} != {expected[case.label]}"
                        )
            aggregates[backend] = (
                statistics.median(samples), min(samples), max(samples), sources
            )
            sample_map[backend] = samples

        ddc_time = aggregates["ddc"][0]
        winner_key = min(run.BACKENDS, key=lambda key: aggregates[key][0])
        winner = BACKEND_DISPLAY[winner_key]
        wins[winner] += 1
        wide: dict[str, object] = {
            "case": case.label,
            "num_rows": case.rows,
            "cardinality": case.cardinality,
            "winner": winner,
        }
        for backend in run.BACKENDS:
            display = BACKEND_DISPLAY[backend]
            median, minimum, maximum, sources = aggregates[backend]
            ratio = median / ddc_time
            paired_ratios = [
                sample_map[backend][index] / sample_map["ddc"][index]
                for index in range(reps)
            ]
            paired_ratio = statistics.median(paired_ratios)
            ratios[display].append(ratio)
            compact_rows.append(
                {
                    "group": "job",
                    "case": case.label,
                    "num_rows": case.rows,
                    "operation": "COMP",
                    "backend": display,
                    "time_ms": median,
                    "latency_over_ddc": ratio,
                    "winner": winner,
                    "source_operation": run.SOURCE_OPERATION[backend],
                    "source_file": ";".join(sources),
                    "replicates": reps,
                    "min_ms": minimum,
                    "max_ms": maximum,
                    "paired_ratio_median": paired_ratio,
                    "paired_ratio_min": min(paired_ratios),
                    "paired_ratio_max": max(paired_ratios),
                    "expected_result_cardinality": expected[case.label],
                }
            )
            wide[f"{backend}_time_ms"] = median
            wide[f"{backend}_ratio_to_ddc"] = ratio
            wide[f"{backend}_min_ms"] = minimum
            wide[f"{backend}_max_ms"] = maximum
            wide[f"{backend}_paired_ratio_median"] = paired_ratio
        wide_rows.append(wide)

    RESULTS.mkdir(exist_ok=True)
    compact_path = RESULTS / "job_literal_native_logical.csv"
    compact_fields = list(compact_rows[0])
    with compact_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields)
        writer.writeheader()
        writer.writerows(compact_rows)

    wide_path = RESULTS / "job_literal_native_summary.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0]))
        writer.writeheader()
        writer.writerows(wide_rows)

    with FROZEN_LOGICAL.open(newline="", encoding="utf-8") as handle:
        frozen_records = {
            (row["case"], row["backend"]): row
            for row in csv.DictReader(handle)
            if row["group"] == "job" and row["operation"] == "COMP"
        }
    comparison_rows: list[dict[str, object]] = []
    comparison_factors: dict[str, list[float]] = {
        display: [] for display in BACKEND_DISPLAY.values()
    }
    for row in compact_rows:
        key = (str(row["case"]), str(row["backend"]))
        if key not in frozen_records:
            raise RuntimeError(f"missing frozen comparison row {key}")
        frozen = frozen_records[key]
        frozen_time = float(frozen["time_ms"])
        literal_time = float(row["time_ms"])
        factor = literal_time / frozen_time
        comparison_factors[key[1]].append(factor)
        comparison_rows.append(
            {
                "case": key[0],
                "backend": key[1],
                "frozen_time_ms": frozen_time,
                "literal_fresh_time_ms": literal_time,
                "literal_over_frozen": factor,
                "frozen_ratio_to_ddc": float(frozen["latency_over_ddc"]),
                "literal_ratio_to_ddc": float(row["latency_over_ddc"]),
                "frozen_source_operation": frozen["source_operation"],
                "literal_source_operation": row["source_operation"],
            }
        )
    comparison_path = RESULTS / "comparison_to_frozen.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "schema_version": 1,
        "literal_plan": run.LITERAL_PLAN,
        "result_boundary": metadata["result_boundary"],
        "limitations": [
            "Same-column equality operands make literal COMP reduce to NOT B "
            "for C>=3; the C=2 case reduces to NOT (A OR B).",
            "DDC uses the existing default decompressed mixed-result path; "
            "CRoaring, WAH, and EWAH retain compressed native results.",
            "DDC includes destruction of one local intermediate before the "
            "timer stops; the other backends destroy intermediates afterward.",
        ],
        "process_replicates": reps,
        "cases": len(run.cases()),
        "wins": dict(wins),
        "geometric_mean_latency_over_ddc": {
            backend: geomean(values) for backend, values in ratios.items()
        },
        "geometric_mean_literal_over_frozen": {
            backend: geomean(values)
            for backend, values in comparison_factors.items()
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
