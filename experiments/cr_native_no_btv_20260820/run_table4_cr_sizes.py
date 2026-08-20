#!/usr/bin/env python3
"""Remeasure Table 4 CRoaring plain and run-optimized native payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
DEFAULT_BINARY = WORKSPACE / "JOB_R3W1_20260807_v2" / "size" / "build" / "measure_job_sizes"
MEMBERSHIPS = WORKSPACE / "R3W1_realworld_excel_20260809" / "job_unified10" / "memberships"
SOURCE = WORKSPACE / "JOB_R3W1_20260807_v2" / "size" / "measure_job_sizes.cpp"

SPECS = (
    ("movie_companies.company_type_id", "mc_ctype", "mc_ctype.jobcsr"),
    ("title.kind_id", "kind", "kind.jobcsr"),
    ("cast_info.role_id", "role", "role.jobcsr"),
    ("person_info.info_type_id", "pi_itype", "pi_itype.jobcsr"),
    ("movie_info.info_type_id", "mi_itype", "mi_itype.jobcsr"),
    ("title.production_year", "year", "year.jobcsr"),
    ("company_name.country_code", "country", "country.jobcsr"),
    ("movie_keyword.movie_id", "movie_keyword_movie_id", "movie_keyword_movie_id.jobcsr"),
    ("cast_info.movie_id", "cast_info_movie_id", "cast_info_movie_id.jobcsr"),
    ("cast_info.person_id", "cast_info_person_id", "cast_info_person_id.jobcsr"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_single(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one row, found {len(rows)}")
    row = rows[0]
    plain = int(row["cr_plain_payload_bytes"])
    run = int(row["cr_run_payload_bytes"])
    if not (0 < run <= plain):
        raise RuntimeError(f"{path}: invalid CR plain/run sizes {plain}/{run}")
    return row


def main() -> None:
    args = parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    raw_dir = HERE / "raw" / "table4_cr_run"
    log_dir = HERE / "logs" / "table4_cr_run"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "memberships": {
            name: sha256(MEMBERSHIPS / name) for _, _, name in SPECS
        },
        "measurement": "CRoaring getSizeInBytes(false), before and after runOptimize()",
    }
    (HERE / "table4_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    started = time.monotonic()
    result_rows = []
    for index, (field, slug, filename) in enumerate(SPECS, start=1):
        membership = MEMBERSHIPS / filename
        if not membership.is_file():
            raise FileNotFoundError(membership)
        output = raw_dir / f"{slug}.csv"
        log = log_dir / f"{slug}.log"
        if output.exists() and not args.resume:
            raise FileExistsError(output)
        if not output.exists():
            with output.open("w", encoding="utf-8") as sink, log.open(
                "w", encoding="utf-8"
            ) as errors:
                subprocess.run(
                    [str(binary), field, str(membership), "job"],
                    stdout=sink,
                    stderr=errors,
                    check=True,
                )
        row = read_single(output)
        result_rows.append({
            "field": field,
            "slug": slug,
            "n": row["n"],
            "c": row["c"],
            "memberships": row["memberships"],
            "cr_plain_payload_bytes": row["cr_plain_payload_bytes"],
            "cr_run_payload_bytes": row["cr_run_payload_bytes"],
            "cr_plain_mib": int(row["cr_plain_payload_bytes"]) / (1 << 20),
            "cr_run_mib": int(row["cr_run_payload_bytes"]) / (1 << 20),
            "run_reduction_pct": 100.0 * (
                1.0
                - int(row["cr_run_payload_bytes"])
                / int(row["cr_plain_payload_bytes"])
            ),
        })
        print(
            f"[{index}/{len(SPECS)}] {slug} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )

    result_path = HERE / "results" / "table4_cr_run_sizes.csv"
    result_path.parent.mkdir(exist_ok=True)
    totals = {
        "field": "TOTAL",
        "slug": "TOTAL",
        "n": "",
        "c": "",
        "memberships": "",
        "cr_plain_payload_bytes": sum(
            int(row["cr_plain_payload_bytes"]) for row in result_rows
        ),
        "cr_run_payload_bytes": sum(
            int(row["cr_run_payload_bytes"]) for row in result_rows
        ),
    }
    totals["cr_plain_mib"] = totals["cr_plain_payload_bytes"] / (1 << 20)
    totals["cr_run_mib"] = totals["cr_run_payload_bytes"] / (1 << 20)
    totals["run_reduction_pct"] = 100.0 * (
        1.0 - totals["cr_run_payload_bytes"] / totals["cr_plain_payload_bytes"]
    )
    output_rows = result_rows + [totals]
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
