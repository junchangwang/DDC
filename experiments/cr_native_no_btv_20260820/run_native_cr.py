#!/usr/bin/env python3
"""Run isolated native-CR sensitivity sweeps without changing source inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
DEFAULT_BINARY = REPO / "build-native" / "benchmark_app"
COUNTS = (66, 328, 655, 1311, 3277, 6554, 13107, 19661, 26214, 32768)
REQUIRED_CR_ROWS = {
    "OR_op",
    "AND_op",
    "NOT_op",
    "COMP_op_native",
    "OR_op_conv",
    "AND_op_conv",
    "NOT_op_conv",
    "COMP_op",
}


@dataclass(frozen=True)
class Case:
    group: str
    label: str
    rows: int
    bitmap_dir: str
    pair_and: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def earth_cases() -> list[Case]:
    specs = (("month", 12), ("qflag", 17), ("decade", 33),
             ("temp_bin", 232), ("year", 325))
    return [
        Case(
            "earth",
            field,
            21_844_315,
            str(WORKSPACE / "combit" / "earthdata" / field / "bitmap" /
                f"bm_21844315_c{cardinality}_roaring"),
            True,
        )
        for field, cardinality in specs
    ]


def job_cases() -> list[Case]:
    old_specs = (
        ("mc_ctype", 2_609_129, 2),
        ("kind", 2_528_312, 6),
        ("role", 36_244_344, 11),
        ("pi_itype", 2_963_664, 22),
        ("mi_itype", 14_835_720, 71),
        ("year", 2_528_312, 132),
        ("country", 234_997, 215),
    )
    result = [
        Case(
            "job",
            field,
            rows,
            str(WORKSPACE / "combit" / "realdata" / field / "bitmap" /
                f"bm_{rows}_c{cardinality}_roaring"),
            True,
        )
        for field, rows, cardinality in old_specs
    ]
    new_specs = (
        ("movie_keyword_movie_id", 4_523_930),
        ("cast_info_movie_id", 36_244_344),
        ("cast_info_person_id", 36_244_344),
    )
    for field, rows in new_specs:
        result.append(
            Case(
                "job",
                field,
                rows,
                str(WORKSPACE / "R3W1_realworld_excel_20260809" /
                    "job_unified10" / "logical_top3_datasets" / field /
                    "bitmap" / f"bm_{rows}_c3_roaring"),
                True,
            )
        )
    return result


def cluster_cases() -> list[Case]:
    factors = ("iid", "f2", "f4", "f8", "f16", "f32", "f64",
               "f128", "f256", "f1024", "sorted")
    return [
        Case(
            "cluster",
            factor,
            100_000_000,
            str(WORKSPACE / "R2W1_clustering_micro_100m_20260813" /
                "data" / factor / "bitmap" / "bm_100m_c100_roaring"),
            False,
        )
        for factor in factors
    ]


def density_cases() -> list[Case]:
    result = []
    for i, count_a in enumerate(COUNTS):
        for count_b in COUNTS[i:]:
            result.append(
                Case(
                    "density",
                    f"A{count_a}_B{count_b}",
                    100_000_000,
                    str(WORKSPACE / "combit" / "gridindep" / "bitmap" /
                        f"bm_100m_A{count_a}_B{count_b}_roaring"),
                    True,
                )
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("earth", "job", "cluster", "density"),
        default=("earth", "job", "cluster", "density"),
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--density-reps", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_csv(path: Path, rows: int) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    selected = {
        row["operation"]: row
        for row in records
        if row["backend"] == "CRoaring" and int(row["num_rows"]) == rows
    }
    missing = REQUIRED_CR_ROWS - set(selected)
    if missing:
        raise RuntimeError(f"{path}: missing CR rows {sorted(missing)}")
    native = float(selected["COMP_op_native"]["time_ms"])
    delivered = float(selected["COMP_op"]["time_ms"])
    if not (native > 0.0 and delivered > native):
        raise RuntimeError(
            f"{path}: invalid COMP native/delivered {native}/{delivered}"
        )


def main() -> None:
    args = parse_args()
    if args.reps < 1 or args.density_reps < 1:
        raise ValueError("replicate counts must be positive")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)

    groups = {
        "earth": earth_cases(),
        "job": job_cases(),
        "cluster": cluster_cases(),
        "density": density_cases(),
    }
    raw_root = HERE / "raw"
    log_root = HERE / "logs"
    raw_root.mkdir(exist_ok=True)
    log_root.mkdir(exist_ok=True)

    selected_cases = [case for name in args.groups for case in groups[name]]
    for case in selected_cases:
        if not Path(case.bitmap_dir).is_dir():
            raise FileNotFoundError(case.bitmap_dir)

    metadata = {
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cpu": args.cpu,
        "reps": args.reps,
        "density_reps": args.density_reps,
        "groups": args.groups,
        "cases": [asdict(case) for case in selected_cases],
        "allocator_env": {
            "MALLOC_TRIM_THRESHOLD_": "1073741824",
            "MALLOC_MMAP_THRESHOLD_": "1073741824",
        },
    }
    (HERE / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    started = time.monotonic()
    completed = 0
    for case in selected_cases:
        reps = args.density_reps if case.group == "density" else args.reps
        for rep in range(1, reps + 1):
            stem = f"r{rep}_{case.label}"
            csv_path = raw_root / case.group / f"{stem}.csv"
            log_path = log_root / case.group / f"{stem}.log"
            csv_path.parent.mkdir(exist_ok=True)
            log_path.parent.mkdir(exist_ok=True)
            if csv_path.exists():
                if not args.resume:
                    raise FileExistsError(csv_path)
                validate_csv(csv_path, case.rows)
                completed += 1
                continue

            env = dict(os.environ)
            env["MALLOC_TRIM_THRESHOLD_"] = "1073741824"
            env["MALLOC_MMAP_THRESHOLD_"] = "1073741824"
            if case.pair_and:
                env["DDC_BENCH_AND_MODE"] = "pair"
            else:
                env.pop("DDC_BENCH_AND_MODE", None)
            command = [
                "taskset", "-c", str(args.cpu), str(binary),
                "--backend", "croaring",
                "--compressed-dir", case.bitmap_dir,
                "--num-rows", str(case.rows),
                "--iterations", "100",
                "--csv", str(csv_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=WORKSPACE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            validate_csv(csv_path, case.rows)
            completed += 1
            elapsed = time.monotonic() - started
            print(
                f"[{completed}] {case.group}/{case.label}/r{rep} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
