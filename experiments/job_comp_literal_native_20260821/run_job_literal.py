#!/usr/bin/env python3
"""Run a same-literal-plan, backend-native JOB COMP sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
DEFAULT_BINARY = REPO / "build-native" / "benchmark_app"
VERIFY_BASELINES = WORKSPACE / "R3W1_realworld_excel_20260809" / "verify_earth_baselines"
VERIFY_DDC = (
    WORKSPACE / "R3W1_realworld_excel_20260809" / "job_unified10" /
    "build" / "verify_job_ddc"
)
BACKENDS = ("ddc", "croaring", "wah", "ewah")
CSV_BACKEND = {
    "ddc": "DDC (New)",
    "croaring": "CRoaring",
    "wah": "WAH (FastBit)",
    "ewah": "EWAH",
}
SOURCE_OPERATION = {
    "ddc": "COMP_op",
    "croaring": "COMP_op_native",
    "wah": "COMP_op",
    "ewah": "COMP_op",
}
LITERAL_PLAN = "t1=A|B; t2=B|C; t3=t1&t2; result=~t3"
LOGICAL_FIELD_TO_LABEL = {
    "movie_keyword.movie_id": "movie_keyword_movie_id",
    "cast_info.movie_id": "cast_info_movie_id",
    "cast_info.person_id": "cast_info_person_id",
}


@dataclass(frozen=True)
class Case:
    label: str
    rows: int
    cardinality: int
    full_cardinality: int
    dataset: str
    directories: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def old_case(label: str, rows: int, cardinality: int) -> Case:
    root = WORKSPACE / "combit" / "realdata" / label / "bitmap"
    ewah = (
        WORKSPACE / "R3W1_realworld_excel_20260809" / "ewah_corrected" /
        "job" / "data" / label / "bitmap" / f"bm_{rows}_c{cardinality}_ewah"
    )
    return Case(
        label,
        rows,
        cardinality,
        cardinality,
        str(root.parent / f"dataset_{rows}_{cardinality}"),
        {
            "ddc": str(root / f"bm_{rows}_c{cardinality}_ddc_w8"),
            "croaring": str(root / f"bm_{rows}_c{cardinality}_roaring"),
            "wah": str(root / f"bm_{rows}_c{cardinality}_wah"),
            "ewah": str(ewah),
        },
    )


def top3_case(label: str, rows: int, full_cardinality: int) -> Case:
    root = (
        WORKSPACE / "R3W1_realworld_excel_20260809" / "job_unified10" /
        "logical_top3_datasets" / label / "bitmap"
    )
    dataset = root.parent / f"dataset_{rows}_{full_cardinality}_top3"
    return Case(
        label,
        rows,
        3,
        full_cardinality,
        str(dataset),
        {
            "ddc": str(root / f"bm_{rows}_c3_ddc_w8"),
            "croaring": str(root / f"bm_{rows}_c3_roaring"),
            "wah": str(root / f"bm_{rows}_c3_wah"),
            "ewah": str(root / f"bm_{rows}_c3_ewah"),
        },
    )


def cases() -> tuple[Case, ...]:
    return (
        old_case("mc_ctype", 2_609_129, 2),
        old_case("kind", 2_528_312, 6),
        old_case("role", 36_244_344, 11),
        old_case("pi_itype", 2_963_664, 22),
        old_case("mi_itype", 14_835_720, 71),
        old_case("year", 2_528_312, 132),
        old_case("country", 234_997, 215),
        top3_case("movie_keyword_movie_id", 4_523_930, 476_794),
        top3_case("cast_info_movie_id", 36_244_344, 2_331_601),
        top3_case("cast_info_person_id", 36_244_344, 4_051_810),
    )


def numeric_bitmaps(directory: Path) -> list[Path]:
    result = [path for path in directory.glob("*.bm") if path.stem.isdigit()]
    return sorted(result, key=lambda path: int(path.stem))


def expected_cardinalities() -> dict[str, int]:
    source = (
        WORKSPACE / "R3W1_realworld_excel_20260809" / "job_unified10" /
        "job-unified10-logical.csv"
    )
    result: dict[str, int] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["operation"] == "COMP" and row["algorithm"] == "DDC":
                label = LOGICAL_FIELD_TO_LABEL.get(row["field"], row["field"])
                result[label] = int(row["expected_result_cardinality"])
    missing = {case.label for case in cases()} - result.keys()
    if missing:
        raise RuntimeError(f"missing expected cardinalities: {sorted(missing)}")
    return result


def expand_cpu_list(specification: str) -> list[int]:
    result: list[int] = []
    for item in specification.strip().split(","):
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            result.extend(range(first, last + 1))
        elif item:
            result.append(int(item))
    return sorted(set(result))


def host_idle_preflight(cpu: int, allow_busy: bool) -> dict[str, object]:
    sibling_path = Path(
        f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )
    if not sibling_path.is_file():
        raise FileNotFoundError(sibling_path)
    siblings = expand_cpu_list(sibling_path.read_text(encoding="utf-8"))
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    sample = subprocess.check_output(
        ["mpstat", "-P", "ALL", "1", "3"], text=True, env=environment
    )
    idle: dict[str, float] = {}
    for line in sample.splitlines():
        columns = line.split()
        if (
            len(columns) >= 4
            and columns[0] == "Average:"
            and columns[1] != "CPU"
        ):
            idle[columns[1]] = float(columns[-1])
    required = ["all", *(str(value) for value in siblings)]
    missing = [key for key in required if key not in idle]
    if missing:
        raise RuntimeError(f"mpstat did not report CPUs {missing}")
    load = [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]
    acceptable = idle["all"] >= 85.0 and all(
        idle[str(value)] >= 95.0 for value in siblings
    )
    result: dict[str, object] = {
        "load_average": load,
        "logical_cpu": cpu,
        "thread_siblings": siblings,
        "idle_percent": {key: idle[key] for key in required},
        "acceptable": acceptable,
    }
    if not acceptable and not allow_busy:
        raise RuntimeError(
            "host is busy; require >=85% overall idle and >=95% idle on "
            f"CPU siblings, observed {result}"
        )
    return result


def select_row(path: Path, backend: str, rows: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    matches = [
        row for row in records
        if row["backend"] == CSV_BACKEND[backend]
        and row["operation"] == SOURCE_OPERATION[backend]
        and int(row["num_rows"]) == rows
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{path}: expected one {backend}/{SOURCE_OPERATION[backend]} row, "
            f"found {len(matches)}"
        )
    if float(matches[0]["time_ms"]) <= 0.0:
        raise RuntimeError(f"{path}: non-positive timing")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--cpu", type=int, default=14)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reps < 3 or args.reps % 2 == 0:
        raise ValueError("--reps must be an odd integer >= 3")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    for verifier in (VERIFY_DDC, VERIFY_BASELINES):
        if not verifier.is_file():
            raise FileNotFoundError(verifier)
    host_preflight = host_idle_preflight(args.cpu, args.allow_busy)

    all_cases = cases()
    expected = expected_cardinalities()
    operand_manifest: list[dict[str, object]] = []
    for case in all_cases:
        if not Path(case.dataset).is_file():
            raise FileNotFoundError(case.dataset)
        for backend in BACKENDS:
            directory = Path(case.directories[backend])
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            bitmaps = numeric_bitmaps(directory)
            needed = 2 if case.cardinality == 2 else 3
            if len(bitmaps) < needed:
                raise RuntimeError(f"{directory}: only {len(bitmaps)} bitmaps")
            operand_manifest.append(
                {
                    "case": case.label,
                    "backend": backend,
                    "directory": str(directory),
                    "bitmap_count": len(bitmaps),
                    "total_bm_bytes": sum(path.stat().st_size for path in bitmaps),
                    "operands": [
                        {
                            "name": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                        for path in bitmaps[:needed]
                    ],
                }
            )

    raw_dir = HERE / "raw"
    log_dir = HERE / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    metadata = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "verifiers": {
            "ddc": {"path": str(VERIFY_DDC), "sha256": sha256(VERIFY_DDC)},
            "baselines": {
                "path": str(VERIFY_BASELINES),
                "sha256": sha256(VERIFY_BASELINES),
            },
        },
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cpu": args.cpu,
        "host_preflight": host_preflight,
        "process_replicates": args.reps,
        "inner_measurements": 100,
        "outer_iterations": 1,
        "literal_plan": LITERAL_PLAN,
        "result_boundary": (
            "backend-native: DDC/WAH/EWAH COMP_op; "
            "CRoaring COMP_op_native; no dense delivery"
        ),
        "ddc_compress_results": False,
        "allocator_env": {
            "MALLOC_TRIM_THRESHOLD_": "1073741824",
            "MALLOC_MMAP_THRESHOLD_": "1073741824",
        },
        "cases": [asdict(case) for case in all_cases],
        "expected_result_cardinality": expected,
        "operand_manifest": operand_manifest,
    }
    metadata_path = HERE / "run_metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(metadata_path)
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        stable_keys = (
            "binary", "binary_sha256", "git_commit", "verifiers", "cpu",
            "process_replicates", "inner_measurements", "outer_iterations",
            "literal_plan", "result_boundary", "ddc_compress_results",
            "allocator_env", "cases", "expected_result_cardinality",
            "operand_manifest",
        )
        mismatches = [
            key for key in stable_keys if existing.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise RuntimeError(f"resume metadata mismatch: {mismatches}")
        metadata = existing
    else:
        existing_raw = sorted(raw_dir.glob("*.csv"))
        if existing_raw:
            raise RuntimeError(
                f"raw data exists without metadata: {existing_raw[:3]}"
            )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    verification_dir = HERE / "verification"
    verification_dir.mkdir(exist_ok=True)
    for case in all_cases:
        commands = {
            "ddc": [
                str(VERIFY_DDC), case.dataset, case.directories["ddc"],
                str(case.rows), str(case.cardinality),
            ],
            "baselines": [
                str(VERIFY_BASELINES), case.dataset, case.directories["wah"],
                case.directories["ewah"], case.directories["croaring"],
                str(case.rows), str(case.cardinality),
            ],
        }
        for label, command in commands.items():
            log_path = verification_dir / f"{case.label}_{label}.log"
            if args.resume and log_path.is_file() and log_path.read_text(
                encoding="utf-8"
            ).startswith("PASS"):
                continue
            completed = subprocess.run(
                command,
                cwd=WORKSPACE,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0 or not completed.stdout.startswith("PASS"):
                raise RuntimeError(f"verification failed: {log_path}")
        print(f"[verify] {case.label} PASS", flush=True)

    tasks: list[tuple[int, Case, str]] = []
    original_index = {case.label: index for index, case in enumerate(all_cases)}
    for rep in range(1, args.reps + 1):
        ordered_cases = all_cases[rep - 1:] + all_cases[:rep - 1]
        for case in ordered_cases:
            shift = (rep - 1 + original_index[case.label]) % len(BACKENDS)
            ordered_backends = BACKENDS[shift:] + BACKENDS[:shift]
            tasks.extend((rep, case, backend) for backend in ordered_backends)

    started = time.monotonic()
    for task_index, (rep, case, backend) in enumerate(tasks, start=1):
        stem = f"r{rep}_{case.label}_{backend}"
        raw_path = raw_dir / f"{stem}.csv"
        log_path = log_dir / f"{stem}.log"
        if raw_path.exists():
            if not args.resume:
                raise FileExistsError(raw_path)
            row = select_row(raw_path, backend, case.rows)
        else:
            env = dict(os.environ)
            env["MALLOC_TRIM_THRESHOLD_"] = "1073741824"
            env["MALLOC_MMAP_THRESHOLD_"] = "1073741824"
            env["DDC_BENCH_AND_MODE"] = "pair"
            command = [
                "taskset", "-c", str(args.cpu), str(binary),
                "--backend", backend,
                "--compressed-dir", case.directories[backend],
                "--num-rows", str(case.rows),
                "--iterations", "1",
                "--csv", str(raw_path),
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
            row = select_row(raw_path, backend, case.rows)

        if backend in {"ddc", "croaring"}:
            observed = int(row["result_cardinality"])
            if observed != expected[case.label]:
                raise RuntimeError(
                    f"{raw_path}: result cardinality {observed} != "
                    f"{expected[case.label]}"
                )
        print(
            f"[{task_index}/{len(tasks)}] r{rep} {case.label}/{backend} "
            f"COMP={float(row['time_ms']) * 1000:.3f}us "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
