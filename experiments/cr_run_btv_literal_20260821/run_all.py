#!/usr/bin/env python3
"""Fresh selected-delivery rerun for Earth, JOB, clustering, and density."""

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
VERIFY_DENSITY = REPO / "build-native" / "verify_density_pair"
CORE_BACKENDS = ("ddc", "croaring", "wah", "ewah")
DENSITY_BACKENDS = (*CORE_BACKENDS, "bitset_avx512", "concise")
CSV_BACKEND = {
    "ddc": "DDC (New)",
    "croaring": "CRoaring",
    "wah": "WAH (FastBit)",
    "ewah": "EWAH",
    "bitset_avx512": "Bitset (AVX512)",
    "concise": "Concise",
}
DISPLAY = {
    "ddc": "DDC",
    "croaring": "CRoaring",
    "wah": "WAH",
    "ewah": "EWAH",
    "bitset_avx512": "Bitset-AVX512",
    "concise": "Concise",
}
LITERAL_PLAN = "t1=A|B; t2=B|C; t3=t1&t2; result=~t3"
COUNTS = (66, 328, 655, 1311, 3277, 6554, 13107, 19661, 26214, 32768)
COUNT_TO_DENSITY = {
    66: "0.1%", 328: "0.5%", 655: "1%", 1311: "2%", 3277: "5%",
    6554: "10%", 13107: "20%", 19661: "30%", 26214: "40%",
    32768: "50%",
}


@dataclass(frozen=True)
class Case:
    group: str
    label: str
    rows: int
    cardinality: int
    dataset: str | None
    operations: tuple[str, ...]
    backends: tuple[str, ...]
    repetitions: int
    directories: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError(f"mpstat did not report {missing}")
    result: dict[str, object] = {
        "load_average": [
            float(value) for value in Path("/proc/loadavg").read_text().split()[:3]
        ],
        "logical_cpu": cpu,
        "thread_siblings": siblings,
        "idle_percent": {key: idle[key] for key in required},
    }
    result["acceptable"] = idle["all"] >= 85.0 and all(
        idle[str(value)] >= 95.0 for value in siblings
    )
    if not result["acceptable"] and not allow_busy:
        raise RuntimeError(f"host is busy: {result}")
    return result


def numeric_bitmaps(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.glob("*.bm") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )


def target_operation(backend: str, operation: str) -> str:
    if backend == "croaring":
        return "COMP_op" if operation == "COMP" else f"{operation}_op_conv"
    return f"{operation}_op"


def earth_cases(core_reps: int) -> list[Case]:
    specs = (("month", 12), ("qflag", 17), ("decade", 33),
             ("temp_bin", 232), ("year", 325))
    result = []
    for label, cardinality in specs:
        rows = 21_844_315
        ddc = (
            WORKSPACE / "R3W1_realworld_excel_20260809" /
            "earth_adaptive_single_replay_20260809" / "data" / label /
            "bitmap" / f"bm_{rows}_c{cardinality}_ddc_w8"
        )
        root = WORKSPACE / "combit" / "earthdata" / label / "bitmap"
        ewah = (
            WORKSPACE / "R3W1_realworld_excel_20260809" / "ewah_corrected" /
            "earth" / "data" / label / "bitmap" /
            f"bm_{rows}_c{cardinality}_ewah"
        )
        result.append(Case(
            "earth", label, rows, cardinality,
            str(
                WORKSPACE / "combit" / "earthdata" / label /
                f"dataset_{rows}_{cardinality}"
            ),
            ("OR", "COMP"), CORE_BACKENDS, core_reps,
            {
                "ddc": str(ddc),
                "croaring": str(root / f"bm_{rows}_c{cardinality}_roaring"),
                "wah": str(root / f"bm_{rows}_c{cardinality}_wah"),
                "ewah": str(ewah),
            },
        ))
    return result


def job_cases(core_reps: int) -> list[Case]:
    old_specs = (
        ("mc_ctype", 2_609_129, 2), ("kind", 2_528_312, 6),
        ("role", 36_244_344, 11), ("pi_itype", 2_963_664, 22),
        ("mi_itype", 14_835_720, 71), ("year", 2_528_312, 132),
        ("country", 234_997, 215),
    )
    result = []
    for label, rows, cardinality in old_specs:
        root = WORKSPACE / "combit" / "realdata" / label
        ewah = (
            WORKSPACE / "R3W1_realworld_excel_20260809" / "ewah_corrected" /
            "job" / "data" / label / "bitmap" /
            f"bm_{rows}_c{cardinality}_ewah"
        )
        result.append(Case(
            "job", label, rows, cardinality,
            str(root / f"dataset_{rows}_{cardinality}"), ("COMP",),
            CORE_BACKENDS, core_reps,
            {
                "ddc": str(root / "bitmap" / f"bm_{rows}_c{cardinality}_ddc_w8"),
                "croaring": str(root / "bitmap" / f"bm_{rows}_c{cardinality}_roaring"),
                "wah": str(root / "bitmap" / f"bm_{rows}_c{cardinality}_wah"),
                "ewah": str(ewah),
            },
        ))
    top3 = (
        ("movie_keyword_movie_id", 4_523_930, 476_794),
        ("cast_info_movie_id", 36_244_344, 2_331_601),
        ("cast_info_person_id", 36_244_344, 4_051_810),
    )
    for label, rows, full_cardinality in top3:
        root = (
            WORKSPACE / "R3W1_realworld_excel_20260809" / "job_unified10" /
            "logical_top3_datasets" / label
        )
        bitmap = root / "bitmap"
        result.append(Case(
            "job", label, rows, 3,
            str(root / f"dataset_{rows}_{full_cardinality}_top3"), ("COMP",),
            CORE_BACKENDS, core_reps,
            {
                "ddc": str(bitmap / f"bm_{rows}_c3_ddc_w8"),
                "croaring": str(bitmap / f"bm_{rows}_c3_roaring"),
                "wah": str(bitmap / f"bm_{rows}_c3_wah"),
                "ewah": str(bitmap / f"bm_{rows}_c3_ewah"),
            },
        ))
    return result


def cluster_cases(core_reps: int) -> list[Case]:
    factors = ("iid", "f2", "f4", "f8", "f16", "f32", "f64", "f128",
               "f256", "f1024", "sorted")
    result = []
    for factor in factors:
        root = (
            WORKSPACE / "R2W1_clustering_micro_100m_20260813" / "data" /
            factor
        )
        bitmap = root / "bitmap"
        result.append(Case(
            "cluster", factor, 100_000_000, 100,
            str(root / "dataset_100000000_100"),
            ("OR", "AND", "NOT", "COMP"), CORE_BACKENDS, core_reps,
            {
                "ddc": str(bitmap / "bm_100m_c100_ddc_w8"),
                "croaring": str(bitmap / "bm_100m_c100_roaring"),
                "wah": str(bitmap / "bm_100m_c100_wah"),
                "ewah": str(bitmap / "bm_100m_c100_ewah"),
            },
        ))
    return result


def density_cases(density_reps: int) -> list[Case]:
    result = []
    root = WORKSPACE / "combit" / "gridindep" / "bitmap"
    for index, count_a in enumerate(COUNTS):
        for count_b in COUNTS[index:]:
            label = f"A{count_a}_B{count_b}"
            prefix = root / f"bm_100m_{label}"
            result.append(Case(
                "density", label, 100_000_000, count_a, None, ("OR",),
                DENSITY_BACKENDS, density_reps,
                {
                    "ddc": str(Path(f"{prefix}_ddc_w8")),
                    "croaring": str(Path(f"{prefix}_roaring")),
                    "wah": str(Path(f"{prefix}_wah")),
                    "ewah": str(Path(f"{prefix}_ewah")),
                    "bitset_avx512": str(Path(f"{prefix}_bitset")),
                    "concise": str(Path(f"{prefix}_concise")),
                },
            ))
    return result


def all_cases(core_reps: int, density_reps: int) -> list[Case]:
    return [
        *earth_cases(core_reps), *job_cases(core_reps),
        *cluster_cases(core_reps), *density_cases(density_reps),
    ]


def selected_rows(path: Path, case: Case, backend: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for operation in case.operations:
        source = target_operation(backend, operation)
        matches = [
            row for row in records
            if row["backend"] == CSV_BACKEND[backend]
            and row["operation"] == source
            and int(row["num_rows"]) == case.rows
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{path}: expected one {backend}/{source}, found {len(matches)}"
            )
        if float(matches[0]["time_ms"]) <= 0.0:
            raise RuntimeError(f"{path}: non-positive {source} timing")
        result[operation] = matches[0]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-reps", type=int, default=5)
    parser.add_argument("--density-reps", type=int, default=3)
    parser.add_argument("--cpu", type=int, default=14)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--groups", nargs="+", choices=("earth", "job", "cluster", "density"),
                        default=("earth", "job", "cluster", "density"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-busy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.core_reps < 3 or args.core_reps % 2 == 0:
        raise ValueError("--core-reps must be odd and >=3")
    if args.density_reps < 1 or args.density_reps % 2 == 0:
        raise ValueError("--density-reps must be odd")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    for verifier in (VERIFY_DDC, VERIFY_BASELINES, VERIFY_DENSITY):
        if not verifier.is_file():
            raise FileNotFoundError(verifier)
    host_preflight = host_idle_preflight(args.cpu, args.allow_busy)
    cases = [
        case for case in all_cases(args.core_reps, args.density_reps)
        if case.group in args.groups
    ]
    raw_dir = HERE / "raw"
    log_dir = HERE / "logs"
    raw_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    operand_manifest: list[dict[str, object]] = []
    for case in cases:
        if case.dataset is not None and not Path(case.dataset).is_file():
            raise FileNotFoundError(case.dataset)
        for backend in case.backends:
            directory = Path(case.directories[backend])
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            bitmaps = numeric_bitmaps(directory)
            needed = 3 if "COMP" in case.operations and len(bitmaps) >= 3 else 2
            if len(bitmaps) < needed:
                raise RuntimeError(f"{directory}: only {len(bitmaps)} bitmaps")
            operand_manifest.append({
                "group": case.group,
                "case": case.label,
                "backend": backend,
                "directory": str(directory),
                "bitmap_count": len(bitmaps),
                "total_bm_bytes": sum(path.stat().st_size for path in bitmaps),
                "operands": [
                    {"name": path.name, "bytes": path.stat().st_size,
                     "sha256": sha256(path)}
                    for path in bitmaps[:needed]
                ],
            })

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
            "density": {
                "path": str(VERIFY_DENSITY),
                "sha256": sha256(VERIFY_DENSITY),
            },
        },
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "host_preflight": host_preflight,
        "cpu": args.cpu,
        "core_reps": args.core_reps,
        "density_reps": args.density_reps,
        "outer_iterations": 1,
        "inner_measurements": 100,
        "literal_comp_plan": LITERAL_PLAN,
        "croaring_optimization": (
            "inputs were generated with an explicit runOptimize() call; "
            "container selection remains automatic"
        ),
        "delivery_boundary": (
            "CRoaring OR/AND/NOT use *_op_conv and COMP uses delivered COMP_op; "
            "other backends use their existing *_op/COMP_op paths"
        ),
        "ddc_compress_results": False,
        "and_mode": "pair",
        "allocator_env": {
            "MALLOC_TRIM_THRESHOLD_": "1073741824",
            "MALLOC_MMAP_THRESHOLD_": "1073741824",
        },
        "groups": list(args.groups),
        "cases": [asdict(case) for case in cases],
        "operand_manifest": operand_manifest,
    }
    metadata_path = HERE / "run_metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(metadata_path)
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        stable = (
            "binary", "binary_sha256", "git_commit", "verifiers", "cpu", "core_reps",
            "density_reps", "outer_iterations", "inner_measurements",
            "literal_comp_plan", "croaring_optimization", "delivery_boundary",
            "ddc_compress_results", "and_mode", "allocator_env", "groups",
            "cases", "operand_manifest",
        )
        normalized = json.loads(json.dumps(metadata))
        mismatches = [
            key for key in stable if existing.get(key) != normalized.get(key)
        ]
        if mismatches:
            raise RuntimeError(f"resume metadata mismatch: {mismatches}")
    else:
        if list(raw_dir.glob("*.csv")):
            raise RuntimeError("raw CSVs exist without metadata")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    verification_dir = HERE / "verification"
    verification_dir.mkdir(exist_ok=True)
    density_root = WORKSPACE / "combit" / "gridindep" / "bitmap"
    for verify_index, case in enumerate(cases, start=1):
        if case.group == "density":
            commands = {
                "all": [
                    str(VERIFY_DENSITY), "--root", str(density_root),
                    "--case", case.label, "--rows", str(case.rows),
                ]
            }
        else:
            commands = {
                "ddc": [
                    str(VERIFY_DDC), str(case.dataset), case.directories["ddc"],
                    str(case.rows), str(case.cardinality),
                ],
                "baselines": [
                    str(VERIFY_BASELINES), str(case.dataset),
                    case.directories["wah"], case.directories["ewah"],
                    case.directories["croaring"], str(case.rows),
                    str(case.cardinality),
                ],
            }
        for label, command in commands.items():
            log_path = verification_dir / f"{case.group}_{case.label}_{label}.log"
            if args.resume and log_path.is_file():
                content = log_path.read_text(encoding="utf-8")
                valid = content.startswith("PASS") or any(
                    line.strip() == "ALL_EXACT_MATCH"
                    for line in content.splitlines()
                )
                if valid:
                    continue
            completed = subprocess.run(
                command, cwd=WORKSPACE, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            valid = completed.stdout.startswith("PASS") or any(
                line.strip() == "ALL_EXACT_MATCH"
                for line in completed.stdout.splitlines()
            )
            if completed.returncode != 0 or not valid:
                raise RuntimeError(f"verification failed: {log_path}")
        print(f"[verify {verify_index}/{len(cases)}] {case.group}/{case.label} PASS",
              flush=True)

    groups: dict[str, list[Case]] = {}
    for case in cases:
        groups.setdefault(case.group, []).append(case)
    tasks: list[tuple[int, Case, str]] = []
    for group, group_cases in groups.items():
        repetitions = group_cases[0].repetitions
        original_index = {case.label: index for index, case in enumerate(group_cases)}
        for rep in range(1, repetitions + 1):
            ordered_cases = group_cases[rep - 1:] + group_cases[:rep - 1]
            for case in ordered_cases:
                shift = (rep - 1 + original_index[case.label]) % len(case.backends)
                order = case.backends[shift:] + case.backends[:shift]
                tasks.extend((rep, case, backend) for backend in order)

    started = time.monotonic()
    for index, (rep, case, backend) in enumerate(tasks, start=1):
        stem = f"{case.group}_r{rep}_{case.label}_{backend}"
        raw_path = raw_dir / f"{stem}.csv"
        log_path = log_dir / f"{stem}.log"
        if raw_path.exists():
            if not args.resume:
                raise FileExistsError(raw_path)
            rows = selected_rows(raw_path, case, backend)
        else:
            environment = dict(os.environ)
            environment["MALLOC_TRIM_THRESHOLD_"] = "1073741824"
            environment["MALLOC_MMAP_THRESHOLD_"] = "1073741824"
            environment["DDC_BENCH_AND_MODE"] = "pair"
            command = [
                "taskset", "-c", str(args.cpu), str(binary),
                "--backend", backend,
                "--compressed-dir", case.directories[backend],
                "--num-rows", str(case.rows),
                "--iterations", "1", "--csv", str(raw_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=WORKSPACE, env=environment,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = selected_rows(raw_path, case, backend)
        details = ",".join(
            f"{operation}={float(row['time_ms']) * 1000:.3f}us"
            for operation, row in rows.items()
        )
        print(
            f"[{index}/{len(tasks)}] {case.group}/r{rep}/{case.label}/{backend} "
            f"{details} elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
