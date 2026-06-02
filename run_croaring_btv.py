#!/usr/bin/env python3
"""
Benchmark CRoaring-bitvector across all cardinalities.
CRoaring-bitvector = CRoaring operation + forced conversion to flat bitset
when the result contains array/run containers.

Generates CRoaring bitmap files if not present, then runs benchmark_app
with --backend croaring and collects storage + operation timing (including
the to-bitset conversion overhead).

Usage (on server):
    cd ~/lee/thesis/ddc
    python3 ../run_croaring_btv.py                        # default: all cards
    python3 ../run_croaring_btv.py --regen                # force regenerate .bm files
    python3 ../run_croaring_btv.py -o my_results.csv      # custom output filename
    python3 ../run_croaring_btv.py --cards 2,100,1000     # specific cardinalities
    python3 ../run_croaring_btv.py --iterations 10        # more iterations
"""

import subprocess
import re
import csv
import os
import sys
import shutil
import argparse

ROWS = 100_000_000
ITERATIONS = 5
CARDS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]

ALGO_LABEL = "CRoaring-bitvector"


def get_bm_dir(base_dir, card):
    return os.path.join(base_dir, "bitmap", f"bm_100m_c{card}_roaring")


# ── Step 1: generate bitmaps ─────────────────────────────────
def generate_bitmaps(base_dir, build_dir, force_regen, cards=None):
    if cards is None:
        cards = CARDS
    gen_bm = os.path.join(build_dir, "gen_bitmap")
    if not os.path.isfile(gen_bm):
        print(f"ERROR: gen_bitmap not found at {gen_bm}")
        sys.exit(1)

    print("=" * 55)
    print(" Step 1: Generating CRoaring bitmaps")
    print("=" * 55)

    for card in cards:
        bm_path = get_bm_dir(base_dir, card)
        done_file = os.path.join(bm_path, "done.txt")

        if force_regen and os.path.isdir(bm_path):
            shutil.rmtree(bm_path)

        if os.path.isfile(done_file):
            print(f"  [SKIP] bm_100m_c{card}_roaring (already exists)")
            continue

        print(f"  [GEN]  CRoaring c={card} ...", end=" ", flush=True)

        cmd = [gen_bm, "-n", str(ROWS), "-c", str(card),
               "roaring", "-d", base_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr[:500])
            sys.exit(1)
        print("OK")

    print("  Bitmap generation complete.\n")


# ── Step 2: parse benchmark output ───────────────────────────
def parse_benchmark_output(output, card):
    """Extract storage + timing from benchmark_app stdout for CRoaring."""
    results = []
    base = {"algorithm": ALGO_LABEL, "word_size": 0, "cardinality": card, "num_rows": ROWS}

    # ── Load time ──
    m = re.search(r'\[Load\] Loaded (\d+) compressed bitmaps: ([\d.]+) ms', output)
    if m:
        results.append({**base, "phase": "Normal", "operation": "load",
                        "time_ms": float(m.group(2)), "num_bitmaps": int(m.group(1))})

    # ── General storage line ──
    m = re.search(r'\[Storage\] Total compressed: (\d+) bytes .* Ratio: ([\d.]+)x', output)
    compressed_bytes = int(m.group(1)) if m else 0
    compression_ratio = float(m.group(2)) if m else 0

    # ── CRoaring aggregate container stats ──
    cr_arr = cr_bytes_arr = cr_run = cr_bytes_run = cr_bs = cr_bytes_bs = cr_total = 0
    m = re.search(
        r'\[CRoaring Storage\]\s*array=(\d+)\s*\(([\d.e+-]+)\s*MB\)'
        r'\s*\|\s*run=(\d+)\s*\(([\d.e+-]+)\s*MB\)'
        r'\s*\|\s*bitset=(\d+)\s*\(([\d.e+-]+)\s*MB\)'
        r'\s*\|\s*total_bytes=(\d+)',
        output)
    if m:
        cr_arr = int(m.group(1))
        cr_bytes_arr = float(m.group(2))
        cr_run = int(m.group(3))
        cr_bytes_run = float(m.group(4))
        cr_bs = int(m.group(5))
        cr_bytes_bs = float(m.group(6))
        cr_total = int(m.group(7))

    storage_row = {**base, "phase": "Storage", "operation": "storage",
                   "compressed_bytes": compressed_bytes,
                   "compression_ratio": compression_ratio,
                   "cr_array_containers": cr_arr,
                   "cr_bytes_array_mb": round(cr_bytes_arr, 1),
                   "cr_run_containers": cr_run,
                   "cr_bytes_run_mb": round(cr_bytes_run, 1),
                   "cr_bitset_containers": cr_bs,
                   "cr_bytes_bitset_mb": round(cr_bytes_bs, 1),
                   "cr_total_bytes": cr_total}
    results.append(storage_row)

    # ── Summary (median over iterations) ──
    summary = re.search(r'\[Summary\] Median over \d+ iterations:', output)
    if summary:
        for op, pat in [
            ("OR",  r'bitOr:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("AND", r'bitAnd:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("XOR", r'bitXor:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("multi-OR", r'multi-OR:\s+([\d.]+) ms \(card: (\d+)\)'),
        ]:
            m = re.search(pat, output[summary.start():])
            if m:
                results.append({**base, "phase": "Normal-Median", "operation": op,
                                "time_ms": float(m.group(1)),
                                "result_cardinality": int(m.group(2))})
        m = re.search(r'Load:\s+([\d.]+) ms', output[summary.start():])
        if m:
            results.append({**base, "phase": "Normal-Median", "operation": "load",
                            "time_ms": float(m.group(1))})
    else:
        # Single iteration fallback
        for op, pat in [
            ("OR",  r'bitOr:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("AND", r'bitAnd:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("XOR", r'bitXor:\s+([\d.]+) ms \(card: (\d+)\)'),
        ]:
            m = re.search(pat, output)
            if m:
                results.append({**base, "phase": "Normal", "operation": op,
                                "time_ms": float(m.group(1)),
                                "result_cardinality": int(m.group(2))})
        m = re.search(r'Multi-way OR of \d+ bitmaps: ([\d.]+) ms \(card: (\d+)\)', output)
        if m:
            results.append({**base, "phase": "Normal", "operation": "multi-OR",
                            "time_ms": float(m.group(1)),
                            "result_cardinality": int(m.group(2))})

    # ── Pure Ops (CRoaring with to-bitset) ──
    pure_section = re.search(r'\[Pure Ops\] CRoaring', output)
    if pure_section:
        pure_text = output[pure_section.start():]
        for op, pat in [
            ("AND", r'AND(?:\s*\([^)]*\))?:\s+([\d.]+) ms'),
            ("OR",  r'OR(?:\s*\([^)]*\))?:\s+([\d.]+) ms'),
            ("XOR", r'XOR(?:\s*\([^)]*\))?:\s+([\d.]+) ms'),
        ]:
            m = re.search(pat, pure_text)
            if m:
                results.append({**base, "phase": "PureOps", "operation": op,
                                "time_ms": float(m.group(1))})

    return results


# ── Step 3: run benchmarks & write CSV ───────────────────────
def run_benchmarks(base_dir, build_dir, csv_path, cards=None, iterations=None):
    if cards is None:
        cards = CARDS
    if iterations is None:
        iterations = ITERATIONS
    bench = os.path.join(build_dir, "benchmark_app")
    if not os.path.isfile(bench):
        print(f"ERROR: benchmark_app not found at {bench}")
        sys.exit(1)

    fieldnames = [
        "algorithm", "word_size", "cardinality", "num_rows",
        "phase", "operation", "time_ms",
        "compressed_bytes", "compression_ratio",
        "cr_array_containers", "cr_bytes_array_mb",
        "cr_run_containers", "cr_bytes_run_mb",
        "cr_bitset_containers", "cr_bytes_bitset_mb",
        "cr_total_bytes",
        "result_cardinality", "num_bitmaps",
    ]

    all_rows = []
    total = len(cards)
    count = 0

    print("=" * 55)
    print(" Step 2: Running CRoaring-bitvector benchmarks")
    print(f" Cards      : {cards}")
    print(f" Iterations : {iterations}")
    print("=" * 55)

    for card in cards:
        count += 1
        bm_path = get_bm_dir(base_dir, card)

        if not os.path.isdir(bm_path):
            print(f"  [{count}/{total}] SKIP c={card} (dir not found: {bm_path})")
            continue

        print(f"  [{count}/{total}] CRoaring-btv c={card} ...", end=" ", flush=True)

        cmd = [bench, "--compressed-dir", bm_path,
               "--backend", "croaring",
               "--iterations", str(iterations)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("FAILED")
            print(result.stderr[:500])
            continue

        rows = parse_benchmark_output(result.stdout, card)
        all_rows.extend(rows)

        # Show brief summary
        medians = {r["operation"]: r["time_ms"]
                   for r in rows if r["phase"] == "Normal-Median" and "time_ms" in r}
        storage = next((r for r in rows if r["phase"] == "Storage"), None)
        parts = []
        if storage and storage.get("compressed_bytes"):
            sz_kb = storage["compressed_bytes"] / 1024.0
            parts.append(f"size={sz_kb:.1f}KB")
            parts.append(f"ratio={storage['compression_ratio']:.4f}x")
        for op in ["OR", "AND", "XOR"]:
            if op in medians:
                parts.append(f"{op}={medians[op]:.2f}ms")
        print(f"  {', '.join(parts)}" if parts else "  done")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n{'=' * 55}")
    print(f" Done! {len(all_rows)} rows written to:")
    print(f"   {csv_path}")
    print(f"{'=' * 55}")


def main():
    parser = argparse.ArgumentParser(
        description="CRoaring-bitvector benchmark runner (op + forced to-bitset conversion)")
    parser.add_argument("--base-dir", default=".",
                        help="ddc project root (default: .)")
    parser.add_argument("-o", "--output", default="results_croaring_btv.csv",
                        help="output CSV filename (default: results_croaring_btv.csv)")
    parser.add_argument("--regen", action="store_true",
                        help="force regenerate all CRoaring .bm files")
    parser.add_argument("--skip-gen", action="store_true",
                        help="skip bitmap generation entirely")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help=f"benchmark iterations (default: {ITERATIONS})")
    parser.add_argument("--cards", type=str, default=None,
                        help="comma-separated cardinalities, e.g. '2,10,100,1000'")
    args = parser.parse_args()

    iters = args.iterations
    card_list = [int(x) for x in args.cards.split(",")] if args.cards else list(CARDS)

    base_dir = os.path.abspath(args.base_dir)
    build_dir = os.path.join(base_dir, "build")
    csv_path = os.path.join(base_dir, args.output)

    print(f"\n Base dir   : {base_dir}")
    print(f" Build dir  : {build_dir}")
    print(f" Output     : {csv_path}")
    print(f" Cards      : {card_list}")
    print(f" Iterations : {iters}")
    print(f" Regen      : {args.regen}")
    print()

    if not args.skip_gen:
        generate_bitmaps(base_dir, build_dir, args.regen, card_list)

    run_benchmarks(base_dir, build_dir, csv_path, card_list, iters)


if __name__ == "__main__":
    main()
