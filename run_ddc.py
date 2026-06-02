#!/usr/bin/env python3
"""
Benchmark DDC (w8) across all cardinalities.

Generates DDC compressed bitmap files if not present, then runs
benchmark_app with --backend ddc and collects storage + operation timing.

Usage (on server):
    cd ~/lee/thesis/check/lee/ddc
    python3 ../run_ddc.py                          # default: all cards
    python3 ../run_ddc.py --regen                  # force regenerate .bm files
    python3 ../run_ddc.py -o my_results.csv        # custom output filename
    python3 ../run_ddc.py --cards 2,100,1000       # specific cardinalities
    python3 ../run_ddc.py --iterations 10          # more iterations
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
CARDS = [2, 5, 10, 20, 50, 100, 200, 500, 1000,
         2000, 3000, 4000, 5000, 8000, 10000]

ALGO_LABEL = "DDC-w8"


def get_bm_dir(base_dir, card):
    return os.path.join(base_dir, "bitmap", f"bm_100m_c{card}_ddc_w8")


# ── Step 1: generate bitmaps ─────────────────────────────────
def generate_bitmaps(base_dir, build_dir, force_regen, cards=None):
    if cards is None:
        cards = CARDS
    gen_bm = os.path.join(build_dir, "gen_bitmap")
    if not os.path.isfile(gen_bm):
        print(f"ERROR: gen_bitmap not found at {gen_bm}")
        sys.exit(1)

    print("=" * 55)
    print(" Step 1: Generating DDC bitmaps")
    print("=" * 55)

    for card in cards:
        bm_path = get_bm_dir(base_dir, card)
        done_file = os.path.join(bm_path, "done.txt")

        if force_regen and os.path.isdir(bm_path):
            shutil.rmtree(bm_path)

        if os.path.isfile(done_file):
            print(f"  [SKIP] bm_100m_c{card}_ddc (already exists)")
            continue

        print(f"  [GEN]  DDC c={card} ...", end=" ", flush=True)

        cmd = [gen_bm, "-n", str(ROWS), "-c", str(card),
               "ddc", "-d", base_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FAILED")
            print(result.stderr[:500])
            sys.exit(1)
        print("OK")

    print("  Bitmap generation complete.\n")


# ── Step 2: parse benchmark output ───────────────────────────
def parse_benchmark_output(output, card):
    """Extract storage + timing from benchmark_app stdout for DDC."""
    results = []
    base = {"algorithm": ALGO_LABEL, "word_size": 8, "cardinality": card, "num_rows": ROWS}

    # ── Load time ──
    m = re.search(r'\[Load\] Loaded (\d+) compressed bitmaps: ([\d.]+) ms', output)
    if m:
        results.append({**base, "phase": "Normal", "operation": "load",
                        "time_ms": float(m.group(2)), "num_bitmaps": int(m.group(1))})

    # ── General storage line ──
    m = re.search(r'\[Storage\] Total compressed: (\d+) bytes .* Ratio: ([\d.]+)x', output)
    compressed_bytes = int(m.group(1)) if m else 0
    compression_ratio = float(m.group(2)) if m else 0

    # ── DDC storage breakdown (MB) ──
    cb_l3_mb = 0.0
    cb_l2_mb = 0.0
    cb_l1_mb = 0.0
    cb_total_mb = 0.0
    m = re.search(
        r'\[DDC Storage\]\s*l3_bytes:\s*([\d.e+-]+)\s*MB'
        r'\s*\|\s*l2_bytes:\s*([\d.e+-]+)\s*MB'
        r'\s*\|\s*l1_bytes:\s*([\d.e+-]+)\s*MB'
        r'\s*\|\s*total_bytes:\s*([\d.e+-]+)\s*MB',
        output)
    if m:
        cb_l3_mb = float(m.group(1))
        cb_l2_mb = float(m.group(2))
        cb_l1_mb = float(m.group(3))
        cb_total_mb = float(m.group(4))

    storage_row = {**base, "phase": "Storage", "operation": "storage",
                   "compressed_bytes": compressed_bytes,
                   "compression_ratio": compression_ratio,
                   "cb_l3_mb": round(cb_l3_mb, 4),
                   "cb_l2_mb": round(cb_l2_mb, 4),
                   "cb_l1_mb": round(cb_l1_mb, 4),
                   "cb_total_mb": round(cb_total_mb, 4)}
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

    # ── Pure Ops (DDC) ──
    pure_section = re.search(r'\[Pure Ops\] DDC', output)
    if pure_section:
        pure_text = output[pure_section.start():]
        for op, pat in [
            ("AND", r'AND:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("OR",  r'OR:\s+([\d.]+) ms \(card: (\d+)\)'),
            ("XOR", r'XOR:\s+([\d.]+) ms \(card: (\d+)\)'),
        ]:
            m = re.search(pat, pure_text)
            if m:
                results.append({**base, "phase": "PureOps", "operation": op,
                                "time_ms": float(m.group(1)),
                                "result_cardinality": int(m.group(2))})

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
        "cb_l3_mb", "cb_l2_mb", "cb_l1_mb", "cb_total_mb",
        "result_cardinality", "num_bitmaps",
    ]

    all_rows = []
    total = len(cards)
    count = 0

    print("=" * 55)
    print(" Step 2: Running DDC benchmarks")
    print(f" Cards      : {cards}")
    print(f" Iterations : {iterations}")
    print("=" * 55)

    for card in cards:
        count += 1
        bm_path = get_bm_dir(base_dir, card)

        if not os.path.isdir(bm_path):
            print(f"  [{count}/{total}] SKIP c={card} (dir not found: {bm_path})")
            continue

        print(f"  [{count}/{total}] DDC c={card} ...", end=" ", flush=True)

        cmd = [bench, "--compressed-dir", bm_path,
               "--backend", "ddc",
               "--iterations", str(iterations)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FAILED (exit code {result.returncode})")
            if result.stderr:
                print("  STDERR:", result.stderr[:500])
            if result.stdout:
                # Print last 500 chars of stdout to see where it crashed
                print("  STDOUT (tail):", result.stdout[-500:])
            continue

        rows = parse_benchmark_output(result.stdout, card)
        all_rows.extend(rows)

        # Show brief summary
        medians = {r["operation"]: r["time_ms"]
                   for r in rows if r["phase"] == "Normal-Median" and "time_ms" in r}
        storage = next((r for r in rows if r["phase"] == "Storage" and r["operation"] == "storage"), None)
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
        description="DDC (w8) benchmark runner")
    parser.add_argument("--base-dir", default=".",
                        help="ddc project root (default: .)")
    parser.add_argument("-o", "--output", default="results_ddc.csv",
                        help="output CSV filename (default: results_ddc.csv)")
    parser.add_argument("--regen", action="store_true",
                        help="force regenerate all DDC .bm files")
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
