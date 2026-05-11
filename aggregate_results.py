#!/usr/bin/env python3
"""Aggregate results_full.csv → results_summary.csv.

For each (backend, cardinality) we report:
  - load: median time to load all bitmaps
  - OR/AND/XOR: median per-pair time **with handle wrapper alloc**
  - OR_op/AND_op/XOR_op: median per-pair time **op-only (pre-alloc, post-warmup)**
  - multi-OR: median time to OR all bitmaps via chained binary OR
  - compressed_MB / ratio: storage of all bitmaps
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "results_full.csv"
DST = ROOT / "results_summary.csv"

# (backend, cardinality, operation) → list of time_ms
times = defaultdict(list)
storage = {}  # (backend, cardinality) → (compressed_bytes, ratio)

with SRC.open() as f:
    rd = csv.DictReader(f)
    for row in rd:
        key = (row["backend"], int(row["cardinality"]), row["operation"])
        times[key].append(float(row["time_ms"]))
        if row["operation"] == "load":
            storage[(row["backend"], int(row["cardinality"]))] = (
                int(row["compressed_bytes"]), float(row["compression_ratio"])
            )

ops_pair = ["OR", "AND", "XOR"]
ops_pure = ["OR_op", "AND_op", "XOR_op"]
backend_order = ["ComBIT (New)", "CRoaring", "EWAH", "Concise", "WAH (FastBit)"]
cards = sorted({k[1] for k in times.keys()})

with DST.open("w", newline="") as f:
    w = csv.writer(f)
    header = (["backend", "cardinality", "compressed_MB", "ratio", "load_ms"]
              + [f"{op}_pair_ms" for op in ops_pair]
              + [f"{op}_op_ms" for op in ops_pure]
              + ["multiOR_ms"])
    w.writerow(header)
    for bend in backend_order:
        for c in cards:
            comp_bytes, ratio = storage.get((bend, c), (0, 0.0))
            row = [bend, c, comp_bytes / (1024.0 * 1024.0), ratio]
            # load
            load_t = times.get((bend, c, "load"), [])
            row.append(statistics.median(load_t) if load_t else "")
            # pairwise
            for op in ops_pair:
                t = times.get((bend, c, op), [])
                row.append(statistics.median(t) if t else "")
            # pure ops
            for op in ops_pure:
                t = times.get((bend, c, op), [])
                row.append(statistics.median(t) if t else "")
            # multi-OR
            t = times.get((bend, c, "multi-OR"), [])
            row.append(statistics.median(t) if t else "")
            w.writerow(row)

print(f"Wrote {DST}")
print(f"  {len(cards)} cards x {len(backend_order)} backends")
print(f"  cols: load + 3 pairwise + 3 op-only + multi-OR")
