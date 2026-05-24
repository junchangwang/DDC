#!/usr/bin/env python3
"""
Aggregate L2/L3/L4/L5 variant measurements into bench_L.xlsx.

Reads two CSVs:
  - bench_L_sizes.csv  (Phase 1 — analytical per-layer sizes)
  - bench_L_ops.csv    (Phase 2 — real depth-N compress + op times +
                                  correctness flags)

The Excel has three sheets:
  - "Sizes (analytical)"  — per cardinality / variant / layer bytes
  - "Ops (measured)"      — per cardinality / variant total size,
                            AND / OR / XOR ms, correctness flags
  - "Notes"               — methodology
"""

import csv
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError:
    sys.exit("Please install openpyxl: pip install openpyxl")


def make_header(ws, headers, row=1):
    bold = Font(bold=True)
    centre = Alignment(horizontal="center")
    fill = PatternFill("solid", fgColor="D8E4F0")
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=row, column=col, value=h)
        cell.font = bold; cell.alignment = centre; cell.fill = fill


def main():
    here = Path(__file__).resolve().parent
    sizes_csv = here / "bench_L_sizes.csv"
    ops_csv   = here / "bench_L_ops.csv"

    wb = Workbook()

    # ---- Sheet 1: Sizes (analytical) ----
    ws = wb.active
    ws.title = "Sizes (analytical)"
    if sizes_csv.exists():
        rows = list(csv.DictReader(sizes_csv.open()))
        headers = ["Cardinality", "Variant", "L1 bytes", "L2 bytes",
                   "L3 bytes", "L4 bytes", "L5 bytes",
                   "Total bytes", "Total MiB"]
        make_header(ws, headers)
        cards = sorted({int(r["cardinality"]) for r in rows})
        variants = ["L2", "L3", "L4", "L5"]
        by_key = {(int(r["cardinality"]), r["variant"]): r for r in rows}
        r = 2
        for c in cards:
            for v in variants:
                row = by_key.get((c, v))
                if not row: continue
                ws.cell(row=r, column=1, value=c)
                ws.cell(row=r, column=2, value=v).font = Font(bold=(v == "L4"))
                ws.cell(row=r, column=3, value=float(row["l1_bytes"]))
                ws.cell(row=r, column=4, value=float(row["l2_bytes"]))
                ws.cell(row=r, column=5, value=float(row["l3_bytes"]))
                ws.cell(row=r, column=6, value=float(row["l4_bytes"]))
                ws.cell(row=r, column=7, value=float(row["l5_bytes"]))
                ws.cell(row=r, column=8, value=float(row["total_bytes"]))
                ws.cell(row=r, column=9, value=float(row["total_MiB"]))
                for col in range(3, 10):
                    ws.cell(row=r, column=col).number_format = '#,##0.00'
                r += 1
            r += 1
        for i in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(i)].width = 14
    else:
        ws.cell(row=1, column=1, value="bench_L_sizes.csv missing")

    # ---- Sheet 2: Ops (measured) ----
    ops = wb.create_sheet("Ops (measured)")
    if ops_csv.exists():
        rows = list(csv.DictReader(ops_csv.open()))
        headers = ["Cardinality", "Variant", "Total bytes", "Total MiB",
                   "AND ms", "OR ms", "XOR ms",
                   "AND ok", "OR ok", "XOR ok", "Round-trip ok"]
        make_header(ops, headers)
        cards = sorted({int(r["cardinality"]) for r in rows})
        variants = ["L2", "L3", "L4", "L5"]
        by_key = {(int(r["cardinality"]), r["variant"]): r for r in rows}
        r = 2
        for c in cards:
            for v in variants:
                row = by_key.get((c, v))
                if not row: continue
                ops.cell(row=r, column=1, value=c)
                ops.cell(row=r, column=2, value=v).font = Font(bold=(v == "L4"))
                ops.cell(row=r, column=3, value=int(row["total_bytes"]))
                ops.cell(row=r, column=4, value=float(row["total_MiB"]))
                ops.cell(row=r, column=5, value=float(row["and_ms"]))
                ops.cell(row=r, column=6, value=float(row["or_ms"]))
                ops.cell(row=r, column=7, value=float(row["xor_ms"]))
                ops.cell(row=r, column=8, value=int(row["and_ok"]))
                ops.cell(row=r, column=9, value=int(row["or_ok"]))
                ops.cell(row=r, column=10, value=int(row["xor_ok"]))
                ops.cell(row=r, column=11, value=int(row["roundtrip_ok"]))
                for col in [4, 5, 6, 7]:
                    ops.cell(row=r, column=col).number_format = '0.000'
                r += 1
            r += 1
        for i in range(1, len(headers) + 1):
            ops.column_dimensions[get_column_letter(i)].width = 14
    else:
        ops.cell(row=1, column=1, value="bench_L_ops.csv missing — Phase 2 not yet run")

    # ---- Sheet 3: Notes ----
    notes = wb.create_sheet("Notes")
    text = [
        "bench_L.xlsx — L2 / L3 / L4 / L5 marker-depth study",
        "",
        "Variants (top layer always present and stored raw; lower layers",
        "compressed via the layer above):",
        "  • L2 — L1 literals + L2 raw bit stream (no L3, no L4)",
        "  • L3 — L1 + L2_lit + L3 raw",
        "  • L4 — current production ComBit  (L1 + L2_lit + L3_lit + L4 raw)",
        "  • L5 — L1 + L2_lit + L3_lit + L4_lit + L5 raw",
        "",
        "Sheet 1 (Sizes, analytical):",
        "  Computed from existing combit_w8 .bm files (depth-4 native compress).",
        "  l4_lit derived by walking the L4 byte stream and counting bytes",
        "  that differ from the rarer of {0x00, 0xFF}.",
        "",
        "Sheet 2 (Ops, measured):",
        "  Re-compresses each input as a depth-N ComBitN (see combit_n.cpp,",
        "  scalar implementation isolated from the AVX-512 ComBit code so",
        "  the depth comparison is apples-to-apples).  AND / OR / XOR",
        "  times are median over 5 iterations of the full op (decode + op",
        "  + recompress at the same depth).  '*_ok' columns are 1 iff the",
        "  decompressed result matches a raw bit-vector reference; 'Round-",
        "  trip ok' is 1 iff decompress(compress(bits)) == bits.",
        "",
        "Methodology note: the scalar baseline does NOT take advantage of",
        "marker skipping (deep variants would skip large zero subtrees in",
        "an optimised implementation), so times across variants are very",
        "close.  This sheet establishes the size winners; a future Phase 3",
        "would add skip-aware ops to expose the depth-vs-speed trade-off.",
    ]
    for i, line in enumerate(text, start=1):
        notes.cell(row=i, column=1, value=line)
    notes.column_dimensions["A"].width = 80

    out_path = here / "bench_L.xlsx"
    wb.save(out_path)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
