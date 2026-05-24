#!/usr/bin/env python3
"""
Aggregate L2/L3/L4/L5 variant measurements into bench_L.xlsx.

Phase 1 (this script today): SIZE columns only — populated from
bench_L_sizes.csv (which the C++ tool bench_L_sizes produces).
Phase 2 (later): AND/OR/XOR speed columns will be appended once the
L2/L3/L5 operator implementations land.
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


def main():
    here = Path(__file__).resolve().parent
    sizes_csv = here / "bench_L_sizes.csv"
    if not sizes_csv.exists():
        sys.exit(f"missing {sizes_csv} — run ./build/bench_L_sizes first")

    rows = list(csv.DictReader(sizes_csv.open()))

    cards = sorted({int(r["cardinality"]) for r in rows})
    variants = ["L2", "L3", "L4", "L5"]

    # Build dict: (card, variant) -> row
    by_key = {(int(r["cardinality"]), r["variant"]): r for r in rows}

    wb = Workbook()
    ws = wb.active
    ws.title = "Sizes"

    bold = Font(bold=True)
    centre = Alignment(horizontal="center")
    hdr_fill = PatternFill("solid", fgColor="D8E4F0")

    headers = ["Cardinality", "Variant",
               "L1 bytes", "L2 bytes", "L3 bytes", "L4 bytes", "L5 bytes",
               "Total bytes", "Total MiB"]
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = bold
        cell.alignment = centre
        cell.fill = hdr_fill

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
        r += 1  # blank row between cardinalities for readability

    # Auto-fit column widths
    for col_idx in range(1, len(headers) + 1):
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = 14

    # ---- Notes sheet ----
    notes = wb.create_sheet("Notes")
    notes_lines = [
        "bench_L.xlsx — L2 / L3 / L4 / L5 marker-depth study",
        "",
        "Phase 1 (this version):",
        "  • Sizes only — derived analytically from existing L4-compressed",
        "    bitmaps (one ComBit per cardinality, 100M rows, seg_bits=2^16).",
        "  • Per-bitmap byte counts averaged across all .bm files in",
        "    bm_100m_c{N}_combit_w8/.",
        "",
        "Variants (top layer must be present; lower layers compressed):",
        "  • L2 — L1 literals + L2 raw bit stream (no L3, no L4)",
        "  • L3 — L1 + L2_lit (gated by L3) + L3 raw",
        "  • L4 — current implementation (L1 + L2_lit + L3_lit + L4 raw)",
        "  • L5 — L1 + L2_lit + L3_lit + L4_lit (gated by L5) + L5 raw",
        "",
        "l4_lit is computed by walking the L4 byte stream and counting bytes",
        "that differ from l4_fill (chosen 0x00 or 0xFF, whichever is more",
        "common).  l5_count = number of L4 BYTES = ⌈l4_count/8⌉.",
        "",
        "Phase 2 (TBD): append AND / OR / XOR op-only timing columns once",
        "the L2/L3/L5 operator implementations are wired in.",
    ]
    for i, line in enumerate(notes_lines, start=1):
        notes.cell(row=i, column=1, value=line)
    notes.column_dimensions["A"].width = 80

    out_path = here / "bench_L.xlsx"
    wb.save(out_path)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
