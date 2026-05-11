#!/usr/bin/env python3
"""
Export TPC-H Q6 bitmap data from DuckDB (publication-grade version).

Generates raw .bm bitmap files (LSB-first packed bits) for:
  - l_discount:  11 equality-encoded bitmaps (values 0..10, representing 0.00..0.10)
  - l_quantity:  50 equality-encoded bitmaps (values 1..50)
  - l_shipdate:  2526 equality-encoded bitmaps (one per distinct date)
                 Date IDs are days since 1992-01-01 (epoch).

All bitmaps use equality encoding — no pre-computed range results.
Q6 must OR the relevant bitmaps at query time, matching real bitmap index behavior.

Usage:
  python3 export_tpch_q6.py <duckdb_binary> <db_path> <output_base_dir>

Example:
  python3 util/export_tpch_q6.py \
      ../duckdb-dev/build/release/duckdb \
      ../duckdb-dev/tpch_sf10.db \
      bitmap/tpch_q6
"""

import sys
import os
import subprocess
import tempfile
import numpy as np
import time


def export_csv(duckdb_bin, db_path, csv_path):
    """Export Q6-relevant columns from DuckDB to CSV."""
    # shipdate encoded as integer: days since 1992-01-01 (TPC-H min date)
    sql = f"""
COPY (
    SELECT
        CAST(l_discount * 100 AS INTEGER) AS discount,
        CAST(l_quantity AS INTEGER) AS quantity,
        l_shipdate - DATE '1992-01-01' AS shipdate_days
    FROM lineitem
) TO '{csv_path}' (HEADER false);
"""
    print(f"[export] Exporting Q6 columns to {csv_path} ...")
    t0 = time.time()
    result = subprocess.run(
        [duckdb_bin, db_path],
        input=sql,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[export] DuckDB error:\n{result.stderr}")
        sys.exit(1)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"[export] Done: {size_mb:.1f} MB in {elapsed:.1f}s")


def read_csv_columns(csv_path):
    """Read 3-column CSV into numpy arrays."""
    print(f"[read] Loading CSV into memory ...")
    t0 = time.time()
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.int32)
    elapsed = time.time() - t0
    print(f"[read] Loaded {data.shape[0]} rows in {elapsed:.1f}s")
    return data[:, 0], data[:, 1], data[:, 2]  # discount, quantity, shipdate_days


def gen_eq_bitmaps(column_data, output_dir, val_min, val_max):
    """
    Generate equality-encoded bitmaps (raw packed bits, LSB-first).
    For each value v in [val_min, val_max], writes output_dir/v.bm
    """
    os.makedirs(output_dir, exist_ok=True)
    num_rows = len(column_data)
    print(f"[gen_eq] Generating {val_max - val_min + 1} bitmaps "
          f"(values {val_min}..{val_max}) for {num_rows} rows → {output_dir}")
    t0 = time.time()

    for v in range(val_min, val_max + 1):
        mask = (column_data == v).astype(np.uint8)
        packed = np.packbits(mask, bitorder='little')
        out_path = os.path.join(output_dir, f"{v}.bm")
        packed.tofile(out_path)

    elapsed = time.time() - t0
    print(f"[gen_eq] Done in {elapsed:.1f}s")


def gen_range_bitmap(bool_data, output_dir, filename):
    """
    Generate a single range bitmap from a boolean (0/1) column.
    (Legacy — kept for reference but not used in publication-grade version.)
    """
    os.makedirs(output_dir, exist_ok=True)
    num_rows = len(bool_data)
    card = int(np.sum(bool_data))
    print(f"[gen_range] Generating range bitmap: {card}/{num_rows} set bits → {output_dir}/{filename}")
    t0 = time.time()

    mask = bool_data.astype(np.uint8)
    packed = np.packbits(mask, bitorder='little')
    out_path = os.path.join(output_dir, filename)
    packed.tofile(out_path)

    elapsed = time.time() - t0
    print(f"[gen_range] Done in {elapsed:.1f}s")


def compute_date_range(epoch_str="1992-01-01"):
    """
    Compute shipdate day-IDs for Q6 range predicate:
      l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'
    Returns (day_start, day_end) as days since epoch (inclusive start, exclusive end).
    """
    from datetime import date
    epoch = date.fromisoformat(epoch_str)
    d_start = (date(1994, 1, 1) - epoch).days   # 730
    d_end   = (date(1995, 1, 1) - epoch).days    # 1095
    return d_start, d_end


def verify_q6(discount_data, quantity_data, shipdate_days):
    """Verify Q6 result count matches expected."""
    d_start, d_end = compute_date_range()
    mask = ((discount_data >= 5) & (discount_data <= 7) &
            (quantity_data < 24) &
            (shipdate_days >= d_start) & (shipdate_days < d_end))
    count = int(np.sum(mask))
    print(f"[verify] Q6 result rows (all 3 conditions AND): {count}")
    print(f"[verify] Shipdate range: day {d_start}..{d_end-1} (365 days)")
    return count


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    duckdb_bin = os.path.abspath(sys.argv[1])
    db_path = os.path.abspath(sys.argv[2])
    output_base = sys.argv[3]

    if not os.path.isfile(duckdb_bin):
        print(f"Error: DuckDB binary not found: {duckdb_bin}")
        sys.exit(1)
    if not os.path.isfile(db_path):
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    # Step 1: Export from DuckDB
    csv_path = tempfile.mktemp(suffix='.csv', prefix='tpch_q6_')
    try:
        export_csv(duckdb_bin, db_path, csv_path)
        discount, quantity, shipdate_days = read_csv_columns(csv_path)
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"[cleanup] Removed temp CSV")

    num_rows = len(discount)
    num_dates = len(np.unique(shipdate_days))
    date_min, date_max = int(shipdate_days.min()), int(shipdate_days.max())
    d_start, d_end = compute_date_range()

    print(f"\n[info] Total rows: {num_rows}")
    print(f"[info] Discount range: {discount.min()} .. {discount.max()} ({len(np.unique(discount))} values)")
    print(f"[info] Quantity range: {quantity.min()} .. {quantity.max()} ({len(np.unique(quantity))} values)")
    print(f"[info] Shipdate days: {date_min} .. {date_max} ({num_dates} distinct dates)")
    print(f"[info] Q6 shipdate range: day {d_start}..{d_end-1} (365 dates to OR)")

    # Verify Q6 expected result
    expected = verify_q6(discount, quantity, shipdate_days)

    # Step 2: Generate equality-encoded bitmaps for ALL columns
    print()
    gen_eq_bitmaps(discount, os.path.join(output_base, "discount"), 0, 10)
    gen_eq_bitmaps(quantity, os.path.join(output_base, "quantity"), 1, 50)
    gen_eq_bitmaps(shipdate_days, os.path.join(output_base, "shipdate"), date_min, date_max)

    # Step 3: Write metadata
    meta_path = os.path.join(output_base, "metadata.txt")
    os.makedirs(output_base, exist_ok=True)
    with open(meta_path, 'w') as f:
        f.write(f"source=TPC-H SF10 lineitem\n")
        f.write(f"num_rows={num_rows}\n")
        f.write(f"encoding=equality (one bitmap per distinct value)\n")
        f.write(f"format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")
        f.write(f"\n")
        f.write(f"discount: values 0..10, {11} bitmaps\n")
        f.write(f"  meaning: l_discount * 100 (0.00 to 0.10)\n")
        f.write(f"quantity: values 1..50, {50} bitmaps\n")
        f.write(f"shipdate: days since 1992-01-01, values {date_min}..{date_max}, {num_dates} bitmaps\n")
        f.write(f"  epoch: 1992-01-01\n")
        f.write(f"\n")
        f.write(f"Q6 filter (at query time, OR then AND):\n")
        f.write(f"  discount BETWEEN 0.05 AND 0.07 → OR(5.bm, 6.bm, 7.bm)\n")
        f.write(f"  quantity < 24 → OR(1.bm .. 23.bm)\n")
        f.write(f"  shipdate >= 1994-01-01 AND < 1995-01-01 → OR({d_start}.bm .. {d_end-1}.bm) = 365 bitmaps\n")
        f.write(f"  final = AND(discount_or, quantity_or, shipdate_or)\n")
        f.write(f"\n")
        f.write(f"Expected Q6 result: {expected} rows\n")
    print(f"[meta] Written {meta_path}")

    # Step 4: Summary
    expected_bytes = (num_rows + 7) // 8
    total_bm = 11 + 50 + num_dates
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  Output:    {os.path.abspath(output_base)}")
    print(f"  Rows:      {num_rows}")
    print(f"  Bitmaps:   {total_bm} ({11} discount + {50} quantity + {num_dates} shipdate)")
    print(f"  Each .bm:  {expected_bytes} bytes ({expected_bytes/1024/1024:.2f} MB)")
    print(f"  Total raw: {total_bm * expected_bytes / 1024/1024:.1f} MB")
    print(f"{'='*60}")
    print(f"  Q6 expected result: {expected} rows")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
