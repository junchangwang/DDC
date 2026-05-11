#!/usr/bin/env python3
"""
Export TPC-H Q1 bitmap data from DuckDB.

Generates raw .bm bitmap files (LSB-first packed bits) for:
  - l_returnflag: 3 equality-encoded bitmaps (A=0, N=1, R=2)
  - l_linestatus: 2 equality-encoded bitmaps (F=0, O=1)
  - l_shipdate:   reuse from Q6 export (days since 1992-01-01)

Usage:
  python3 export_tpch_q1.py <duckdb_binary> <db_path> <output_dir>

Example:
  python3 util/export_tpch_q1.py \
      ../duckdb-dev/build/release/duckdb \
      ../duckdb-dev/tpch_sf10.db \
      bitmap/tpch_q1
"""

import sys
import os
import subprocess
import numpy as np
import time


def export_csv(duckdb_bin, db_path, csv_path):
    """Export Q1-relevant columns from DuckDB to CSV."""
    sql = f"""
COPY (
    SELECT
        CASE l_returnflag WHEN 'A' THEN 0 WHEN 'N' THEN 1 WHEN 'R' THEN 2 END AS returnflag,
        CASE l_linestatus WHEN 'F' THEN 0 WHEN 'O' THEN 1 END AS linestatus,
        l_shipdate - DATE '1992-01-01' AS shipdate_days
    FROM lineitem
) TO '{csv_path}' (HEADER false);
"""
    print(f"[export] Exporting Q1 columns to {csv_path} ...")
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


def gen_eq_bitmaps(column_data, output_dir, val_min, val_max):
    """Generate equality-encoded bitmaps (raw packed bits, LSB-first)."""
    os.makedirs(output_dir, exist_ok=True)
    num_rows = len(column_data)
    print(f"[gen_eq] Generating {val_max - val_min + 1} bitmaps "
          f"(values {val_min}..{val_max}) for {num_rows} rows -> {output_dir}")
    t0 = time.time()
    for v in range(val_min, val_max + 1):
        mask = (column_data == v).astype(np.uint8)
        packed = np.packbits(mask, bitorder='little')
        out_path = os.path.join(output_dir, f"{v}.bm")
        packed.tofile(out_path)
    elapsed = time.time() - t0
    print(f"[gen_eq] Done in {elapsed:.1f}s")


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q1.py <duckdb_binary> <db_path> <output_dir>")
        sys.exit(1)

    duckdb_bin = sys.argv[1]
    db_path = sys.argv[2]
    output_dir = sys.argv[3]

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "_q1_export.csv")
    export_csv(duckdb_bin, db_path, csv_path)

    print(f"[read] Loading CSV ...")
    t0 = time.time()
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.int32)
    print(f"[read] {data.shape[0]} rows in {time.time()-t0:.1f}s")

    returnflag = data[:, 0]
    linestatus = data[:, 1]
    shipdate   = data[:, 2]
    num_rows = len(returnflag)

    # returnflag: A=0, N=1, R=2
    gen_eq_bitmaps(returnflag, os.path.join(output_dir, "returnflag"), 0, 2)

    # linestatus: F=0, O=1
    gen_eq_bitmaps(linestatus, os.path.join(output_dir, "linestatus"), 0, 1)

    # shipdate: reuse from Q6 if exists, otherwise generate
    ship_dir = os.path.join(output_dir, "shipdate")
    q6_ship_dir = output_dir.replace("tpch_q1", "tpch_q6") + "/shipdate"
    if os.path.isdir(q6_ship_dir) and len(os.listdir(q6_ship_dir)) > 2000:
        print(f"[shipdate] Symlinking from Q6: {q6_ship_dir}")
        if os.path.exists(ship_dir):
            if os.path.islink(ship_dir):
                os.unlink(ship_dir)
            else:
                import shutil
                shutil.rmtree(ship_dir)
        os.symlink(os.path.abspath(q6_ship_dir), ship_dir)
    else:
        print(f"[shipdate] Q6 shipdate not found at {q6_ship_dir}, generating...")
        min_day = int(shipdate.min())
        max_day = int(shipdate.max())
        gen_eq_bitmaps(shipdate, ship_dir, min_day, max_day)

    # Metadata
    min_day = int(shipdate.min())
    max_day = int(shipdate.max())
    cutoff_day = 2436  # 1998-09-02 - 1992-01-01

    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"source=TPC-H lineitem\n")
        f.write(f"num_rows={num_rows}\n")
        f.write(f"encoding=equality (one bitmap per distinct value)\n")
        f.write(f"format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n\n")
        f.write(f"returnflag: A=0, N=1, R=2 (3 bitmaps)\n")
        f.write(f"linestatus: F=0, O=1 (2 bitmaps)\n")
        f.write(f"shipdate: days since 1992-01-01, values {min_day}..{max_day}\n")
        f.write(f"  epoch: 1992-01-01\n\n")
        f.write(f"Q1 filter:\n")
        f.write(f"  l_shipdate <= 1998-09-02 (day {cutoff_day})\n")
        f.write(f"  Complement approach: OR days {cutoff_day+1}..{max_day} = {max_day-cutoff_day} bitmaps, then NOT\n")
        f.write(f"  GROUP BY returnflag, linestatus (3x2=6 groups, AND with group bitmaps)\n")

    os.remove(csv_path)
    print(f"\n[done] Output: {output_dir}")
    print(f"  returnflag: 3 bitmaps")
    print(f"  linestatus: 2 bitmaps")
    print(f"  shipdate: {max_day - min_day + 1} bitmaps (day {min_day}..{max_day})")
    print(f"  Q1 complement OR: {max_day - cutoff_day} bitmaps")


if __name__ == '__main__':
    main()
