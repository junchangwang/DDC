#!/usr/bin/env python3
"""
Export TPC-H Q3 bitmap data from DuckDB.

Generates:
  - join_result/0.bm: raw bitmap marking lineitem rows from qualifying orders
    (c_mktsegment='BUILDING' AND o_orderdate < '1995-03-15')
  - Shipdate bitmaps: reuse from Q1 export
  - orders_meta.csv: orderkey → orderdate, shippriority for qualifying orders

Usage:
  python3 export_tpch_q3.py <duckdb_binary> <db_path> <output_dir>
"""

import sys, os, subprocess, time
import numpy as np


def run_sql_to_csv(duckdb_bin, db_path, sql, csv_path):
    result = subprocess.run(
        [duckdb_bin, db_path],
        input=sql, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[error] DuckDB:\n{result.stderr}")
        sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q3.py <duckdb_binary> <db_path> <output_dir>")
        sys.exit(1)

    duckdb_bin = sys.argv[1]
    db_path = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    # 1. Export join condition: for each lineitem row, 1 if in qualifying join, 0 otherwise
    join_csv = os.path.join(output_dir, "_q3_join.csv")
    print(f"[export] Computing Q3 join result...")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT CASE WHEN l.l_orderkey IN (
        SELECT o.o_orderkey FROM orders o
        JOIN customer c ON o.o_custkey = c.c_custkey
        WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate < DATE '1995-03-15'
    ) THEN 1 ELSE 0 END AS in_join
    FROM lineitem l
    ORDER BY l.rowid
) TO '{join_csv}' (HEADER false);
"""
    run_sql_to_csv(duckdb_bin, db_path, sql, join_csv)
    elapsed = time.time() - t0
    print(f"[export] Done in {elapsed:.1f}s")

    # 2. Read and pack into raw bitmap
    print(f"[read] Loading join column...")
    t0 = time.time()
    data = np.loadtxt(join_csv, dtype=np.int32)
    num_rows = len(data)
    ones = int(np.sum(data))
    print(f"[read] {num_rows} rows, {ones} qualifying ({100*ones/num_rows:.1f}%) in {time.time()-t0:.1f}s")

    join_dir = os.path.join(output_dir, "join_result")
    os.makedirs(join_dir, exist_ok=True)
    packed = np.packbits(data.astype(np.uint8), bitorder='little')
    bm_path = os.path.join(join_dir, "0.bm")
    packed.tofile(bm_path)
    print(f"[gen] join_result/0.bm: {os.path.getsize(bm_path)} bytes")

    # 3. Export orders metadata for qualifying orders
    meta_csv = os.path.join(output_dir, "orders_meta.csv")
    print(f"[export] Exporting qualifying orders metadata...")
    sql = f"""
COPY (
    SELECT o.o_orderkey, o.o_orderdate - DATE '1970-01-01' AS orderdate_epoch,
           o.o_shippriority
    FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey
    WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate < DATE '1995-03-15'
    ORDER BY o.o_orderkey
) TO '{meta_csv}' (HEADER false);
"""
    run_sql_to_csv(duckdb_bin, db_path, sql, meta_csv)

    # 4. Symlink shipdate from Q1 if available
    ship_dir = os.path.join(output_dir, "shipdate")
    q1_ship_dir = output_dir.replace("tpch_q3", "tpch_q1") + "/shipdate"
    if os.path.isdir(q1_ship_dir) and len(os.listdir(q1_ship_dir)) > 2000:
        print(f"[shipdate] Symlinking from Q1: {q1_ship_dir}")
        if os.path.exists(ship_dir):
            if os.path.islink(ship_dir):
                os.unlink(ship_dir)
            else:
                import shutil
                shutil.rmtree(ship_dir)
        os.symlink(os.path.abspath(q1_ship_dir), ship_dir)
    else:
        print(f"[shipdate] Q1 shipdate not found, please generate Q1 data first")
        sys.exit(1)

    # 5. Write metadata
    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"q3_join_qualifying={ones}\n")
        f.write(f"shipdate_cutoff=1169 (1995-03-15, days since 1992-01-01)\n")
        f.write(f"complement_or_count=1169 (days 1..1169, then NOT)\n")

    os.remove(join_csv)
    print(f"\n[done] Output: {output_dir}")
    print(f"  join_result: 1 bitmap ({ones} set bits)")
    print(f"  shipdate: symlinked from Q1")
    print(f"  orders_meta.csv: qualifying orders metadata")


if __name__ == '__main__':
    main()
