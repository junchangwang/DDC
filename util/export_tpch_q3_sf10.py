#!/usr/bin/env python3
# Export TPC-H Q3 bitmap data at SF10 from DuckDB.
#
# Same as export_tpch_q3.py, except:
#   - Does NOT require raw Q1 shipdate dir (SF10 only ships compressed
#     Q1 in this checkout; shipdate symlink is set up at the compressed
#     level by compress_tpch_q3.sh).
#   - Uses a faster numpy reader (np.fromfile of a single-byte-per-row
#     binary blob) to keep the 60 M-row CSV import under a minute.
#
# Usage:
#   python3 export_tpch_q3_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


def run_sql(duckdb_bin, db_path, sql):
    r = subprocess.run([duckdb_bin, db_path], input=sql,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q3_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Q3 join column (1 byte per lineitem row).  Output as a CSV with
    #    a single integer column; np.loadtxt would be too slow at SF10
    #    (60 M rows), so we route through a binary export instead by
    #    asking DuckDB to write 'true'/'false' and parsing the raw bytes.
    join_csv = os.path.join(out_dir, "_q3_join.csv")
    print(f"[export] computing join_result …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT CASE WHEN l.l_orderkey IN (
        SELECT o.o_orderkey FROM orders o
        JOIN customer c ON o.o_custkey = c.c_custkey
        WHERE c.c_mktsegment = 'BUILDING'
          AND o.o_orderdate < DATE '1995-03-15'
    ) THEN 1 ELSE 0 END AS in_join
    FROM lineitem l
    ORDER BY l.rowid
) TO '{join_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] done in {time.time()-t0:.1f} s")

    # 2. Read the CSV and pack — pandas avoids np.loadtxt's per-line cost.
    print(f"[read] loading join column …")
    t0 = time.time()
    try:
        import pandas as pd
        data = pd.read_csv(join_csv, header=None, dtype=np.uint8).iloc[:, 0].to_numpy()
    except ImportError:
        # Fallback: np.fromstring of the whole file (one byte per char +
        # newline ≈ 2 bytes/row).  Still O(N) but slower than pandas.
        with open(join_csv, "rb") as fh:
            buf = fh.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data[(data == ord('0')) | (data == ord('1'))] - ord('0')

    n = data.shape[0]
    ones = int(data.sum())
    print(f"[read] {n:,} rows, {ones:,} qualifying ({100*ones/n:.2f}%) in {time.time()-t0:.1f} s")

    join_dir = os.path.join(out_dir, "join_result")
    os.makedirs(join_dir, exist_ok=True)
    packed = np.packbits(data, bitorder="little")
    bm_path = os.path.join(join_dir, "0.bm")
    packed.tofile(bm_path)
    print(f"[gen] {bm_path}: {os.path.getsize(bm_path):,} bytes")

    # 3. orders_meta.csv (small — qualifying orders only).
    meta_csv = os.path.join(out_dir, "orders_meta.csv")
    print(f"[export] orders_meta …")
    sql = f"""
COPY (
    SELECT o.o_orderkey,
           o.o_orderdate - DATE '1970-01-01' AS orderdate_epoch,
           o.o_shippriority
    FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey
    WHERE c.c_mktsegment = 'BUILDING' AND o.o_orderdate < DATE '1995-03-15'
    ORDER BY o.o_orderkey
) TO '{meta_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] {meta_csv}: {os.path.getsize(meta_csv):,} bytes")

    # 4. metadata.txt (no shipdate symlink at SF10 — see header comment).
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={n}\n")
        f.write(f"q3_join_qualifying={ones}\n")
        f.write(f"shipdate_cutoff=1169 (1995-03-15, days since 1992-01-01)\n")
        f.write(f"complement_or_count=1169 (days 1..1169, then NOT)\n")
        f.write("note=raw shipdate not exported at SF10; compress step "
                "symlinks shipdate from tpch_q1_<fmt>/shipdate.\n")

    os.remove(join_csv)
    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
