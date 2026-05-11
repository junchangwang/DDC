#!/usr/bin/env python3
# Export TPC-H Q4 bitmap data at SF10 from DuckDB.
#
# Q4 (TPC-H spec v3.0.1 §2.4.4):
#   SELECT o_orderpriority, count(*) AS order_count
#   FROM orders
#   WHERE o_orderdate >= DATE '1993-07-01'
#     AND o_orderdate <  DATE '1993-10-01'
#     AND EXISTS (SELECT * FROM lineitem
#                 WHERE l_orderkey = o_orderkey
#                   AND l_commitdate < l_receiptdate)
#   GROUP BY o_orderpriority
#   ORDER BY o_orderpriority;
#
# Bitmaps are aligned to the orders table (rowid order):
#
#   orderdate/D.bm       D in [547..638]  (1993-07-01 .. 1993-09-30
#                                          since 1992-01-01) — 92 days
#   orderpriority/P.bm   P in [1..5]      (1-URGENT, 2-HIGH, 3-MEDIUM,
#                                          4-NOT SPECIFIED, 5-LOW)
#   late_lineitem/0.bm   pre-encoded EXISTS predicate:
#                        bit[i] = 1 iff order rowid=i has at least one
#                        lineitem with l_commitdate < l_receiptdate.
#
# The pre-encoded EXISTS bitmap is consistent with Q3's `join_result/0.bm`
# pattern: the relational join is computed once at export time, and the
# bitmap operations measured at query time are the standard
# OR (date)  →  AND (EXISTS)  →  AND (priority)  →  popcount
# pipeline that all backends share.
#
# Usage:
#   python3 export_tpch_q4_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
DAY_START = 547   # 1993-07-01 since 1992-01-01
DAY_END   = 638   # 1993-09-30 since 1992-01-01 (inclusive)


def run_sql(duckdb_bin, db_path, sql):
    r = subprocess.run([duckdb_bin, db_path], input=sql,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(1)


def export_column_csv(duckdb_bin, db_path, sql, csv_path):
    """Run a SQL COPY ... TO csv and return the CSV path."""
    run_sql(duckdb_bin, db_path, sql)
    return csv_path


def read_uint8_csv(csv_path):
    """Fast 1-byte-per-row CSV reader.  Returns numpy uint8 array."""
    try:
        import pandas as pd
        return pd.read_csv(csv_path, header=None, dtype=np.uint8).iloc[:, 0].to_numpy()
    except ImportError:
        with open(csv_path, "rb") as fh:
            buf = fh.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        return (data[(data == ord("0")) | (data == ord("1"))] - ord("0"))


def write_bm(out_path, mask_uint8):
    """Pack a 0/1 uint8 array into a raw .bm file (LSB-first)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    packed = np.packbits(mask_uint8, bitorder="little")
    packed.tofile(out_path)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q4_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Orders row count.
    # ---------------------------------------------------------------
    print("[probe] orders row count …")
    cnt_csv = os.path.join(out_dir, "_q4_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM orders) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # ---------------------------------------------------------------
    # 2. orderdate/D.bm for D in [DAY_START..DAY_END].
    #
    #    Single SQL pass: stream o_orderdate for every order in rowid
    #    order; for each row emit the day-since-1992 as a small int.
    #    We then bucket into per-day boolean masks and packbits each.
    # ---------------------------------------------------------------
    date_csv = os.path.join(out_dir, "_q4_orderdate.csv")
    print("[export] orderdate (one int per order, rowid order) …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT (o_orderdate - DATE '1992-01-01')::INTEGER AS day
    FROM orders
    ORDER BY rowid
) TO '{date_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] orderdate column in {time.time()-t0:.1f} s")

    print("[read] loading orderdate column …")
    t0 = time.time()
    try:
        import pandas as pd
        days = pd.read_csv(date_csv, header=None, dtype=np.int32).iloc[:, 0].to_numpy()
    except ImportError:
        days = np.loadtxt(date_csv, dtype=np.int32)
    assert days.shape[0] == num_rows, (days.shape, num_rows)
    print(f"[read] {num_rows:,} orderdates in {time.time()-t0:.1f} s")
    os.remove(date_csv)

    print(f"[gen] orderdate/D.bm for D in [{DAY_START}..{DAY_END}] "
          f"({DAY_END - DAY_START + 1} days) …")
    t0 = time.time()
    for d in range(DAY_START, DAY_END + 1):
        mask = (days == d).astype(np.uint8)
        write_bm(os.path.join(out_dir, "orderdate", f"{d}.bm"), mask)
    print(f"[gen] {DAY_END - DAY_START + 1} orderdate bitmaps "
          f"in {time.time()-t0:.1f} s")

    # ---------------------------------------------------------------
    # 3. orderpriority/P.bm for P in [1..5].
    # ---------------------------------------------------------------
    print("[export] orderpriority (CASE → small int per order) …")
    t0 = time.time()
    prio_csv = os.path.join(out_dir, "_q4_priority.csv")
    sql = f"""
COPY (
    SELECT CASE o_orderpriority
        WHEN '1-URGENT'        THEN 1
        WHEN '2-HIGH'          THEN 2
        WHEN '3-MEDIUM'        THEN 3
        WHEN '4-NOT SPECIFIED' THEN 4
        WHEN '5-LOW'           THEN 5
    END AS p
    FROM orders
    ORDER BY rowid
) TO '{prio_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    try:
        import pandas as pd
        prio = pd.read_csv(prio_csv, header=None, dtype=np.int32).iloc[:, 0].to_numpy()
    except ImportError:
        prio = np.loadtxt(prio_csv, dtype=np.int32)
    assert prio.shape[0] == num_rows
    os.remove(prio_csv)
    print(f"[export] priority column in {time.time()-t0:.1f} s")

    print("[gen] orderpriority/P.bm for P in [1..5] …")
    for p in range(1, 6):
        mask = (prio == p).astype(np.uint8)
        write_bm(os.path.join(out_dir, "orderpriority", f"{p}.bm"), mask)
        print(f"[gen]   {p} ({PRIORITIES[p-1]}): {int(mask.sum()):,} orders")

    # ---------------------------------------------------------------
    # 4. late_lineitem/0.bm — pre-encoded EXISTS predicate.
    #
    #    For each order (in rowid order), bit=1 iff at least one lineitem
    #    has l_orderkey = o_orderkey AND l_commitdate < l_receiptdate.
    #    Done as a single SQL pass using a SEMI-JOIN; orders are read in
    #    rowid order so the boolean stream lines up with the other
    #    orders/* bitmaps.
    # ---------------------------------------------------------------
    print("[export] late_lineitem (EXISTS pre-join, rowid order) …")
    t0 = time.time()
    late_csv = os.path.join(out_dir, "_q4_late.csv")
    sql = f"""
COPY (
    SELECT CASE WHEN o.o_orderkey IN (
        SELECT DISTINCT l_orderkey FROM lineitem
        WHERE l_commitdate < l_receiptdate
    ) THEN 1 ELSE 0 END AS late
    FROM orders o
    ORDER BY o.rowid
) TO '{late_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] late_lineitem column in {time.time()-t0:.1f} s")

    late = read_uint8_csv(late_csv)
    assert late.shape[0] == num_rows
    os.remove(late_csv)

    write_bm(os.path.join(out_dir, "late_lineitem", "0.bm"), late)
    print(f"[gen] late_lineitem: {int(late.sum()):,} / {num_rows:,} "
          f"({100*int(late.sum())/num_rows:.2f}%)")

    # ---------------------------------------------------------------
    # 5. metadata.txt.
    # ---------------------------------------------------------------
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"orderdate_day_start={DAY_START} (1993-07-01, days since 1992-01-01)\n")
        f.write(f"orderdate_day_end={DAY_END} (1993-09-30, days since 1992-01-01)\n")
        f.write(f"orderdate_bitmaps={DAY_END - DAY_START + 1}\n")
        f.write("orderpriority_bitmaps=5 (1-URGENT, 2-HIGH, 3-MEDIUM, 4-NOT SPECIFIED, 5-LOW)\n")
        f.write(f"late_lineitem_count={int(late.sum())}\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
