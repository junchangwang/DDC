#!/usr/bin/env python3
# Export TPC-H Q12 bitmap data at SF10 from DuckDB.
#
# Q12 (TPC-H spec v3.0.1 §2.4.12) — Shipping Modes and Order Priority:
#   SELECT l_shipmode,
#          SUM(CASE WHEN o_orderpriority = '1-URGENT'
#                     OR o_orderpriority = '2-HIGH'
#                   THEN 1 ELSE 0 END) AS high_line_count,
#          SUM(CASE WHEN o_orderpriority <> '1-URGENT'
#                    AND o_orderpriority <> '2-HIGH'
#                   THEN 1 ELSE 0 END) AS low_line_count
#   FROM   orders, lineitem
#   WHERE  o_orderkey = l_orderkey
#     AND  l_shipmode IN ('MAIL', 'SHIP')
#     AND  l_commitdate < l_receiptdate
#     AND  l_shipdate   < l_commitdate
#     AND  l_receiptdate >= DATE '1994-01-01'
#     AND  l_receiptdate <  DATE '1995-01-01'
#   GROUP BY l_shipmode
#   ORDER BY l_shipmode;
#
# Bitmaps are aligned to lineitem.rowid:
#
#   receiptdate/D.bm    D in [731..1095] (1994-01-01..1994-12-31,
#                                         days since 1992-01-01) — 365 days
#   shipmode/MAIL.bm    l_shipmode = 'MAIL'
#   shipmode/SHIP.bm    l_shipmode = 'SHIP'
#   commit_lt_receipt/0.bm  l_commitdate < l_receiptdate
#   ship_lt_commit/0.bm     l_shipdate   < l_commitdate
#   priority_high/0.bm      orders.o_orderpriority IN ('1-URGENT', '2-HIGH')
#   priority_low/0.bm       NOT priority_high
#
# Pipeline (per backend, computed at query time):
#   1. OR_many over 365 receiptdate days  → date_mask
#   2. AND with commit_lt_receipt          → c_mask
#   3. AND with ship_lt_commit             → cs_mask
#   4. For each shipmode m in {MAIL, SHIP}:
#        sm_filt = cs_mask AND shipmode[m]
#        high[m] = popcount(sm_filt AND priority_high)
#        low[m]  = popcount(sm_filt AND priority_low)
#
# Usage:
#   python3 export_tpch_q12_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


DAY_START = 731    # 1994-01-01 (days since 1992-01-01)
DAY_END   = 1095   # 1994-12-31 (inclusive; 365 days)


def run_sql(duckdb_bin, db_path, sql):
    r = subprocess.run([duckdb_bin, db_path], input=sql,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(1)


def write_bm(out_path, mask_uint8):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    packed = np.packbits(mask_uint8, bitorder="little")
    packed.tofile(out_path)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q12_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. lineitem row count ---
    print("[probe] lineitem row count …")
    cnt_csv = os.path.join(out_dir, "_q12_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM lineitem) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # --- 2. Single-pass join SQL — emit per-lineitem fields needed
    #        for every bitmap.  All joins are inner; lineitem.rowid order
    #        is preserved with `ORDER BY l.rowid`.
    rows_csv = os.path.join(out_dir, "_q12_rows.csv")
    print("[export] full-join SQL → CSV …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT
        (l_receiptdate - DATE '1992-01-01')::INTEGER AS receipt_day,
        CAST((l.l_shipmode = 'MAIL') AS INTEGER) AS sm_mail,
        CAST((l.l_shipmode = 'SHIP') AS INTEGER) AS sm_ship,
        CAST((l_commitdate < l_receiptdate) AS INTEGER) AS c_lt_r,
        CAST((l_shipdate   < l_commitdate)  AS INTEGER) AS s_lt_c,
        CAST((o_orderpriority = '1-URGENT'
           OR o_orderpriority = '2-HIGH')   AS INTEGER) AS prio_high
    FROM lineitem l
        JOIN orders o ON l.l_orderkey = o.o_orderkey
    ORDER BY l.rowid
) TO '{rows_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] CSV in {time.time()-t0:.1f} s")

    # --- 3. Read CSV ---
    print("[read] loading per-lineitem fields …")
    t0 = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(rows_csv, header=None,
                         names=["receipt_day", "sm_mail", "sm_ship",
                                "c_lt_r", "s_lt_c", "prio_high"],
                         dtype={"receipt_day": np.int32,
                                "sm_mail": np.uint8,
                                "sm_ship": np.uint8,
                                "c_lt_r":  np.uint8,
                                "s_lt_c":  np.uint8,
                                "prio_high": np.uint8})
        receipt_day = df["receipt_day"].to_numpy()
        sm_mail     = df["sm_mail"].to_numpy()
        sm_ship     = df["sm_ship"].to_numpy()
        c_lt_r      = df["c_lt_r"].to_numpy()
        s_lt_c      = df["s_lt_c"].to_numpy()
        prio_high   = df["prio_high"].to_numpy()
    except ImportError:
        arr = np.loadtxt(rows_csv, delimiter=",", dtype=np.int32)
        receipt_day = arr[:, 0].astype(np.int32)
        sm_mail     = arr[:, 1].astype(np.uint8)
        sm_ship     = arr[:, 2].astype(np.uint8)
        c_lt_r      = arr[:, 3].astype(np.uint8)
        s_lt_c      = arr[:, 4].astype(np.uint8)
        prio_high   = arr[:, 5].astype(np.uint8)

    assert receipt_day.shape[0] == num_rows
    print(f"[read] {num_rows:,} lineitem fields in {time.time()-t0:.1f} s")
    os.remove(rows_csv)

    # --- 4. receiptdate/D.bm ---
    print(f"[gen] receiptdate/D.bm for D in [{DAY_START}..{DAY_END}] "
          f"({DAY_END - DAY_START + 1} days) …")
    t0 = time.time()
    n_in_window = 0
    for d in range(DAY_START, DAY_END + 1):
        mask = (receipt_day == d).astype(np.uint8)
        n_in_window += int(mask.sum())
        write_bm(os.path.join(out_dir, "receiptdate", f"{d}.bm"), mask)
    print(f"[gen] {DAY_END - DAY_START + 1} day bitmaps in {time.time()-t0:.1f} s "
          f"({n_in_window:,} rows in 1994 window)")

    # --- 5. shipmode/{MAIL,SHIP}.bm ---
    n_mail = int(sm_mail.sum())
    n_ship = int(sm_ship.sum())
    write_bm(os.path.join(out_dir, "shipmode", "MAIL.bm"), sm_mail)
    write_bm(os.path.join(out_dir, "shipmode", "SHIP.bm"), sm_ship)
    print(f"[gen] shipmode: MAIL={n_mail:,}  SHIP={n_ship:,}")

    # --- 6. commit_lt_receipt / ship_lt_commit ---
    n_c = int(c_lt_r.sum())
    n_s = int(s_lt_c.sum())
    write_bm(os.path.join(out_dir, "commit_lt_receipt", "0.bm"), c_lt_r)
    write_bm(os.path.join(out_dir, "ship_lt_commit",    "0.bm"), s_lt_c)
    print(f"[gen] commit_lt_receipt: {n_c:,}  ship_lt_commit: {n_s:,}")

    # --- 7. priority_high / priority_low ---
    prio_low = (1 - prio_high).astype(np.uint8)
    n_h = int(prio_high.sum())
    n_l = int(prio_low.sum())
    write_bm(os.path.join(out_dir, "priority_high", "0.bm"), prio_high)
    write_bm(os.path.join(out_dir, "priority_low",  "0.bm"), prio_low)
    print(f"[gen] priority_high: {n_h:,}  priority_low: {n_l:,} "
          f"(should sum to num_rows: {n_h + n_l == num_rows})")

    # --- 8. metadata.txt ---
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"receiptdate_day_start={DAY_START} (1994-01-01, days since 1992-01-01)\n")
        f.write(f"receiptdate_day_end={DAY_END} (1994-12-31)\n")
        f.write(f"receiptdate_bitmaps={DAY_END - DAY_START + 1}\n")
        f.write(f"shipmode_count_MAIL={n_mail}\n")
        f.write(f"shipmode_count_SHIP={n_ship}\n")
        f.write(f"commit_lt_receipt_count={n_c}\n")
        f.write(f"ship_lt_commit_count={n_s}\n")
        f.write(f"priority_high_count={n_h}\n")
        f.write(f"priority_low_count={n_l}\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
