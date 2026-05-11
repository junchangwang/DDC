#!/usr/bin/env python3
# Export TPC-H Q8 bitmap data at SF10 from DuckDB.
#
# Q8 (TPC-H spec v3.0.1 §2.4.8) — National Market Share Query:
#   SELECT
#       o_year,
#       SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume)
#   FROM (
#     SELECT EXTRACT(YEAR FROM o_orderdate) AS o_year,
#            l_extendedprice * (1 - l_discount) AS volume,
#            n2.n_name AS nation
#     FROM part, supplier, lineitem, orders, customer, nation n1, region, nation n2
#     WHERE p_partkey = l_partkey
#       AND s_suppkey = l_suppkey
#       AND l_orderkey = o_orderkey
#       AND o_custkey = c_custkey
#       AND c_nationkey = n1.n_nationkey
#       AND n1.n_regionkey = r_regionkey
#       AND r_name = 'AMERICA'
#       AND s_nationkey = n2.n_nationkey
#       AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
#       AND p_type = 'ECONOMY ANODIZED STEEL'
#   ) AS all_nations
#   GROUP BY o_year
#   ORDER BY o_year;
#
# Bitmaps are aligned to the lineitem table (rowid order) so each bitmap
# is a per-lineitem mask:
#
#   orderdate/D.bm       D in [1095..1825]  (1995-01-01..1996-12-31,
#                                            days since 1992-01-01) — 731 days
#   year/Y.bm            Y in {1995, 1996}  — orderdate falls in that year
#   join_result/0.bm     pre-encoded:
#                          (p_type = 'ECONOMY ANODIZED STEEL')
#                        AND (customer's nation is in AMERICA region)
#   brazil/0.bm          supplier's s_nationkey = BRAZIL nation_key
#
# Volume column (l_extendedprice * (1 - l_discount)) is loaded at runtime
# directly from DuckDB by Q8.cpp — same pattern as Q5/Q10.
#
# Pipeline (per backend, computed at query time):
#   1. OR_many over 731 day bitmaps           → date_mask
#   2. AND with join_result                    → joined_filt
#   3. For each year y in {1995, 1996}:
#        year_filt   = joined_filt AND year_y
#        denom_y     = SUM(volume[r] for r in year_filt)
#        year_brazil = year_filt AND brazil
#        numer_y     = SUM(volume[r] for r in year_brazil)
#        mkt_share_y = numer_y / denom_y
#
# Usage:
#   python3 export_tpch_q8_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


# Date constants (days since 1992-01-01).  TPC-H dbgen uses calendar
# date arithmetic with leap-year corrections, so 1992 contributes 366
# days, then 1993/1994/1995 each contribute 365.  Hence 1995-01-01 is
# day 1096 (not 1095) and 1996-12-31 is day 1826.
DAY_START = 1096   # 1995-01-01
DAY_END   = 1826   # 1996-12-31  (inclusive; 731 days)
YEARS     = [1995, 1996]


def run_sql(duckdb_bin, db_path, sql):
    r = subprocess.run([duckdb_bin, db_path], input=sql,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(1)


def write_bm(out_path, mask_uint8):
    """Pack a 0/1 uint8 array into a raw .bm file (LSB-first)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    packed = np.packbits(mask_uint8, bitorder="little")
    packed.tofile(out_path)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q8_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. lineitem row count.  Used by metadata.txt and to size masks.
    # ---------------------------------------------------------------
    print("[probe] lineitem row count …")
    cnt_csv = os.path.join(out_dir, "_q8_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM lineitem) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # ---------------------------------------------------------------
    # 2. Single full-join SQL pass: emit one row per lineitem with the
    #    five fields we need to build all bitmaps:
    #      day         INT32  — orderdate days since 1992-01-01
    #      yr          INT16  — EXTRACT(YEAR FROM o_orderdate)
    #      in_america  0/1    — 1 iff customer's region is AMERICA
    #      part_match  0/1    — 1 iff p_type = 'ECONOMY ANODIZED STEEL'
    #      brazil_supp 0/1    — 1 iff supplier.nation = 'BRAZIL'
    #
    #    All joins are inner — every lineitem row's FKs (orderkey,
    #    custkey, suppkey, partkey, nationkey, regionkey) must resolve
    #    in TPC-H, so the row count of the output equals num_rows.
    #
    #    `ORDER BY l.rowid` is critical — bitmaps must be aligned to the
    #    same rowid order DuckDB uses when scanning lineitem at query
    #    time, otherwise per-row masks are meaningless.
    # ---------------------------------------------------------------
    rows_csv = os.path.join(out_dir, "_q8_rows.csv")
    print("[export] full-join SQL → CSV (one-shot, all lineitem rows preserved) …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT
        (o_orderdate - DATE '1992-01-01')::INTEGER AS day,
        EXTRACT(YEAR FROM o_orderdate)::INTEGER AS yr,
        CAST((r.r_name = 'AMERICA') AS INTEGER) AS in_america,
        CAST((p.p_type = 'ECONOMY ANODIZED STEEL') AS INTEGER) AS part_match,
        CAST((n2.n_name = 'BRAZIL') AS INTEGER) AS brazil_supp
    FROM lineitem l
        JOIN orders   o  ON l.l_orderkey = o.o_orderkey
        JOIN customer c  ON o.o_custkey  = c.c_custkey
        JOIN nation   n1 ON c.c_nationkey = n1.n_nationkey
        JOIN region   r  ON n1.n_regionkey = r.r_regionkey
        JOIN supplier s  ON l.l_suppkey   = s.s_suppkey
        JOIN nation   n2 ON s.s_nationkey = n2.n_nationkey
        JOIN part     p  ON l.l_partkey   = p.p_partkey
    ORDER BY l.rowid
) TO '{rows_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] full-row CSV in {time.time()-t0:.1f} s")

    # ---------------------------------------------------------------
    # 3. Read CSV with pandas (fast).
    # ---------------------------------------------------------------
    print("[read] loading per-lineitem fields …")
    t0 = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(rows_csv, header=None,
                         names=["day", "yr", "in_america", "part_match", "brazil_supp"],
                         dtype={"day": np.int32,
                                "yr":  np.int16,
                                "in_america":  np.uint8,
                                "part_match":  np.uint8,
                                "brazil_supp": np.uint8})
        days        = df["day"].to_numpy()
        years       = df["yr"].to_numpy()
        in_america  = df["in_america"].to_numpy()
        part_match  = df["part_match"].to_numpy()
        brazil_supp = df["brazil_supp"].to_numpy()
    except ImportError:
        # Fallback without pandas (slower).
        arr = np.loadtxt(rows_csv, delimiter=",", dtype=np.int32)
        days        = arr[:, 0].astype(np.int32)
        years       = arr[:, 1].astype(np.int16)
        in_america  = arr[:, 2].astype(np.uint8)
        part_match  = arr[:, 3].astype(np.uint8)
        brazil_supp = arr[:, 4].astype(np.uint8)

    assert days.shape[0] == num_rows, (days.shape, num_rows)
    print(f"[read] {num_rows:,} lineitem fields in {time.time()-t0:.1f} s")
    os.remove(rows_csv)

    # ---------------------------------------------------------------
    # 4. orderdate/D.bm for D in [DAY_START..DAY_END].
    # ---------------------------------------------------------------
    print(f"[gen] orderdate/D.bm for D in [{DAY_START}..{DAY_END}] "
          f"({DAY_END - DAY_START + 1} days) …")
    t0 = time.time()
    n_set_total = 0
    for d in range(DAY_START, DAY_END + 1):
        mask = (days == d).astype(np.uint8)
        n_set = int(mask.sum())
        n_set_total += n_set
        write_bm(os.path.join(out_dir, "orderdate", f"{d}.bm"), mask)
    print(f"[gen] {DAY_END - DAY_START + 1} day bitmaps in {time.time()-t0:.1f} s "
          f"(total set bits across all days = {n_set_total:,})")

    # ---------------------------------------------------------------
    # 5. year/Y.bm for Y in {1995, 1996}.
    # ---------------------------------------------------------------
    print("[gen] year/Y.bm for Y in {1995, 1996} …")
    for y in YEARS:
        mask = (years == y).astype(np.uint8)
        n_set = int(mask.sum())
        write_bm(os.path.join(out_dir, "year", f"{y}.bm"), mask)
        print(f"[gen]   {y}: {n_set:,} lineitem rows")

    # ---------------------------------------------------------------
    # 6. join_result/0.bm  =  in_america AND part_match.
    # ---------------------------------------------------------------
    join_result = (in_america & part_match).astype(np.uint8)
    n_join = int(join_result.sum())
    write_bm(os.path.join(out_dir, "join_result", "0.bm"), join_result)
    print(f"[gen] join_result: {n_join:,} / {num_rows:,} lineitem rows "
          f"({100*n_join/num_rows:.3f}%)")

    # ---------------------------------------------------------------
    # 7. brazil/0.bm  =  brazil_supp.
    # ---------------------------------------------------------------
    n_brazil = int(brazil_supp.sum())
    write_bm(os.path.join(out_dir, "brazil", "0.bm"), brazil_supp)
    print(f"[gen] brazil: {n_brazil:,} / {num_rows:,} lineitem rows "
          f"({100*n_brazil/num_rows:.3f}%)")

    # ---------------------------------------------------------------
    # 8. metadata.txt.
    # ---------------------------------------------------------------
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"orderdate_day_start={DAY_START} (1995-01-01, days since 1992-01-01)\n")
        f.write(f"orderdate_day_end={DAY_END} (1996-12-31, days since 1992-01-01)\n")
        f.write(f"orderdate_bitmaps={DAY_END - DAY_START + 1}\n")
        f.write("year_bitmaps=2 (1995, 1996)\n")
        f.write(f"join_result_count={n_join}\n")
        f.write(f"brazil_count={n_brazil}\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
