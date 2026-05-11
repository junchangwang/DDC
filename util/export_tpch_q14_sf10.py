#!/usr/bin/env python3
# Export TPC-H Q14 bitmap data at SF10 from DuckDB.
#
# Q14 (TPC-H spec v3.0.1 §2.4.14) — Promotion Effect Query:
#   SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%'
#                              THEN l_extendedprice * (1 - l_discount)
#                              ELSE 0 END)
#                / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
#   FROM   lineitem, part
#   WHERE  l_partkey  = p_partkey
#     AND  l_shipdate >= DATE '[DATE]'
#     AND  l_shipdate <  DATE '[DATE]' + INTERVAL '1' MONTH;
#
#   Qualification DATE = '1995-09-01' (spec §2.4.14.4).
#
# Bitmaps are aligned to lineitem.rowid:
#
#   is_promo/0.bm   l_partkey -> part.p_type LIKE 'PROMO%'  (1 bitmap)
#
# Q14 also needs lineitem.l_shipdate equality bitmaps for days
# [1339..1368] (= [1995-09-01, 1995-10-01)).  Those bitmaps are
# already produced by export_tpch_q6.py and shipped under
# `tpch_q6_<fmt>/shipdate/{day}.bm`, so this script does NOT
# regenerate them — the Q14 benchmark loads them directly from the
# Q6 shipdate directory.  Likewise the post-export compression
# `bash util/compress_tpch_q14.sh` symlinks Q6's compressed
# shipdate folders into Q14's compressed dirs.
#
# Pipeline (per backend, computed at query time):
#   1. ship_filter = OR_many(shipdate[1339..1369))    (30 day bitmaps)
#   2. promo_set   = ship_filter & is_promo
#   3. iterate ship_filter rows: total_rev += price * (10000 - disc)
#      iterate promo_set    rows: promo_rev += price * (10000 - disc)
#   4. promo_revenue = 100.0 * promo_rev / total_rev
#
# Usage:
#   python3 export_tpch_q14_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


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
        print("Usage: export_tpch_q14_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. lineitem row count ---
    print("[probe] lineitem row count …")
    cnt_csv = os.path.join(out_dir, "_q14_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM lineitem) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # --- 2. lineitem JOIN part — emit is_promo per lineitem row,
    #        ordered by lineitem.rowid so the resulting bitmap is
    #        aligned to lineitem's natural row positions.
    rows_csv = os.path.join(out_dir, "_q14_rows.csv")
    print("[export] lineitem JOIN part SQL → CSV …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT
        CAST((p.p_type LIKE 'PROMO%') AS INTEGER) AS is_promo
    FROM lineitem l
        JOIN part p ON l.l_partkey = p.p_partkey
    ORDER BY l.rowid
) TO '{rows_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] CSV in {time.time()-t0:.1f} s")

    # --- 3. Read CSV ---
    print("[read] loading is_promo column …")
    t0 = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(rows_csv, header=None,
                         names=["is_promo"],
                         dtype={"is_promo": np.uint8})
        is_promo = df["is_promo"].to_numpy()
    except ImportError:
        is_promo = np.loadtxt(rows_csv, dtype=np.uint8)

    assert is_promo.shape[0] == num_rows, \
        f"row mismatch: csv={is_promo.shape[0]} db={num_rows}"
    print(f"[read] {num_rows:,} lineitem rows in {time.time()-t0:.1f} s")
    os.remove(rows_csv)

    # --- 4. is_promo/0.bm ---
    n_promo = int(is_promo.sum())
    write_bm(os.path.join(out_dir, "is_promo", "0.bm"), is_promo)
    print(f"[gen] is_promo: {n_promo:,} / {num_rows:,} "
          f"({100.0 * n_promo / num_rows:.2f}%)")

    # --- 5. metadata.txt ---
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"is_promo_count={n_promo}\n")
        f.write(f"shipdate_range=[1339..1368] (days since 1992-01-01) "
                f"= [1995-09-01, 1995-10-01)\n")
        f.write(f"shipdate_source=tpch_q6_<fmt>/shipdate/  (reused via symlink)\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
