#!/usr/bin/env python3
# Export TPC-H Q17 bitmap data at SF10 from DuckDB.
#
# Q17 (TPC-H spec v3.0.1 §2.4.17) — Small-Quantity-Order Revenue Query:
#
#   SELECT sum(l_extendedprice) / 7.0 AS avg_yearly
#     FROM lineitem, part
#    WHERE p_partkey = l_partkey
#      AND p_brand = '[BRAND]'
#      AND p_container = '[CONTAINER]'
#      AND l_quantity < (
#            SELECT 0.2 * avg(l_quantity)
#              FROM lineitem
#             WHERE l_partkey = p_partkey
#          );
#
#   Validation: BRAND='Brand#23', CONTAINER='MED BOX' (spec §2.4.17.4).
#
# Bitmaps are aligned to lineitem.rowid:
#
#   is_q17_part/0.bm    l_partkey -> part.p_brand='Brand#23'
#                                AND part.p_container='MED BOX'
#                       (1 bitmap, ~0.1% density at SF10)
#
# l_partkey, l_quantity and l_extendedprice are read directly from
# DuckDB storage at query time (mirrors Q1/Q14/Q15's pre-load pattern),
# so no further column .bin files are exported.
#
# Pipeline (per backend, computed at query time):
#   1. Pass1: walk is_q17_part rows → accumulate
#               (sum_qty[partkey], count[partkey])
#             then threshold[pk] = 0.2 * sum_qty[pk] / count[pk]
#   2. Pass2: walk is_q17_part rows → if l_quantity[r] < threshold[partkey[r]]
#               sum_ep += l_extendedprice[r]
#   3. avg_yearly = sum_ep / 7.0
#
# Usage:
#   python3 export_tpch_q17_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


BRAND     = "Brand#23"
CONTAINER = "MED BOX"


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
        print("Usage: export_tpch_q17_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. lineitem row count ---
    print("[probe] lineitem row count …")
    cnt_csv = os.path.join(out_dir, "_q17_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM lineitem) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # --- 2. lineitem JOIN part — emit is_q17_part per lineitem row,
    #        ordered by lineitem.rowid so the resulting bitmap is
    #        aligned to lineitem's natural row positions.
    rows_csv = os.path.join(out_dir, "_q17_rows.csv")
    print("[export] lineitem JOIN part SQL → CSV …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT
        CAST((p.p_brand     = '{BRAND}'
          AND p.p_container = '{CONTAINER}') AS INTEGER) AS is_q17_part
    FROM lineitem l
        JOIN part p ON l.l_partkey = p.p_partkey
    ORDER BY l.rowid
) TO '{rows_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] CSV in {time.time()-t0:.1f} s")

    # --- 3. Read CSV ---
    print("[read] loading is_q17_part column …")
    t0 = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(rows_csv, header=None,
                         names=["is_q17_part"],
                         dtype={"is_q17_part": np.uint8})
        is_q17_part = df["is_q17_part"].to_numpy()
    except ImportError:
        is_q17_part = np.loadtxt(rows_csv, dtype=np.uint8)

    assert is_q17_part.shape[0] == num_rows, \
        f"row mismatch: csv={is_q17_part.shape[0]} db={num_rows}"
    print(f"[read] {num_rows:,} lineitem rows in {time.time()-t0:.1f} s")
    os.remove(rows_csv)

    # --- 4. is_q17_part/0.bm ---
    n_q17 = int(is_q17_part.sum())
    write_bm(os.path.join(out_dir, "is_q17_part", "0.bm"), is_q17_part)
    print(f"[gen] is_q17_part: {n_q17:,} / {num_rows:,} "
          f"({100.0 * n_q17 / num_rows:.4f}%)")

    # --- 5. metadata.txt ---
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"is_q17_part_count={n_q17}\n")
        f.write(f"brand={BRAND}\n")
        f.write(f"container={CONTAINER}\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    # --- 6. done.txt for Q17.cpp num_rows lookup ---
    with open(os.path.join(out_dir, "done.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
