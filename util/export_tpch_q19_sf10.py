#!/usr/bin/env python3
# Export TPC-H Q19 bitmap data at SF10 from DuckDB.
#
# Q19 (TPC-H spec v3.0.1 §2.4.19) — Discounted Revenue Query:
#
#   SELECT sum(l_extendedprice * (1 - l_discount)) AS revenue
#     FROM lineitem, part
#    WHERE
#      ( p_partkey = l_partkey
#        AND p_brand = '[BRAND1]'
#        AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
#        AND l_quantity >= [QUANTITY1]
#        AND l_quantity <= [QUANTITY1] + 10
#        AND p_size BETWEEN 1 AND 5
#        AND l_shipmode IN ('AIR','AIR REG')
#        AND l_shipinstruct = 'DELIVER IN PERSON' )
#      OR ( ... [BRAND2], MED *, [Q2..Q2+10], size 1..10 ... )
#      OR ( ... [BRAND3], LG *,  [Q3..Q3+10], size 1..15 ... );
#
#   Validation: BRAND1=Brand#12, BRAND2=Brand#23, BRAND3=Brand#34,
#               QUANTITY1=1, QUANTITY2=10, QUANTITY3=20.
#
# Bitmaps generated (all aligned to lineitem.rowid):
#
#   branch1_part/0.bm    l_partkey -> p with BRAND1 + SM container + size 1..5
#   branch2_part/0.bm    l_partkey -> p with BRAND2 + MED container + size 1..10
#   branch3_part/0.bm    l_partkey -> p with BRAND3 + LG container + size 1..15
#   shipmode_air/0.bm    l_shipmode IN ('AIR', 'AIR REG')
#   shipinstruct_dip/0.bm  l_shipinstruct = 'DELIVER IN PERSON'
#
# l_quantity equality bitmaps for the three ranges
#   [1..11], [10..20], [20..30] are reused via symlink from Q6's
#   quantity/{1..50} export — Q6 already exports the full TPC-H
#   l_quantity domain (1..50) which strictly covers Q19's needs.
#
# l_extendedprice / l_discount are read directly from DuckDB storage
# at query time (mirrors Q1/Q14/Q15/Q17 pattern) — no extra .bin file.
#
# Pipeline (per backend, computed at query time):
#   1. q1_or = OR(quantity[1..11])     (11 bitmaps)
#      q2_or = OR(quantity[10..20])    (11 bitmaps)
#      q3_or = OR(quantity[20..30])    (11 bitmaps)
#   2. b1 = branch1_part AND q1_or
#      b2 = branch2_part AND q2_or
#      b3 = branch3_part AND q3_or
#   3. branches = b1 OR b2 OR b3
#   4. filter   = branches AND shipmode_air AND shipinstruct_dip
#   5. walk filter rows → revenue += pp[r] * (100 - dp[r])
#
# Usage:
#   python3 export_tpch_q19_sf10.py <duckdb_binary> <db_path> <output_dir>
import sys, os, subprocess, time
import numpy as np


BRAND1 = "Brand#12"
BRAND2 = "Brand#23"
BRAND3 = "Brand#34"
SM_CONTAINERS  = ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
MED_CONTAINERS = ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
LG_CONTAINERS  = ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')


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


def in_list_sql(values):
    """Return SQL IN-list `('a','b',...)` for a list of strings."""
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + ")"


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q19_sf10.py <duckdb_bin> <db_path> <out_dir>")
        sys.exit(1)
    duckdb_bin, db_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. lineitem row count ---
    print("[probe] lineitem row count …")
    cnt_csv = os.path.join(out_dir, "_q19_count.csv")
    run_sql(duckdb_bin, db_path,
            f"COPY (SELECT count(*) FROM lineitem) TO '{cnt_csv}' (HEADER false);")
    with open(cnt_csv) as fh:
        num_rows = int(fh.read().strip())
    os.remove(cnt_csv)
    print(f"[probe] num_rows = {num_rows:,}")

    # --- 2. Single-pass JOIN — emit per-lineitem flags for all 5 own bitmaps
    rows_csv = os.path.join(out_dir, "_q19_rows.csv")
    print("[export] lineitem JOIN part SQL → CSV …")
    t0 = time.time()
    sql = f"""
COPY (
    SELECT
        CAST((p.p_brand = '{BRAND1}'
          AND p.p_container IN {in_list_sql(SM_CONTAINERS)}
          AND p.p_size BETWEEN 1 AND 5)        AS INTEGER) AS b1,
        CAST((p.p_brand = '{BRAND2}'
          AND p.p_container IN {in_list_sql(MED_CONTAINERS)}
          AND p.p_size BETWEEN 1 AND 10)       AS INTEGER) AS b2,
        CAST((p.p_brand = '{BRAND3}'
          AND p.p_container IN {in_list_sql(LG_CONTAINERS)}
          AND p.p_size BETWEEN 1 AND 15)       AS INTEGER) AS b3,
        CAST((l.l_shipmode IN ('AIR','AIR REG'))           AS INTEGER) AS sm_air,
        CAST((l.l_shipinstruct = 'DELIVER IN PERSON')      AS INTEGER) AS si_dip
    FROM lineitem l
        JOIN part p ON l.l_partkey = p.p_partkey
    ORDER BY l.rowid
) TO '{rows_csv}' (HEADER false);
"""
    run_sql(duckdb_bin, db_path, sql)
    print(f"[export] CSV in {time.time()-t0:.1f} s")

    # --- 3. Read CSV ---
    print("[read] loading per-lineitem flags …")
    t0 = time.time()
    try:
        import pandas as pd
        df = pd.read_csv(rows_csv, header=None,
                         names=["b1", "b2", "b3", "sm_air", "si_dip"],
                         dtype={"b1": np.uint8, "b2": np.uint8, "b3": np.uint8,
                                "sm_air": np.uint8, "si_dip": np.uint8})
        b1     = df["b1"].to_numpy()
        b2     = df["b2"].to_numpy()
        b3     = df["b3"].to_numpy()
        sm_air = df["sm_air"].to_numpy()
        si_dip = df["si_dip"].to_numpy()
    except ImportError:
        arr = np.loadtxt(rows_csv, delimiter=",", dtype=np.int32)
        b1, b2, b3 = (arr[:, i].astype(np.uint8) for i in range(3))
        sm_air = arr[:, 3].astype(np.uint8)
        si_dip = arr[:, 4].astype(np.uint8)

    assert b1.shape[0] == num_rows
    print(f"[read] {num_rows:,} lineitem rows in {time.time()-t0:.1f} s")
    os.remove(rows_csv)

    # --- 4. Write 5 own bitmaps ---
    n_b1 = int(b1.sum())
    n_b2 = int(b2.sum())
    n_b3 = int(b3.sum())
    n_sm = int(sm_air.sum())
    n_si = int(si_dip.sum())
    write_bm(os.path.join(out_dir, "branch1_part",     "0.bm"), b1)
    write_bm(os.path.join(out_dir, "branch2_part",     "0.bm"), b2)
    write_bm(os.path.join(out_dir, "branch3_part",     "0.bm"), b3)
    write_bm(os.path.join(out_dir, "shipmode_air",     "0.bm"), sm_air)
    write_bm(os.path.join(out_dir, "shipinstruct_dip", "0.bm"), si_dip)
    pct = lambda n: 100.0 * n / num_rows
    print(f"[gen] branch1_part:      {n_b1:>10,} ({pct(n_b1):.4f}%)")
    print(f"[gen] branch2_part:      {n_b2:>10,} ({pct(n_b2):.4f}%)")
    print(f"[gen] branch3_part:      {n_b3:>10,} ({pct(n_b3):.4f}%)")
    print(f"[gen] shipmode_air:      {n_sm:>10,} ({pct(n_sm):.4f}%)")
    print(f"[gen] shipinstruct_dip:  {n_si:>10,} ({pct(n_si):.4f}%)")

    # --- 5. metadata.txt + done.txt ---
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"brand1={BRAND1}\nbrand2={BRAND2}\nbrand3={BRAND3}\n")
        f.write(f"sm_containers={','.join(SM_CONTAINERS)}\n")
        f.write(f"med_containers={','.join(MED_CONTAINERS)}\n")
        f.write(f"lg_containers={','.join(LG_CONTAINERS)}\n")
        f.write(f"branch1_part_count={n_b1}\n")
        f.write(f"branch2_part_count={n_b2}\n")
        f.write(f"branch3_part_count={n_b3}\n")
        f.write(f"shipmode_air_count={n_sm}\n")
        f.write(f"shipinstruct_dip_count={n_si}\n")
        f.write("quantity_source=tpch_q6_<fmt>/quantity (reused via symlink)\n")
        f.write("format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")
    with open(os.path.join(out_dir, "done.txt"), "w") as f:
        f.write(f"num_rows={num_rows}\n")

    print(f"\n[done] {out_dir}")


if __name__ == "__main__":
    main()
