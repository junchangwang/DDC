#!/usr/bin/env python3
"""
Export TPC-H Q5 bitmap data from DuckDB.

Generates:
  - orderdate/D.bm: equality bitmap on lineitem via FK mapping
    (bit[i]=1 if lineitem row i's order has o_orderdate = D days since 1992-01-01)
  - nation_join/N.bm: per-nation join bitmap
    (bit[i]=1 if lineitem row i's order-customer nation = N AND supplier nation = N)
  - nation_names.csv: nationkey → nation_name mapping for ASIA nations

Q5 filter: o_orderdate >= '1994-01-01' AND o_orderdate < '1995-01-01', r_name = 'ASIA'
           c_nationkey = s_nationkey

Usage:
  python3 export_tpch_q5.py <duckdb_binary> <db_path> <output_dir>
"""

import sys, os, subprocess, time
import numpy as np


def run_sql_csv(duckdb_bin, db_path, sql, csv_path):
    """Run SQL in DuckDB and export to CSV."""
    result = subprocess.run(
        [duckdb_bin, db_path],
        input=sql, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[error] DuckDB:\n{result.stderr}")
        sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print("Usage: export_tpch_q5.py <duckdb_binary> <db_path> <output_dir>")
        sys.exit(1)

    duckdb_bin = sys.argv[1]
    db_path = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # 1. Export lineitem FK columns + lookup tables, join in Python
    #    (avoids duplicate-row issues from SQL JOINs)
    # ================================================================
    # Export lineitem l_orderkey and l_suppkey
    li_csv = os.path.join(output_dir, "_q5_lineitem.csv")
    print(f"[export] Exporting lineitem FK columns (l_orderkey, l_suppkey)...")
    t0 = time.time()
    sql = f"""COPY (SELECT l_orderkey, l_suppkey FROM lineitem ORDER BY rowid) TO '{li_csv}' (HEADER false);"""
    run_sql_csv(duckdb_bin, db_path, sql, li_csv)
    print(f"[export] lineitem done in {time.time()-t0:.1f}s")

    # Export orders (deduplicated)
    ord_csv = os.path.join(output_dir, "_q5_orders.csv")
    sql = f"""COPY (SELECT DISTINCT o_orderkey, o_custkey, o_orderdate - DATE '1992-01-01' AS day FROM orders) TO '{ord_csv}' (HEADER false);"""
    run_sql_csv(duckdb_bin, db_path, sql, ord_csv)

    # Export customer
    cust_csv = os.path.join(output_dir, "_q5_customer.csv")
    sql = f"""COPY (SELECT c_custkey, c_nationkey FROM customer) TO '{cust_csv}' (HEADER false);"""
    run_sql_csv(duckdb_bin, db_path, sql, cust_csv)

    # Export supplier
    supp_csv = os.path.join(output_dir, "_q5_supplier.csv")
    sql = f"""COPY (SELECT s_suppkey, s_nationkey FROM supplier) TO '{supp_csv}' (HEADER false);"""
    run_sql_csv(duckdb_bin, db_path, sql, supp_csv)

    # ================================================================
    # Build lookup tables in Python
    # ================================================================
    print(f"[read] Building lookup tables...")
    t0 = time.time()

    li_data = np.loadtxt(li_csv, delimiter=',', dtype=np.int64)
    li_orderkey = li_data[:, 0]
    li_suppkey = li_data[:, 1]
    num_rows = len(li_orderkey)
    print(f"  lineitem: {num_rows} rows")

    ord_data = np.loadtxt(ord_csv, delimiter=',', dtype=np.int64)
    orderkey_to_custkey = dict(zip(ord_data[:, 0], ord_data[:, 1]))
    orderkey_to_day = dict(zip(ord_data[:, 0], ord_data[:, 2]))
    print(f"  orders: {len(orderkey_to_day)} unique orderkeys")

    cust_data = np.loadtxt(cust_csv, delimiter=',', dtype=np.int64)
    custkey_to_nation = dict(zip(cust_data[:, 0], cust_data[:, 1]))

    supp_data = np.loadtxt(supp_csv, delimiter=',', dtype=np.int64)
    suppkey_to_nation = dict(zip(supp_data[:, 0], supp_data[:, 1]))

    print(f"[read] Lookups built in {time.time()-t0:.1f}s")

    # Map orderdate to each lineitem row
    print(f"[map] Mapping orderdate + nations to lineitem rows...")
    t0 = time.time()
    odate = np.array([orderkey_to_day[ok] for ok in li_orderkey], dtype=np.int32)
    cust_nation = np.array([custkey_to_nation[orderkey_to_custkey[ok]] for ok in li_orderkey], dtype=np.int32)
    supp_nation = np.array([suppkey_to_nation[sk] for sk in li_suppkey], dtype=np.int32)
    min_day = int(odate.min())
    max_day = int(odate.max())
    print(f"[map] Done in {time.time()-t0:.1f}s, day range [{min_day}, {max_day}]")

    # Clean up temp CSVs
    for f in [li_csv, ord_csv, cust_csv, supp_csv]:
        os.remove(f)

    # ================================================================
    # 2. Generate orderdate equality bitmaps
    # ================================================================
    odate_dir = os.path.join(output_dir, "orderdate")
    os.makedirs(odate_dir, exist_ok=True)
    print(f"\n[gen] Creating orderdate equality bitmaps...")
    t0 = time.time()
    unique_days = np.unique(odate)
    for i, d in enumerate(unique_days):
        mask = (odate == d).astype(np.uint8)
        packed = np.packbits(mask, bitorder='little')
        packed.tofile(os.path.join(odate_dir, f"{int(d)}.bm"))
        if (i+1) % 500 == 0:
            print(f"  {i+1}/{len(unique_days)} days...")
    print(f"[gen] {len(unique_days)} orderdate bitmaps in {time.time()-t0:.1f}s")

    # ================================================================
    # 3. Generate nation_join bitmaps (per ASIA nation)
    # ================================================================
    # ASIA nations: regionkey=2 → nationkeys 8(INDIA), 9(INDONESIA), 12(JAPAN), 18(CHINA), 21(VIETNAM)
    asia_nations = {8: 'INDIA', 9: 'INDONESIA', 12: 'JAPAN', 18: 'CHINA', 21: 'VIETNAM'}

    nation_dir = os.path.join(output_dir, "nation_join")
    os.makedirs(nation_dir, exist_ok=True)
    print(f"\n[gen] Creating per-nation join bitmaps (customer_nation = supplier_nation = N)...")
    for nk, name in asia_nations.items():
        mask = ((cust_nation == nk) & (supp_nation == nk)).astype(np.uint8)
        ones = int(mask.sum())
        packed = np.packbits(mask, bitorder='little')
        packed.tofile(os.path.join(nation_dir, f"{nk}.bm"))
        print(f"  nation {nk} ({name}): {ones} rows ({100*ones/num_rows:.2f}%)")

    # ================================================================
    # 4. Export nation names mapping
    # ================================================================
    names_path = os.path.join(output_dir, "nation_names.csv")
    with open(names_path, 'w') as f:
        for nk, name in sorted(asia_nations.items()):
            f.write(f"{nk},{name}\n")

    # ================================================================
    # 5. Write metadata
    # ================================================================
    meta_path = os.path.join(output_dir, "metadata.txt")
    with open(meta_path, 'w') as f:
        f.write(f"num_rows={num_rows}\n")
        f.write(f"orderdate_min={min_day}\n")
        f.write(f"orderdate_max={max_day}\n")
        f.write(f"orderdate_count={len(unique_days)}\n")
        f.write(f"q5_date_range=731..1095 (1994-01-01 to 1994-12-31, days since 1992-01-01)\n")
        f.write(f"q5_date_bitmaps=365\n")
        f.write(f"asia_nations=8(INDIA),9(INDONESIA),12(JAPAN),18(CHINA),21(VIETNAM)\n")
        f.write(f"format=raw packed bits, LSB-first (numpy packbits bitorder='little')\n")

    print(f"\n[done] Output: {output_dir}")
    print(f"  orderdate: {len(unique_days)} bitmaps (days {min_day}..{max_day})")
    print(f"  nation_join: 5 bitmaps (ASIA nations)")
    print(f"  Total rows: {num_rows}")


if __name__ == '__main__':
    main()
