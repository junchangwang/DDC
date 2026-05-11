#!/bin/bash
# =============================================================================
# Set up TPC-H Q15 bitmap directories at SF10.
#
# Q15 (TPC-H spec v3.0.1 §2.4.15) — Top Supplier Query:
#
#   create view revenue0 (supplier_no, total_revenue) as
#     select l_suppkey, sum(l_extendedprice * (1 - l_discount))
#     from   lineitem
#     where  l_shipdate >= date '[DATE]'
#       and  l_shipdate <  date '[DATE]' + interval '3' month
#     group by l_suppkey;
#
#   select s_suppkey, s_name, s_address, s_phone, total_revenue
#   from   supplier, revenue0
#   where  s_suppkey = supplier_no
#     and  total_revenue = (select max(total_revenue) from revenue0)
#   order by s_suppkey;
#
#   Qualification DATE = '1996-01-01' (spec §2.4.15.4).
#
# Bitmap data needed:
#   shipdate/{day}.bm  : equality bitmaps for days [1461..1551]
#                        (= [1996-01-01, 1996-04-01), 1996 leap year, 91 days)
#
# These shipdate bitmaps are already produced by export_tpch_q6.py for
# the full TPC-H shipdate domain [date_min..date_max].  This script
# simply creates Q15 bitmap directories whose `shipdate/` subdirectory
# is a symlink to the corresponding Q6 directory — both raw and the
# four compressed formats (combit/wah/croaring/ewah).
#
# l_suppkey, l_extendedprice and l_discount are read directly from
# DuckDB storage at query time (mirrors Q1's per-row column load) and
# therefore do not need an export step.
#
# Usage (from any cwd):
#   bash setup_tpch_q15_sf10.sh <bitmap_base_dir>
#
# Example:
#   bash util/setup_tpch_q15_sf10.sh /home/lichenhang/lee/thesis/check/lee/duckdb-dev
#
# After running this, the Q15 benchmark (`PRAGMA bm_tpch(15);`) will
# find its bitmaps under tpch_q15{,_combit,_wah,_croaring,_ewah}/shipdate/.
# =============================================================================
set -euo pipefail

BASE_DIR="${1:-$PWD}"
BASE_DIR="$(realpath "$BASE_DIR")"

echo "==========================================="
echo "  Set up TPC-H Q15 bitmap directories"
echo "==========================================="
echo "  Base dir: $BASE_DIR"

# All five flavors: raw + 4 compressed formats
FORMATS=("" "_combit" "_wah" "_croaring" "_ewah")

for fmt in "${FORMATS[@]}"; do
    Q6_DIR="${BASE_DIR}/tpch_q6${fmt}"
    Q15_DIR="${BASE_DIR}/tpch_q15${fmt}"

    if [[ ! -d "$Q6_DIR" ]]; then
        echo "  [skip] $Q6_DIR not found — Q6 has not been exported/compressed for this format yet."
        continue
    fi

    if [[ ! -d "$Q6_DIR/shipdate" ]]; then
        echo "  [skip] $Q6_DIR/shipdate missing — run export_tpch_q6.py / compress_tpch_q6 first."
        continue
    fi

    mkdir -p "$Q15_DIR"

    # Replace any existing Q15 shipdate (file, dir, or stale symlink)
    SHIP_LINK="$Q15_DIR/shipdate"
    rm -rf "$SHIP_LINK"
    ln -s "$Q6_DIR/shipdate" "$SHIP_LINK"
    echo "  Symlinked $SHIP_LINK -> $Q6_DIR/shipdate"
done

echo ""
echo "  Done.  Q15 reads l_suppkey/l_extendedprice/l_discount from"
echo "  DuckDB storage at query time; no further data export needed."
