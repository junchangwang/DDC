#!/bin/bash
# Compress Q5 bitmaps using the existing compress_tpch_q1 tool
# Q5 has: orderdate/ (many bitmaps, like shipdate) + nation_join/ (5 bitmaps)
#
# Strategy: run compress_tpch_q1 twice with temp dir mapping
#   Pass 1: orderdate/ → mapped as shipdate/ → output to orderdate/
#   Pass 2: nation_join/ → mapped as shipdate/ → output to nation_join/
#
# Usage: compress_tpch_q5.sh <input_dir> <output_base> <num_rows>

set -e

INPUT_DIR="${1:?Usage: compress_tpch_q5.sh <input_dir> <output_base> <num_rows>}"
OUTPUT_BASE="${2:?}"
NUM_ROWS="${3:?}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMBIT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPRESS_TOOL="$COMBIT_ROOT/build/compress_tpch_q1"

if [ ! -f "$COMPRESS_TOOL" ]; then
    echo "Error: $COMPRESS_TOOL not found. Build combit first."
    exit 1
fi

echo "==========================================="
echo "  Compress TPC-H Q5 Bitmaps"
echo "==========================================="
echo "  Input: $INPUT_DIR"
echo "  Rows:  $NUM_ROWS"
echo ""

# Create output dirs
for fmt in combit wah croaring ewah; do
    mkdir -p "${OUTPUT_BASE}_${fmt}"
done

# =======================================
# Pass 1: orderdate (many bitmaps)
# =======================================
echo "=== Pass 1: Compressing orderdate bitmaps ==="
TMPDIR=$(mktemp -d)
ln -s "$(realpath "$INPUT_DIR/orderdate")" "$TMPDIR/shipdate"
"$COMPRESS_TOOL" "$TMPDIR" "${OUTPUT_BASE}_odate_tmp" "$NUM_ROWS"
rm -rf "$TMPDIR"

# Move shipdate → orderdate in output dirs
for fmt in combit wah croaring ewah; do
    if [ -d "${OUTPUT_BASE}_odate_tmp_${fmt}/shipdate" ]; then
        mv "${OUTPUT_BASE}_odate_tmp_${fmt}/shipdate" "${OUTPUT_BASE}_${fmt}/orderdate"
    fi
    rm -rf "${OUTPUT_BASE}_odate_tmp_${fmt}"
done

# =======================================
# Pass 2: nation_join (5 bitmaps)
# =======================================
echo ""
echo "=== Pass 2: Compressing nation_join bitmaps ==="
TMPDIR=$(mktemp -d)
ln -s "$(realpath "$INPUT_DIR/nation_join")" "$TMPDIR/shipdate"
"$COMPRESS_TOOL" "$TMPDIR" "${OUTPUT_BASE}_nj_tmp" "$NUM_ROWS"
rm -rf "$TMPDIR"

# Move shipdate → nation_join in output dirs
for fmt in combit wah croaring ewah; do
    if [ -d "${OUTPUT_BASE}_nj_tmp_${fmt}/shipdate" ]; then
        mv "${OUTPUT_BASE}_nj_tmp_${fmt}/shipdate" "${OUTPUT_BASE}_${fmt}/nation_join"
    fi
    rm -rf "${OUTPUT_BASE}_nj_tmp_${fmt}"
done

# =======================================
# Write done.txt with correct metadata
# =======================================
ODATE_COUNT=$(ls "$INPUT_DIR/orderdate/"*.bm 2>/dev/null | wc -l)
NJ_COUNT=$(ls "$INPUT_DIR/nation_join/"*.bm 2>/dev/null | wc -l)
TOTAL=$((ODATE_COUNT + NJ_COUNT))

for fmt in combit wah croaring ewah; do
    cat > "${OUTPUT_BASE}_${fmt}/done.txt" << EOF
num_rows=$NUM_ROWS
num_bitmaps=$TOTAL
orderdate_count=$ODATE_COUNT
nation_join_count=$NJ_COUNT
EOF
done

# Copy nation_names.csv if it exists
if [ -f "$INPUT_DIR/nation_names.csv" ]; then
    for fmt in combit wah croaring ewah; do
        cp "$INPUT_DIR/nation_names.csv" "${OUTPUT_BASE}_${fmt}/"
    done
fi

echo ""
echo "==========================================="
echo "  Done! $TOTAL bitmaps ($ODATE_COUNT orderdate + $NJ_COUNT nation_join)"
echo "  Output: ${OUTPUT_BASE}_{combit,wah,croaring,ewah}"
echo "==========================================="
