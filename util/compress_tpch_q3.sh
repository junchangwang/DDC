#!/bin/bash
# Compress Q3 bitmaps using the existing compress_tpch_q1 tool
# Q3 only has join_result (1 bitmap) + shipdate (reuse from Q1)
# We compress join_result, then symlink shipdate from Q1 compressed dirs
#
# Usage: compress_tpch_q3.sh <input_dir> <output_base> <num_rows>

set -e

INPUT_DIR="${1:-bitmap/tpch_q3}"
OUTPUT_BASE="${2:-bitmap/tpch_q3}"
NUM_ROWS="${3:-12002430}"
DDC_DIR="${OUTPUT_BASE}_ddc"
WAH_DIR="${OUTPUT_BASE}_wah"
CROARING_DIR="${OUTPUT_BASE}_croaring"
EWAH_DIR="${OUTPUT_BASE}_ewah"

# Q1 compressed dirs (for shipdate symlinks)
Q1_BASE="${OUTPUT_BASE/tpch_q3/tpch_q1}"

echo "==========================================="
echo "  Compress TPC-H Q3 Bitmaps"
echo "==========================================="
echo "  Input: $INPUT_DIR"
echo "  Rows:  $NUM_ROWS"
echo "  Q1 base: $Q1_BASE"
echo ""

# Use compress_tpch_q1 to compress just the join_result bitmap
# We create a temp dir with only the join_result subdirectory
TMPDIR=$(mktemp -d)
cp -r "$INPUT_DIR/join_result" "$TMPDIR/join_result"
mkdir -p "$TMPDIR/shipdate"  # empty shipdate so the tool doesn't fail

# Run the compress tool (it will process join_result + empty shipdate)
./build/compress_tpch_q1 "$TMPDIR" "$OUTPUT_BASE" "$NUM_ROWS"

rm -rf "$TMPDIR"

# Now symlink shipdate from Q1 compressed dirs
for fmt in ddc wah croaring ewah; do
    DST="${OUTPUT_BASE}_${fmt}/shipdate"
    SRC="$(realpath ${Q1_BASE}_${fmt}/shipdate)"
    rm -rf "$DST"
    ln -s "$SRC" "$DST"
    echo "  Symlinked: $DST -> $SRC"
done

echo ""
echo "  Done! Output dirs: ${OUTPUT_BASE}_{ddc,wah,croaring,ewah}"
