#!/bin/bash
# run_not_sweep.sh — sweep all 7 backends across the c=2..1000 standard
# cardinalities plus the three CRoaring-engineered points (t3500, A2500_B100,
# o2200) and append NOT_op + OR_op + AND_op + XOR_op rows to plot_results.csv.
#
# We intentionally re-record OR/AND/XOR while we're here so the CSV row set
# stays internally consistent — same iteration, same machine state, same
# warmup.  The motivation.py plot scripts only consume OR_op / AND_op / NOT_op
# rows, so the extra OR/AND/XOR rows are harmless duplicates.
#
# Output: appends to tools/plot_results.csv (same file motivation_eva.py reads).

set -e
cd "$(dirname "$0")/.."

BENCH=./build/benchmark_app
OUT=tools/plot_results.csv

if [ ! -x "$BENCH" ]; then
    echo "ERROR: $BENCH not found.  Run cmake --build build first." >&2
    exit 1
fi
if [ ! -f "$OUT" ]; then
    echo "ERROR: $OUT not found.  Aborting (need an existing CSV to append to)." >&2
    exit 1
fi

# Standard cardinality sweep (c=2..1000 per user request) + three engineered
# points for CRoaring container-transition stress (t3500/A2500_B100/o2200).
CARDS_STD="2 5 10 20 50 100 200 500 1000"
ENG_DIRS="bm_100m_t3500 bm_100m_A2500_B100 bm_100m_o2200"

# (algo_dir_suffix, backend_name)
declare -a BACKENDS=(
    "combit_w8:combit"
    "roaring:croaring"
    "wah:wah"
    "ewah:ewah"
    "concise:concise"
    "bitset:bitset"           # bitset dir also drives bitset_avx512 backend
)

run_one() {
    local dir="$1" backend="$2"
    if [ ! -d "bitmap/$dir" ]; then
        printf "  SKIP (missing dir): bitmap/%s\n" "$dir"
        return
    fi
    printf "  %s [%s] ... " "$dir" "$backend"
    # --iterations 1 because the pure-ops blocks already do their own N_ITER=5
    # median internally.  --csv appends to plot_results.csv.
    "$BENCH" --compressed-dir "bitmap/$dir" --backend "$backend" \
             --iterations 1 --csv "$OUT" > /dev/null 2>&1
    printf "OK\n"
}

echo "=========================================="
echo " NOT sweep: 7 backends × (c=2..1000 + 3 engineered points)"
echo "=========================================="

for spec in "${BACKENDS[@]}"; do
    algo="${spec%%:*}"; bend="${spec##*:}"
    echo ""
    echo "[$bend] standard cardinalities:"
    for c in $CARDS_STD; do
        run_one "bm_100m_c${c}_${algo}" "$bend"
    done
    # bitset directory is shared by bitset + bitset_avx512 backends; second pass
    if [ "$algo" = "bitset" ]; then
        for c in $CARDS_STD; do
            run_one "bm_100m_c${c}_bitset" "bitset_avx512"
        done
    fi
    echo "[$bend] engineered points:"
    for dir_pre in $ENG_DIRS; do
        run_one "${dir_pre}_${algo}" "$bend"
    done
    if [ "$algo" = "bitset" ]; then
        for dir_pre in $ENG_DIRS; do
            run_one "${dir_pre}_bitset" "bitset_avx512"
        done
    fi
done

echo ""
echo "=========================================="
echo " Done.  NOT_op rows appended to $OUT"
echo "=========================================="
