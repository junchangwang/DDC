#!/bin/bash
# run_comp_sweep.sh — sweep 7 backends across c=5..1000 for the COMP_op
# (comprehensive predicate `~((A | B) & (B | C))`).
#
# COMP needs 3 distinct bitmaps in the directory; c=2 has only 2 .bm files
# and the engineered points (t3500/A2500_B100/o2200) also have only 2, so
# they are skipped.  Output: appends COMP_op rows to tools/plot_results.csv.

set -e
cd "$(dirname "$0")/.."

BENCH=./build/benchmark_app
OUT=tools/plot_results.csv

if [ ! -x "$BENCH" ]; then echo "ERROR: $BENCH missing"; exit 1; fi
if [ ! -f "$OUT" ];  then echo "ERROR: $OUT missing";   exit 1; fi

# COMP needs ≥3 bitmaps per dir; c=2 has 2 .bm files (partition style A∪B
# = universe).  The benchmark falls back to C=A at c=2, so the 4-op chain
# still runs — predicate mathematically degenerates to ~(A|B) but timer
# captures the same 4 ops for all backends.
CARDS_STD="2 5 10 20 50 100 200 500 1000"

declare -a BACKENDS=(
    "ddc_w8:ddc"
    "roaring:croaring"
    "wah:wah"
    "ewah:ewah"
    "concise:concise"
    "bitset:bitset"           # bitset dir is also used by bitset_avx512
)

run_one() {
    local dir="$1" backend="$2"
    if [ ! -d "bitmap/$dir" ]; then
        printf "  SKIP (missing): bitmap/%s\n" "$dir"; return
    fi
    printf "  %s [%s] ... " "$dir" "$backend"
    "$BENCH" --compressed-dir "bitmap/$dir" --backend "$backend" \
             --iterations 1 --csv "$OUT" > /dev/null 2>&1
    printf "OK\n"
}

echo "=========================================="
echo " COMP sweep: 7 backends × c=5..1000"
echo "=========================================="

for spec in "${BACKENDS[@]}"; do
    algo="${spec%%:*}"; bend="${spec##*:}"
    echo ""
    echo "[$bend]"
    for c in $CARDS_STD; do
        run_one "bm_100m_c${c}_${algo}" "$bend"
    done
    if [ "$algo" = "bitset" ]; then
        for c in $CARDS_STD; do
            run_one "bm_100m_c${c}_bitset" "bitset_avx512"
        done
    fi
done

echo ""
echo "=========================================="
echo " Done.  COMP_op rows appended to $OUT"
echo "=========================================="
