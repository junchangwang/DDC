#!/bin/bash
# Capture per-combo storage info for Excel report.
# Runs benchmark_app once per (algo, card) with stdout to logs.

set -e
cd "$(dirname "$0")"
mkdir -p storage_info

CARDS="2 5 10 20 50 100 200 500 1000 2000 3000 4000 5000 8000 10000 20000 50000"
for c in $CARDS; do
    for algo in combit roaring ewah concise wah; do
        bend="$algo"
        [ "$algo" = "roaring" ] && bend="croaring"
        dir="bitmap/bm_100m_c${c}_${algo}"
        [ "$algo" = "combit" ] && dir="${dir}_w8"
        log="storage_info/c${c}_${algo}.log"

        if [ ! -f "$dir/done.txt" ]; then continue; fi
        if [ -f "$log" ] && [ "$(stat -c %s "$log")" -gt 200 ]; then continue; fi

        printf "[c=%5s %s] " "$c" "$algo"
        ./build/benchmark_app --compressed-dir "$dir" --backend "$bend" \
            --iterations 1 > "$log" 2>&1
        echo "OK"
    done
done
echo "Storage pass complete."
