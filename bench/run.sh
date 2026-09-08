#!/usr/bin/env bash
# Times the three workloads (3 runs, min wall time) and prints a markdown table.
# loop_sum reads its iteration count from stdin; the others hard-code N.
#
# Usage: bench/run.sh <outdir>
set -euo pipefail

OUT=${1:?outdir}

# time_ms <bin> [arg ...] [--stdin VAL]
time_ms() {
    local bin=$1; shift
    local ps=()
    local stdin_val=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--stdin" ]; then
            stdin_val=$2
            shift 2
        else
            ps+=("$1")
            shift
        fi
    done
    local best=-1
    for _ in 1 2 3; do
        local start end d
        start=$(date +%s%N)
        if [ -n "$stdin_val" ]; then
            "$bin" "${ps[@]}" <<<"$stdin_val" >/dev/null 2>&1
        else
            "$bin" "${ps[@]}" >/dev/null 2>&1
        fi
        end=$(date +%s%N)
        d=$(( (end - start) / 1000000 ))
        if [ "$best" -lt 0 ] || [ "$d" -lt "$best" ]; then best=$d; fi
    done
    echo "$best"
}

RSBIN="$OUT/rust/target/release/glyph_bench"
LOOP_IN=${LOOP_IN:-400000000}

printf '%-14s %12s %12s %12s %s\n' "workload" "glyph" "c" "rust" "ratio"
for w in loop_sum list_append map_put_get; do
    gh=$(time_ms "$OUT/g_${w}" --stdin "$LOOP_IN")
    c=$(time_ms "$OUT/b_${w}" --stdin "$LOOP_IN")
    if [ "$w" = "loop_sum" ]; then
        rs=$(time_ms "$RSBIN" "$w" --stdin "$LOOP_IN")
    else
        rs=$(time_ms "$RSBIN" "$w")
    fi
    ratio=""
    if [ "$c" -gt 0 ]; then
        ratio="g/c $(awk "BEGIN{printf \"%.2fx\", $gh/$c}")"
    fi
    printf '%-14s %12s %12s %12s %s\n' "$w" "${gh}ms" "${c}ms" "${rs}ms" "$ratio"
done