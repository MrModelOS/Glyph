#!/usr/bin/env bash
# Compile and run every Glyph example, printing a per-file summary.
# Usage: examples/run_all.sh   (optionally set GLYPHC to a custom glyphc binary)
set -u
cd "$(dirname "$0")"

GLYPHC="${GLYPHC:-../target/debug/glyphc}"
if [ ! -x "${GLYPHC}" ]; then
    echo "glyphc not found at ${GLYPHC}. Run 'cargo build' first," >&2
    echo "or point GLYPHC at the build output (e.g. target/release/glyphc)." >&2
    exit 2
fi

pass=0
fail=0
failed=()

run_one() {
    if "${GLYPHC}" run --input "$1" >/dev/null 2>&1; then
        printf 'OK   %s\n' "$1"
        pass=$((pass + 1))
    else
        printf 'FAIL %s\n' "$1"
        failed+=("$1")
        fail=$((fail + 1))
    fi
}

for f in *.glyph; do
    run_one "$f"
done
run_one "project/src/main.glyph"

printf '\nResult: %d passed, %d failed\n' "$pass" "$fail"
if [ "${#failed[@]}" -gt 0 ]; then
    printf 'Failed: %s\n' "${failed[*]}"
    exit 1
fi