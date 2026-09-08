#!/usr/bin/env bash
# Generates three workloads (loop_sum, list_append, map_put_get) in Glyph, C
# and Rust, then builds each into a binary. Every language gets the same
# algorithm and the same iteration counts. loop_sum reads its bound from
# stdin (so the compiler cannot fold the loop into a constant), the others
# hard-code N.
#
# Usage: bench/gen.sh <outdir> <N>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT=${1:?outdir}
N=${2:?iteration count}
LOOP_N=$((N * 20))   # scalar adds are cheap; scale up for measurable time
LIST_N=$N
MAP_N=$((N / 10))    # string-key map ops are the slowest; keep < 1s
mkdir -p "$OUT"/glyph "$OUT"/c "$OUT"/rust/src

GCC_BIN="$(command -v gcc || true)"

# ---------- Glyph ----------
cat > "$OUT/glyph/loop_sum.glyph" <<EOF
@fn main() -> Void {
    let line: String = read_line();
    let parsed: Option<Int64> = parse_int(line);
    let mut n: Int64 = match parsed {
        | Some(v) => v
        | None => 0
    };
    drop(parsed);
    let mut total: Int64 = 0;
    for i in 0..n {
        total = total + (i % 7);
    }
    print_int(total);
}
EOF

cat > "$OUT/glyph/list_append.glyph" <<EOF
@fn main() -> Void {
    let mut xs: List<Int64> = [];
    for i in 0..$LIST_N {
        xs.append(i);
    }
    let mut total: Int64 = 0;
    for x in xs {
        total = total + x;
    }
    xs.free();
    print_int(total);
}
EOF

cat > "$OUT/glyph/map_put_get.glyph" <<EOF
@fn main() -> Void {
    let mut m: Map<String, Int64> = #{ "seed": 0 };
    for i in 0..$MAP_N {
        m.put(int_to_string(i % 100), i);
    }
    let mut total: Int64 = 0;
    for i in 0..$MAP_N {
        total = total + m[int_to_string(i % 100)];
    }
    m.free();
    print_int(total);
}
EOF

# ---------- C ----------
cat > "$OUT/c/loop_sum.c" <<EOF
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
int main(void) {
    char buf[32];
    if (!fgets(buf, sizeof buf, stdin)) return 1;
    int64_t n = strtoll(buf, 0, 10);
    int64_t total = 0;
    for (int64_t i = 0; i < n; i++) total += i % 7;
    printf("%lld\n", (long long)total);
    return 0;
}
EOF

cat > "$OUT/c/list_append.c" <<EOF
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
int main(void) {
    int64_t cap = 8, len = 0;
    int64_t* a = malloc((size_t)cap * sizeof(int64_t));
    for (int64_t i = 0; i < $LIST_N; i++) {
        if (len == cap) { cap *= 2; a = realloc(a, (size_t)cap * sizeof(int64_t)); }
        a[len++] = i;
    }
    int64_t total = 0;
    for (int64_t i = 0; i < len; i++) total += a[i];
    free(a);
    printf("%lld\n", (long long)total);
    return 0;
}
EOF

cat > "$OUT/c/map_put_get.c" <<EOF
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BUCKETS 256

static uint64_t hash_str(const char* k) {
    uint64_t h = 1469598103934665603ULL;
    for (; *k; k++) { h ^= (uint64_t)*k; h *= 1099511628211ULL; }
    return h;
}

typedef struct { char* key; int64_t val; char* next; } Entry;

int main(void) {
    Entry* buckets[BUCKETS] = {0};
    for (int64_t i = 0; i < $MAP_N; i++) {
        char k[16];
        snprintf(k, sizeof k, "%lld", (long long)(i % 100));
        uint64_t b = hash_str(k) & (BUCKETS - 1);
        Entry* e = buckets[b];
        for (; e; e = (Entry*)e->next)
            if (strcmp(e->key, k) == 0) break;
        if (e) { e->val = i; continue; }
        e = malloc(sizeof(Entry));
        e->key = strdup(k);
        e->val = i;
        e->next = (char*)buckets[b];
        buckets[b] = e;
    }
    int64_t total = 0;
    for (int64_t i = 0; i < $MAP_N; i++) {
        char k[16];
        snprintf(k, sizeof k, "%lld", (long long)(i % 100));
        uint64_t b = hash_str(k) & (BUCKETS - 1);
        Entry* e = buckets[b];
        for (; e; e = (Entry*)e->next)
            if (strcmp(e->key, k) == 0) { total += e->val; break; }
    }
    for (int b = 0; b < BUCKETS; b++)
        for (Entry* e = buckets[b]; e; ) { Entry* n = (Entry*)e->next; free(e->key); free(e); e = n; }
    printf("%lld\n", (long long)total);
    return 0;
}
EOF

# ---------- Rust ----------
cat > "$OUT/rust/Cargo.toml" <<EOF
[package]
name = "glyph_bench"
version = "0.1.0"
edition = "2021"

[profile.release]
lto = true
codegen-units = 1
EOF

cat > "$OUT/rust/src/main.rs" <<EOF
use std::collections::HashMap;

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "loop_sum" => {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).ok();
            let n: i64 = line.trim().parse().unwrap_or(0);
            let mut total: i64 = 0;
            for i in 0..n { total += i % 7; }
            println!("{total}");
        }
        "list_append" => {
            let mut xs: Vec<i64> = Vec::new();
            for i in 0..$LIST_N { xs.push(i); }
            let mut total: i64 = 0;
            for x in &xs { total += x; }
            println!("{total}");
        }
        "map_put_get" => {
            let mut m: HashMap<String, i64> = HashMap::new();
            for i in 0..$MAP_N { m.insert(format!("{}", i % 100), i); }
            let mut total: i64 = 0;
            for i in 0..$MAP_N { total += m.get(&format!("{}", i % 100)).copied().unwrap_or(0); }
            println!("{total}");
        }
        _ => eprintln!("usage: glyph_bench <loop_sum|list_append|map_put_get>"),
    }
}
EOF

# ---------- Build ----------
if [ -x "$GCC_BIN" ]; then
    for w in loop_sum list_append map_put_get; do
        "$GCC_BIN" -O2 -o "$OUT/b_${w}" "$OUT/c/${w}.c"
    done
fi

if command -v cargo >/dev/null 2>&1; then
    (cd "$OUT/rust" && cargo build --release --quiet)
fi

(cd "$REPO_ROOT" && cargo build --release --quiet)
GLYPHC="$REPO_ROOT/target/release/glyphc"
for w in loop_sum list_append map_put_get; do
    if "$GLYPHC" compile --input "$OUT/glyph/${w}.glyph" --output "$OUT/g_${w}.c"; then
        "$GCC_BIN" -O2 -o "$OUT/g_${w}" "$OUT/g_${w}.c" -lm
    fi
done

echo "generated in $OUT (N=$N; loop=${LOOP_N} list=${LIST_N} map=${MAP_N})"