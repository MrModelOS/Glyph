# Benchmarks

These are small, reproducible microbenchmarks, not a language-wide shootout.
They compare the same three workloads in Glyph-generated C, hand-written C,
and Rust:

- `loop_sum` — integer induction and loop codegen;
- `list_append` — dynamic-list growth and element copies;
- `map_put_get` — string keys, hashing, and map operations.

## Run locally

```bash
cargo build --release
cc --version
./bench/gen.sh /tmp/glyph-bench
./bench/run.sh /tmp/glyph-bench
```

`bench/gen.sh` generates the C and Rust comparison programs. The runner uses
three repetitions and reports the minimum wall-clock time. For publication or
performance discussions, run multiple times on an idle machine and retain the
raw output; compiler flags, CPU, and power settings affect the numbers.

The table in the top-level README is intentionally labelled as a small local
microbenchmark. It is not a claim that Glyph is faster than C or Rust in
general.
