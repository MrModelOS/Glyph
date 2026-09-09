# Glyph Language Compiler (glyphc)

[![CI](https://github.com/MrModelOS/Glyph/actions/workflows/ci.yml/badge.svg)](https://github.com/MrModelOS/Glyph/actions/workflows/ci.yml)
![version](https://img.shields.io/badge/glyphc-v1.3.0-blue)

The **Glyph** programming language compiler, written in Rust.

Glyph transpiles to C (GNU statement expressions) and builds with GCC or clang into
plain native binaries.

```
┌──────────┐   glyphc    ┌────────────┐   gcc/clang   ┌─────────┐
│ .glyph   │ ──────────▶ │     .c     │ ─────────────▶ │ binary  │
└──────────┘   (Rust)    └────────────┘               └─────────┘
```

## Features

- Static typing: `Int64`, `UInt64`, `Float64`, `Bool`, `String`, `Bytes`
- Flow-sensitive static analysis: using a `List`/`Map` after `.free()`
  (or freeing it twice) is rejected at compile time until reassignment
- User types: `@struct`, `@enum` with payload data in variants
- `@impl` methods: `obj.method(args)`, receiver is the first parameter
- Control flow: `if`/`else`, `match` (including variant patterns with payload), `while`, `loop`, `for .. in`
- Guards: `#guard(cond) else { ... };`
- Modules: `@module`, `@use`, `@pub`, qualified calls `math::sqrt`
- `@const` named constants (compiled to `static const`)
- Typed lists `List<T>`: literals `[..]`, indexing `arr[i]`, slices `arr[a..b]`/`arr[a..=b]`,
  ranges `0..10`, `.len()`, `append`, `for x in xs`, `++` concatenation,
  `==` for POD lists, refcounted buffers with `xs.free()`
- Maps `Map<String, V>`: literal `#{ "k": v }`, indexing `m["k"]`
  read/write, `.put()`, `.get() -> Option<V>`, `.len()`, `.free()`,
  iteration `for k in m` yields keys
- `Result`/`Option` with payloads and `drop(box)` freeing
- Generic functions: `@fn identity<T>(x: T) -> T` — monomorphization, type inference
  by argument and `let` annotations, nested types (`Option<T>`, `List<T>`)
- Concurrency (M:N worker pool): `@fn async`, lazy handles `Async<T>`, `spawn`/`await`,
  typed channels `Channel<T>(capacity)`, `send`/`recv`/`close`, `select` over
  recv/await with `timeout(ms)`/`default` arms
- Built-in test framework: `@test`, asserts, `glyphc test`
- `glyphc fmt`: comment-preserving canonical indentation + `--check`/`--write`

## Install

Requires Rust (1.70+) and `gcc` (or `clang`) on `PATH`.

```bash
git clone https://github.com/MrModelOS/Glyph.git
cd Glyph/glyphc
cargo build --release
```

The binary appears at `target/release/glyphc`. Prebuilt binaries for tagged releases
are published on the [Releases](/MrModelOS/Glyph/releases) page.

## Quick start

`hello.glyph`:

```glyph
@fn main() -> Void {
    let name: String = "World";
    let greeting: String = "Hello, " ++ name ++ "!";
    print(greeting);
}
```

```bash
glyphc run --input hello.glyph
# Hello, World!
```

## CLI

| Subcommand | Description |
|---|---|
| `compile --input f.glyph [-o out.c] [--emit-ir] [--no-typecheck]` | Transpile to C (default `output.c`) |
| `check --input f.glyph` | Check syntax and types without codegen |
| `tokens --input f.glyph` | Show the token stream |
| `ast --input f.glyph` | Show the AST |
| `run --input f.glyph [--compiler gcc] [--opt -O2]` | Compile and run (inherits stdio) |
| `build [--profile dev\|release]` | Build a project described by `glyph.toml` |
| `fmt --input f.glyph [--write] [--check]` | Canonical indentation/whitespace; comments preserved (prints to stdout by default) |
| `test [-i f.glyph] [--compiler gcc] [--opt -O2]` | Run `@test` functions (scans `./src` without `-i`) |
| `glyphc --lsp` | LSP server: live diagnostics (lexer/parser/typechecker, full spans), hover and completion |

Common flags: `-h/--help`, `-V/--version`.

## Language tour

Full reference (in Russian): [docs/language.md](docs/language.md). Highlights:

```glyph
@struct Point { x: Float64, y: Float64 }

@impl Point {
    @fn norm(p: Point) -> Float64 {
        return sqrt(p.x * p.x + p.y * p.y);
    }
}

@enum Shape {
    Circle(Float64),
    Rectangle(Float64, Float64),
    Point,
}

@fn area(s: Shape) -> Float64 {
    match s {
        | Shape::Circle(r) => 3.14159 * r * r
        | Shape::Rectangle(w, h) => w * h
        | Shape::Point => 0.0
    }
}

@fn distance(a: Point, b: Point) -> Float64 {
    let dx: Float64 = b.x - a.x;
    let dy: Float64 = b.y - a.y;
    return sqrt(dx * dx + dy * dy);
}

@fn main() -> Void {
    let p: Point = Point { x: 3.0, y: 4.0 };
    print_float(p.norm());          // 5.0

    let arr: List<Int64> = [1, 2, 3];
    arr.append(4);
    let both: List<Int64> = arr ++ arr;
    let slice: List<Int64> = arr[1..3];
    print_float(area(Shape::Circle(5.0)));
}
```

Errors and generics:

```glyph
@fn divide(a: Int64, b: Int64) -> Result<Int64, String> {
    #guard(b != 0) else {
        return Result::Err("division by zero");
    };
    return Result::Ok(a / b);
}

@fn identity<T>(x: T) -> T { return x; }

@fn main() {
    let q: Int64 = match divide(10, 2) {
        | Ok(v) => v,
        | Err(e) => -1
    };
    let s: String = identity("hello");
    print_int(q);
}
```

Concurrency, modules, and asserts are documented in [docs/language.md](docs/language.md).

## Testing

```bash
cargo build
cargo test          # 51 unit tests (lexer, parser, typechecker, codegen)
examples/run_all.sh # compiles and runs every example
glyphc test -i examples/fifteen.glyph
```

`examples/fifteen.glyph` is a real interactive program (15-puzzle) written in Glyph:
lists, slicing, functions, `read_line`, and `@test`s. `examples/list_refs.glyph`
demonstrates the refcounted list semantics, and `examples/maps.glyph` shows the
`Map<String, V>` runtime (`#{}` literal, indexing, `put`/`get`/`len`/`free`, `for k in m`).

## Performance

Glyph compiles to C and inherits the C toolchain: scalar code compiles to the
same machine code, and the runtime overhead is proportional to the
allocation/GC features you use. The table below compares identical algorithms
on one machine (min of 3 runs, gcc 16.2.1 `-O2`, rustc 1.98.1 `--release` with
LTO, meant to show the ballpark, not to be a rigorous benchmark).

```
workload        glyph      c      rust   ratio (glyph/c)
loop_sum         440ms   443ms   503ms        0.99x
list_append       35ms    29ms    44ms        1.21x
map_put_get      157ms   144ms   161ms        1.09x
```

Hardware: 11th Gen Intel Core i5-1135G7, 2026-09-09. `loop_sum` sums
`i % 7` over a runtime size (400M iterations); `list_append` appends 20M
int64s to a dynamic array and sums them; `map_put_get` puts+gets 2M string-key
entries into a 100-key churn. Reproduce with `bench/gen.sh` + `bench/run.sh`.

Notes:

- `loop_sum` is at parity with C — the loop, modulo and integer arithmetic emit
  the same code gcc would write by hand.
- `list_append` uses a refcounted growable buffer that doubles its capacity
  geometrically, so appends amortize to O(1) like C's `realloc` array; the
  runtime helpers are `static inline`, so the hot loop inlines to a plain
  store. The ~20% residual is the refcount and capacity bookkeeping.
- `map_put_get` rehashes geometrically (load factor ≤ 0.75) and keys built
  temporarily by `int_to_string`/`++`/`substring`... are freed right after the
  put/get/index call, so the string churn that used to be measured (and leak)
  is gone. At 1.09x it sits between C (fixed 256 buckets) and Rust's
  hashbrown.

## Limitations

- Runtime is unmanaged: `Result`/`Option` payloads are heap boxes freed via `drop(box)`;
  async handles and channels are not auto-released
- Generics cover functions only (no generic structs/enums/impl, no trait bounds);
  calling a generic function inside a generic body needs concrete types
- `List<T>`: `==`/`!=` works only for POD scalars; slices are copies
- `Map<String, V>`: iteration (`for k in m`) walks hash buckets, so key order
  is not insertion order; keys are `String` only
- The use-after-free / double-free detector is statement-flow based and does not
  track aliases: `let y = xs; xs.free(); y.len()` is not yet caught, and a free
  inside a `match` arm is treated as having happened after the `match`
- Concurrency: no GC; async generic functions unsupported; `select` arms must be
  `recv()`/`await` and its wait loop polls at ~1 ms granularity
- LSP server: diagnostics from lexer/parser/typechecker with full source spans,
  cursor hover and keyword completion; textDocumentSync = Full

## License

MIT