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
- Concurrency (pthreads): `@fn async`, lazy handles `Async<T>`, `spawn`/`await`,
  typed channels `Channel<T>(capacity)`, `send`/`recv`/`close`
- Built-in test framework: `@test`, asserts, `glyphc test`

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
| `test [-i f.glyph] [--compiler gcc] [--opt -O2]` | Run `@test` functions (scans `./src` without `-i`) |
| `glyphc --lsp` | Experimental LSP server |

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

## Limitations

- Runtime is unmanaged: `Result`/`Option` payloads are heap boxes freed via `drop(box)`;
  async handles and channels are not auto-released
- Generics cover functions only (no generic structs/enums/impl, no trait bounds);
  calling a generic function inside a generic body needs concrete types
- `List<T>`: `==`/`!=` works only for POD scalars; slices are copies
- `Map<String, V>`: iteration (`for k in m`) walks hash buckets, so key order
  is not insertion order; keys are `String` only
- Concurrency: no GC, `select`, or timeouts; async generic functions unsupported
- LSP server is experimental

## License

MIT