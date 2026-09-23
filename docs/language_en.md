# Glyph Language Reference

Status: v2.1.0. This document describes the actually working syntax
(verified against `examples/` and tests). Undocumented constructs are not considered part of the language.

## Table of Contents

1. [File Structure](#1-file-structure)
2. [Types](#2-types)
3. [Variables and Constants](#3-variables-and-constants)
4. [Functions](#4-functions)
5. [Operators](#5-operators)
6. [Control Flow](#6-control-flow)
7. [Strings and Lists](#7-strings-and-lists)
8. [Structs and Methods](#8-structs-and-methods)
9. [Enums](#9-enums)
10. [Modules](#10-modules)
11. [Tests and Asserts](#11-tests-and-asserts)
12. [Concurrency](#12-concurrency)
13. [Standard Library](#13-standard-library)

---

## 1. File Structure

A file is a sequence of top-level declarations:

```glyph
@module examples.demo        // top-level module name
@use other.module;            // import another module

@const SCALE: Float64 = 1.5;

@struct Point { x: Float64, y: Float64 }

@enum Op { Plus, Minus, MulDiv(Float64, Float64) }

@fn twice(x: Int64) -> Int64 { return x * 2; }

@test @fn test_twice() -> Void { assert_eq(twice(2), 4, "2 * 2 == 4"); }
```

Attributes (written with `@`):

| Attribute | Applies to | Purpose |
|---|---|---|
| `@module` | file | module name |
| `@use` | file | module import |
| `@fn` | function | function declaration |
| `@struct` | struct | struct declaration |
| `@enum` | enum | enum declaration |
| `@const` | constant | named constant |
| `@pub` | fn/struct/enum | public visibility (for imported modules) |
| `@test` | fn | test function (see docs/testing.md) |

Comments: `// line`, `/* block */`.

---

## 2. Types

| Type | C representation | Notes |
|---|---|---|
| `Int64` | `int64_t` | signed integer |
| `UInt64` | `uint64_t` | unsigned |
| `Float64` | `double` | floating point |
| `Bool` | `int` | `true`/`false` |
| `String` | `char*` | strings |
| `Bytes` | `char*` | raw bytes |
| `List<T>` | `GlyphList` (refcounted fat-struct) | literal `[..]`, range `..`, `.len()`, `.append()`, `.free()`, slices, `for x in xs`, `++`, `==` for POD |
| `Option<T>` | `GlyphBox*` (C: `void*`) | box `{tag, data}`; `Some(x)`/`None` |
| `Result<T, E>` | `GlyphBox*` (C: `void*`) | box `{tag, data}`; `Ok(x)`/`Err(e)` |
| `&T` | `T*` | reference to a value (for enums in `match`) |
| `MyStruct`, `MyEnum` | struct/tagged union | user-defined types |

### `Result` / `Option` with payloads

`Result<Ok, E>` and `Option<T>` are built from variants and matched in `match`.
Payload is copied to the heap (box `{ tag, data }`); tag `1` = `Ok`/`Some`,
`0` = `Err`/`None`:

```glyph
@fn divide(a: Int64, b: Int64) -> Result<Int64, String> {
    if b == 0 {
        return Result::Err("division by zero");
    }
    return Result::Ok(a / b);
}

@fn main() {
    let q: Int64 = match divide(10, 2) {
        | Ok(v) => v,
        | Err(e) => -1
    };
    print_int(q);

    let parsed: Option<Float64> = parse_float("2.5");
    let f: Float64 = match parsed {
        | Some(v) => v,
        | None => 0.0
    };
    print_float(f);
}
```

Notes:
- `Option::None` / `Result::Ok(x)` without context yields an unknown parameter (`Type::Void`);
  in `let` it unifies with the declared type (`let n: Option<Int64> = Option::None`)
- `parse_int` / `parse_float` / `to_int` / `to_float` return boxed `Option`
  (`Some`/`None`) suitable for `match`

### Generic functions

Type parameters are declared in angle brackets after the function name and can
be used in the signature and body, including nested (`List<T>`, `Option<T>`,
`Result<T, E>`). The compiler monomorphizes each instantiation into a
separate C function (`identity_i64`, `first_str_i64`):

```glyph
@fn identity<T>(x: T) -> T {
    return x;
}

@fn first<T, K>(a: T, b: K) -> T {
    return a;
}

@fn main() {
    let a: Int64 = identity(5);          // T = Int64 from argument
    let s: String = first("hi", 42);     // T = String, K = Int64
    print_int(a);
}
```

Type inference:
- from call argument types (including nested: `List<Int64>` gives `T = Int64`);
- from `let` annotation for parameters appearing only in the return type
  (`let r: Result<String, Int64> = ok_wrap("fine")` infers `E = Int64`).

Limitations in v2.1: only functions are generic (structs/enums/impl are
concrete); explicit type arguments (`f::<Int64>`) are not supported;
calling a generic function inside a generic body requires concrete types
(otherwise — error `Cannot infer type arguments`); type `[T; N]` is described
but the literal always produces `List<T>`, so arrays are not constructible.

Literals:

```glyph
42         // Int64
498_351_8   // digit separators
3.14       // Float64
true false // Bool
"hello"    // String
0xFF       // hex
```

---

## 3. Variables and Constants

```glyph
let x: Int64 = 42;            // immutable
let mut acc: Float64 = 0.0;   // mutable
let y = 42;                   // untyped let: type inferred (Int64, String, List<T>, ...)
let r = 0..5;                 // List<Int64]
let s = "a" ++ "b";           // String

x = 10;        // ❌ type error: x is immutable
acc = acc + 1; // ✅

@const PI: Float64 = 3.141592653589793;
@const MAX_RETRIES: Int64 = 3;
```

Untyped `let` infers the type from the value (literals, `List`, `Range`,
concatenation, calls). Unknown type is a codegen error
(`cannot infer type of untyped let`), not garbage in C.

Constants compile to `static const` expressions and may reference other
constants and literals.

---

## 4. Functions

```glyph
@fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

@fn log_message(msg: String) -> Void { println(msg); }

@fn default() -> Int64 { 0 }   // expression body
@fn make() -> Float64 { return 2.0; }
```

- Parameters are typed; return type is indicated via `->`, `Void` means no value.
- Arguments are passed by value; references `&T` are available.
- Unqualified calls inside the same module work as-is.
- `@fn async` (and `@fn async` methods in `@impl`) returns a lazy handle —
  see "Concurrency" below. Async-generic functions are not supported.

## 5. Operators

Arithmetic: `+ - * / %`.

Comparisons: `== != < > <= >=`. Logic: `&& || !`.

String concatenation: `++` — `"Hello, " ++ "World!"`. When concatenating
with numbers, use conversion: `"x=" ++ int_to_string(x)`.

Type cast `as`: `let f: Float64 = 42 as Float64;`.

Reference `&`: `let r: &Shape = &circle;` (used for `match` by value
without copying).

## 6. Control Flow

### if / else

```glyph
if x > 0 {
    print("positive");
} else if x < 0 {
    print("negative");
} else {
    print("zero");
}
```

`if` is also an expression: when both branches evaluate to a value of the same
scalar type (`Int64`, `UInt64`, `Float64`, `Bool`), the whole `if/else` has that
type and can be assigned. An integer literal branch promotes to a sibling's
`Float64`/`UInt64` type. Heap-typed branches (`String`, `List<T>`, structs) are
not supported as `if` expression values.

```glyph
let hi: Int64 = if coins > 100 { 2; } else { 1; };
let scale: Float64 = if fast { 1.5; } else { 1; };
```

### while

```glyph
let i: Int64 = 0;
while i < 5 {
    i = i + 1;
}
```

### loop / break / continue

```glyph
let i: Int64 = 0;
loop {
    i = i + 1;
    if i >= 100 { break; }
}
```

### for .. in (integer range)

```glyph
let sum: Int64 = 0;
for i in 0..5 {        // 0..5 — excluding 5 → sum 10
    sum = sum + i;
}
for j in 1..=5 {       // 1..=5 — inclusive
    sum = sum + j;
}
```

### match

```glyph
match value {
    | 1 => print("one")
    | 2 => print("two")
    | _ => print("other")
}

match status {
    | HttpStatus::Ok => "OK"
    | HttpStatus::NotFound => "Not Found"
    | _ => "Unknown"
}

match shape {
    | Shape::Circle(r) => 3.14159 * r * r
    | Shape::Rectangle(w, h) => w * h
    | Shape::Point => 0.0
}
```

Patterns: literals, `Enum::Variant`, `Enum::Variant(bindings)`, `_` (wildcard).
Arm bodies are expressions (including calls terminated with `;`).

### guard

```glyph
@fn divide(a: Float64, b: Float64) -> DivisionResult {
    #guard(b != 0.0) else {
        return DivisionResult::DivisionByZero;
    };
    return DivisionResult::Success(a / b);
}
```

`#guard(condition) else { ... };` — if the condition is false, the `else`
block executes and the current function returns; `;` after `else` is mandatory.

## 7. Strings and Lists

```glyph
let greeting: String = "Hello, " ++ "Glyph!";
let n: Int64 = len(greeting);                // 13
let sub: String = substring(greeting, 0, 5); // "Hello"
let has: Bool = contains(greeting, "Glyph"); // true
let upper: String = to_upper(greeting);      // "HELLO, GLYPH!"
let words: String = split("a,b,c", ",");     // CSV fragment
let num: Option<Int64> = parse_int("42");    // Option<Int64>
let s: String = int_to_string(42);

let arr: List<Int64> = [10, 20, 30];          // list literal
print_int(arr[1]);                            // 20 — indexing
let points: List<P> = [P { x: 1.0, y: 2.0 }];
print_float(points[0].y);                     // 2.0 — struct indexing
let range: List<Int64> = 0..10;   // [0, 1, ..., 9]
print_int(range.len());           // 10 — length
range.append(10);                 // buffer growth
let both: List<Int64> = arr ++ range;  // list concatenation
for x in arr {                    // iteration
    print_int(x);
}
```

Lists are fat-structs `{data, len, elem_size, refs}` over a heap buffer:
literals and ranges allocate, `append` grows the buffer, indexing `arr[i]`
returns element `T`. Copying a list (`let b = a`, passing to a function)
bumps the refcount: the buffer lives while at least one copy exists, so
`a.free(); b[0]` is safe. `append` with multiple owners does copy-on-write
(separate buffer for this owner; `append` inside a function is not visible
outside). The element type of a literal is inferred from the first element.
`let` without annotation also infers the type (`let xs = [1, 2]` gives `List<Int64>`).

Slices `xs[a..b]` / `xs[a..=b]` return a new `List<T>` (range copy, bounds
clamped). POD-scalar lists (`Int64`, `UInt64`, `Float64`, `Bool`) compare by
value (`==`/`!=` via `len` + `memcmp`); others are a loud codegen error.
`xs.free()` releases one ref and zeroes the list (buffer stays while other
copies exist).

### 7.1. Map

`Map<String, V>` — chained hash map (FNV-1a, key always `String`), values
copied by `elem_size(V)`. Literal `#{ "key": value, ... }` builds a map;
value type is inferred from the first pair (empty → `Map<String, Void>`).

```glyph
@fn main() -> Void {
    let m: Map<String, Int64> = #{ "alice": 90, "bob": 75 };
    print_int(m["alice"]);          // read; missing key — abort
    m["carol"] = 60;                // write via index
    m.put("bob", 80);               // write via method
    let got: Option<Int64> = m.get("dave");   // Option instead of abort
    let v: Int64 = match got {
        | Some(x) => x
        | None => 0
    };
    drop(got);                      // Option box requires drop(box)
    print_int(m.len());

    for k in m {                    // iteration yields keys (String)
        print(k);
    }

    m.free();                       // free storage
}
```

Note: `m["k"]` on a missing key aborts — for checked access use `m.get(k)`
(returns `Option<V>`; `Some` box is freed via `drop(box)`). Iteration
`for k in m` walks bucket chains (order = hash, not insertion order) and yields
keys; values are read via `m[k]`.

### 7.2. Static use-after-free protection

The compiler tracks `v.free()` calls for `List` and `Map` in statement flow
and rejects subsequent use of `v` until reassignment:

```glyph
let xs: List<Int64> = [1, 2, 3];
xs.free();
xs.append(4);        // ERROR: use after free: 'xs'
xs.free();           // ERROR: double free: 'xs'
```

Legal path — reassign after `free()` (slot revives):

```glyph
xs.free();
xs = [4, 5];         // reassignment resets the flag
xs.append(6);
```

Analyzer limitations: statement-flow, per-function — does not track aliases
(`let y = xs; xs.free(); y.len()` is not caught), and `free()` inside a
`match` arm is treated as executed after the `match`. Coverage: reads,
indexing, iteration, argument passing, and repeated `.free()`.

## 8. Structs and Methods

```glyph
@struct Point {
    x: Float64,
    y: Float64,
}

let p: Point = Point { x: 0.0, y: 0.0 };
let dx: Float64 = p2.x - p1.x;
```

### Methods via `@impl`

`@impl <Type> { @fn ... }` declares methods for a type. The receiver is
passed as the **first parameter**; call `obj.method(a, b)` desugars to
`Type_method(obj, a, b)`.

```glyph
@impl Point {
    @fn norm(p: Point) -> Float64 {
        return sqrt(p.x * p.x + p.y * p.y);
    }
    @fn scaled(p: Point, k: Float64) -> Point {
        return Point { x: p.x * k, y: p.y * k };
    }
}

let p: Point = Point { x: 3.0, y: 4.0 };
let n: Float64 = p.norm();          // → Point_norm(p)
let q: Point = p.scaled(2.0);       // → Point_scaled(p, 2.0)
```

Rules:
- first method parameter must have type (or `&`-reference to) `Type` —
  same as the call object;
- remaining parameters are regular arguments; `args.len() == params.len() - 1`;
- method body accesses receiver fields via the parameter (`p.x`), no `self`;
- methods from other modules are not prefixed (§10), method names are
  `Type_method`; methods cannot be imported as functions.

## 9. Enums

```glyph
@enum Color { Red, Green, Blue }              // unit variants

@enum Shape {
    Circle(Float64),                          // variant with data
    Rectangle(Float64, Float64),
    Point,
}

@enum HttpStatus {                            // discriminants
    Ok = 200,
    NotFound = 404,
}
```

Construction: `Shape::Circle(5.0)`, `HttpStatus::Ok`, `Color::Red`.

Matching:

```glyph
match s {
    | Shape::Circle(r) => 3.14159 * r * r
    | Shape::Rectangle(w, h) => w * h
    | Shape::Point => 0.0
}
```

**`::` parsing rule**: a `path` segment starting with uppercase is an enum
constructor (`Shape::Circle(...)`); starting with lowercase — a qualified
module function call (`math::add(...)`).

## 10. Modules

```glyph
// math.glyph  —— module
@module my.math
@pub @fn add(a: Int64, b: Int64) -> Int64 { return a + b; }

// main.glyph —— entry
@use my.math;
@fn main() -> Void {
    let sum: Int64 = my.math::add(1, 2);
}
```

- Entry file (the one passed to `glyphc run/compile`) is not prefixed.
- Imported module names are prefixed (`add` → `my_math_add`); references
  to own functions inside the module are rewritten automatically.
- `@pub` — public visibility; by default symbols are private, but in v2.1 the
  marking is not a hard restriction.

## 11. Tests and Asserts

See [docs/testing.md](testing.md).

## 12. Concurrency

Calling `@fn async` does not execute the body but returns a lazy handle `Async<T>`.
`spawn h;` enqueues the task in the shared FIFO worker pool (idempotent);
the pool is `M:N` (M tasks on N threads), N = core count, capped at 8,
overridable via `GLYPH_WORKERS=K`. `h await` waits for the result: executes
synchronously if the handle has not yet been launched; otherwise waits on a
condvar. The waiting thread "helps" the pool — while the target is not ready
it drains the queue and executes other tasks (work-sharing), so nested `await`
inside async functions does not deadlock the pool. Repeated `await` returns the
cached value:

```glyph
@fn async fetch(url: String) -> String {
    return url;
}

@fn main() {
    let h: Async<String> = fetch("http://x");
    spawn h;
    println(h await);
}
```

`List`/`Map` across worker boundaries are passed by value-struct:
the trampoline puts the struct in the cell, and each `await` deep-copies it
(`GlyphList`/`GlyphMap` memcpy data + fresh refcount). So the handle keeps a
private result instance: mutation and `free()` of the value obtained via
`await` are safe and only corrupt that copy; repeated `await` yields the
original value again. (Handle result cell is not freed — same leak class as
`Int64`/`String`: runtime without GC.)

Typed channels are created with `Channel<T>(capacity)`:
capacity `0` — rendezvous (direct handoff without buffer), `> 0` — buffered.
`send(v)` blocks when the buffer is full and returns `false` if the channel is closed;
`recv()` blocks when empty and returns `None` when the channel is closed
and empty; `close()` wakes all waiters:

```glyph
@fn async produce(ch: Channel<Int64>) -> Int64 {
    ch.send(1);
    ch.send(2);
    ch.close();
    return 2;
}

@fn main() {
    let ch: Channel<Int64> = Channel<Int64>(4);
    spawn produce(ch);
    let m: Option<Int64> = ch.recv();
    let v: Int64 = match m {
        | Some(x) => x,
        | None => -1
    };
    print_int(v);
}
```

### select: waiting for the first ready event

`select { ... }` waits for one of several events and executes the matching
block. An arm is either `| name: Type <- channel.recv()`, or
`| name: Type <- handle await`, or `| timeout(ms)`, or `| default`.
The first source in order that already has data fires;
for `recv` from a buffered channel and for `await`, task submission to the pool
happens automatically. `timeout(ms)` fires if no event becomes ready within
`ms` milliseconds; `default` — if no ready event exists right now.
`timeout` and `default` cannot be specified together (in each `select`),
and there cannot be multiple `timeout` arms. Waiting logic polls at ~1 ms
steps, so timeouts are honest and the worker pool keeps running:

```glyph
@fn async worker(ch: Channel<Int64>) -> Int64 {
    ch.send(42);
    ch.close();
    return 42;
}

@fn main() {
    let ch: Channel<Int64> = Channel<Int64>(1);
    let h: Async<Int64> = worker(ch);
    select {
        | v: Int64 <- ch.recv() => print_int(v),
        | v: Int64 <- h await => print_int(v),
        | timeout(100) => print_int(-1)
    }
}
```

`recv` from a closed and drained channel is immediately ready and returns
the zero value of the type (analogous to `None`).

Limitations: runtime without GC (handles, channels and lists are not freed);
`Option`/`Result` boxes: temporaries (call result directly in `match`)
are freed automatically, named ones via `drop(box)`; `select` arm must be
exactly `recv()` or `await` (arbitrary expressions including `recv_timeout`
are not allowed); threads are not preempted at task level — a long compute
at the task root occupies the worker entirely (via nested `await` the worker
switches to other tasks); async-generic functions are forbidden;
`send(&ref)` is forbidden by type (stack address must not be passed
between threads). `List`/`Map` values in a channel are passed as struct copies
with shared buffer (COW, like `String`): after `send` the sender must not
reuse or free the value — the receiver owns it.

## 13. Standard Library

The library is available without `@use`; signatures are declared in the
compiler, bodies in `stdlib/`.

```glyph
print(msg: String) -> Void
println(msg: String) -> Void
eprintln(msg: String) -> Void
print_int(value: Int64) -> Void
print_float(value: Float64) -> Void
print_bool(value: Bool) -> Void
read_line() -> String

sqrt(x: Float64) -> Float64
pow(base: Float64, exp: Float64) -> Float64
abs(x: Float64) -> Float64
floor(x: Float64) -> Int64
ceil(x: Float64) -> Int64
round(x: Float64) -> Int64
min(a: Float64, b: Float64) -> Float64
max(a: Float64, b: Float64) -> Float64
clamp(value: Float64, low: Float64, high: Float64) -> Float64
log(x: Float64) -> Float64      log2(x: Float64) -> Float64
log10(x: Float64) -> Float64    sin(x: Float64) -> Float64
cos(x: Float64) -> Float64      tan(x: Float64) -> Float64

len(s: String) -> Int64
substring(s: String, start: Int64, end: Int64) -> String
contains(s: String, sub: String) -> Bool
starts_with(s: String, prefix: String) -> Bool
ends_with(s: String, suffix: String) -> Bool
replace(s: String, from: String, to: String) -> String
split(s: String, delimiter: String) -> String
trim(s: String) -> String
to_upper(s: String) -> String    to_lower(s: String) -> String
char_at(s: String, index: Int64) -> String
parse_int(s: String) -> Option<Int64>
parse_float(s: String) -> Option<Float64>
int_to_string(value: Int64) -> String
float_to_string(value: Float64) -> String
bool_to_string(value: Bool) -> String

to_int(s: String) -> Option<Int64>
to_float(s: String) -> Option<Float64>
to_bool(s: String) -> Option<Bool>

file_read(path: String) -> String
file_write(path: String, content: String) -> Bool
file_append(path: String, content: String) -> Bool
file_exists(path: String) -> Bool
create_dir(path: String) -> Bool
remove_file(path: String) -> Bool
list_dir(path: String) -> String
file_copy(src: String, dst: String) -> Bool
file_rename(old: String, new: String) -> Bool

alloc(size: Int64) -> Bytes
free(ptr: Bytes) -> Void
memcpy(dst: Bytes, src: Bytes, size: Int64) -> Void
memset(dst: Bytes, value: Int64, size: Int64) -> Void

assert(condition: Bool, msg: String) -> Void
assert_eq(a: Int64|UInt64|Float64|Bool|String, b: ..., msg: String) -> Void
assert_ne(a: ..., b: ..., msg: String) -> Void
assert_true(condition: Bool, msg: String) -> Void
assert_false(condition: Bool, msg: String) -> Void
```
