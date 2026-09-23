# Glyph vs Zig / Odin / Mojo / tinygrad — Honest Comparison

> Scope: where Glyph actually sits among its closest neighbours. Not a
> benchmark shootout and not FUD — each tool optimises for a different
> trade-off. Read the per-system notes before picking.

## Summary table

| Dimension | Glyph | Zig | Odin | Mojo | tinygrad |
|-----------|-------|-----|------|------|----------|
| **Primary goal** | Small, C-transpiled application language with an integrated tensor DSL (.ns) and M:N concurrency | Systems language, explicit control, no hidden allocations | Pragmatic systems language (Jai heritage), batteries-included | Python-superset for AI, CPU/GPU kernels, Python interop | Minimal deep-learning framework (Python), <10k LOC |
| **Runtime model** | No GC, refcounted `List`/`Map`, `drop(box)` for `Option`/`Result`; async handles/channels not auto-freed | Manual memory, no GC, explicit allocators | Manual + optional allocators, no GC | Borrowed Python runtime; Mojo structs can be `__del__`-managed | Python runtime, lazy UOp graph |
| **Compilation** | Transpiles to C (GNU statement exprs) → `gcc`/`clang` (`-O2`, `-pthread`) | LLVM via own toolchain, self-hosted | LLVM | MLIR → LLVM / custom GPU codegen | Python JIT → LLVM / METAL / CUDA / CL / HIP |
| **Concurrency** | `M:N` worker pool, `Async<T>`, `spawn`/`await`, `Channel<T>`, `select { timeout/default }` | `std.Thread`, atomics, no built-in async | `core:thread`, `core:sync` | Mojo async (evolving), Python `asyncio` interop | Single-threaded Python loop + device queues |
| **Type system** | Static, monomorphized generics on functions only, `Result`/`Option` as boxed unions | `comptime` generics, error unions, optionals | Parametric polymorphism, `Maybe` / `Error` | Strong + grad. typing, traits, ownership | Python dynamic + shape tracking |
| **Tensor / ML story** | First-class: `.ns` (`Tensor[Dims]`, `network`/`layer`/`forward`/`train{grad{}}`, shape checker, fused MLIR → C++/CUDA, C-ABI with MoE + checkpoint) | Libraries only (no DSL) | Libraries only | First-class: `Tensor`, `SIMD`, `autotune`, MAX Engine, GPU | First-class: tinygrad is the framework |
| **Tooling** | `glyphc check/tokens/ast/run/build/test/fmt/nns --check/--cpp/--cuda/--runtime/--fp16` + `glyphc lsp` (hover, completion, diagnostics) | `zig build`, LSP (`zls`), `zig fmt` | `odin build`, `odlsp` | `mojo build/run`, `mojo` LSP, `pixi` | `python -m pytest`, `TINY_BACKEND` env |
| **C interop** | Generated C is plain `gnu11`; host links directly against `ns_runtime.h` | `@cImport`, `@extern` | `foreign import` | `external_call`, C FFI | `ctypes` / `cffi` via Python |
| **Maturity** | v2.1, single `glyphc` binary, 100 unit + 25 integration tests + `examples/run_all.sh`; former `nsc` merged in | Production — self-hosted compiler, many users | Production (used at Janga, FMOD) | Public preview → GA track; breaking changes still happen | Production for hobby/research, used in prod by some |
| **Best fit** | You want one binary that compiles both application logic and a parameterised model with compile-time shape checks and a C/CUDA host loop | You want a C replacement with comptime and tight control | You want a concise C replacement with pragmatic stdlib | You want Python-compatible AI code that lowers to fast kernels | You want to read/hack the whole DL stack in Python |

---

## Per-system notes

### Glyph

Strengths: single `glyphc` binary covers app + model; `.ns` catches rank/dim
errors before codegen (symbolic + `Dynamic` unify, MoE liveness at runtime);
emitted kernels are plain C++/CUDA that link against a minimal `ns_runtime.h`
(C-ABI version 1.2.0) — see `docs/nns.md` and `examples/nns/host.cpp`.

Trade-offs: runtime is unmanaged (no GC) — `xs.free()` and `drop(box)` are
manual and the use-after-free detector is statement-flow only (aliases inside
`match` etc. not tracked). Generics only on functions (no generic structs/enums,
no trait bounds; `f::<T>` not supported). `Map` iterates in hash order, keys are
`String` only. `select` polls at ~1 ms, async-generic `fn` unsupported.
LSP is Full-sync, per-file diagnostics.

### Zig

Strengths: strongest story for explicit, auditable systems code; `comptime` is
more expressive than Glyph's monomorphization; error handling via error unions
is ergonomic; allocator-aware stdlib.

Trade-offs: no built-in async or channels; no tensor DSL — all ML is via
external libs. Build system and language churn faster than C. No GC either,
but memory errors are caught differently (no statement-flow alias tracking for
collections — you just use allocators).

When to prefer Zig: you need resource-constrained, allocation-transparent
systems code and do not need in-language ML.

### Odin

Strengths: very pragmatic, small language surface, good C FFI, concise.
Comparable C-replacement niche to Zig but with different ergonomics.

Trade-offs: similar to Zig — no built-in tensor/ML, no M:N async, no `select`
multiplexer. Generics are limited functors/procedures.

When to prefer Odin: you value brevity and a stdlib that feels like a game
engine toolkit.

### Mojo

Strengths: Python syntax + systems speed; owns the full AI stack (SIMD,
autotune, GPU) with Python interop. For pure ML workloads it has a wider
kernel library than Glyph's `.ns`.

Trade-offs: larger runtime/toolchain, faster-moving spec, ecosystem still
stabilising. Module/trait system is more general than Glyph's `@module`/`@use`
prefix scheme but heavier for tiny programs.

When to prefer Mojo: the workload is AI-centric and you need Python interop or
MAX Engine. Glyph wins when the program is a small native binary with a
parameterised model and minimal dependencies.

### tinygrad

Strengths: smallest DL stack you can read end-to-end; excellent for research,
custom backends, and hacking the UOp graph. Python-native.

Trade-offs: it is a library, not a compiled language — performance and
deployment depend on Python + the chosen backend. No Glyph-style static shape
checker or generated C++ host; no `glyphc fmt/test/lsp`.

When to prefer tinygrad: you want to own the framework, in Python.

---

## Guidance

- **App + embedded model, one binary, static shapes, C/CUDA host** → Glyph.
- **Systems code with comptime and no hidden costs** → Zig.
- **Game/tools systems code, pragmatic stdlib** → Odin.
- **Python-compatible AI with kernels + interop** → Mojo.
- **Minimal Python DL framework to read/hack** → tinygrad.

No system here is a superset of another; pick the one whose primary goal
matches yours. Contributions that improve the `glyphc --lsp`, `.ns` fusion
passes, and the `Map`/`List` runtime are especially welcome — see
`docs/language_en.md` and `docs/nns.md`.
