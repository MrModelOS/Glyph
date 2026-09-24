# Changelog

## [Unreleased]

### Public-release preparation

- Added the official [Glyphc philosophy](docs/philosophy.md), including the
  Glyph/NeuralScript roles, AI-readable and hardware-first principles, current
  capability boundaries, and product roadmap.
- Added `glyphc new`/`init`, a release installer, Dockerfile, and reproducible
  `cargo fmt`/Clippy/docs CI checks.
- Added English language/NNS/editor/comparison documentation, issue templates,
  contribution/security policies, Dependabot, and cross-platform release archives.
- Made `nns` output/errors robust, added `-o` and an explicit CPU SIMD target,
  `lsp` as a real subcommand, and rejected unsupported fp8/fp4/float16 code
  generation instead of emitting misleading types.
- Fixed `glyphc run`/`test` temporary-directory collisions and made the example
  suite self-consistent after the NeuralScript merge.
- Corrected the transformer example to use token IDs (`Tensor[Seq]`) consistently
  with its Embedding layer; all shipped NNS examples now pass strict CI checks.
- Made generic and constant C emission deterministic across runs and added a
  regression test for reproducible output.

## [2.1.0] — 2026-09-22

### NeuralScript (nns) merged

- Ported NeuralScript C++ implementation to Rust with byte-identical codegen.
- Merged into `glyphc` as `glyphc nns` — single binary, no separate toolchain.
- Backends: `--cpp` and `--cuda` with `--runtime` header emission; `--check` for typecheck-only.
- Examples: `glyphc nns examples/nns/mlp.ns --cpp --runtime | gcc -x c - -o a.out && ./a.out`, `glyphc nns file.ns --cuda --runtime`, `glyphc nns file.ns --check`.

## [2.0.0] — 2026-09-10

### If/else as an expression

- `if/else` branches unified to a common scalar type (`Int64`, `UInt64`,
  `Float64`, `Bool`); the whole chain is a value and can be assigned:

  ```glyph
  let hi: Int64 = if coins > 100 { 2; } else { 1; };
  let scale: Float64 = if fast { 1.5; } else { 1; };
  ```

- Integer literal branches promote to a sibling's `Float64`/`UInt64` type;
  incompatible or non-scalar branches (e.g. `String`) are rejected with a clear
  type error instead of a C-level failure.
- Value-producing codegen via a temp statement-expression; statement-position
  `if..else if..return` chains keep their tolerant checking.
- Fixed a pre-existing codegen bug where the implicit return of a non-`Void`
  function wrapped a `Void` expression in `return (..);` (visible as a GCC error
  when the last statement was a `print_int`/`print`/`read_line` call).

### LSP: go-to-definition

- `definitionProvider` for functions, structs, parameters (including `@impl`
  methods), `let` bindings and `for`-loop variables; binding name spans are
  tracked in the AST and resolved to the definition nearest before the cursor.
- `Document` keeps the last successful parse to serve definitions across edits.

### Fixes & polish

- `resolved_expr_type` extended to `If`/`Match`/`Block` so implicit returns and
  value-flow analysis stay correct for trailing expression statements.

Regression base at this release: 83 unit tests, 26/26 examples.

## [1.3.0] — 2026-09-09

- `glyphc fmt`: comment-preserving canonical indentation with `--check`/`--write`.
- Benchmark harness (`bench/`) comparing Glyph vs identical C and Rust workloads.