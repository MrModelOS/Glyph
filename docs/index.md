# Glyph

Glyph is a small systems language that transpiles to readable GNU C. The
compiler also embeds NeuralScript (`.ns`), a typed tensor DSL with CPU/CUDA AOT
code generation.

## Start here

1. [Install `glyphc`](https://github.com/MrModelOS/Glyph#install).
2. Read the [English language reference](language_en.md).
3. Run `glyphc new hello` and `cd hello && glyphc build`.
4. For tensor graphs, read the [NeuralScript guide](nns.md).

The project is MIT-licensed and the compiler, LSP, tests, examples, and release
artifacts live in the same repository.
