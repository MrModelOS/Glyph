# Contributing to Glyph

Thanks for your interest in contributing!

## Getting started

```bash
git clone https://github.com/MrModelOS/Glyph.git
cd Glyph
cargo build
cargo test
examples/run_all.sh
```

Requires Rust 1.85+ and `gcc` (or `clang`) on `PATH`.

## Workflow

1. Fork and create a feature branch (`git checkout -b feat/my-change`).
2. Make changes with tests where practical.
3. Ensure CI passes locally:
   ```bash
   cargo fmt --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test --locked
   ```
4. Open a pull request against `main` using the PR template.

## Code style

- Run `cargo fmt` before committing.
- Keep public APIs documented; prefer small focused PRs.
- Add examples under `examples/` for language features when relevant.

## Reporting issues

Use the bug report / feature request templates in `.github/ISSUE_TEMPLATE/`.

## License

By contributing you agree that your contributions will be licensed under the MIT License.
