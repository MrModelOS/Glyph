// glyphc binary — thin wrapper over the library crate.
// All compiler modules live in `src/lib.rs` (canonical root); this binary
// simply calls `glyphc::cli::run()` so `cargo test` and `cargo check` share
// a single module tree and docs/tests can `use glyphc::…`.

use glyphc::cli::run;

fn main() {
    run();
}
