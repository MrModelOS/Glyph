//! MLIR-like IR layer (textual ns.tensor dialect) — submodules filled in
//! during the port: dialect, compiler, eval, fusion.

pub mod dialect;
pub mod mlir_compiler;
pub mod eval;
pub mod fusion;