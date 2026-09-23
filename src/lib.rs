//! glyphc — library root (canonical module tree).
//! The binary (`src/main.rs`) re-uses this tree via `use glyphc::…` to avoid
//! dual-root duplication (`mod` in both bin and lib).

pub mod ast;
pub mod cli;
pub mod codegen;
pub mod formatter;
pub mod generics;
pub mod lexer;
pub mod lsp;
pub mod modules;
pub mod nns;
pub mod parser;
pub mod typechecker;
