//! NeuralScript (`nns`) subsystem — the neural-network DSL compiler core,
//! ported 1:1 from the C++ «NeuralScript» compiler (nsc) into glyphc.
//!
//! Module tree mirrors the original C++ layout:
//! `lexer -> parser -> typechecker -> mlir -> optim/training -> codegen`,
//! plus a C-ABI runtime driver (`runtime`) spliced into emitted code.
//!
//! The pipe is: `.ns` source -> tokens -> AST -> shape-check -> MLIR
//! (textual IR + reverse-mode lowering) -> fusion -> C++/CUDA source.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod mlir;
pub mod optim;
pub mod parser;
pub mod runtime;
pub mod shape_checker;
pub mod token;
pub mod training;
pub mod type_system;

use std::fmt;

/// Errors raised by the nns pipeline (lexer/parser/shape/compile errors).
#[derive(Debug, Clone)]
pub struct NsError(pub String);

impl fmt::Display for NsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for NsError {}

impl From<String> for NsError {
    fn from(s: String) -> Self {
        NsError(s)
    }
}

impl From<&str> for NsError {
    fn from(s: &str) -> Self {
        NsError(s.to_string())
    }
}

pub type NsResult<T> = Result<T, NsError>;

/// Raise an nns error with a formatted message.
#[macro_export]
macro_rules! ns_error {
    ($($arg:tt)*) => {
        $crate::nns::NsError(format!($($arg)*))
    };
}

/// Format a double the way C++ `std::to_string(double)` does for the
/// optimizer constants that get spliced into emitted source (6 decimals).
pub fn fmt_float(v: f64) -> String {
    format!("{:.6}", v)
}
