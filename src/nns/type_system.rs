//! Type-system helpers — port of `ns/typechecker/type_system.cpp`.

use super::ast::Dtype;

/// Highest-precedence dtype promotion for elementwise binary ops.
/// e.g. float16 + float32 -> float32 (and so on).
pub fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
    // Simple ranking: Bool < Int-ish < FP8 < FP16 < FP32 < FP64
    fn rank(d: Dtype) -> i32 {
        match d {
            Dtype::Bool => 0,
            Dtype::Int8 => 1,
            Dtype::Int16 => 2,
            Dtype::Int32 => 3,
            Dtype::Int64 => 4,
            Dtype::Fp4 => 5,
            Dtype::Fp8 => 6,
            Dtype::Float16 => 7,
            Dtype::Float32 => 8,
            Dtype::Float64 => 9,
        }
    }
    if rank(a) >= rank(b) {
        a
    } else {
        b
    }
}