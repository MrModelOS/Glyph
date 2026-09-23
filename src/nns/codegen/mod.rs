//! Codegen: emits self-contained C++ (CPU) and CUDA sources.

// The nested namespace mirrors the source layout (`nns::codegen::codegen`).
#[allow(clippy::module_inception)]
pub mod codegen;
pub mod cpu_blobs;
pub mod cuda_backend;
