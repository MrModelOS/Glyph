//! MLIR-like IR types — port of `ns/mlir/mlir_compiler.hpp` IR section.
//!
//! A simplified MLIR-style IR that captures the high-level tensor graph.
//! This represents the "ns.tensor" / "ns.layer" dialect before lowering.

use super::super::ast::{Dtype, TensorType};

/// Operation kinds of the ns.tensor / ns.layer dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLIROp {
    // Tensor ops
    TensorAlloc,      // allocate a tensor
    TensorFree,       // deallocate a tensor
    Matmul,           // matrix multiply (GEMM)
    ElementwiseBinop, // elementwise binary op (+,-,*,/)
    Activation,       // activation function
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Swish,
    Gelu,
    Silu,
    Identity,
    Softmax,
    Dropout,
    Layernorm,
    CrossEntropy, // loss function
    Concat,
    Reshape,
    Transpose,
    Slice,   // extract a range: operands {in}, attribute "axis:start:stop"
    Index,   // column/row gather: operands {in}, int_attr=axis, ints_attr=indices
    Scatter, // scatter updates: operands {in, upd}, int_attr=axis, ints_attr=indices
    Constant,

    // Reverse-mode (AOT backward pass) ops
    MatmulGradA,     // dA = dC @ B^T      (operands: dC, B)
    MatmulGradW,     // dB = A^T @ dC      (operands: A, dC, Bweight)
    ActivationGrad,  // elementwise jacobian backward (operands: dOut, actInput)
    LossGrad,        // loss seed: d(preds) from preds+labels (cross-entropy)
    BinopGrad,       // elementwise binop backward (operands: dC, lhs, rhs; attribute = operator)
    // (int_attr = 0 for lhs grad, 1 for rhs grad)

    // Control flow
    FnCall,   // function call
    Forward,  // forward pass
    Grad,     // gradient computation block
    OptStep,  // optimizer step

    // Layer declarations
    LayerDense,
    LayerDropout,
    LayerAttention,
    LayerEmbedding,
    LayerLayernorm,
    LayerMoe,

    // Layer backward ops
    LayernormGrad,

    // Embedding backward: scatter-add dOut rows into dW
    EmbeddingGradW,

    // MoE backward: input / router-weight / expert-weight gradients
    MoeGradX,
    MoeGradWg,
    MoeGradWe1,
    MoeGradWe2,

    // Multi-head attention backward ops
    AttentionGradX,
    AttentionGradWq,
    AttentionGradWk,
    AttentionGradWv,
    AttentionGradWo,

    // Storage
    AllocBuffer, // allocate GPU buffer
    FreeBuffer,  // free GPU buffer

    // Fused kernel (introduced by the fusion pass)
    Fused, // fused GEMM + (activation | layernorm | elementwise tail)
}

/// A fused group produced by the fusion pass: GEMM (A@B->C) followed by an
/// elementwise epilogue (activation / layernorm / binop chain) that has no
/// consumers other than the fusion boundary.
#[derive(Debug, Clone)]
pub struct FusedopGroup {
    pub result_id: String,            // final fused result value name
    pub c: String,                    // intermediate GEMM result (usually internal)
    pub ops: Vec<String>,             // epilogue op names: "relu","gelu","layernorm","+",...
    pub epilogue_operands: Vec<String>, // extra operands for binops
    pub result_type: TensorType,
    pub has_bias: bool,
    pub bias_operand: String,
}

impl Default for FusedopGroup {
    fn default() -> Self {
        FusedopGroup {
            result_id: String::new(),
            c: String::new(),
            ops: Vec::new(),
            epilogue_operands: Vec::new(),
            result_type: TensorType::new(Vec::new(), Dtype::Float32),
            has_bias: false,
            bias_operand: String::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MLIRValue {
    pub id: String,
    pub type_: TensorType,
    pub is_temporary: bool,
}

impl Default for MLIRValue {
    fn default() -> Self {
        MLIRValue {
            id: String::new(),
            type_: TensorType::new(Vec::new(), Dtype::Float32),
            is_temporary: false,
        }
    }
}

impl MLIRValue {
    pub fn new(id: String, type_: TensorType) -> Self {
        MLIRValue {
            id,
            type_,
            is_temporary: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MLIRInstr {
    pub op: MLIROp,
    pub result_id: String,
    pub operands: Vec<String>,
    pub result_type: TensorType,

    // Auxiliary data
    pub comment: String,
    pub attribute: String, // e.g., activation type, loss type, "axis:start:stop"
    pub int_attr: i64,
    pub float_attr: f64,
    pub ints_attr: Vec<i64>, // index lists for Index/Scatter
}

impl MLIRInstr {
    pub fn new(op: MLIROp, result_id: String) -> Self {
        MLIRInstr {
            op,
            result_id,
            operands: Vec::new(),
            result_type: TensorType::new(Vec::new(), Dtype::Float32),
            comment: String::new(),
            attribute: String::new(),
            int_attr: 0,
            float_attr: 0.0,
            ints_attr: Vec::new(),
        }
    }
}

impl Default for MLIRInstr {
    fn default() -> Self {
        MLIRInstr::new(MLIROp::Constant, String::new())
    }
}

#[derive(Debug, Clone, Default)]
pub struct MLIRFunction {
    pub name: String,
    pub instructions: Vec<MLIRInstr>,
    pub return_id: String,
    pub is_train: bool, // network train() method (forward+backward+opt)
}

#[derive(Debug, Clone, Default)]
pub struct MLIRModule {
    pub functions: Vec<MLIRFunction>,
    pub global_buffers: Vec<String>,
    // Fusion pass output: groups keyed by the fused result value id.
    pub fused_groups: Vec<FusedopGroup>,
}