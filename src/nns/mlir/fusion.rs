//! Kernel fusion pass — port of `ns/mlir/fusion.cpp`.
//!
//! Groups elementwise epilogues (activations, layer-norm) onto their
//! single-consumer GEMM producer into a single `FUSED` op.

use std::collections::HashMap;

use super::dialect::{FusedopGroup, MLIRFunction, MLIRInstr, MLIRModule, MLIROp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionKind {
    GemmActivation,  // GEMM -> single activation
    GemmLayernorm,   // GEMM -> layernorm
    GemmElementwise, // GEMM -> one elementwise binop (e.g. bias add)
    GemmChain,       // GEMM -> multiple elementwise tail ops
}

#[derive(Debug, Clone)]
pub struct FuseDecision {
    pub start: usize,      // index of the leader (MATMUL)
    pub end: usize,        // index of the last fused op (exclusive)
    pub kind: FusionKind,
    pub leader_result: String, // GEMM result id
    pub fuse_result: String,   // final result id of the fused run
}

/// Map a MLIROp to a stable textual op name (used in fused group descriptors).
fn op_name(op: MLIROp) -> &'static str {
    match op {
        MLIROp::Relu => "relu",
        MLIROp::LeakyRelu => "leaky_relu",
        MLIROp::Sigmoid => "sigmoid",
        MLIROp::Tanh => "tanh",
        MLIROp::Swish => "swish",
        MLIROp::Gelu => "gelu",
        MLIROp::Silu => "silu",
        MLIROp::Identity => "identity",
        MLIROp::Softmax => "softmax",
        MLIROp::Layernorm => "layernorm",
        MLIROp::ElementwiseBinop => "binop",
        _ => "op",
    }
}

pub struct FusionPass {
    decisions: Vec<FuseDecision>,
}

impl FusionPass {
    pub fn new() -> Self {
        FusionPass {
            decisions: Vec::new(),
        }
    }

    pub fn decisions(&self) -> &[FuseDecision] {
        &self.decisions
    }

    pub fn is_fusible_epilogue(op: MLIROp) -> bool {
        matches!(
            op,
            MLIROp::Relu
                | MLIROp::LeakyRelu
                | MLIROp::Sigmoid
                | MLIROp::Tanh
                | MLIROp::Swish
                | MLIROp::Gelu
                | MLIROp::Silu
                | MLIROp::Identity
                | MLIROp::Softmax
                | MLIROp::Layernorm
                | MLIROp::ElementwiseBinop
        )
    }

    fn use_count(fn_: &MLIRFunction, id: &str) -> usize {
        let mut uses = 0usize;
        for instr in fn_.instructions.iter() {
            for op in instr.operands.iter() {
                if op == id {
                    uses += 1;
                }
            }
        }
        uses
    }

    fn try_fuse(fn_: &MLIRFunction, start: usize) -> FuseDecision {
        let leader = &fn_.instructions[start];
        let leader_result = leader.result_id.clone();

        // The GEMM result must be consumed exactly once (by the first epilogue)
        // to be safely removed.
        if Self::use_count(fn_, &leader_result) != 1 {
            return FuseDecision {
                start,
                end: start,
                kind: FusionKind::GemmChain,
                leader_result: leader_result.clone(),
                fuse_result: leader_result,
            };
        }

        let mut end = start + 1;
        let mut consumed = leader_result.clone(); // value feeding the current candidate
        let mut kind = FusionKind::GemmActivation;
        let mut any = false;

        while end < fn_.instructions.len() {
            let cand = &fn_.instructions[end];
            if !Self::is_fusible_epilogue(cand.op) {
                break;
            }
            if cand.operands.is_empty() || cand.operands[0] != consumed {
                break;
            }

            let uses = Self::use_count(fn_, &cand.result_id);
            let is_last = end + 1 >= fn_.instructions.len()
                || !Self::is_fusible_epilogue(fn_.instructions[end + 1].op)
                || fn_.instructions[end + 1].operands.is_empty()
                || fn_.instructions[end + 1].operands[0] != cand.result_id;
            if uses > 1 || (!is_last && uses == 0) {
                break;
            }

            if cand.op == MLIROp::Layernorm {
                kind = FusionKind::GemmLayernorm;
            } else if cand.op == MLIROp::ElementwiseBinop && !any && end == start + 1 {
                kind = FusionKind::GemmElementwise;
            }

            consumed = cand.result_id.clone();
            end += 1;
            any = true;
        }

        if !any {
            return FuseDecision {
                start,
                end: start,
                kind: FusionKind::GemmChain,
                leader_result: leader_result.clone(),
                fuse_result: leader_result,
            };
        }
        FuseDecision {
            start,
            end,
            kind,
            leader_result,
            fuse_result: consumed,
        }
    }

    /// Run the pass in place; returns number of fusion groups introduced.
    pub fn run(&mut self, module: &mut MLIRModule) -> usize {
        self.decisions.clear();
        let mut fused_groups = 0usize;

        for fn_ in module.functions.iter_mut() {
            let mut out: Vec<MLIRInstr> = Vec::new();
            out.reserve(fn_.instructions.len());
            // Redirect any later consumer of an internal (fused-away) value.
            let mut redirect: HashMap<String, String> = HashMap::new();
            let mut fuse_of: HashMap<String, String> = HashMap::new();

            let count = fn_.instructions.len();
            let mut i = 0usize;
            while i < count {
                let instr = fn_.instructions[i].clone();

                if instr.op == MLIROp::Matmul {
                    let d = Self::try_fuse(fn_, i);
                    if d.end > d.start {
                        // Build the fused group descriptor.
                        let mut group = FusedopGroup::default();
                        group.result_id = d.fuse_result.clone();
                        group.c = d.leader_result.clone();
                        group.result_type = fn_.instructions[d.end - 1].result_type.clone();

                        for j in (d.start + 1)..d.end {
                            let ep = fn_.instructions[j].clone();
                            // For binops, use the actual operator symbol.
                            let nm = if ep.op == MLIROp::ElementwiseBinop {
                                ep.attribute.clone()
                            } else {
                                op_name(ep.op).to_string()
                            };
                            group.ops.push(nm);
                            for k in 1..ep.operands.len() {
                                group.epilogue_operands.push(ep.operands[k].clone());
                                if ep.op == MLIROp::ElementwiseBinop && ep.attribute == "+" {
                                    group.has_bias = true;
                                    group.bias_operand = ep.operands[k].clone();
                                }
                            }
                            // Redirect future consumers of this internal value.
                            fuse_of.insert(ep.result_id.clone(), d.fuse_result.clone());
                        }
                        fuse_of.insert(d.leader_result.clone(), d.fuse_result.clone());
                        redirect = fuse_of.clone();
                        module.fused_groups.push(group);
                        fused_groups += 1;

                        // Emit a single FUSED instruction.
                        let mut fused = MLIRInstr::new(MLIROp::Fused, d.fuse_result.clone());
                        fused.operands = instr.operands.clone(); // A, B from the MATMUL leader
                        for g in module.fused_groups.last().unwrap().epilogue_operands.iter() {
                            fused.operands.push(g.clone());
                        }
                        fused.result_type = fn_.instructions[d.end - 1].result_type.clone();
                        fused.comment = format!("fused {} ops", d.end - d.start);
                        out.push(fused);

                        i = d.end;
                        continue;
                    }
                }

                // Copy, rewriting operands that were fused away.
                let mut copy = instr;
                for op in copy.operands.iter_mut() {
                    if let Some(new_op) = redirect.get(op) {
                        *op = new_op.clone();
                    }
                }
                out.push(copy);
                i += 1;
            }

            fn_.instructions = out;
        }

        fused_groups
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}