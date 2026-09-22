#![allow(dead_code)]
//! Numeric interpreter over the MLIR module — port of `ns/mlir/eval.cpp`.
//!
//! Used by the fusion test to prove that the fused module computes the
//! identical result as the unfused one.

use std::collections::HashMap;

use super::dialect::{FusedopGroup, MLIRInstr, MLIRFunction, MLIRModule, MLIROp};

/// A concrete tensor buffer for evaluation.
#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub data: Vec<f64>,
    pub shape: Vec<i64>, // dims; empty => scalar
}

impl TensorBuffer {
    pub fn numel(&self) -> i64 {
        if self.shape.is_empty() {
            return 1;
        }
        self.shape.iter().product()
    }
}

/// Minimal numeric interpreter over the MLIR module.
pub struct ModuleEvaluator {
    values: HashMap<String, TensorBuffer>,
    scalars: HashMap<String, f64>,
    errors: Vec<String>,
}

impl ModuleEvaluator {
    pub fn new() -> Self {
        ModuleEvaluator {
            values: HashMap::new(),
            scalars: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn bind(&mut self, id: &str, data: Vec<f64>, shape: Vec<i64>) {
        self.values.insert(id.to_string(), TensorBuffer { data, shape });
    }

    /// Evaluate an instruction list starting at `from`, returning the value of
    /// `want` (or fn.return_id if empty). Uses module.fused_groups to interpret
    /// FUSED instructions.
    pub fn run(&mut self, fn_: &MLIRFunction, mod_: &MLIRModule, want: &str, out: &mut TensorBuffer) -> bool {
        self.errors.clear();
        for instr in fn_.instructions.iter() {
            if !self.eval_instr(instr, mod_) {
                return false;
            }
        }
        let want_id = if want.is_empty() {
            fn_.return_id.clone()
        } else {
            want.to_string()
        };
        match self.values.get(&want_id) {
            Some(v) => {
                *out = v.clone();
                true
            }
            None => {
                self.errors.push(format!("no value for {}", want_id));
                false
            }
        }
    }

    pub fn values(&self) -> &HashMap<String, TensorBuffer> {
        &self.values
    }

    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    fn getv(&self, id: &str) -> Option<&TensorBuffer> {
        self.values.get(id)
    }

    fn elementwise(a: &TensorBuffer, b: Option<&TensorBuffer>, op: &str, out: &mut TensorBuffer) {
        out.shape = a.shape.clone();
        out.data.resize(a.data.len(), 0.0);
        for (i, av) in a.data.iter().enumerate() {
            let rhs = match b {
                Some(r) if !r.data.is_empty() => r.data[i % r.data.len()],
                _ => 0.0,
            };
            out.data[i] = match op {
                "+" => av + rhs,
                "-" => av - rhs,
                "*" => av * rhs,
                "/" => av / rhs,
                _ => *av,
            };
        }
    }

    fn activation(act: &str, m: &mut TensorBuffer) {
        for v in m.data.iter_mut() {
            match act {
                "relu" => *v = if *v > 0.0 { *v } else { 0.0 },
                "leaky_relu" => *v = if *v > 0.0 { *v } else { 0.01 * *v },
                "sigmoid" => *v = 1.0 / (1.0 + (-*v).exp()),
                "tanh" => *v = v.tanh(),
                "swish" => *v = *v / (1.0 + (-*v).exp()),
                "gelu" => *v = 0.5 * *v * (1.0 + libm_erf(*v / 2f64.sqrt())),
                "silu" => *v = *v / (1.0 + (-*v).exp()),
                _ => {}
            }
        }
    }

    fn layernorm(m: &mut TensorBuffer) {
        let last = *m.shape.last().unwrap_or(&1);
        if last <= 0 {
            return;
        }
        let rows = m.numel() / last;
        for r in 0..rows {
            let mut mean = 0.0;
            let mut var = 0.0;
            for j in 0..last {
                mean += m.data[(r * last + j) as usize];
            }
            mean /= last as f64;
            for j in 0..last {
                let d = m.data[(r * last + j) as usize] - mean;
                var += d * d;
            }
            var /= last as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..last {
                let idx = (r * last + j) as usize;
                m.data[idx] = (m.data[idx] - mean) * inv;
            }
        }
    }

    fn softmax(m: &mut TensorBuffer) {
        let last = *m.shape.last().unwrap_or(&1);
        if last <= 0 {
            return;
        }
        let rows = m.numel() / last;
        for r in 0..rows {
            let base = (r * last) as usize;
            let mut mx = m.data[base];
            for j in 1..last {
                mx = mx.max(m.data[base + j as usize]);
            }
            let mut s = 0.0;
            for j in 0..last {
                m.data[base + j as usize] = (m.data[base + j as usize] - mx).exp();
                s += m.data[base + j as usize];
            }
            for j in 0..last {
                m.data[base + j as usize] /= s;
            }
        }
    }

    fn matmul(a: &TensorBuffer, b: &TensorBuffer, c: &mut TensorBuffer) {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];
        c.shape = vec![m, n];
        c.data = vec![0.0; (m * n) as usize];
        for i in 0..m {
            for kk in 0..k {
                let av = a.data[(i * k + kk) as usize];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c.data[(i * n + j) as usize] += av * b.data[(kk * n + j) as usize];
                }
            }
        }
    }

    fn eval_instr(&mut self, instr: &MLIRInstr, mod_: &MLIRModule) -> bool {
        match instr.op {
            MLIROp::Matmul => {
                let (Some(a), Some(b)) = (self.getv(&instr.operands[0]), self.getv(&instr.operands[1])) else {
                    self.errors.push("matmul missing operand".to_string());
                    return false;
                };
                let mut c = TensorBuffer {
                    data: Vec::new(),
                    shape: Vec::new(),
                };
                Self::matmul(a, b, &mut c);
                self.values.insert(instr.result_id.clone(), c);
                true
            }
            MLIROp::Dropout => {
                let Some(a) = self.getv(&instr.operands[0]) else {
                    return false;
                };
                self.values.insert(instr.result_id.clone(), a.clone()); // inference: identity
                true
            }
            MLIROp::Relu | MLIROp::LeakyRelu | MLIROp::Sigmoid | MLIROp::Tanh
            | MLIROp::Swish | MLIROp::Gelu | MLIROp::Silu | MLIROp::Identity => {
                let Some(a) = self.getv(&instr.operands[0]) else {
                    return false;
                };
                let act = match instr.op {
                    MLIROp::Relu => "relu",
                    MLIROp::LeakyRelu => "leaky_relu",
                    MLIROp::Sigmoid => "sigmoid",
                    MLIROp::Tanh => "tanh",
                    MLIROp::Swish => "swish",
                    MLIROp::Gelu => "gelu",
                    MLIROp::Silu => "silu",
                    _ => "identity",
                };
                let mut out = a.clone();
                Self::activation(act, &mut out);
                self.values.insert(instr.result_id.clone(), out);
                true
            }
            MLIROp::ElementwiseBinop => {
                let Some(a) = self.getv(&instr.operands[0]) else {
                    return false;
                };
                let b = if instr.operands.len() > 1 {
                    self.getv(&instr.operands[1])
                } else {
                    None
                };
                let mut out = TensorBuffer {
                    data: Vec::new(),
                    shape: Vec::new(),
                };
                Self::elementwise(a, b, &instr.attribute, &mut out);
                self.values.insert(instr.result_id.clone(), out);
                true
            }
            MLIROp::Layernorm => {
                let Some(a) = self.getv(&instr.operands[0]) else {
                    return false;
                };
                let mut out = a.clone();
                Self::layernorm(&mut out);
                self.values.insert(instr.result_id.clone(), out);
                true
            }
            MLIROp::Softmax => {
                let Some(a) = self.getv(&instr.operands[0]) else {
                    return false;
                };
                let mut out = a.clone();
                Self::softmax(&mut out);
                self.values.insert(instr.result_id.clone(), out);
                true
            }
            MLIROp::Fused => self.eval_fused(instr, mod_),
            _ => true, // non-tensor markers (GRAD, FORWARD, layer decls) skipped
        }
    }

    fn eval_fused(&mut self, instr: &MLIRInstr, mod_: &MLIRModule) -> bool {
        let (Some(a), Some(b)) = (self.getv(&instr.operands[0]), self.getv(&instr.operands[1])) else {
            return false;
        };
        let mut c = TensorBuffer {
            data: Vec::new(),
            shape: Vec::new(),
        };
        Self::matmul(a, b, &mut c);

        let mut group: Option<FusedopGroup> = None;
        for g in mod_.fused_groups.iter() {
            if g.result_id == instr.result_id {
                group = Some(g.clone());
                break;
            }
        }
        let Some(group) = group else {
            self.errors.push(format!("no fusion group for {}", instr.result_id));
            return false;
        };

        let mut extra = 0usize;
        for opname in group.ops.iter() {
            if opname == "+" || opname == "-" || opname == "*" || opname == "/" {
                let rhs = if extra < group.epilogue_operands.len() {
                    self.getv(&group.epilogue_operands[extra])
                } else {
                    None
                };
                let mut out = TensorBuffer {
                    data: Vec::new(),
                    shape: Vec::new(),
                };
                Self::elementwise(&c, rhs, opname, &mut out);
                c = out;
                extra += 1;
            } else if opname == "layernorm" {
                Self::layernorm(&mut c);
            } else if opname == "softmax" {
                Self::softmax(&mut c);
            } else {
                Self::activation(opname, &mut c);
            }
        }
        self.values.insert(instr.result_id.clone(), c);
        true
    }
}

impl Default for ModuleEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// libm-style error function for GELU (std::erf equivalent).
pub fn libm_erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation (max error 1.5e-7).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}