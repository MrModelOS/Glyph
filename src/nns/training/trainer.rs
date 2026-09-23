#![allow(dead_code)]
// NNS port: public API preserved for parity with C++ nsc; not all items are used in current pipeline — intentional, not tech debt
//! Numeric training driver — port of `ns/training/trainer.cpp`.
//!
//! Executes the *unfused* MLIR forward graph of a network with a reverse-mode
//! numeric tape, accumulates gradients into the trainable weight buffers, and
//! hands them per-parameter to the optimizer (Muon for 2D matrices, AdamW
//! otherwise).

use std::collections::HashMap;

use super::super::mlir::dialect::{MLIRFunction, MLIRModule, MLIROp};
use super::super::mlir::eval::TensorBuffer;
use super::super::optim::muon::MuonOptimizer;
use super::super::optim::optim_params;

#[derive(Debug, Clone)]
pub struct Param {
    pub id: String, // MLIR value id (e.g. "fc1_w")
    pub rows: i64,
    pub cols: i64,
    pub data: Vec<f64>, // current weights, row-major rows*cols
    pub grad: Vec<f64>, // accumulated batch gradient
}

#[derive(Debug, Clone)]
struct Rec {
    op: MLIROp,
    result_id: String,
    operands: Vec<String>,
    attribute: String,
    out: TensorBuffer,      // forward output (saved)
    ins: Vec<TensorBuffer>, // forward operand values
}

/// Numeric bridge between the forward graph and the optimizer library.
pub struct NumericTrainer<'a> {
    fn_: &'a MLIRFunction,
    mod_: &'a MLIRModule,
    values: HashMap<String, TensorBuffer>,
    inputs: HashMap<String, TensorBuffer>, // bound inputs (survive forwards)
    grads: HashMap<String, TensorBuffer>,
    tape: Vec<Rec>,
    params: Vec<Param>,
    last_out: Option<TensorBuffer>,
    errors: Vec<String>,
    params_registered: bool,
}

impl<'a> NumericTrainer<'a> {
    pub fn new(fn_: &'a MLIRFunction, mod_: &'a MLIRModule) -> Self {
        let mut t = NumericTrainer {
            fn_,
            mod_,
            values: HashMap::new(),
            inputs: HashMap::new(),
            grads: HashMap::new(),
            tape: Vec::new(),
            params: Vec::new(),
            last_out: None,
            errors: Vec::new(),
            params_registered: false,
        };
        t.register_params();
        t
    }

    fn register_params(&mut self) {
        for instr in self.fn_.instructions.iter() {
            if instr.op == MLIROp::TensorAlloc {
                let mut p = Param {
                    id: instr.result_id.clone(),
                    rows: 0,
                    cols: 0,
                    data: Vec::new(),
                    grad: Vec::new(),
                };
                if let Some(d0) = instr.result_type.dims.first() {
                    if d0.is_const() {
                        p.rows = d0.const_value;
                    }
                }
                if let Some(d1) = instr.result_type.dims.get(1) {
                    if d1.is_const() {
                        p.cols = d1.const_value;
                    }
                }
                p.data = vec![0.0; (p.rows * p.cols) as usize];
                p.grad = vec![0.0; p.data.len()];
                self.params.push(p);
            }
        }
    }

    pub fn bind_input(&mut self, id: &str, data: Vec<f64>, shape: Vec<i64>) {
        let tb = TensorBuffer { data, shape };
        self.inputs.insert(id.to_string(), tb.clone());
        self.values.insert(id.to_string(), tb);
    }

    pub fn params(&self) -> &[Param] {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut Vec<Param> {
        &mut self.params
    }

    pub fn param(&self, id: &str) -> Option<&Param> {
        self.params.iter().find(|p| p.id == id)
    }

    pub fn param_mut(&mut self, id: &str) -> Option<&mut Param> {
        self.params.iter_mut().find(|p| p.id == id)
    }

    pub fn value(&self, id: &str) -> Option<&TensorBuffer> {
        self.values.get(id)
    }

    pub fn errors(&self) -> &[String] {
        &self.errors
    }

    /// Execute the forward graph, recording the tape. Returns the value of the
    /// function return id (logits, [B, C]).
    pub fn forward(&mut self) -> Option<&TensorBuffer> {
        self.tape.clear();
        self.values = self.inputs.clone(); // restore bound inputs; clear leftovers
        for instr in self.fn_.instructions.iter() {
            if instr.op == MLIROp::TensorAlloc {
                // Materialize the (possibly overridden) weight into a buffer.
                let p = self.param(&instr.result_id).cloned();
                if let Some(p) = p {
                    self.values.insert(
                        p.id.clone(),
                        TensorBuffer {
                            data: p.data,
                            shape: vec![p.rows, p.cols],
                        },
                    );
                }
                continue;
            }
            self.record(instr);
        }
        let out = self.value(&self.fn_.return_id).cloned();
        self.last_out = out;
        self.last_out.as_ref()
    }

    fn record(&mut self, instr: &super::super::mlir::dialect::MLIRInstr) {
        let mut r = Rec {
            op: instr.op,
            result_id: instr.result_id.clone(),
            operands: instr.operands.clone(),
            attribute: instr.attribute.clone(),
            out: TensorBuffer {
                data: Vec::new(),
                shape: Vec::new(),
            },
            ins: Vec::new(),
        };

        for opnd in instr.operands.iter() {
            if let Some(v) = self.values.get(opnd) {
                if r.ins.len() < 2 {
                    r.ins.push(v.clone());
                }
            }
        }
        // Every (first up-to-2) operand must resolve, in declared order.
        if r.ins.len() != instr.operands.len().min(2) {
            self.errors
                .push(format!("forward: missing operand for {}", instr.result_id));
            return;
        }

        let ok = match instr.op {
            MLIROp::Matmul => self.eval_matmul(&mut r),
            MLIROp::Relu
            | MLIROp::LeakyRelu
            | MLIROp::Sigmoid
            | MLIROp::Tanh
            | MLIROp::Swish
            | MLIROp::Gelu
            | MLIROp::Silu
            | MLIROp::Identity => self.eval_activation(&mut r),
            MLIROp::Dropout => {
                if !r.ins.is_empty() {
                    r.out = r.ins[0].clone();
                    true
                } else {
                    false
                }
            }
            MLIROp::ElementwiseBinop => self.eval_binop(&mut r),
            MLIROp::Layernorm => self.eval_layernorm(&mut r),
            MLIROp::Softmax => self.eval_softmax(&mut r),
            _ => {
                // GRAD / FORWARD / layer markers: no tensor result.
                return;
            }
        };
        if !ok {
            self.errors
                .push(format!("forward failed at {}", r.result_id));
            return;
        }
        self.values.insert(r.result_id.clone(), r.out.clone());
        self.tape.push(r);
    }

    fn eval_matmul(&mut self, r: &mut Rec) -> bool {
        if r.ins.len() < 2 {
            return false;
        }
        matmul_local(&r.ins[0], &r.ins[1], &mut r.out);
        true
    }

    fn eval_activation(&mut self, r: &mut Rec) -> bool {
        if r.ins.is_empty() {
            return false;
        }
        let mut out = r.ins[0].clone();
        let act = match r.op {
            MLIROp::Relu => "relu",
            MLIROp::LeakyRelu => "leaky_relu",
            MLIROp::Sigmoid => "sigmoid",
            MLIROp::Tanh => "tanh",
            MLIROp::Swish => "swish",
            MLIROp::Gelu => "gelu",
            MLIROp::Silu => "silu",
            _ => "identity",
        };
        for v in out.data.iter_mut() {
            match act {
                "relu" => *v = if *v > 0.0 { *v } else { 0.0 },
                "leaky_relu" => *v = if *v > 0.0 { *v } else { 0.01 * *v },
                "sigmoid" => *v = 1.0 / (1.0 + (-*v).exp()),
                "tanh" => *v = v.tanh(),
                "swish" => *v = *v / (1.0 + (-*v).exp()),
                "gelu" => {
                    *v = 0.5 * *v * (1.0 + super::super::mlir::eval::libm_erf(*v / 2f64.sqrt()))
                }
                "silu" => *v = *v / (1.0 + (-*v).exp()),
                _ => {}
            }
        }
        r.out = out;
        true
    }

    fn eval_binop(&mut self, r: &mut Rec) -> bool {
        if r.ins.is_empty() {
            return false;
        }
        let a = r.ins[0].clone();
        let b = if r.ins.len() > 1 {
            Some(r.ins[1].clone())
        } else {
            None
        };
        let mut out = TensorBuffer {
            data: Vec::new(),
            shape: a.shape.clone(),
        };
        out.data.resize(a.data.len(), 0.0);
        let op = r.attribute.clone();
        for (i, av) in a.data.iter().enumerate() {
            let bv = match &b {
                Some(buf) if !buf.data.is_empty() => buf.data[i % buf.data.len()],
                _ => 0.0,
            };
            out.data[i] = match op.as_str() {
                "+" => av + bv,
                "-" => av - bv,
                "*" => av * bv,
                "/" => {
                    if bv == 0.0 {
                        0.0
                    } else {
                        av / bv
                    }
                }
                _ => *av,
            };
        }
        r.out = out;
        true
    }

    fn eval_layernorm(&mut self, r: &mut Rec) -> bool {
        if r.ins.is_empty() {
            return false;
        }
        let input = &r.ins[0];
        let mut out = input.clone();
        let last = input.shape.last().copied().unwrap_or(1);
        if last <= 0 {
            return true;
        }
        let rows = out.data.len() as i64 / last;
        for row in 0..rows {
            let mut mean = 0.0;
            let mut var = 0.0;
            for j in 0..last {
                mean += out.data[(row * last + j) as usize];
            }
            mean /= last as f64;
            for j in 0..last {
                let d = out.data[(row * last + j) as usize] - mean;
                var += d * d;
            }
            var /= last as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for j in 0..last {
                let idx = (row * last + j) as usize;
                out.data[idx] = (out.data[idx] - mean) * inv;
            }
        }
        r.out = out;
        true
    }

    fn eval_softmax(&mut self, r: &mut Rec) -> bool {
        if r.ins.is_empty() {
            return false;
        }
        let input = &r.ins[0];
        let mut out = input.clone();
        let last = input.shape.last().copied().unwrap_or(1);
        if last <= 0 {
            return true;
        }
        let rows = out.data.len() as i64 / last;
        for row in 0..rows {
            let base = (row * last) as usize;
            let mut mx = out.data[base];
            for j in 1..last {
                mx = mx.max(out.data[base + j as usize]);
            }
            let mut s = 0.0;
            for j in 0..last {
                out.data[base + j as usize] = (out.data[base + j as usize] - mx).exp();
                s += out.data[base + j as usize];
            }
            for j in 0..last {
                out.data[base + j as usize] /= s;
            }
        }
        r.out = out;
        true
    }

    /// Seed the reverse pass: expects the gradient of the loss wrt the return
    /// value and accumulates dL/dw into every registered weight.
    pub fn backward(&mut self, grad_return: Vec<f64>, b: i64, c: i64) {
        self.grads.clear();
        self.grads.insert(
            self.fn_.return_id.clone(),
            TensorBuffer {
                shape: vec![b, c],
                data: grad_return,
            },
        );
        // Snapshot the tape: the back_* methods mutate `self` (grads/params).
        // Process in reverse order (reverse-mode).
        let tape = self.tape.clone();
        for r in tape.iter().rev() {
            match r.op {
                MLIROp::Matmul => self.back_matmul(r),
                MLIROp::Relu
                | MLIROp::LeakyRelu
                | MLIROp::Sigmoid
                | MLIROp::Tanh
                | MLIROp::Swish
                | MLIROp::Gelu
                | MLIROp::Silu
                | MLIROp::Identity => self.back_activation(r),
                MLIROp::Dropout => {
                    if let Some(g) = self.grads.get(&r.result_id).cloned() {
                        if let Some(first) = r.operands.first() {
                            self.accumulate(first, &g);
                        }
                    }
                }
                MLIROp::ElementwiseBinop => self.back_binop(r),
                MLIROp::Layernorm => self.back_layernorm(r),
                MLIROp::Softmax => self.back_softmax(r),
                _ => {}
            }
        }

        // Fold accumulated grads into params.
        for p in self.params.iter_mut() {
            if let Some(it) = self.grads.get(&p.id) {
                if it.data.len() == p.grad.len() {
                    for i in 0..p.grad.len() {
                        p.grad[i] += it.data[i];
                    }
                }
            }
        }
        self.grads.clear();
    }

    fn back_matmul(&mut self, r: &Rec) {
        let g_c = match self.grads.get(&r.result_id).cloned() {
            Some(g) => g,
            None => return,
        };
        if r.ins.len() < 2 {
            return;
        }
        let a = r.ins[0].clone();
        let b = r.ins[1].clone();

        let m = g_c.shape[0];
        let n = b.shape[0];
        let k = b.shape[1];
        // dA = gC @ B^T  (dims: [M, K], K = B.rows)
        let mut ga = TensorBuffer {
            data: vec![0.0; (m * n) as usize],
            shape: vec![m, n],
        };
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += g_c.data[(i * k + kk) as usize] * b.data[(j * k + kk) as usize];
                }
                ga.data[(i * n + j) as usize] = s;
            }
        }
        if let Some(first) = r.operands.first() {
            self.accumulate(first, &ga);
        }

        // dB = A^T @ gC   (sum over M = rows of A)
        let mm = a.shape[0];
        let mut gb = TensorBuffer {
            data: vec![0.0; (n * g_c.shape[1]) as usize],
            shape: vec![n, g_c.shape[1]],
        };
        for i in 0..n {
            for j in 0..g_c.shape[1] {
                let mut s = 0.0;
                for p in 0..mm {
                    s += a.data[(p * a.shape[1] + i) as usize]
                        * g_c.data[(p * g_c.shape[1] + j) as usize];
                }
                gb.data[(i * gb.shape[1] + j) as usize] = s;
            }
        }
        if let Some(second) = r.operands.get(1) {
            self.accumulate(second, &gb);
        }
    }

    fn back_activation(&mut self, r: &Rec) {
        let g = match self.grads.get(&r.result_id).cloned() {
            Some(g) => g,
            None => return,
        };
        if r.ins.is_empty() {
            return;
        }
        let input = &r.ins[0];
        let mut ga = g;
        for i in 0..ga.data.len().min(input.data.len()) {
            let x = input.data[i];
            let d = match r.op {
                MLIROp::Relu => {
                    if x > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                MLIROp::LeakyRelu => {
                    if x > 0.0 {
                        1.0
                    } else {
                        0.01
                    }
                }
                MLIROp::Sigmoid => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                }
                MLIROp::Tanh => {
                    let t = x.tanh();
                    1.0 - t * t
                }
                MLIROp::Swish | MLIROp::Silu => {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s + x * s * (1.0 - s)
                }
                MLIROp::Gelu => {
                    0.5 * (1.0 + super::super::mlir::eval::libm_erf(x / 2f64.sqrt()))
                        + x * (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
                }
                _ => 1.0,
            };
            ga.data[i] *= d;
        }
        if let Some(first) = r.operands.first() {
            self.accumulate(first, &ga);
        }
    }

    fn back_binop(&mut self, r: &Rec) {
        let g = match self.grads.get(&r.result_id).cloned() {
            Some(g) => g,
            None => return,
        };
        if r.ins.is_empty() {
            return;
        }
        let a = r.ins[0].clone();
        if let Some(first) = r.operands.first() {
            self.accumulate(first, &g);
        }
        if r.operands.len() > 1 && r.ins.len() > 1 {
            let b = &r.ins[1];
            let op = r.attribute.clone();
            let mut gb = TensorBuffer {
                data: vec![0.0; b.data.len()],
                shape: b.shape.clone(),
            };
            for i in 0..g.data.len() {
                if b.data.is_empty() {
                    continue;
                }
                let k = i % b.data.len();
                let d = match op.as_str() {
                    "+" => 1.0,
                    "-" => -1.0,
                    "*" => a.data[i],
                    "/" => {
                        if b.data[k] == 0.0 {
                            0.0
                        } else {
                            -a.data[i] / (b.data[k] * b.data[k])
                        }
                    }
                    _ => 0.0,
                };
                gb.data[k] += g.data[i] * d;
            }
            if let Some(second) = r.operands.get(1) {
                self.accumulate(second, &gb);
            }
        }
    }

    fn back_layernorm(&mut self, r: &Rec) {
        let g = match self.grads.get(&r.result_id).cloned() {
            Some(g) => g,
            None => return,
        };
        if r.ins.is_empty() {
            return;
        }
        let input = &r.ins[0];
        let mut ga = r.out.clone();
        let last = r.out.shape.last().copied().unwrap_or(1);
        if last <= 0 {
            return;
        }
        let rows = g.data.len() as i64 / last;
        for row in 0..rows {
            let mut mean = 0.0;
            let mut var = 0.0;
            for j in 0..last {
                mean += input.data[(row * last + j) as usize];
            }
            mean /= last as f64;
            for j in 0..last {
                let d = input.data[(row * last + j) as usize] - mean;
                var += d * d;
            }
            var /= last as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            let mut wsum = 0.0;
            for j in 0..last {
                wsum += g.data[(row * last + j) as usize]
                    * (input.data[(row * last + j) as usize] - mean);
            }
            wsum /= last as f64;
            for j in 0..last {
                let idx = (row * last + j) as usize;
                let gd = g.data[idx];
                let xc = input.data[idx] - mean;
                ga.data[idx] = (gd - wsum - xc * wsum / (var + 1e-5)) * inv;
            }
        }
        if let Some(first) = r.operands.first() {
            self.accumulate(first, &ga);
        }
    }

    fn back_softmax(&mut self, r: &Rec) {
        let g = match self.grads.get(&r.result_id).cloned() {
            Some(g) => g,
            None => return,
        };
        if r.ins.is_empty() {
            return;
        }
        let sm = r.out.clone();
        let mut ga = g.clone();
        let last = sm.shape.last().copied().unwrap_or(1);
        if last <= 0 {
            return;
        }
        let rows = g.data.len() as i64 / last;
        for row in 0..rows {
            let base = (row * last) as usize;
            let mut dot = 0.0;
            for j in 0..last {
                dot += g.data[base + j as usize] * sm.data[base + j as usize];
            }
            for j in 0..last {
                ga.data[base + j as usize] =
                    sm.data[base + j as usize] * (g.data[base + j as usize] - dot);
            }
        }
        if let Some(first) = r.operands.first() {
            self.accumulate(first, &ga);
        }
    }

    fn accumulate(&mut self, id: &str, g: &TensorBuffer) {
        let entry = self
            .grads
            .entry(id.to_string())
            .or_insert_with(|| TensorBuffer {
                data: Vec::new(),
                shape: g.shape.clone(),
            });
        if entry.data.is_empty() {
            *entry = g.clone();
            return;
        }
        let n = entry.data.len().min(g.data.len());
        for i in 0..n {
            entry.data[i] += g.data[i];
        }
    }

    pub fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            p.grad.fill(0.0);
        }
    }

    /// Apply the registered weights through `opt` (per-parameter bridging into
    /// the Muon/AdamW optimizer). step_index = 0-based step counter.
    pub fn apply_step(&mut self, opt: &mut MuonOptimizer, step_index: usize) {
        if !self.params_registered {
            for p in self.params.iter() {
                opt.add_param(p.data.clone(), p.rows as usize, p.cols as usize);
            }
            self.params_registered = true;
        }
        // Mirror the AOT runtime wrapper: scale the base LR by the schedule.
        let base = opt.params().lr;
        opt.set_lr(base * optim_params::lr_scale(step_index as i64) as f64);
        opt.set_step(step_index);
        for i in 0..self.params.len() {
            let grad = self.params[i].grad.clone();
            opt.step_param(i, &grad);
            // Sync the optimizer-owned buffer back into the trainer params.
            let new_data = opt.param_data(i).to_vec();
            self.params[i].data = new_data;
        }
        opt.set_lr(base);
    }
}

fn matmul_local(a: &TensorBuffer, b: &TensorBuffer, c: &mut TensorBuffer) {
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
