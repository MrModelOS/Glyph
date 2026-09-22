//! Code generation port of `NeuralScript/src/codegen/codegen.cpp`:
//! emits a self-contained C++ (CPU) or CUDA source implementing the compiled
//! MLIR graph, plus the runtime driver that main.cpp relied on.
//!
//! The emitter is a faithful 1:1 port: output must be byte-identical to the
//! C++ `CodeGenerator`. Every string literal that the original wrote is
//! reproduced exactly; orderings follow `std::map`/`std::set` (here
//! `BTreeMap`/`BTreeSet`) so iteration order matches. Shape errors that the
//! original raised with `throw std::runtime_error` surface as `NsResult`
//! errors.

use std::collections::{BTreeMap, BTreeSet};

use crate::nns::ast::{DimExpr, Dtype, TensorType};
use crate::nns::codegen::{cpu_blobs, cuda_backend};
use crate::nns::mlir::dialect::{MLIRFunction, MLIRInstr, MLIRModule, MLIROp};
use crate::nns::optim::optim_params as optim;
use crate::nns::{fmt_float, NsResult};
use crate::ns_error;

// ---------------------------------------------------------------------------
// fmt_float9: C++ `ostringstream` with `defaultfloat` and `precision(9)`.
// libstdc++ emits `%.9g` on the exact value of the float: the exact decimal
// expansion is rounded to 9 significant digits (round-half-even) and laid out
// with `%g` rules — fixed notation when -4 <= X < 9 (X = floor(log10 v)),
// otherwise scientific with a sign and a two-digit (at minimum) exponent;
// trailing zeros of the fraction are dropped, and with it the decimal point.
// -0.0f prints as "-0".
// ---------------------------------------------------------------------------

/// Format `v` the way C++ `ostringstream << setprecision(9) << v` (default
/// floatfield) prints it — see module docs above.
pub fn fmt_float9(v: f32) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() {
            String::from("-0")
        } else {
            String::from("0")
        };
    }

    let bits = v.to_bits();
    let sign = (bits >> 31) != 0;
    let raw_exp = ((bits >> 23) & 0xff) as i32;
    let mant = (bits & 0x7f_ffff) as u64;

    // Infinity / NaN never occur for the constants we format, but mirror the
    // C++ iostream text for the differential test.
    if raw_exp == 255 {
        return if mant == 0 {
            if sign {
                String::from("-inf")
            } else {
                String::from("inf")
            }
        } else if sign {
            String::from("-nan")
        } else {
            String::from("nan")
        };
    }

    // value = m * 2^e with m a 24-bit significand (subnormals omit the
    // implicit bit).
    let (m, e) = if raw_exp == 0 {
        (mant, -149)
    } else {
        ((1 << 23) | mant, raw_exp - 127 - 23)
    };

    // Exact decimal expansion: value = N * 10^-k.
    //   e >= 0: N = m * 2^e, k = 0   (m < 2^24, e <= 104 -> fits in u128)
    //   e <  0: N = m * 5^-e, k = -e (bignum: 5^149 ~ 7e104)
    let nstr: String;
    let k: i64;
    if e >= 0 {
        nstr = ((m as u128) << e).to_string();
        k = 0;
    } else {
        k = -(e as i64);
        let mut num = BigDec::from_u64(m);
        for _ in 0..k {
            num.mul_small(5);
        }
        nstr = num.to_digits();
    }

    // Split into integer / fraction digit strings, padding the fraction with
    // leading zeros to exactly k digits so the decimal point is implicit.
    let len_n = nstr.len() as i64;
    let (int_part, frac_part) = if len_n > k {
        let split = (len_n - k) as usize;
        (nstr[..split].to_string(), nstr[split..].to_string())
    } else {
        (
            String::new(),
            format!("{}{}", "0".repeat((k - len_n) as usize), nstr),
        )
    };

    let int_l = int_part.len() as i64; // number of digits before the point
    let total = format!("{}{}", int_part, frac_part);
    let total_b = total.as_bytes();
    let n = total_b.len() as i64;

    // First significant digit and its place value X = floor(log10 v).
    let mut sig_start: i64 = -1;
    for (i, &c) in total_b.iter().enumerate() {
        if c != b'0' {
            sig_start = i as i64;
            break;
        }
    }
    debug_assert!(sig_start >= 0, "zero handled above");
    let x_pre = if sig_start < int_l {
        int_l - 1 - sig_start
    } else {
        -1 - (sig_start - int_l)
    };

    // Round to 9 significant digits (keep the digits at places x_pre
    // down to x_pre - 8; round-half-even at the boundary).
    const P: usize = 9;
    let end = sig_start as usize + P; // one past the last kept digit
    let digit = |c: u8| c - b'0';
    let mut kept: Vec<u8> = if n <= end as i64 {
        total_b[sig_start as usize..].iter().map(|&c| digit(c)).collect()
    } else {
        total_b[sig_start as usize..end].iter().map(|&c| digit(c)).collect()
    };
    let mut x = x_pre;
    if n > end as i64 {
        let d0 = total_b[end] - b'0';
        let rest_nonzero = total_b[end + 1..].iter().any(|&c| c != b'0');
        let round_up = d0 > 5 || (d0 == 5 && (rest_nonzero || kept[kept.len() - 1] % 2 == 1));
        if round_up {
            let mut i = kept.len();
            let mut carry = true;
            while i > 0 && carry {
                i -= 1;
                if kept[i] == 9 {
                    kept[i] = 0;
                } else {
                    kept[i] += 1;
                    carry = false;
                }
            }
            if carry {
                // All kept digits were 9: value is exactly 10^(x+1); drop the
                // now-insignificant trailing zero.
                kept.insert(0, 1);
                kept.pop();
                x += 1;
            }
        }
    }

    // Lay out per %g.
    let mut out = String::new();
    if sign {
        out.push('-');
    }
    if (-4..9).contains(&x) {
        // Fixed notation.
        if x >= 0 {
            let il = x as usize + 1;
            for &d in kept.iter().take(il) {
                out.push((b'0' + d) as char);
            }
            let tail: String = kept[il..].iter().map(|&d| (b'0' + d) as char).collect();
            let trimmed = tail.trim_end_matches('0');
            if !trimmed.is_empty() {
                out.push('.');
                out.push_str(trimmed);
            }
        } else {
            out.push('0');
            out.push('.');
            for _ in 0..(-x - 1) {
                out.push('0');
            }
            for &d in &kept {
                out.push((b'0' + d) as char);
            }
            let trimmed = out.trim_end_matches('0').to_string();
            out = trimmed;
        }
    } else {
        // Scientific notation.
        debug_assert!(!kept.is_empty());
        out.push((b'0' + kept[0]) as char);
        let tail: String = kept[1..].iter().map(|&d| (b'0' + d) as char).collect();
        let trimmed = tail.trim_end_matches('0');
        if !trimmed.is_empty() {
            out.push('.');
            out.push_str(trimmed);
        }
        out.push('e');
        out.push(if x < 0 { '-' } else { '+' });
        let ax = x.abs();
        if ax < 10 {
            out.push('0');
            out.push((b'0' + ax as u8) as char);
        } else {
            out.push_str(&ax.to_string());
        }
    }
    out
}

/// Little-endian base-10^9 bignum, big enough for m * 5^149 (~7e104).

struct BigDec(Vec<u32>);

impl BigDec {
    fn from_u64(mut n: u64) -> Self {
        let mut d = Vec::new();
        while n > 0 {
            d.push((n % 1_000_000_000) as u32);
            n /= 1_000_000_000;
        }
        BigDec(d)
    }

    fn mul_small(&mut self, s: u32) {
        let mut carry: u64 = 0;
        for c in self.0.iter_mut() {
            let t = (*c as u64) * (s as u64) + carry;
            *c = (t % 1_000_000_000) as u32;
            carry = t / 1_000_000_000;
        }
        while carry > 0 {
            self.0.push((carry % 1_000_000_000) as u32);
            carry /= 1_000_000_000;
        }
    }

    fn to_digits(&self) -> String {
        if self.0.is_empty() {
            return String::from("0");
        }
        let mut s = self.0[self.0.len() - 1].to_string();
        for &c in self.0[..self.0.len() - 1].iter().rev() {
            s.push_str(&format!("{:09}", c));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Codegen options (port of `ns/codegen/codegen.hpp`).
// ---------------------------------------------------------------------------

/// Target backend of the emitted source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetBackend {
    CpuCxx,
    CpuSimd,
    Cuda,
    Rocm,
    Metal,
}

/// Options controlling a `CodeGenerator::generate` call.
#[derive(Debug, Clone)]
pub struct CodegenOptions {
    pub backend: TargetBackend,
    pub fuse_kernels: bool,
    pub enable_fp16: bool,
    pub emit_runtime_driver: bool,
    pub function_name: String,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        CodegenOptions {
            backend: TargetBackend::Cuda,
            fuse_kernels: true,
            enable_fp16: false,
            emit_runtime_driver: false,
            function_name: String::from("forward"),
        }
    }
}

/// The C++ `CodeGenerator`. Stateless: `generate` dispatches on the backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodeGenerator;

impl CodeGenerator {
    /// Port of `CodeGenerator::generate` — dispatch on `opts.backend`.
    pub fn generate(&self, module: &MLIRModule, opts: &CodegenOptions) -> NsResult<String> {
        match opts.backend {
            TargetBackend::Cuda => self.gen_cuda(module, opts),
            TargetBackend::CpuCxx | TargetBackend::CpuSimd => self.gen_cpu(module, opts),
            TargetBackend::Rocm | TargetBackend::Metal => self.gen_cuda(module, opts),
        }
    }

    /// Port of `CodeGenerator::dtype_c_name`.
    pub fn dtype_c_name(&self, d: Dtype) -> &'static str {
        match d {
            Dtype::Float16 => "half",
            Dtype::Float32 => "float",
            Dtype::Float64 => "double",
            Dtype::Int8 => "int8_t",
            Dtype::Int16 => "int16_t",
            Dtype::Int32 => "int32_t",
            Dtype::Int64 => "int64_t",
            Dtype::Fp8 => "int8_t",  // placeholder fp8
            Dtype::Fp4 => "int8_t",  // placeholder fp4
            Dtype::Bool => "bool",
        }
    }

    /// Port of `CodeGenerator::numel` (dynamic/symbolic dims count as 1).
    pub fn numel(&self, t: &TensorType) -> u64 {
        let mut n: u64 = 1;
        for d in &t.dims {
            if d.is_const() {
                n = n.wrapping_mul(d.const_value as u64);
            }
        }
        n
    }

    /// Port of `CodeGenerator::emit_train_core`.
    fn emit_train_core(&self, tfn: &MLIRFunction, in_cols: i64) -> NsResult<String> {
        emit_train_core(tfn, in_cols)
    }

    /// Port of `CodeGenerator::emit_train_core_cuda`.
    fn emit_train_core_cuda(&self, tfn: &MLIRFunction, in_cols: i64, out_cols: i64) -> NsResult<String> {
        emit_train_core_cuda(tfn, in_cols, out_cols)
    }

    /// Port of `CodeGenerator::gen_cuda`.
    fn gen_cuda(&self, module: &MLIRModule, opts: &CodegenOptions) -> NsResult<String> {
        gen_cuda(module, opts)
    }

    /// Port of `CodeGenerator::gen_cpu`.
    fn gen_cpu(&self, module: &MLIRModule, opts: &CodegenOptions) -> NsResult<String> {
        gen_cpu(module, opts)
    }
}

// ---------------------------------------------------------------------------
// Free helpers (port of the anonymous-namespace helpers of codegen.cpp).
// ---------------------------------------------------------------------------

/// Emit the compiled-in LR schedule as a standalone helper for the runtime
/// driver. The body is a constant-fold of optim_params.hpp so the generated TU
/// stays self-contained (no library linkage). Port of `lr_schedule_source`.
pub fn lr_schedule_source() -> String {
    let mut o = String::new();
    o.push_str("static float ns_lr_schedule(int64_t step) {\n");
    o.push_str(&format!(
        "    if ({} == 0) return 1.0f;\n",
        optim::K_LR_SCHEDULE as i32
    ));
    o.push_str(&format!(
        "    if ({} <= 0) return 1.0f;\n",
        optim::K_LR_WARMUP_STEPS
    ));
    o.push_str(&format!(
        "    if (step < {}) return (float)step / (float){};\n",
        optim::K_LR_WARMUP_STEPS, optim::K_LR_WARMUP_STEPS
    ));
    o.push_str(&format!(
        "    const int64_t end = {} > {} ? {} : {};\n",
        optim::K_LR_TOTAL_STEPS,
        optim::K_LR_WARMUP_STEPS,
        optim::K_LR_TOTAL_STEPS,
        optim::K_LR_WARMUP_STEPS + 1
    ));
    o.push_str(&format!(
        "    const int64_t s = step >= end ? end : step;\n"
    ));
    o.push_str(&format!(
        "    const float t = (float)(s - {}) / (float)(end - {});\n",
        optim::K_LR_WARMUP_STEPS, optim::K_LR_WARMUP_STEPS
    ));
    o.push_str(&format!(
        "    return {} + 0.5f * (1.0f - {}) * (1.0f + cosf(3.14159265f * t));\n",
        fmt_float9(optim::K_LR_MIN_FACTOR),
        fmt_float9(optim::K_LR_MIN_FACTOR)
    ));
    o.push_str("}\n");
    o
}

/// Identify the runtime data input of an inference instruction. GEMMs consume
/// it as A; v1.2 layer ops consume it via their tensor operand (embedding: the
/// index vector, operand[1]). Port of `input_op_of`.
pub fn input_op_of(instr: &MLIRInstr) -> String {
    if instr.op == MLIROp::Matmul || instr.op == MLIROp::Fused {
        return if instr.operands.len() >= 2 {
            instr.operands[0].clone()
        } else {
            String::new()
        };
    }
    if instr.op == MLIROp::LayerEmbedding {
        return if instr.operands.len() >= 2 {
            instr.operands[1].clone()
        } else {
            String::new()
        };
    }
    if instr.op == MLIROp::LayerAttention
        || instr.op == MLIROp::Layernorm
        || instr.op == MLIROp::LayerMoe
        || instr.op == MLIROp::Softmax
        || instr.op == MLIROp::Transpose
        || instr.op == MLIROp::Concat
        || instr.op == MLIROp::Reshape
        || instr.op == MLIROp::Slice
        || instr.op == MLIROp::Index
        || instr.op == MLIROp::Scatter
        || instr.op == MLIROp::Dropout
        || instr.op == MLIROp::Activation
        || instr.op == MLIROp::Relu
        || instr.op == MLIROp::LeakyRelu
        || instr.op == MLIROp::Sigmoid
        || instr.op == MLIROp::Tanh
        || instr.op == MLIROp::Swish
        || instr.op == MLIROp::Gelu
        || instr.op == MLIROp::Silu
        || instr.op == MLIROp::Identity
    {
        return if instr.operands.len() >= 1 {
            instr.operands[0].clone()
        } else {
            String::new()
        };
    }
    String::new()
}

/// Per-id static shape metadata used by the CUDA emitters (mirrors the CPU
/// reference emitter, but keeps the forward and train function views separate).
/// Port of `CuIdInfo`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CuIdInfo {
    pub is_weight: bool,
    pub is_input: bool,
    pub rows: i64,
    pub cols: i64,
    pub static_numel: i64,
}

/// Port of the anonymous `cu_numel`.
fn cu_numel(t: &TensorType) -> u64 {
    let mut n: u64 = 1;
    for d in &t.dims {
        if d.is_const() {
            n = n.wrapping_mul(d.const_value as u64);
        }
    }
    n
}

/// Port of `fill_cu_ids` (uses `std::map` ordering → `BTreeMap`).
pub fn fill_cu_ids(fn_: &MLIRFunction, ids: &mut BTreeMap<String, CuIdInfo>) {
    let all_const = |d: &Vec<DimExpr>| d.is_empty() || d.iter().all(|x| x.is_const());
    for instr in &fn_.instructions {
        if instr.op == MLIROp::TensorAlloc {
            let inf = ids.entry(instr.result_id.clone()).or_default();
            inf.is_weight = true;
            let d = &instr.result_type.dims;
            if !d.is_empty() && d[0].is_const() {
                inf.rows = d[0].const_value;
            }
            if d.len() > 1 && d[1].is_const() {
                inf.cols = d[1].const_value;
            }
            if all_const(d) {
                inf.static_numel = cu_numel(&instr.result_type) as i64;
            }
        } else if !instr.result_id.is_empty() {
            let inf = ids.entry(instr.result_id.clone()).or_default();
            let d = &instr.result_type.dims;
            if !d.is_empty() && d[0].is_const() {
                inf.rows = d[0].const_value;
            }
            if d.len() > 1 && d[1].is_const() {
                inf.cols = d[1].const_value;
            }
            if all_const(d) {
                inf.static_numel = cu_numel(&instr.result_type) as i64;
            }
        }
    }
    for instr in &fn_.instructions {
        for op in &instr.operands {
            if !ids.contains_key(op) {
                ids.insert(op.clone(), CuIdInfo::default());
            }
        }
    }
}

/// Activation-forward code matching gen_cpu's ns_act2. Port of `cu_act_code`.
pub fn cu_act_code(op: MLIROp) -> &'static str {
    match op {
        MLIROp::Relu => "1",
        MLIROp::LeakyRelu => "2",
        MLIROp::Sigmoid => "3",
        MLIROp::Tanh => "4",
        MLIROp::Swish | MLIROp::Silu => "5",
        MLIROp::Gelu => "6",
        _ => "0",
    }
}

/// Port of `cu_act_code_attr`.
pub fn cu_act_code_attr(a: &str) -> &'static str {
    match a {
        "relu" => "1",
        "leaky_relu" => "2",
        "sigmoid" => "3",
        "tanh" => "4",
        "swish" | "silu" => "5",
        "gelu" => "6",
        _ => "0",
    }
}

/// Shared helper: select the inference function (port of the four-step
/// selection in gen_cuda/gen_cpu) plus the first train function (if any).
fn select_functions<'a>(
    module: &'a MLIRModule,
) -> (Option<&'a MLIRFunction>, Option<&'a MLIRFunction>) {
    let mut fn_ = None;
    // NB: the original's "infer" scan has NO break: the LAST function named
    // "infer" wins, mirroring `for (..) if (f.name == "infer") fn = &f;`.
    for f in &module.functions {
        if f.name == "infer" {
            fn_ = Some(f);
        }
    }
    if fn_.is_none() {
        for f in &module.functions {
            if f.name != "main" && f.name != "train_step" && !f.is_train {
                fn_ = Some(f);
                break;
            }
        }
    }
    if fn_.is_none() {
        for f in &module.functions {
            if f.name != "main" && !f.is_train {
                fn_ = Some(f);
                break;
            }
        }
    }
    if fn_.is_none() {
        for f in &module.functions {
            fn_ = Some(f);
            break;
        }
    }
    let mut tfn = None;
    for f in &module.functions {
        if f.is_train {
            tfn = Some(f);
            break;
        }
    }
    (fn_, tfn)
}

// ---------------------------------------------------------------------------
// gen_cuda — port of `CodeGenerator::gen_cuda` (codegen.cpp 306-1136).
// ---------------------------------------------------------------------------

fn gen_cuda(module: &MLIRModule, opts: &CodegenOptions) -> NsResult<String> {
    // ---- Select the function to lower (mirrors gen_cpu). ----
    let (fn_, tfn) = select_functions(module);
    let fn_ = match fn_ {
        Some(f) => f,
        None => {
            let mut o = String::new();
            o.push_str("// Generated by NeuralScript compiler\n// Target: CUDA\n");
            o.push_str("#include <cuda_runtime.h>\n");
            o.push_str(&format!(
                "extern \"C\" void {}(const float*, float*, size_t) {{}}\n",
                opts.function_name
            ));
            return Ok(o);
        }
    };

    // ---- Per-id metadata for the inference function. ----
    let mut ids: BTreeMap<String, CuIdInfo> = BTreeMap::new();
    fill_cu_ids(fn_, &mut ids);

    // Identify the runtime data input. GEMMs consume it as A; v1.2 layer ops
    // consume it via their tensor operand (embedding: the index vector).
    let mut x_id = String::new();
    for instr in &fn_.instructions {
        let op0 = input_op_of(instr);
        if !op0.is_empty() && !ids.get(&op0).map_or(false, |i| i.is_weight) {
            x_id = op0;
            break;
        }
    }
    if !x_id.is_empty() {
        ids.entry(x_id.clone()).or_default().is_input = true;
    }

    let mut worder: Vec<String> = Vec::new();
    for instr in &fn_.instructions {
        if instr.op == MLIROp::TensorAlloc {
            worder.push(instr.result_id.clone());
        }
    }
    let mut woff: BTreeMap<String, u64> = BTreeMap::new();
    let mut weight_total: u64 = 0;
    {
        let mut off: u64 = 0;
        for w in &worder {
            woff.insert(w.clone(), off);
            let n = ids
                .get(w)
                .map_or(1, |i| if i.static_numel > 0 { i.static_numel as u64 } else { 1 });
            off += n;
            weight_total += n;
        }
    }

    // MoE lifecycle metadata (one MoE layer per model). Computed from the
    // forward function's LAYER_MOE instruction + the weight layout.
    let mut moe_e: i32 = 0;
    let mut moe_d: i32 = 0;
    let mut moe_h: i32 = 0;
    let mut moe_k0: i32 = 0;
    let mut moe_off_g: u64 = 0;
    let mut moe_off_e1: u64 = 0;
    let mut moe_off_e2: u64 = 0;
    for instr in &fn_.instructions {
        if instr.op != MLIROp::LayerMoe || instr.operands.len() < 4 {
            continue;
        }
        moe_e = if instr.attribute.is_empty() {
            4
        } else {
            instr.attribute.parse::<i32>().unwrap_or(4)
        };
        moe_d = instr.int_attr as i32;
        moe_h = if instr.float_attr > 0.0 {
            instr.float_attr as i32
        } else {
            (4.0 * instr.int_attr as f64) as i32
        };
        moe_k0 = if !instr.ints_attr.is_empty() {
            instr.ints_attr[0] as i32
        } else {
            moe_e
        };
        if moe_k0 <= 0 || moe_k0 > moe_e {
            moe_k0 = moe_e;
        }
        if woff.contains_key(&instr.operands[1]) {
            moe_off_g = woff[&instr.operands[1]];
        }
        if woff.contains_key(&instr.operands[2]) {
            moe_off_e1 = woff[&instr.operands[2]];
        }
        if woff.contains_key(&instr.operands[3]) {
            moe_off_e2 = woff[&instr.operands[3]];
        }
    }
    let moe_e: i32 = moe_e;
    let moe_d: i32 = moe_d;
    let moe_h: i32 = moe_h;
    let moe_k0: i32 = moe_k0;

    // Static input/output geometry.
    let mut in_cols: i64 = -1;
    let mut out_cols: i64 = -1;
    for instr in &fn_.instructions {
        let gemm = instr.op == MLIROp::Matmul || instr.op == MLIROp::Fused;
        if gemm && instr.operands.len() >= 2 {
            if in_cols < 0 {
                in_cols = ids.get(&instr.operands[1]).map_or(0, |i| i.rows);
            }
            out_cols = ids.get(&instr.operands[1]).map_or(0, |i| i.cols);
        }
        // v1.2: non-GEMM heads (embedding/attention/layernorm/concat/softmax)
        // also fix geometry. Embedding consumes one scalar index per row, so
        // the input's per-row width is 1 (M == number of indices).
        if (instr.op == MLIROp::LayerEmbedding
            || instr.op == MLIROp::LayerAttention
            || instr.op == MLIROp::LayerMoe
            || instr.op == MLIROp::Layernorm
            || instr.op == MLIROp::Softmax
            || instr.op == MLIROp::Concat
            || instr.op == MLIROp::Slice
            || instr.op == MLIROp::Index
            || instr.op == MLIROp::Scatter
            || instr.op == MLIROp::Transpose
            || instr.op == MLIROp::Reshape)
            && !instr.result_id.is_empty()
        {
            if instr.op == MLIROp::LayerEmbedding && in_cols < 0 {
                in_cols = 1;
            }
            let c = ids.get(&instr.result_id).map_or(0, |i| i.cols);
            if c > 0 {
                out_cols = c;
            }
        }
    }
    if in_cols < 0 {
        in_cols = 1;
    }
    if out_cols < 0 {
        out_cols = 1;
    }
    if !x_id.is_empty() && ids.get(&x_id).map_or(false, |i| i.is_input) {
        if let Some(inf) = ids.get_mut(&x_id) {
            inf.cols = in_cols;
        }
    }

    let mut os = String::new();
    os.push_str("// Generated by NeuralScript compiler (CUDA backend)\n");
    os.push_str(&format!("// Lowered function: @{}\n", fn_.name));
    os.push_str("#include <cuda_runtime.h>\n");
    os.push_str("#include <curand_kernel.h>\n");
    os.push_str("#include <stdint.h>\n");
    os.push_str("#include <math.h>\n");
    os.push_str("#include <stdio.h>\n");
    os.push_str("#include <stdlib.h>\n");
    os.push_str("#include <cstring>\n");
    os.push_str("#include <algorithm>\n\n");

    // ---- Device scalar helpers + uniform forward kernels (owned by the CUDA
    //      target layer in cuda_backend.cpp). ----
    os.push_str(&cuda_backend::device_helpers_source());
    os.push_str(cuda_backend::forward_kernels_source());

    if tfn.is_some() {
        // ---- Reverse-mode + optimizer kernels (AOT training, owned by the
        //      CUDA target layer). ----
        os.push_str(cuda_backend::train_kernels_source());
    }

    // ---- Runtime utilities: canonical buffers + lazy device allocator. ----
    os.push_str(cuda_backend::runtime_utils_source());
    if moe_e > 0 {
        os.push_str(cuda_backend::runtime_utils_u8_source());
    }

    // ---- Per-model device context (weights + scratch + optimizer state).
    //      Every allocation is owned by the model and released in the context
    //      destructor, so multiple models are independent and ns_free() frees
    //      ALL device memory (no more file-scope statics => no shared-state /
    //      VRAM-leak hazard). ----
    {
        let mut ty_id = String::new();
        let cef = tfn.unwrap_or(fn_);
        for instr in &cef.instructions {
            if instr.op == MLIROp::CrossEntropy && instr.operands.len() >= 2 {
                ty_id = instr.operands[1].clone();
            }
        }
        if !ty_id.is_empty() {
            ids.entry(ty_id.clone()).or_default().is_input = true;
        }
        let mut allids: BTreeSet<String> = BTreeSet::new();
        let collect = |f: &MLIRFunction, all: &mut BTreeSet<String>| {
            for instr in &f.instructions {
                if instr.op == MLIROp::CrossEntropy {
                    continue;
                }
                for op in &instr.operands {
                    all.insert(op.clone());
                }
                if !instr.result_id.is_empty() {
                    all.insert(instr.result_id.clone());
                }
            }
        };
        collect(fn_, &mut allids);
        if let Some(t) = tfn {
            collect(t, &mut allids);
        }
        let sorted: Vec<String> = allids.iter().cloned().collect();

        let mut frees: Vec<String> = Vec::new(); // per-field cudaFree statement for the dtor
        let add_free = |field: &str, cap: &str, frees: &mut Vec<String>| {
            frees.push(format!("    cudaFree({}); {} = 0;\n", field, cap));
        };

        os.push_str("// ---- Per-model device context (weights + scratch + optimizer state) ----\n");
        os.push_str("typedef struct NSContext {\n");
        os.push_str("  float* d_wb = nullptr; size_t d_wb_cap = 0;   // canonical device weights\n");
        os.push_str("  float* d_xi = nullptr; size_t d_xi_cap = 0;   // input upload\n");
        os.push_str("  float* d_yl = nullptr; size_t d_yl_cap = 0;   // labels upload\n");
        os.push_str("  float* d_row = nullptr; size_t d_row_cap = 0; // CE row losses\n");
        os.push_str("  float* hrow = nullptr;  size_t hrow_cap = 0;  // CE host reduction\n");
        os.push_str("  float* d_asr = nullptr; size_t d_asr_cap = 0; // attention grad scratch\n");
        if moe_e > 0 {
            os.push_str(&format!(
                "  uint8_t* h_moe_a = nullptr; size_t h_moe_a_cap = 0; // host expert liveness\n"
            ));
            os.push_str(&format!(
                "  uint8_t* d_moe_a = nullptr; size_t d_moe_a_cap = 0; // device expert liveness\n"
            ));
            os.push_str(&format!(
                "  unsigned moe_cap = {}; unsigned moe_dim = {}; unsigned moe_ffn = {}; unsigned moe_k0 = {};\n",
                moe_e, moe_d, moe_h, moe_k0
            ));
            os.push_str(&format!(
                "  size_t moe_off_g = {}, moe_off_e1 = {}, moe_off_e2 = {};\n  unsigned moe_count = 0;\n",
                moe_off_g, moe_off_e1, moe_off_e2
            ));
        }
        for id in &sorted {
            let is_weight = ids.get(id).map_or(false, |i| i.is_weight);
            if is_weight {
                continue;
            }
            if *id == x_id {
                continue;
            }
            let is_input = ids.get(id).map_or(false, |i| i.is_input);
            if is_input {
                continue;
            }
            os.push_str(&format!("  float* d_{} = nullptr; size_t d_{}_cap = 0;\n", id, id));
            add_free(&format!("d_{}", id), &format!("d_{}_cap", id), &mut frees);
        }
        // Dropout masks and optimizer moments are only touched by the training
        // core, but they are per-model state, so they live in the context too.
        if let Some(t) = tfn {
            for instr in &t.instructions {
                if instr.op == MLIROp::Dropout {
                    os.push_str(&format!(
                        "  float* d_dm_{} = nullptr; size_t d_dm_{}_cap = 0; // dropout mask (scaled)\n",
                        instr.result_id, instr.result_id
                    ));
                    add_free(
                        &format!("d_dm_{}", instr.result_id),
                        &format!("d_dm_{}_cap", instr.result_id),
                        &mut frees,
                    );
                }
            }
            for instr in &t.instructions {
                if instr.op != MLIROp::OptStep {
                    continue;
                }
                let mut i = 0;
                while i + 1 < instr.operands.len() {
                    let wt = &instr.operands[i];
                    os.push_str(&format!(
                        "  float* d_am{} = nullptr; size_t d_am{}_cap = 0; // AdamW/Muon momentum\n",
                        wt, wt
                    ));
                    os.push_str(&format!(
                        "  float* d_av{} = nullptr; size_t d_av{}_cap = 0; // AdamW variance\n",
                        wt, wt
                    ));
                    add_free(&format!("d_am{}", wt), &format!("d_am{}_cap", wt), &mut frees);
                    add_free(&format!("d_av{}", wt), &format!("d_av{}_cap", wt), &mut frees);
                    i += 2;
                }
            }
        }
        os.push_str("  size_t adam_step = 0;    // AdamW bias-correction step counter\n");
        os.push_str("  int64_t lr_step = 0;     // LR schedule step counter (runtime wrapper)\n");
        os.push_str("  unsigned drop_seed = 0;  // dropout RNG seed counter\n");
        os.push_str("  NSContext() {}\n");
        os.push_str("  ~NSContext() {\n");
        os.push_str("    cudaFree(d_wb); d_wb_cap = 0;\n");
        os.push_str("    cudaFree(d_xi); d_xi_cap = 0;\n");
        os.push_str("    cudaFree(d_yl); d_yl_cap = 0;\n");
        os.push_str("    cudaFree(d_row); d_row_cap = 0;\n");
        os.push_str("    cudaFree(d_asr); d_asr_cap = 0;\n");
        os.push_str("    if (hrow) { free((void*)hrow); hrow = nullptr; hrow_cap = 0; }\n");
        if moe_e > 0 {
            os.push_str("    cudaFree(d_moe_a); d_moe_a_cap = 0;\n");
            os.push_str("    if (h_moe_a) { free((void*)h_moe_a); h_moe_a = nullptr; h_moe_a_cap = 0; }\n");
        }
        for s in &frees {
            os.push_str(s);
        }
        os.push_str("  }\n");
        os.push_str("  NSContext(const NSContext&) = delete;\n");
        os.push_str("  NSContext& operator=(const NSContext&) = delete;\n");
        os.push_str("} NSContext;\n\n");
        if moe_e > 0 {
            os.push_str("// Allocate the host mask (first K0 experts live), upload it, and\n");
            os.push_str("// keep the device copy in sync. Idempotent per context.\n");
            os.push_str("static void ns_moe_mask_sync(NSContext* ctx) {\n");
            os.push_str("  if (ctx->h_moe_a) return;\n");
            os.push_str("  ctx->h_moe_a = (uint8_t*)calloc(ctx->moe_cap, 1);\n");
            os.push_str("  ctx->h_moe_a_cap = ctx->moe_cap;\n");
            os.push_str("  for (unsigned e = 0; e < ctx->moe_k0; e++) ctx->h_moe_a[e] = 1;\n");
            os.push_str("  ctx->moe_count = ctx->moe_k0;\n");
            os.push_str(
                "  if (ns_cu_reserve_u8(&ctx->d_moe_a, &ctx->d_moe_a_cap, ctx->moe_cap, 0)) return;\n",
            );
            os.push_str(
                "  cudaMemcpy(ctx->d_moe_a, ctx->h_moe_a, ctx->moe_cap, cudaMemcpyHostToDevice);\n",
            );
            os.push_str("}\n\n");
        }
    }

    let id_info = |id: &str| ids.get(id).copied().unwrap_or_default();
    let dbuf = |id: &str| -> String {
        let is_w = ids.get(id).map_or(false, |i| i.is_weight);
        if is_w && woff.contains_key(id) {
            format!("(ctx->d_wb + {})", woff[id])
        } else if ids.get(id).map_or(false, |i| i.is_input) {
            String::from("ctx->d_xi")
        } else {
            format!("ctx->d_{}", id)
        }
    };

    // INDEX/SCATTER carry constant int64 index lists; materialize them as
    // device-side constants (referenced by the launched kernels).
    for instr in &fn_.instructions {
        if (instr.op == MLIROp::Index || instr.op == MLIROp::Scatter)
            && !instr.ints_attr.is_empty()
            && !instr.result_id.is_empty()
        {
            os.push_str(&format!(
                "__device__ const int64_t ns_idx_{}[{}] = {{ ",
                instr.result_id,
                instr.ints_attr.len()
            ));
            for (k, v) in instr.ints_attr.iter().enumerate() {
                os.push_str(&format!("{}", v));
                if k + 1 < instr.ints_attr.len() {
                    os.push_str(", ");
                }
            }
            os.push_str(" };\n");
        }
    }

    // ---- Forward launcher (runs entirely on device scratch buffers). ----
    os.push_str(&format!(
        "extern \"C\" void {}(\n    NSContext* ctx, const float* input, float* output, size_t n) {{\n",
        opts.function_name
    ));
    os.push_str("  if (!ctx || !ctx->d_wb || n == 0) return;\n");
    os.push_str(&format!("  const int M = (int)(n / {});\n", in_cols));
    os.push_str("  if (M <= 0) return;\n");
    os.push_str("  if (ns_cu_reserve(&ctx->d_xi, &ctx->d_xi_cap, n * sizeof(float), 0)) return;\n");
    os.push_str("  cudaMemcpy(ctx->d_xi, input, n * sizeof(float), cudaMemcpyHostToDevice);\n");

    // Feature width of each per-row tensor, propagated forward over the
    // instruction list (ops are in topological order). Ops lowered from bare
    // function calls (dropout, plain activations, ...) carry no result_type in
    // CuIdInfo, so width comes from the producing instruction instead.
    let wcols: BTreeMap<String, i64> = {
        let mut cur: BTreeMap<String, i64> = BTreeMap::new();
        for a in &fn_.instructions {
            let ida = a.result_id.clone();
            // Seed widths from statically typed operands first; do NOT use
            // operator[] (it would insert 0 entries that break row_dims).
            for op in &a.operands {
                let inf = ids.get(op).copied().unwrap_or_default();
                if inf.cols > 0 {
                    cur.insert(op.clone(), inf.cols);
                }
            }
            if (a.op == MLIROp::Matmul || a.op == MLIROp::Fused) && a.operands.len() >= 2 {
                let inf = ids.get(&a.operands[1]).copied().unwrap_or_default();
                if inf.cols > 0 {
                    cur.insert(ida.clone(), inf.cols);
                }
            } else if !ida.is_empty() && !a.operands.is_empty() {
                if let Some(&c) = cur.get(&a.operands[0]) {
                    if c > 0 {
                        cur.insert(ida.clone(), c);
                    }
                }
            }
        }
        cur
    };
    let row_dims = |v: &str| -> i64 {
        if !v.is_empty() {
            if let Some(&c) = wcols.get(v) {
                if c > 0 {
                    return c;
                }
            }
        }
        if !v.is_empty() {
            let inf = ids.get(v).copied().unwrap_or_default();
            if inf.cols > 0 {
                return inf.cols;
            }
        }
        for a in &fn_.instructions {
            if (a.op == MLIROp::Matmul || a.op == MLIROp::Fused)
                && a.operands.len() >= 2
                && a.operands[0] == v
            {
                let r = ids.get(&a.operands[1]).map_or(0, |i| i.rows);
                if r > 0 {
                    return r;
                }
            }
        }
        1
    };

    let mut result_id = String::new();
    for instr in &fn_.instructions {
        let id = instr.result_id.clone();
        let op = instr.op;
        match op {
            MLIROp::Matmul => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&b);
                let (ks, ns) = (inf.rows, inf.cols);
                if ks <= 0 || ns <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for matmul '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, M, {}, {});\n",
                    ns,
                    dbuf(&a),
                    dbuf(&b),
                    id,
                    ks,
                    ns
                ));
            }
            MLIROp::Fused => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&b);
                let (ks, ns) = (inf.rows, inf.cols);
                if ks <= 0 || ns <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for fused group '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, M, {}, {});\n",
                    ns,
                    dbuf(&a),
                    dbuf(&b),
                    id,
                    ks,
                    ns
                ));
                let mut group_idx: Option<usize> = None;
                for (gi, g) in module.fused_groups.iter().enumerate() {
                    if g.result_id == instr.result_id {
                        group_idx = Some(gi);
                    }
                }
                if let Some(gi) = group_idx {
                    let g = &module.fused_groups[gi];
                    let mut extra: usize = 0;
                    for opname in &g.ops {
                        if opname == "+" || opname == "-" || opname == "*" || opname == "/" {
                            let code: i32 = if opname == "-" {
                                1
                            } else if opname == "*" {
                                2
                            } else if opname == "/" {
                                3
                            } else {
                                0
                            };
                            let rhs = if extra < g.epilogue_operands.len() {
                                g.epilogue_operands[extra].clone()
                            } else {
                                String::new()
                            };
                            let nb = if id_info(&rhs).static_numel > 0 {
                                id_info(&rhs).static_numel as i32
                            } else {
                                1
                            };
                            os.push_str(&format!(
                                "  NS_LAUNCH1(ns_binop_kernel, M * {}, ctx->d_{}, {}, ctx->d_{}, M * {}, {}, {});\n",
                                ns, id, dbuf(&rhs), id, ns, nb, code
                            ));
                            extra += 1;
                        } else if opname == "layernorm" {
                            os.push_str(&format!(
                                "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_{}, M * {}, {});\n",
                                id, ns, ns
                            ));
                        } else if opname == "softmax" {
                            os.push_str(&format!(
                                "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_{}, M * {}, {});\n",
                                id, ns, ns
                            ));
                        } else {
                            os.push_str(&format!(
                                "  NS_LAUNCH1(ns_act_kernel, M * {}, ctx->d_{}, ctx->d_{}, M * {}, {});\n",
                                ns,
                                id,
                                id,
                                ns,
                                cu_act_code_attr(opname)
                            ));
                        }
                    }
                }
            }
            MLIROp::Relu
            | MLIROp::LeakyRelu
            | MLIROp::Sigmoid
            | MLIROp::Tanh
            | MLIROp::Swish
            | MLIROp::Gelu
            | MLIROp::Silu
            | MLIROp::Identity => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&in_);
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_act_kernel, M * {}, {}, ctx->d_{}, M * {}, {});\n",
                    ns,
                    dbuf(&in_),
                    id,
                    ns,
                    cu_act_code(op)
                ));
            }
            MLIROp::Dropout => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&in_);
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_copy_kernel, M * {}, {}, ctx->d_{}, M * {});\n",
                    ns,
                    dbuf(&in_),
                    id,
                    ns
                ));
            }
            MLIROp::ElementwiseBinop => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&a);
                let mut code: i32 = 0;
                if instr.attribute == "-" {
                    code = 1;
                } else if instr.attribute == "*" {
                    code = 2;
                } else if instr.attribute == "/" {
                    code = 3;
                }
                let nb = if id_info(&b).static_numel > 0 {
                    id_info(&b).static_numel as i32
                } else {
                    1
                };
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_binop_kernel, M * {}, {}, {}, ctx->d_{}, M * {}, {}, {});\n",
                    ns,
                    dbuf(&a),
                    dbuf(&b),
                    id,
                    ns,
                    nb,
                    code
                ));
            }
            MLIROp::Layernorm | MLIROp::Softmax => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let last = row_dims(&in_);
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, last
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_copy_kernel, M * {}, {}, ctx->d_{}, M * {});\n",
                    last,
                    dbuf(&in_),
                    id,
                    last
                ));
                if op == MLIROp::Layernorm {
                    os.push_str(&format!(
                        "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_{}, M * {}, {});\n",
                        id, last, last
                    ));
                } else {
                    os.push_str(&format!(
                        "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_{}, M * {}, {});\n",
                        id, last, last
                    ));
                }
            }
            MLIROp::Constant => {
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, sizeof(float), 0)) return;\n",
                    id, id
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_fill_kernel, 1, ctx->d_{}, 1, ({})f);\n",
                    id, instr.attribute
                ));
            }
            MLIROp::Transpose => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&in_);
                let r = if inf.rows > 0 { inf.rows } else { 1 };
                let c = if inf.cols > 0 { inf.cols } else { 1 };
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({} * {}) * sizeof(float), 0)) return;\n",
                    id, id, c, r
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_transpose2d_kernel, (({} + 15) / 16) * (({} + 15) / 16), {}, ctx->d_{}, {}, {});\n",
                    c,
                    r,
                    dbuf(&in_),
                    id,
                    r,
                    c
                ));
            }
            MLIROp::Concat => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let axis = if !instr.attribute.is_empty() && instr.attribute == "0" {
                    0
                } else {
                    1
                };
                let ia = id_info(&a);
                let ib = id_info(&b);
                let ca = if ia.cols > 0 { ia.cols } else { 1 };
                let cb = if ib.cols > 0 { ib.cols } else { 1 };
                let co = ca + cb;
                result_id = id.clone();
                if axis == 1 {
                    os.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, co
                    ));
                    os.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS(ns_concat2_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, M, {}, {});\n",
                        co,
                        dbuf(&a),
                        dbuf(&b),
                        id,
                        ca,
                        co
                    ));
                } else {
                    let ba = if ia.rows > 0 { ia.rows } else { 1 };
                    let bb = if ib.rows > 0 { ib.rows } else { 1 };
                    let d = if ia.cols > 0 { ia.cols } else { 1 };
                    os.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({} * {}) * sizeof(float), 0)) return;\n",
                        id, id, ba + bb, d
                    ));
                    os.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS(ns_concat0_kernel, (({} + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, {}, {}, {});\n",
                        ba + bb,
                        d,
                        dbuf(&a),
                        dbuf(&b),
                        id,
                        ba,
                        bb,
                        d
                    ));
                }
            }
            MLIROp::Reshape => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&in_);
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_copy_kernel, M * {}, {}, ctx->d_{}, M * {});\n",
                    ns,
                    dbuf(&in_),
                    id,
                    ns
                ));
            }
            MLIROp::Slice => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let mut axis: i64 = 1;
                let mut s: i64 = 0;
                let mut e: i64 = 0;
                {
                    let parts: Vec<&str> = instr.attribute.split(':').collect();
                    if parts.first().map_or(false, |p| !p.is_empty()) {
                        if let Ok(v) = parts[0].trim().parse::<i64>() {
                            axis = v;
                        }
                    }
                    if parts.len() > 1 {
                        if let Ok(v) = parts[1].trim().parse::<i64>() {
                            s = v;
                        }
                    }
                    if parts.len() > 2 {
                        if let Ok(v) = parts[2].trim().parse::<i64>() {
                            e = v;
                        }
                    }
                }
                let inf = id_info(&in_);
                let c = if inf.cols > 0 { inf.cols } else { 1 };
                let sr = if inf.rows > 0 { inf.rows } else { -1 };
                let rows_e = if sr > 0 {
                    format!("{}", sr)
                } else {
                    String::from("M")
                };
                result_id = id.clone();
                if axis == 0 {
                    let ow = (e - s) * c;
                    os.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * sizeof(float), 0)) return;\n",
                        id, id, ow
                    ));
                    os.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS(ns_slicerows_kernel, (({} + 255) / 256), {}, ctx->d_{}, {}, {}, {});\n",
                        ow,
                        dbuf(&in_),
                        id,
                        c,
                        s,
                        e
                    ));
                } else {
                    let ow = e - s;
                    os.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * {} * sizeof(float), 0)) return;\n",
                        id, id, rows_e, ow
                    ));
                    os.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS(ns_slice2_kernel, (({} + 15) / 16) * (({} + 15) / 16), {}, ctx->d_{}, {}, {}, {}, {});\n",
                        rows_e,
                        ow,
                        dbuf(&in_),
                        id,
                        rows_e,
                        c,
                        s,
                        e
                    ));
                }
            }
            MLIROp::Index | MLIROp::Scatter => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let axis = instr.int_attr as i32;
                let inf = id_info(&in_);
                let c = if inf.cols > 0 { inf.cols } else { 1 };
                let l = instr.ints_attr.len() as i64;
                let sr = if inf.rows > 0 { inf.rows } else { -1 };
                let rows_e = if sr > 0 {
                    format!("{}", sr)
                } else {
                    String::from("M")
                };
                result_id = id.clone();
                let upd = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                if op == MLIROp::Index {
                    if axis == 0 {
                        let ow = l * c;
                        os.push_str(&format!(
                            "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * sizeof(float), 0)) return;\n",
                            id, id, ow
                        ));
                        os.push_str(&format!(
                            "  NS_LAUNCH_BLOCKS(ns_indexrows_kernel, (({} + 255) / 256), {}, ns_idx_{}, ctx->d_{}, {}, {});\n",
                            ow,
                            dbuf(&in_),
                            id,
                            id,
                            l,
                            c
                        ));
                    } else {
                        os.push_str(&format!(
                            "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * {} * sizeof(float), 0)) return;\n",
                            id, id, rows_e, l
                        ));
                        os.push_str(&format!(
                            "  NS_LAUNCH_BLOCKS(ns_index_kernel, (({} + 15) / 16) * (({} + 15) / 16), {}, ns_idx_{}, ctx->d_{}, {}, {}, {});\n",
                            rows_e,
                            l,
                            dbuf(&in_),
                            id,
                            id,
                            rows_e,
                            c,
                            l
                        ));
                    }
                } else {
                    if axis == 0 {
                        let ow = (if sr > 0 { sr } else { 0 }) * c;
                        os.push_str(&format!(
                            "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * sizeof(float), 0)) return;\n",
                            id, id, ow
                        ));
                        os.push_str(&format!(
                            "  NS_LAUNCH_BLOCKS(ns_scatterrows_kernel, (({} + 255) / 256), {}, ns_idx_{}, {}, ctx->d_{}, {}, {}, {});\n",
                            ow,
                            dbuf(&in_),
                            id,
                            dbuf(&upd),
                            id,
                            if sr > 0 { sr } else { 1 },
                            c,
                            l
                        ));
                    } else {
                        os.push_str(&format!(
                            "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t){} * {} * sizeof(float), 0)) return;\n",
                            id, id, rows_e, c
                        ));
                        os.push_str(&format!(
                            "  NS_LAUNCH_BLOCKS(ns_scatter_kernel, (({} + 15) / 16) * (({} + 15) / 16), {}, ns_idx_{}, {}, ctx->d_{}, {}, {}, {});\n",
                            rows_e,
                            c,
                            dbuf(&in_),
                            id,
                            dbuf(&upd),
                            id,
                            rows_e,
                            c,
                            l
                        ));
                    }
                }
            }
            MLIROp::LayerEmbedding => {
                let wt = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let idx = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&wt);
                let v = if inf.rows > 0 { inf.rows } else { 1 };
                let d = if inf.cols > 0 { inf.cols } else { 1 };
                result_id = id.clone();
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_embedding_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, M, {}, {});\n",
                    d,
                    dbuf(&wt),
                    dbuf(&idx),
                    id,
                    v,
                    d
                ));
            }
            MLIROp::LayerAttention => {
                let x = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let wq = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let wk = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let wv = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let wo = if instr.operands.len() > 4 {
                    instr.operands[4].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 {
                    instr.int_attr
                } else if id_info(&x).cols > 0 {
                    id_info(&x).cols
                } else {
                    1
                };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                result_id = id.clone();
                // Scratch: Q, K, V and the pre-projection context (M*D floats).
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                os.push_str("  { float* qs = 0; float* ks = 0; float* vs = 0; float* cs = 0;\n");
                os.push_str(&format!(
                    "    if (cudaMalloc(&qs, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                os.push_str(&format!(
                    "        cudaMalloc(&ks, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                os.push_str(&format!(
                    "        cudaMalloc(&vs, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                os.push_str(&format!(
                    "        cudaMalloc(&cs, (size_t)M * {} * sizeof(float))) return;\n",
                    d
                ));
                os.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, qs, M, {}, {});\n",
                    d,
                    dbuf(&x),
                    dbuf(&wq),
                    d,
                    d
                ));
                os.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ks, M, {}, {});\n",
                    d,
                    dbuf(&x),
                    dbuf(&wk),
                    d,
                    d
                ));
                os.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, vs, M, {}, {});\n",
                    d,
                    dbuf(&x),
                    dbuf(&wv),
                    d,
                    d
                ));
                os.push_str("    __attribute__((unused)) static bool ns_a_ok_ = (cudaFuncSetAttribute(ns_attention_core_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536), true);\n");
                os.push_str(&format!(
                    "    ns_attention_core_kernel<<<{}, 256, (size_t)M * M * sizeof(float)>>>(qs, ks, vs, cs, M, {}, {}, 1.0f / sqrtf((float)({} / {})){});\n",
                    h,
                    d,
                    h,
                    d,
                    h,
                    if !instr.ints_attr.is_empty() {
                        if instr.ints_attr[0] != 0 {
                            ", 1"
                        } else {
                            ", 0"
                        }
                    } else {
                        ", 0"
                    }
                ));
                os.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), cs, {}, ctx->d_{}, M, {}, {});\n",
                    d,
                    dbuf(&wo),
                    id,
                    d,
                    d
                ));
                os.push_str("    cudaFree(qs); cudaFree(ks); cudaFree(vs); cudaFree(cs); }\n");
            }
            MLIROp::LayerMoe => {
                let x = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let wg = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let we1 = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let we2 = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 {
                    instr.int_attr
                } else if id_info(&x).cols > 0 {
                    id_info(&x).cols
                } else {
                    1
                };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                result_id = id.clone();
                os.push_str("  ns_moe_mask_sync(ctx);\n");
                os.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                os.push_str(&format!(
                    "  NS_LAUNCH1(ns_moe_kernel, M, {}, {}, {}, {}, ctx->d_moe_a, ctx->d_{}, M, {}, {}, {});\n",
                    dbuf(&x),
                    dbuf(&wg),
                    dbuf(&we1),
                    dbuf(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            _ => {}
        }
    }
    os.push_str("  cudaDeviceSynchronize();\n");
    os.push_str("  if (output) {\n");
    if result_id.is_empty() {
        os.push_str("    (void)output;\n");
    } else {
        os.push_str(&format!(
            "    cudaMemcpy(output, ctx->d_{}, (size_t)M * {} * sizeof(float), cudaMemcpyDeviceToHost);\n",
            result_id, out_cols
        ));
    }
    os.push_str("  }\n");
    os.push_str("}\n\n");

    if opts.emit_runtime_driver {
        os.push_str("// ---- CUDA C-ABI runtime driver ----\n");
        os.push_str("typedef struct ns_model { float* h_w; NSContext* ctx; size_t n; } ns_model;\n");
        os.push_str("typedef struct ns_weight_desc { const char* name; size_t offset; size_t count; } ns_weight_desc;\n");
        os.push_str("typedef struct ns_weight_layout { size_t num_weights; const ns_weight_desc* desc; } ns_weight_layout;\n\n");
        os.push_str("namespace { \n");
        if worder.is_empty() {
            os.push_str("static const ns_weight_desc ns_desc[1] = {};\n");
        } else {
            os.push_str(&format!(
                "static const ns_weight_desc ns_desc[{}] = {{\n",
                worder.len()
            ));
            for (i, w) in worder.iter().enumerate() {
                let szn = if id_info(w).static_numel > 0 {
                    id_info(w).static_numel
                } else {
                    1
                };
                os.push_str(&format!("    {{\"{}\", {}, {}}}", w, woff[w], szn));
                os.push_str(if i + 1 < worder.len() { ",\n" } else { "\n" });
            }
            os.push_str("};\n");
        }
        os.push_str(&format!(
            "static const ns_weight_layout ns_layout = {{ {}, ns_desc }};\n",
            worder.len()
        ));
        os.push_str(&format!("static const size_t ns_weight_total = {};\n", weight_total));
        os.push_str(&format!(
            "static const int64_t ns_in_cols = {}, ns_out_cols = {};\n",
            in_cols, out_cols
        ));
        if moe_e > 0 {
            os.push_str(&format!(
                "static const size_t ns_moe_cap = {}, ns_moe_dim = {}, ns_moe_ffn = {}, ns_moe_k0 = {}, ns_moe_off_g = {}, ns_moe_off_e1 = {}, ns_moe_off_e2 = {};\n",
                moe_e, moe_d, moe_h, moe_k0, moe_off_g, moe_off_e1, moe_off_e2
            ));
        }
        os.push_str("}\n\n");
        os.push_str("extern \"C\" ns_model* ns_runtime_init(const float* weights, size_t num_floats) {\n");
        os.push_str("    if (num_floats != ns_weight_total) return nullptr;\n");
        os.push_str("    float* h_w = new float[ns_weight_total];\n");
        os.push_str("    NSContext* ctx = new NSContext();\n");
        os.push_str("    if (!h_w || !ctx) { delete[] h_w; delete ctx; return nullptr; }\n");
        os.push_str("    std::memcpy(h_w, weights, ns_weight_total * sizeof(float));\n");
        os.push_str("    if (ns_cu_reserve(&ctx->d_wb, &ctx->d_wb_cap, ns_weight_total * sizeof(float), 0)) { delete[] h_w; delete ctx; return nullptr; }\n");
        os.push_str("    cudaMemcpy(ctx->d_wb, weights, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n");
        if moe_e > 0 {
            os.push_str("    ns_moe_mask_sync(ctx);\n");
        }
        os.push_str("    ns_model* m = new ns_model{ h_w, ctx, ns_weight_total };\n");
        os.push_str("    if (!m) { delete[] h_w; delete ctx; return nullptr; }\n");
        os.push_str("    return m;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" int ns_eval_infer(ns_model* m, const float* input, float* output, size_t input_numel) {\n");
        os.push_str("    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n");
        os.push_str(&format!("    {}(m->ctx, input, output, input_numel);\n", opts.function_name));
        os.push_str("    return 0;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" size_t ns_model_output_numel(const ns_model* m, size_t input_numel) {\n");
        os.push_str("    (void)m; return (size_t)((int64_t)input_numel / ns_in_cols) * (size_t)ns_out_cols;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" size_t ns_model_weight_count(const ns_model* m) {\n");
        os.push_str("    (void)m; return ns_weight_total;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" size_t ns_weight_count_static(void) {\n");
        os.push_str("    return ns_weight_total;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" int ns_model_get_weights(const ns_model* m, float* out, size_t n) {\n");
        os.push_str("    if (!m || !m->ctx || !out || n != ns_weight_total) return -1;\n");
        os.push_str("    std::memcpy(out, m->h_w, n * sizeof(float));\n");
        os.push_str("    return 0;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" const ns_weight_layout* ns_model_layout(const ns_model* m) {\n");
        os.push_str("    (void)m; return &ns_layout;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" void ns_free(ns_model* m) {\n");
        os.push_str("    if (!m) return; delete m->ctx; delete[] m->h_w; delete m;\n");
        os.push_str("}\n\n");
        os.push_str("// Persist / restore the weight blob + the MoE liveness mask. Format:\n");
        os.push_str("// 4-byte magic \"NSM2\", size_t float count, raw weights, then (MoE\n");
        os.push_str("// models) uint32 n_layers, uint32 capacity, <capacity> mask bytes.\n");
        os.push_str("// Legacy \"NSM1\" files (weights only) load with all experts alive.\n");
        os.push_str("extern \"C\" int ns_save_checkpoint(const ns_model* m, const char* path) {\n");
        os.push_str("    if (!m || !m->h_w || !path) return -1;\n");
        os.push_str("    FILE* fp = fopen(path, \"wb\");\n");
        os.push_str("    if (!fp) return -1;\n");
        os.push_str("    const unsigned magic = 0x4E534D32u; /* \"NSM2\" */\n");
        os.push_str("    if (fwrite(&magic, sizeof(magic), 1, fp) != 1 ||\n");
        os.push_str("        fwrite(&ns_weight_total, sizeof(ns_weight_total), 1, fp) != 1 ||\n");
        os.push_str("        fwrite(m->h_w, sizeof(float), ns_weight_total, fp) != ns_weight_total) {\n");
        os.push_str("        fclose(fp); return -1;\n");
        os.push_str("    }\n");
        if moe_e > 0 {
            os.push_str("    if (!m->ctx->h_moe_a) { fclose(fp); return -1; }\n");
            os.push_str("    { const unsigned n_layers = 1U;\n");
            os.push_str("      if (fwrite(&n_layers, sizeof(n_layers), 1, fp) != 1 ||\n");
            os.push_str("          fwrite(&ns_moe_cap, sizeof(ns_moe_cap), 1, fp) != 1 ||\n");
            os.push_str("          fwrite(m->ctx->h_moe_a, 1, ns_moe_cap, fp) != ns_moe_cap) {\n");
            os.push_str("          fclose(fp); return -1;\n");
            os.push_str("      } }\n");
        }
        os.push_str("    fclose(fp); return 0;\n");
        os.push_str("}\n\n");
        os.push_str("extern \"C\" int ns_load_checkpoint(ns_model* m, const char* path) {\n");
        os.push_str("    if (!m || !m->ctx || !m->h_w || !path) return -1;\n");
        os.push_str("    FILE* fp = fopen(path, \"rb\");\n");
        os.push_str("    if (!fp) return -1;\n");
        os.push_str("    unsigned magic = 0; size_t n = 0;\n");
        os.push_str("    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||\n");
        os.push_str("        fread(&n, sizeof(n), 1, fp) != 1 ||\n");
        os.push_str("        (magic != 0x4E534D31u && magic != 0x4E534D32u) || n != ns_weight_total ||\n");
        os.push_str("        fread(m->h_w, sizeof(float), n, fp) != n) {\n");
        os.push_str("        fclose(fp); return -1;\n");
        os.push_str("    }\n");
        if moe_e > 0 {
            os.push_str("    ns_moe_mask_sync(m->ctx);\n");
            os.push_str("    if (magic == 0x4E534D32u) {\n");
            os.push_str("      unsigned n_layers = 0; size_t cap = 0;\n");
            os.push_str("      if (fread(&n_layers, sizeof(n_layers), 1, fp) != 1 ||\n");
            os.push_str("          fread(&cap, sizeof(cap), 1, fp) != 1 ||\n");
            os.push_str("          n_layers != 1U || cap != ns_moe_cap ||\n");
            os.push_str("          fread(m->ctx->h_moe_a, 1, cap, fp) != cap) {\n");
            os.push_str("          fclose(fp); return -1;\n");
            os.push_str("      }\n");
            os.push_str("    } else {\n");
            os.push_str("      memset(m->ctx->h_moe_a, 0, ns_moe_cap);\n");
            os.push_str("      for (size_t e = 0; e < ns_moe_k0; e++) m->ctx->h_moe_a[e] = 1;\n");
            os.push_str("    }\n");
            os.push_str("    m->ctx->moe_count = 0;\n");
            os.push_str("    for (size_t e = 0; e < ns_moe_cap; e++) m->ctx->moe_count += m->ctx->h_moe_a[e];\n");
            os.push_str("    cudaMemcpy(m->ctx->d_moe_a, m->ctx->h_moe_a, ns_moe_cap, cudaMemcpyHostToDevice);\n");
        }
        os.push_str("    fclose(fp);\n");
        os.push_str("    cudaMemcpy(m->ctx->d_wb, m->h_w, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n");
        os.push_str("    return 0;\n");
        os.push_str("}\n\n");
        if moe_e > 0 {
            os.push_str("// ---- MoE expert lifecycle (capacity is compile-time; liveness\n");
            os.push_str("//      is runtime state in the context) ----\n");
            os.push_str("extern \"C\" size_t ns_expert_count(const ns_model* m) {\n");
            os.push_str("    if (!m || !m->ctx) return 0;\n");
            os.push_str("    return m->ctx->moe_count;\n");
            os.push_str("}\n\n");
            os.push_str("extern \"C\" size_t ns_expert_birth(ns_model* m, int n) {\n");
            os.push_str("    if (!m || !m->ctx || !m->h_w || n <= 0) return m && m->ctx ? m->ctx->moe_count : 0;\n");
            os.push_str("    ns_moe_mask_sync(m->ctx);\n");
            os.push_str("    if (m->ctx->moe_count >= ns_moe_cap) return m->ctx->moe_count;\n");
            os.push_str("    int src = -1;\n");
            os.push_str("    for (unsigned e = 0; e < ns_moe_cap && src < 0; e++)\n");
            os.push_str("      if (m->ctx->h_moe_a[e]) src = (int)e;\n");
            os.push_str("    if (src < 0) return 0;\n");
            os.push_str("    const size_t s1 = (size_t)src * ns_moe_dim * ns_moe_ffn;\n");
            os.push_str("    const size_t s2 = (size_t)src * ns_moe_ffn * ns_moe_dim;\n");
            os.push_str("    for (unsigned e = 0; e < ns_moe_cap && n > 0; e++) {\n");
            os.push_str("      if (m->ctx->h_moe_a[e]) continue;\n");
            os.push_str("      const size_t g2 = (size_t)e * ns_moe_dim * ns_moe_ffn;\n");
            os.push_str("      const size_t g3 = (size_t)e * ns_moe_ffn * ns_moe_dim;\n");
            os.push_str("      for (size_t k = 0; k < ns_moe_dim; k++)\n");
            os.push_str("        m->h_w[ns_moe_off_g + k * ns_moe_cap + e] = m->h_w[ns_moe_off_g + k * ns_moe_cap + (size_t)src];\n");
            os.push_str("      for (size_t q = 0; q < ns_moe_dim * ns_moe_ffn; q++)\n");
            os.push_str("        m->h_w[ns_moe_off_e1 + g2 + q] = m->h_w[ns_moe_off_e1 + s1 + q];\n");
            os.push_str("      for (size_t q = 0; q < ns_moe_ffn * ns_moe_dim; q++)\n");
            os.push_str("        m->h_w[ns_moe_off_e2 + g3 + q] = m->h_w[ns_moe_off_e2 + s2 + q];\n");
            os.push_str("      m->ctx->h_moe_a[e] = 1;\n");
            os.push_str("      m->ctx->moe_count++;\n");
            os.push_str("      n--;\n");
            os.push_str("    }\n");
            os.push_str("    cudaMemcpy(m->ctx->d_wb, m->h_w, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n");
            os.push_str("    cudaMemcpy(m->ctx->d_moe_a, m->ctx->h_moe_a, ns_moe_cap, cudaMemcpyHostToDevice);\n");
            os.push_str("    return m->ctx->moe_count;\n");
            os.push_str("}\n\n");
            os.push_str("extern \"C\" size_t ns_expert_merge(ns_model* m, int a, int b) {\n");
            os.push_str("    if (!m || !m->ctx || !m->h_w || a < 0 || b < 0 || a == b ||\n");
            os.push_str("        a >= (int)ns_moe_cap || b >= (int)ns_moe_cap ||\n");
            os.push_str("        !m->ctx->h_moe_a[a] || !m->ctx->h_moe_a[b])\n");
            os.push_str("        return m && m->ctx ? m->ctx->moe_count : 0;\n");
            os.push_str("    const size_t s1 = (size_t)a * ns_moe_dim * ns_moe_ffn;\n");
            os.push_str("    const size_t s1b = (size_t)b * ns_moe_dim * ns_moe_ffn;\n");
            os.push_str("    const size_t t2 = (size_t)a * ns_moe_ffn * ns_moe_dim;\n");
            os.push_str("    const size_t t2b = (size_t)b * ns_moe_ffn * ns_moe_dim;\n");
            os.push_str("    for (size_t k = 0; k < ns_moe_dim; k++)\n");
            os.push_str("      m->h_w[ns_moe_off_g + k * ns_moe_cap + (size_t)a] =\n");
            os.push_str("        0.5f * (m->h_w[ns_moe_off_g + k * ns_moe_cap + (size_t)a] +\n");
            os.push_str("                m->h_w[ns_moe_off_g + k * ns_moe_cap + (size_t)b]);\n");
            os.push_str("    for (size_t q = 0; q < ns_moe_dim * ns_moe_ffn; q++)\n");
            os.push_str("      m->h_w[ns_moe_off_e1 + s1 + q] = 0.5f * (m->h_w[ns_moe_off_e1 + s1 + q] + m->h_w[ns_moe_off_e1 + s1b + q]);\n");
            os.push_str("    for (size_t q = 0; q < ns_moe_ffn * ns_moe_dim; q++)\n");
            os.push_str("      m->h_w[ns_moe_off_e2 + t2 + q] = 0.5f * (m->h_w[ns_moe_off_e2 + t2 + q] + m->h_w[ns_moe_off_e2 + t2b + q]);\n");
            os.push_str("    m->ctx->h_moe_a[b] = 0;\n");
            os.push_str("    m->ctx->moe_count--;\n");
            os.push_str("    cudaMemcpy(m->ctx->d_wb, m->h_w, ns_weight_total * sizeof(float), cudaMemcpyHostToDevice);\n");
            os.push_str("    cudaMemcpy(m->ctx->d_moe_a, m->ctx->h_moe_a, ns_moe_cap, cudaMemcpyHostToDevice);\n");
            os.push_str("    return m->ctx->moe_count;\n");
            os.push_str("}\n\n");
            os.push_str("extern \"C\" size_t ns_expert_kill(ns_model* m, int k) {\n");
            os.push_str("    if (!m || !m->ctx || k < 0 || k >= (int)ns_moe_cap || !m->ctx->h_moe_a[k])\n");
            os.push_str("        return m && m->ctx ? m->ctx->moe_count : 0;\n");
            os.push_str("    m->ctx->h_moe_a[k] = 0;\n");
            os.push_str("    m->ctx->moe_count--;\n");
            os.push_str("    cudaMemcpy(m->ctx->d_moe_a, m->ctx->h_moe_a, ns_moe_cap, cudaMemcpyHostToDevice);\n");
            os.push_str("    return m->ctx->moe_count;\n");
            os.push_str("}\n\n");
        }

        if let Some(t) = tfn {
            os.push_str("\n// ---- CUDA training core (forward + backward + optimizer on device) ----\n");
            os.push_str(&emit_train_core_cuda(t, in_cols, out_cols)?);
            os.push_str("\n");
            os.push_str(&lr_schedule_source());
            os.push_str("\nextern \"C\" int ns_runtime_train_step(ns_model* m, const float* input,\n");
            os.push_str("                                        const float* labels, size_t input_numel,\n");
            os.push_str("                                        float* loss_out, float lr) {\n");
            os.push_str("    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n");
            os.push_str("    int64_t st = m->ctx->lr_step;\n");
            os.push_str("    if (st < 9223372036854775807LL) m->ctx->lr_step = st + 1;\n");
            os.push_str("    const float lr_eff = lr * ns_lr_schedule(st);\n");
            os.push_str("    ns_train_core(m->ctx, input, input_numel, labels, m->h_w, m->h_w, loss_out, lr_eff, 1);\n");
            os.push_str("    return 0;\n");
            os.push_str("}\n\n");
            os.push_str("extern \"C\" int ns_objective_loss(ns_model* m, const float* input,\n");
            os.push_str("                                   const float* labels, size_t input_numel,\n");
            os.push_str("                                   float* loss_out) {\n");
            os.push_str("    if (!m || !m->ctx || !m->ctx->d_wb) return -1;\n");
            os.push_str("    ns_train_core(m->ctx, input, input_numel, labels, m->h_w, (float*)0, loss_out, 0.f, 0);\n");
            os.push_str("    return 0;\n");
            os.push_str("}\n");
        }
    }

    os.push_str("// All device buffers live in the per-model NSContext; they are released\n");
    os.push_str("// when ns_free() destroys the context.\n");
    Ok(os)
}

// ---------------------------------------------------------------------------
// emit_train_core_cuda — port of `CodeGenerator::emit_train_core_cuda`
// (codegen.cpp 3218-3758): the AOT training core (forward + backward +
// optimizer on device) as a C-ABI slave of ns_runtime_train_step.
// ---------------------------------------------------------------------------

fn emit_train_core_cuda(tfn: &MLIRFunction, in_cols: i64, out_cols: i64) -> NsResult<String> {
    let mut ids: BTreeMap<String, CuIdInfo> = BTreeMap::new();
    fill_cu_ids(tfn, &mut ids);

    // Dynamic inputs of the train function: batch_x (first non-weight GEMM A),
    // batch_y (CE labels).
    let mut x_id = String::new();
    let mut y_id = String::new();
    for instr in &tfn.instructions {
        if instr.op == MLIROp::LayerEmbedding
            && instr.operands.len() >= 2
            && !ids.get(&instr.operands[1]).map_or(false, |i| i.is_weight)
            && x_id.is_empty()
        {
            x_id = instr.operands[1].clone();
        }
        if instr.op == MLIROp::Matmul
            && instr.operands.len() >= 2
            && !ids.get(&instr.operands[0]).map_or(false, |i| i.is_weight)
            && x_id.is_empty()
        {
            x_id = instr.operands[0].clone();
        }
        if instr.op == MLIROp::CrossEntropy && instr.operands.len() >= 2 {
            y_id = instr.operands[1].clone();
        }
    }

    // Weight buffers in train-fn ALLOC order (layout-compatible with the
    // network's forward blob for network train methods).
    let mut worder_t: Vec<String> = Vec::new();
    for instr in &tfn.instructions {
        if instr.op == MLIROp::TensorAlloc {
            worder_t.push(instr.result_id.clone());
        }
    }
    let mut woff: BTreeMap<String, u64> = BTreeMap::new();
    let mut weight_total: u64 = 0;
    {
        let mut off: u64 = 0;
        for wt in &worder_t {
            woff.insert(wt.clone(), off);
            let n = ids
                .get(wt)
                .map_or(1, |i| if i.static_numel > 0 { i.static_numel as u64 } else { 1 });
            off += n;
            weight_total += n;
        }
    }

    let inf_x = ids.get(&x_id).copied().unwrap_or_default();
    let mut k1 = if inf_x.cols > 0 { inf_x.cols } else { in_cols };
    if k1 <= 0 {
        k1 = 1;
    }
    let c = if out_cols > 0 { out_cols } else { 2 };

    let id_info = |id: &str| ids.get(id).copied().unwrap_or_default();
    let bufv = |id: &str| -> String {
        if id == x_id {
            return String::from("ctx->d_xi");
        }
        if id == y_id {
            return String::from("ctx->d_yl");
        }
        let is_w = ids.get(id).map_or(false, |i| i.is_weight);
        if is_w && woff.contains_key(id) {
            return format!("(ctx->d_wb + {})", woff[id]);
        }
        format!("ctx->d_{}", id)
    };

    // Feature width of each per-row tensor, propagated forward over the
    // instruction list (ops are in topological order). Ops lowered from bare
    // function calls (dropout, plain activations, ...) carry no result_type in
    // CuIdInfo, so width comes from the producing instruction instead. Do NOT
    // use operator[] here: it would insert 0 entries that break row_dims.
    let wcols: BTreeMap<String, i64> = {
        let mut cur: BTreeMap<String, i64> = BTreeMap::new();
        for a in &tfn.instructions {
            let ida = a.result_id.clone();
            for op in &a.operands {
                let inf = ids.get(op).copied().unwrap_or_default();
                if inf.cols > 0 {
                    cur.insert(op.clone(), inf.cols);
                }
            }
            if (a.op == MLIROp::Matmul || a.op == MLIROp::Fused) && a.operands.len() >= 2 {
                let inf = ids.get(&a.operands[1]).copied().unwrap_or_default();
                if inf.cols > 0 {
                    cur.insert(ida.clone(), inf.cols);
                }
            } else if !ida.is_empty() && !a.operands.is_empty() {
                if let Some(&c2) = cur.get(&a.operands[0]) {
                    if c2 > 0 {
                        cur.insert(ida.clone(), c2);
                    }
                }
            }
        }
        cur
    };
    let row_dims = |v: &str| -> i64 {
        if !v.is_empty() {
            if let Some(&c2) = wcols.get(v) {
                if c2 > 0 {
                    return c2;
                }
            }
        }
        if !v.is_empty() {
            let inf = ids.get(v).copied().unwrap_or_default();
            if inf.cols > 0 {
                return inf.cols;
            }
        }
        for a in &tfn.instructions {
            if (a.op == MLIROp::Matmul || a.op == MLIROp::Fused)
                && a.operands.len() >= 2
                && a.operands[0] == v
            {
                let r = ids.get(&a.operands[1]).map_or(0, |i| i.rows);
                if r > 0 {
                    return r;
                }
            }
        }
        1
    };

    let mut os = String::new();
    os.push_str("extern \"C\" void ns_train_core(NSContext* ctx, const float* x, size_t nx, const float* y,\n");
    os.push_str("                              const float* w, float* wout, float* loss_out,\n");
    os.push_str("                              float lr, int train_mode) {\n");
    os.push_str("  (void)w; (void)y;\n");
    os.push_str("  if (!ctx || !ctx->d_wb || nx == 0) return;\n");
    os.push_str(&format!("  const int M = (int)(nx / {});\n", k1));
    os.push_str("  if (M <= 0) return;\n");
    os.push_str(&format!("  const size_t ny = (size_t)M * {};\n", c));
    os.push_str("  if (ns_cu_reserve(&ctx->d_xi, &ctx->d_xi_cap, nx * sizeof(float), 0)) return;\n");
    os.push_str("  if (ns_cu_reserve(&ctx->d_yl, &ctx->d_yl_cap, ny * sizeof(float), 0)) return;\n");
    os.push_str("  cudaMemcpy(ctx->d_xi, x, nx * sizeof(float), cudaMemcpyHostToDevice);\n");
    os.push_str("  cudaMemcpy(ctx->d_yl, y, ny * sizeof(float), cudaMemcpyHostToDevice);\n");

    // Batch-element-count helper for emission.
    let mut fwd_text = String::new();
    let mut train_text = String::new();
    for instr in &tfn.instructions {
        let id = instr.result_id.clone();
        let op = instr.op;
        match op {
            MLIROp::Matmul => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&b);
                let (ks, ns) = (inf.rows, inf.cols);
                if ks <= 0 || ns <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for matmul '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ctx->d_{}, M, {}, {});\n",
                    ns,
                    bufv(&a),
                    bufv(&b),
                    id,
                    ks,
                    ns
                ));
            }
            MLIROp::Relu
            | MLIROp::LeakyRelu
            | MLIROp::Sigmoid
            | MLIROp::Tanh
            | MLIROp::Swish
            | MLIROp::Gelu
            | MLIROp::Silu
            | MLIROp::Identity => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let mut ns = row_dims(&in_);
                ns = if ns > 0 { ns } else { 1 };
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_act_kernel, M * {}, {}, ctx->d_{}, M * {}, {});\n",
                    ns,
                    bufv(&in_),
                    id,
                    ns,
                    cu_act_code(op)
                ));
            }
            MLIROp::LayerEmbedding => {
                let wt = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let idx = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&wt);
                let v = if inf.rows > 0 { inf.rows } else { 1 };
                let d = if inf.cols > 0 { inf.cols } else { 1 };
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_embedding_kernel, M * {}, {}, {}, ctx->d_{}, M, {}, {});\n",
                    d,
                    bufv(&wt),
                    bufv(&idx),
                    id,
                    v,
                    d
                ));
            }
            MLIROp::ElementwiseBinop => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&a);
                let mut code: i32 = 0;
                if instr.attribute == "-" {
                    code = 1;
                } else if instr.attribute == "*" {
                    code = 2;
                } else if instr.attribute == "/" {
                    code = 3;
                }
                let mut nb: i32 = 1;
                if !b.is_empty() {
                    let inf = ids.get(&b).copied().unwrap_or_default();
                    if inf.static_numel > 0 {
                        nb = inf.static_numel as i32;
                    }
                }
                if !a.is_empty() && !b.is_empty() {
                    fwd_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, ns
                    ));
                    fwd_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_binop_kernel, M * {}, {}, {}, ctx->d_{}, M * {}, {}, {});\n",
                        ns,
                        bufv(&a),
                        bufv(&b),
                        id,
                        ns,
                        nb,
                        code
                    ));
                }
            }
            MLIROp::LayerMoe => {
                let x = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let wg = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let we1 = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let we2 = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                fwd_text.push_str("  ns_moe_mask_sync(ctx);\n");
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)(M * {}) * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH_BLOCKS(ns_moe_kernel, M, {}, {}, {}, {}, ctx->d_moe_a, ctx->d_{}, M, {}, {}, {});\n",
                    bufv(&x),
                    bufv(&wg),
                    bufv(&we1),
                    bufv(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::LayerAttention => {
                let x = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let wq = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let wk = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let wv = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let wo = if instr.operands.len() > 4 {
                    instr.operands[4].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, d
                ));
                fwd_text.push_str("  { float* qs = 0; float* ks = 0; float* vs = 0; float* cs = 0;\n");
                fwd_text.push_str(&format!(
                    "    if (cudaMalloc(&qs, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                fwd_text.push_str(&format!(
                    "        cudaMalloc(&ks, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                fwd_text.push_str(&format!(
                    "        cudaMalloc(&vs, (size_t)M * {} * sizeof(float)) ||\n",
                    d
                ));
                fwd_text.push_str(&format!(
                    "        cudaMalloc(&cs, (size_t)M * {} * sizeof(float))) return;\n",
                    d
                ));
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, qs, M, {}, {});\n",
                    d,
                    bufv(&x),
                    bufv(&wq),
                    d,
                    d
                ));
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, ks, M, {}, {});\n",
                    d,
                    bufv(&x),
                    bufv(&wk),
                    d,
                    d
                ));
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), {}, {}, vs, M, {}, {});\n",
                    d,
                    bufv(&x),
                    bufv(&wv),
                    d,
                    d
                ));
                fwd_text.push_str("    __attribute__((unused)) static bool ns_a_ok_ = (cudaFuncSetAttribute(ns_attention_core_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536), true);\n");
                fwd_text.push_str(&format!(
                    "    ns_attention_core_kernel<<<{}, 256, (size_t)M * M * sizeof(float)>>>(qs, ks, vs, cs, M, {}, {}, 1.0f / sqrtf((float)({} / {})){});\n",
                    h,
                    d,
                    h,
                    d,
                    h,
                    if !instr.ints_attr.is_empty() && instr.ints_attr[0] != 0 {
                        ", 1"
                    } else {
                        ", 0"
                    }
                ));
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH_BLOCKS(ns_gemm_kernel, ((M + 15) / 16) * (({} + 15) / 16), cs, {}, ctx->d_{}, M, {}, {});\n",
                    d,
                    bufv(&wo),
                    id,
                    d,
                    d
                ));
                fwd_text.push_str("    cudaFree(qs); cudaFree(ks); cudaFree(vs); cudaFree(cs); }\n");
            }
            MLIROp::Layernorm | MLIROp::Softmax => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let mut ns = row_dims(&in_);
                ns = if ns > 0 { ns } else { 1 };
                let kn = format!("M * {}", ns);
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({}) * sizeof(float), 0)) return;\n",
                    id, id, kn
                ));
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_copy_kernel, {}, {}, ctx->d_{}, {});\n",
                    kn,
                    bufv(&in_),
                    id,
                    kn
                ));
                if op == MLIROp::Layernorm {
                    fwd_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_layernorm_kernel, M, ctx->d_{}, {}, {});\n",
                        id, kn, ns
                    ));
                } else {
                    fwd_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_softmax_kernel, M, ctx->d_{}, {}, {});\n",
                        id, kn, ns
                    ));
                }
            }
            MLIROp::Dropout => {
                let in_ = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let mut ns = row_dims(&in_);
                ns = if ns > 0 { ns } else { 1 };
                let mut rate = instr.float_attr;
                if rate < 0.0 {
                    rate = 0.0;
                }
                if rate >= 1.0 {
                    rate = 0.999999;
                }
                let ne = format!("M * {}", ns);
                fwd_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({}) * sizeof(float), 0)) return;\n",
                    id, id, ne
                ));
                fwd_text.push_str("  if (train_mode) {\n");
                fwd_text.push_str(&format!(
                    "    if (ns_cu_reserve(&ctx->d_dm_{}, &ctx->d_dm_{}_cap, (size_t)({}) * sizeof(float), 0)) return;\n",
                    id, id, ne
                ));
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH1(ns_dropout_fwd_kernel, {}, {}, ctx->d_{}, ctx->d_dm_{}, {}, {}f, ctx->drop_seed++);\n",
                    ne,
                    bufv(&in_),
                    id,
                    id,
                    ne,
                    fmt_float(rate)
                ));
                fwd_text.push_str("  } else {\n");
                fwd_text.push_str(&format!(
                    "    NS_LAUNCH1(ns_copy_kernel, {}, {}, ctx->d_{}, {});\n",
                    ne,
                    bufv(&in_),
                    id,
                    ne
                ));
                fwd_text.push_str("  }\n");
            }
            MLIROp::CrossEntropy => {
                let preds = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                fwd_text.push_str("  if (ns_cu_reserve(&ctx->d_row, &ctx->d_row_cap, (size_t)M * sizeof(float), 0)) return;\n");
                fwd_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_ce_row_kernel, M, {}, ctx->d_yl, ctx->d_row, M * {}, {});\n",
                    bufv(&preds),
                    c,
                    c
                ));
                fwd_text.push_str("  cudaDeviceSynchronize();\n");
                fwd_text.push_str("  if (!ctx->hrow || ctx->hrow_cap < (size_t)M) { free((void*)ctx->hrow); ctx->hrow = (float*)malloc((size_t)M * sizeof(float)); ctx->hrow_cap = (size_t)M; }\n");
                fwd_text.push_str("  cudaMemcpy(ctx->hrow, ctx->d_row, (size_t)M * sizeof(float), cudaMemcpyDeviceToHost);\n");
                fwd_text.push_str("  float loss = 0.f;\n");
                fwd_text.push_str("  for (int r = 0; r < M; r++) loss += ctx->hrow[r];\n");
                fwd_text.push_str("  loss /= (float)M;\n");
                fwd_text.push_str("  if (loss_out) *loss_out = loss;\n");
            }
            MLIROp::LossGrad => {
                let preds = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, c
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_loss_grad_kernel, M, {}, ctx->d_yl, ctx->d_{}, M * {}, {});\n",
                    bufv(&preds),
                    id,
                    c,
                    c
                ));
            }
            MLIROp::MatmulGradA => {
                let dc = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&b);
                let (ks, ns) = (inf.rows, inf.cols);
                if ks <= 0 || ns <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for grad_a '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ks
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_grad_a_kernel, M * {}, {}, {}, ctx->d_{}, M, {}, {});\n",
                    ks,
                    bufv(&dc),
                    bufv(&b),
                    id,
                    ks,
                    ns
                ));
            }
            MLIROp::MatmulGradW => {
                let a = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let dc = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&b);
                let (ks, ns) = (inf.rows, inf.cols);
                if ks <= 0 || ns <= 0 || a.is_empty() || dc.is_empty() {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for grad_w '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                let n = ks * ns;
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, {} * sizeof(float), 0)) return;\n",
                    id, id, n
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_grad_w_kernel, {}, {}, {}, ctx->d_{}, M, {}, {});\n",
                    n,
                    bufv(&a),
                    bufv(&dc),
                    id,
                    ks,
                    ns
                ));
            }
            MLIROp::BinopGrad => {
                let dc = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let a = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let b = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let ns = row_dims(&a);
                let mut opcode: i32 = 0;
                if instr.attribute == "-" {
                    opcode = 1;
                } else if instr.attribute == "*" {
                    opcode = 2;
                } else if instr.attribute == "/" {
                    opcode = 3;
                }
                let which = if instr.int_attr != 0 {
                    2 * opcode | 1
                } else {
                    2 * opcode
                };
                if !a.is_empty() && !b.is_empty() {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, ns
                    ));
                    train_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_binop_grad_kernel, M * {}, {}, {}, {}, ctx->d_{}, M * {}, {});\n",
                        ns,
                        bufv(&dc),
                        bufv(&a),
                        bufv(&b),
                        id,
                        ns,
                        which
                    ));
                }
            }
            MLIROp::EmbeddingGradW => {
                let dout = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let idx = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let wt = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let inf = id_info(&wt);
                let v = if inf.rows > 0 { inf.rows } else { 1 };
                let d = if inf.cols > 0 { inf.cols } else { 1 };
                let n = v * d;
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, {} * sizeof(float), 0)) return;\n",
                    id, id, n
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_fill_kernel, {}, ctx->d_{}, {}, 0.0f);\n",
                    n, id, n
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_embedding_grad_w_kernel, M * {}, {}, {}, ctx->d_{}, M, {}, {});\n",
                    d,
                    bufv(&dout),
                    bufv(&idx),
                    id,
                    d,
                    v
                ));
            }
            MLIROp::MoeGradX | MLIROp::MoeGradWg | MLIROp::MoeGradWe1 | MLIROp::MoeGradWe2 => {
                let dout = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let xo = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let wg = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let we1 = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let we2 = if instr.operands.len() > 4 {
                    instr.operands[4].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                if op == MLIROp::MoeGradX {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, d
                    ));
                    train_text.push_str("  if (ns_cu_reserve_u8(&ctx->d_moe_a, &ctx->d_moe_a_cap, ctx->moe_cap, 0)) return;\n");
                    train_text.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS(ns_moe_grad_x_kernel, M, {}, {}, {}, {}, {}, ctx->d_moe_a, ctx->d_{}, M, {}, {}, {});\n",
                        bufv(&dout),
                        bufv(&xo),
                        bufv(&wg),
                        bufv(&we1),
                        bufv(&we2),
                        id,
                        d,
                        h,
                        e
                    ));
                } else {
                    let kernel = match op {
                        MLIROp::MoeGradWg => "ns_moe_grad_wg_kernel",
                        MLIROp::MoeGradWe1 => "ns_moe_grad_we1_kernel",
                        _ => "ns_moe_grad_we2_kernel",
                    };
                    let nwe = if op == MLIROp::MoeGradWg {
                        format!("{} * {}", d, e)
                    } else if op == MLIROp::MoeGradWe1 {
                        format!("{} * {} * {}", e, d, h)
                    } else {
                        format!("{} * {} * {}", e, h, d)
                    };
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({}) * sizeof(float), 0)) return;\n",
                        id, id, nwe
                    ));
                    train_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_fill_kernel, {}, ctx->d_{}, {}, 0.0f);\n",
                        nwe, id, nwe
                    ));
                    train_text.push_str("  if (ns_cu_reserve_u8(&ctx->d_moe_a, &ctx->d_moe_a_cap, ctx->moe_cap, 0)) return;\n");
                    train_text.push_str(&format!(
                        "  NS_LAUNCH_BLOCKS({}, M, {}, {}, {}, {}, {}, ctx->d_moe_a, ctx->d_{}, M, {}, {}, {});\n",
                        kernel,
                        bufv(&dout),
                        bufv(&xo),
                        bufv(&wg),
                        bufv(&we1),
                        bufv(&we2),
                        id,
                        d,
                        h,
                        e
                    ));
                }
            }
            MLIROp::AttentionGradX
            | MLIROp::AttentionGradWq
            | MLIROp::AttentionGradWk
            | MLIROp::AttentionGradWv
            | MLIROp::AttentionGradWo => {
                let dout = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let xo = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let wq = if instr.operands.len() > 2 {
                    instr.operands[2].clone()
                } else {
                    String::new()
                };
                let wk = if instr.operands.len() > 3 {
                    instr.operands[3].clone()
                } else {
                    String::new()
                };
                let wv = if instr.operands.len() > 4 {
                    instr.operands[4].clone()
                } else {
                    String::new()
                };
                let wo = if instr.operands.len() > 5 {
                    instr.operands[5].clone()
                } else {
                    String::new()
                };
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                let dd2 = format!("{} * {}", d, d);
                let is_x = op == MLIROp::AttentionGradX;
                let which = if is_x {
                    0
                } else if op == MLIROp::AttentionGradWq {
                    1
                } else if op == MLIROp::AttentionGradWk {
                    2
                } else if op == MLIROp::AttentionGradWv {
                    3
                } else {
                    4
                };
                if is_x {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, d
                    ));
                } else {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)({}) * sizeof(float), 0)) return;\n",
                        id, id, dd2
                    ));
                }
                if !is_x {
                    train_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_fill_kernel, {}, ctx->d_{}, {}, 0.0f);\n",
                        dd2, id, dd2
                    ));
                }
                // Per-block global scratch for the grad kernel: sc[S*S] + 8*S*Dk.
                // Grid is B*H blocks; one scratch region per block.
                let dk = if d / h > 0 { d / h } else { 1 };
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_asr, &ctx->d_asr_cap, (size_t){} * ((size_t)M * M + 8 * (size_t)M * {}) * sizeof(float), 0)) return;\n",
                    h, dk
                ));
                train_text.push_str(&format!(
                    "  ns_attention_grad_kernel<<<{}, 256>>>({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, ctx->d_asr, M, {}, {}, M, {}{}",
                    h,
                    bufv(&dout),
                    bufv(&xo),
                    bufv(&wq),
                    bufv(&wk),
                    bufv(&wv),
                    bufv(&wo),
                    if is_x { format!("ctx->d_{}", id) } else { String::from("0") },
                    if which == 1 { format!("ctx->d_{}", id) } else { String::from("0") },
                    if which == 2 { format!("ctx->d_{}", id) } else { String::from("0") },
                    if which == 3 { format!("ctx->d_{}", id) } else { String::from("0") },
                    if which == 4 { format!("ctx->d_{}", id) } else { String::from("0") },
                    d,
                    h,
                    which,
                    if !instr.ints_attr.is_empty() && instr.ints_attr[0] != 0 {
                        ", 1);\n"
                    } else {
                        ", 0);\n"
                    }
                ));
            }
            MLIROp::LayernormGrad => {
                let dout = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let act_in = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let mut ns = row_dims(&act_in);
                ns = if ns > 0 { ns } else { 1 };
                train_text.push_str(&format!(
                    "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                    id, id, ns
                ));
                train_text.push_str(&format!(
                    "  NS_LAUNCH1(ns_layernorm_grad_kernel, M, {}, {}, ctx->d_{}, M * {}, {});\n",
                    bufv(&dout),
                    bufv(&act_in),
                    id,
                    ns,
                    ns
                ));
            }
            MLIROp::ActivationGrad => {
                let dout = if instr.operands.len() > 0 {
                    instr.operands[0].clone()
                } else {
                    String::new()
                };
                let act_in = if instr.operands.len() > 1 {
                    instr.operands[1].clone()
                } else {
                    String::new()
                };
                let mut ns = row_dims(&act_in);
                ns = if ns > 0 { ns } else { 1 };
                if instr.attribute == "dropout" {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, ns
                    ));
                    train_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_dropout_mul_kernel, M * {}, {}, ctx->d_dm_{}, ctx->d_{}, M * {});\n",
                        ns,
                        bufv(&dout),
                        act_in,
                        id,
                        ns
                    ));
                } else {
                    train_text.push_str(&format!(
                        "  if (ns_cu_reserve(&ctx->d_{}, &ctx->d_{}_cap, (size_t)M * {} * sizeof(float), 0)) return;\n",
                        id, id, ns
                    ));
                    train_text.push_str(&format!(
                        "  NS_LAUNCH1(ns_act_grad_kernel, M * {}, {}, {}, ctx->d_{}, M * {}, {});\n",
                        ns,
                        bufv(&dout),
                        bufv(&act_in),
                        id,
                        ns,
                        cu_act_code_attr(&instr.attribute)
                    ));
                }
            }
            MLIROp::OptStep => {
                let mut i = 0;
                while i + 1 < instr.operands.len() {
                    let wt = instr.operands[i].clone();
                    let g = instr.operands[i + 1].clone();
                    let inf = id_info(&wt);
                    let r = if inf.rows > 0 { inf.rows } else { 1 };
                    let c2 = if inf.cols > 0 { inf.cols } else { 1 };
                    let n = r * c2;
                    let use_muon = r > 1 && c2 > 1 && r.min(c2) >= 8;
                    train_text.push_str("  {\n");
                    if use_muon {
                        train_text.push_str(&format!(
                            "    if (ns_cu_reserve(&ctx->d_am{}, &ctx->d_am{}_cap, {} * sizeof(float), 1)) return;\n",
                            wt, wt, n
                        ));
                        train_text.push_str("    (void)ctx->adam_step;\n");
                        train_text.push_str(&format!(
                            "    NS_LAUNCH1(ns_muon_ema_kernel, {}, ctx->d_am{}, {}, {}, {}f);\n",
                            n,
                            wt,
                            bufv(&g),
                            n,
                            fmt_float9(optim::K_MUON_MOMENTUM as f32)
                        ));
                        train_text.push_str(&format!(
                            "    ns_orthonom_kernel<<<1, 1>>>(ctx->d_am{}, {}, {});\n",
                            wt, r, c2
                        ));
                        train_text.push_str(&format!(
                            "    float le = lr * sqrtf((float)std::min({}, {}));\n",
                            r, c2
                        ));
                        train_text.push_str(&format!(
                            "    NS_LAUNCH1(ns_muon_step_kernel, {}, {}, ctx->d_am{}, {}, le, {}f * lr);\n",
                            n,
                            bufv(&wt),
                            wt,
                            n,
                            fmt_float9(optim::K_MUON_DECAY as f32)
                        ));
                    } else {
                        train_text.push_str(&format!(
                            "    if (ns_cu_reserve(&ctx->d_am{}, &ctx->d_am{}_cap, {} * sizeof(float), 1)) return;\n",
                            wt, wt, n
                        ));
                        train_text.push_str(&format!(
                            "    if (ns_cu_reserve(&ctx->d_av{}, &ctx->d_av{}_cap, {} * sizeof(float), 1)) return;\n",
                            wt, wt, n
                        ));
                        train_text.push_str("    size_t t = ++ctx->adam_step;\n");
                        train_text.push_str(&format!(
                            "    float b1t = 1.f - powf({}f, (float)t), b2t = 1.f - powf({}f, (float)t);\n",
                            fmt_float9(optim::K_ADAMW_BETA1 as f32),
                            fmt_float9(optim::K_ADAMW_BETA2 as f32)
                        ));
                        train_text.push_str(&format!(
                            "    NS_LAUNCH1(ns_adamw_kernel, {}, {}, {}, ctx->d_am{}, ctx->d_av{}, {}, lr, b1t, b2t, {}f, {}f, {}f, {}f, {}f, {}f * lr);\n",
                            n,
                            bufv(&wt),
                            bufv(&g),
                            wt,
                            wt,
                            n,
                            fmt_float9(optim::K_ADAMW_BETA1 as f32),
                            fmt_float9(optim::K_ADAMW_BETA2 as f32),
                            fmt_float9(optim::K_ONE_MINUS_BETA1 as f32),
                            fmt_float9(optim::K_ONE_MINUS_BETA2 as f32),
                            fmt_float9(optim::K_ADAMW_EPS as f32),
                            fmt_float9(optim::K_ADAMW_DECAY as f32)
                        ));
                    }
                    train_text.push_str("  }\n");
                    i += 2;
                }
            }
            _ => {}
        }
    }

    os.push_str(&fwd_text);
    os.push_str("  if (train_mode) {\n");
    os.push_str(&train_text);
    os.push_str("  }\n");
    os.push_str("  cudaDeviceSynchronize();\n");
    os.push_str(&format!(
        "  if (train_mode && wout)\n    cudaMemcpy(wout, ctx->d_wb, {} * sizeof(float), cudaMemcpyDeviceToHost);\n",
        weight_total
    ));
    os.push_str("}\n");
    Ok(os)
}

// ---------------------------------------------------------------------------
// emit_train_core — port of `CodeGenerator::emit_train_core` (codegen.cpp
// 2683-3216): the AOT training core (forward + backward + optimizer) as a
// C-ABI slave of ns_runtime_train_step, running on the CPU reference kernels.
// ---------------------------------------------------------------------------

fn emit_train_core(tfn: &MLIRFunction, in_cols: i64) -> NsResult<String> {
    let mut ids: BTreeMap<String, CuIdInfo> = BTreeMap::new();
    fill_cu_ids(tfn, &mut ids);

    let mut x_id = String::new();
    let mut y_id = String::new();
    for instr in &tfn.instructions {
        if instr.op == MLIROp::LayerEmbedding
            && instr.operands.len() >= 2
            && !ids.get(&instr.operands[1]).map_or(false, |i| i.is_weight)
            && x_id.is_empty()
        {
            x_id = instr.operands[1].clone();
        }
        if instr.op == MLIROp::Matmul
            && instr.operands.len() >= 2
            && !ids.get(&instr.operands[0]).map_or(false, |i| i.is_weight)
            && x_id.is_empty()
        {
            x_id = instr.operands[0].clone();
        }
        if instr.op == MLIROp::CrossEntropy && instr.operands.len() >= 2 {
            y_id = instr.operands[1].clone();
        }
    }

    let mut worder_t: Vec<String> = Vec::new();
    for instr in &tfn.instructions {
        if instr.op == MLIROp::TensorAlloc {
            worder_t.push(instr.result_id.clone());
        }
    }

    let src_of = |id: &str| -> String {
        if id == x_id.as_str() {
            String::from("x")
        } else {
            format!("buf_{}.data()", id)
        }
    };
    let numel_of = |id: &str| -> String {
        if id == x_id.as_str() {
            String::from("nx")
        } else {
            format!("buf_{}.size()", id)
        }
    };

    let mut wcols: BTreeMap<String, i64> = BTreeMap::new();
    {
        for instr in &tfn.instructions {
            let ida = instr.result_id.clone();
            for op in &instr.operands {
                let inf = ids.get(op).copied().unwrap_or_default();
                if inf.cols > 0 {
                    wcols.insert(op.clone(), inf.cols);
                }
            }
            if (instr.op == MLIROp::Matmul || instr.op == MLIROp::Fused)
                && instr.operands.len() >= 2
            {
                let inf = ids
                    .get(&instr.operands[1])
                    .copied()
                    .unwrap_or_default();
                if inf.cols > 0 {
                    wcols.insert(ida.clone(), inf.cols);
                }
            } else if !ida.is_empty() && !instr.operands.is_empty() {
                if let Some(&c) = wcols.get(&instr.operands[0]) {
                    if c > 0 {
                        wcols.insert(ida.clone(), c);
                    }
                }
            }
        }
    }
    let train_row_dims = |v: &str| -> i64 {
        if !v.is_empty() {
            if let Some(&c) = wcols.get(v) {
                if c > 0 {
                    return c;
                }
            }
        }
        if !v.is_empty() {
            let inf = ids.get(v).copied().unwrap_or_default();
            if inf.cols > 0 {
                return inf.cols;
            }
        }
        1
    };

    let mut os = String::new();
    os.push_str(
        "extern \"C\" void ns_train_core(ns_cpu_ctx* ctx, const float* x, size_t nx, const float* y,\n",
    );
    os.push_str("                              const float* w, float* wout, float* loss_out,\n");
    os.push_str("                              float lr, int train_mode) {\n");
    os.push_str("  (void)y;\n");
    for (id, _inf) in &ids {
        if id == &x_id || id == &y_id {
            continue;
        }
        os.push_str(&format!("  std::vector<float>& buf_{} = ctx->buf_{};\n", id, id));
    }
    for instr in &tfn.instructions {
        if instr.op == MLIROp::Dropout {
            os.push_str(&format!(
                "  std::vector<float>& buf_dm_{} = ctx->buf_dm_{};  // dropout mask (inverted, scaled)\n",
                instr.result_id, instr.result_id
            ));
        }
    }
    {
        let mut off: u64 = 0;
        for wt in &worder_t {
            let n = ids
                .get(wt)
                .map_or(1, |i| if i.static_numel > 0 { i.static_numel } else { 1 });
            os.push_str(&format!("  buf_{}.assign(w + {}, w + {});\n", wt, off, off + n as u64));
            off += n as u64;
        }
    }

    let mut fwd_body = String::new();
    let mut train_body = String::new();
    let mut opt_body = String::new();
    for instr in &tfn.instructions {
        let id = instr.result_id.clone();
        match instr.op {
            MLIROp::Matmul => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let inf = ids.get(&b).copied().unwrap_or_default();
                let ks = inf.rows;
                let ns_ = inf.cols;
                if ks <= 0 || ns_ <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for matmul '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                fwd_body.push_str(&format!("  {{ int64_t K = {}, N = {};\n", ks, ns_));
                fwd_body.push_str(&format!("    int64_t M = (int64_t)({}) / K;\n", numel_of(&a)));
                fwd_body.push_str(&format!("    buf_{}.resize((size_t)(M * N));\n", id));
                fwd_body.push_str(&format!(
                    "    ns_matmul({}, buf_{}.data(), buf_{}.data(), M, K, N); }}\n",
                    src_of(&a),
                    b,
                    id
                ));
            }
            MLIROp::Relu
            | MLIROp::LeakyRelu
            | MLIROp::Sigmoid
            | MLIROp::Tanh
            | MLIROp::Swish
            | MLIROp::Gelu
            | MLIROp::Silu
            | MLIROp::Identity => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let code = match instr.op {
                    MLIROp::Relu => 1,
                    MLIROp::LeakyRelu => 2,
                    MLIROp::Sigmoid => 3,
                    MLIROp::Tanh => 4,
                    MLIROp::Swish | MLIROp::Silu => 5,
                    MLIROp::Gelu => 6,
                    _ => 0,
                };
                fwd_body.push_str(&format!("  {{ size_t zn = {};\n", numel_of(&inp)));
                fwd_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                fwd_body.push_str(&format!("    const float* src = {};\n", src_of(&inp)));
                fwd_body.push_str(&format!(
                    "    for (size_t i = 0; i < zn; i++) buf_{}[i] = ns_act2(src[i], {}); }}\n",
                    id, code
                ));
            }
            MLIROp::LayerEmbedding => {
                let wt = instr.operands.get(0).cloned().unwrap_or_default();
                let idx = instr.operands.get(1).cloned().unwrap_or_default();
                let v = ids.get(&wt).map_or(1, |i| if i.rows > 0 { i.rows } else { 1 });
                let d = ids.get(&wt).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                fwd_body.push_str(&format!("  {{ int64_t n_idx = {};\n", numel_of(&idx)));
                fwd_body.push_str(&format!("    buf_{}.resize((size_t)(n_idx * {}));\n", id, d));
                fwd_body.push_str(&format!(
                    "    ns_embedding({}, {}, buf_{}.data(), n_idx, {}, {}); }}\n",
                    src_of(&wt),
                    src_of(&idx),
                    id,
                    v,
                    d
                ));
            }
            MLIROp::ElementwiseBinop => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let code = if instr.attribute == "-" {
                    1
                } else if instr.attribute == "*" {
                    2
                } else if instr.attribute == "/" {
                    3
                } else {
                    0
                };
                if !a.is_empty() && !b.is_empty() {
                    fwd_body.push_str(&format!("  {{ size_t zn = {};\n", numel_of(&a)));
                    fwd_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                    fwd_body.push_str(&format!(
                        "    ns_binop({}, zn, {}, buf_{}.size(), buf_{}.data(), {}); }}\n",
                        src_of(&a),
                        src_of(&b),
                        b,
                        id,
                        code
                    ));
                }
            }
            MLIROp::LayerMoe => {
                let x = instr.operands.get(0).cloned().unwrap_or_default();
                let wg = instr.operands.get(1).cloned().unwrap_or_default();
                let we = instr.operands.get(2).cloned().unwrap_or_default();
                let we2 = instr.operands.get(3).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                fwd_body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)({}) / {};\n",
                    numel_of(&x),
                    d
                ));
                fwd_body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, d));
                fwd_body.push_str(&format!(
                    "    ns_moe_fwd({}, {}, {}, {}, ctx->moe_active.data(), buf_{}.data(), M, {}, {}, {}); }}\n",
                    src_of(&x),
                    src_of(&wg),
                    src_of(&we),
                    src_of(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::LayerAttention => {
                let x = instr.operands.get(0).cloned().unwrap_or_default();
                let wq = instr.operands.get(1).cloned().unwrap_or_default();
                let wk = instr.operands.get(2).cloned().unwrap_or_default();
                let wv = instr.operands.get(3).cloned().unwrap_or_default();
                let wo = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                let causal = if !instr.ints_attr.is_empty() {
                    instr.ints_attr[0].to_string()
                } else {
                    "0".to_string()
                };
                fwd_body.push_str(&format!(
                    "  {{ int64_t BS = (int64_t)({}) / {};\n",
                    numel_of(&x),
                    d
                ));
                fwd_body.push_str("    int64_t S = BS;\n");
                fwd_body.push_str(&format!(
                    "    std::vector<float> q_b(BS*{}), k_b(BS*{}), v_b(BS*{}), proj_b(BS*{}), sc_b(S*S);\n",
                    d, d, d, d
                ));
                fwd_body.push_str(&format!("    buf_{}.resize((size_t)(BS*{}));\n", id, d));
                fwd_body.push_str(&format!(
                    "    ns_attention_fwd({}, {}, {}, {}, {},\n",
                    src_of(&x),
                    src_of(&wq),
                    src_of(&wk),
                    src_of(&wv),
                    src_of(&wo)
                ));
                fwd_body.push_str(&format!(
                    "        q_b.data(), k_b.data(), v_b.data(), sc_b.data(), proj_b.data(), buf_{}.data(),\n",
                    id
                ));
                fwd_body.push_str(&format!("        BS, {}, {}, S, {}); }}\n", d, h, causal));
            }
            MLIROp::Layernorm | MLIROp::Softmax => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let mut last = train_row_dims(&inp);
                if last <= 0 {
                    last = 1;
                }
                fwd_body.push_str(&format!("  {{ size_t zn = {};\n", numel_of(&inp)));
                fwd_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                fwd_body.push_str(&format!("    const float* src = {};\n", src_of(&inp)));
                fwd_body.push_str(&format!("    for (size_t i = 0; i < zn; i++) buf_{}[i] = src[i];\n", id));
                if instr.op == MLIROp::Layernorm {
                    fwd_body.push_str(&format!(
                        "    ns_layernorm(buf_{}.data(), buf_{}.size(), {}); }}\n",
                        id, id, last
                    ));
                } else {
                    fwd_body.push_str(&format!(
                        "    ns_softmax(buf_{}.data(), buf_{}.size(), {}); }}\n",
                        id, id, last
                    ));
                }
            }
            MLIROp::Dropout => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let mut rate = instr.float_attr;
                if rate < 0.0 {
                    rate = 0.0;
                }
                if rate >= 1.0 {
                    rate = 0.999999;
                }
                let r = fmt_float(rate);
                fwd_body.push_str(&format!("  {{ size_t zn = {};\n", numel_of(&inp)));
                fwd_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                fwd_body.push_str(&format!("    buf_dm_{}.resize(zn);\n", id));
                fwd_body.push_str(&format!("    const float* src = {};\n", src_of(&inp)));
                fwd_body.push_str("    if (train_mode) {\n");
                fwd_body.push_str(&format!("      std::bernoulli_distribution keep(1.0 - {});\n", r));
                fwd_body.push_str(&format!("      const float scale = 1.0f / (float)(1.0 - {});\n", r));
                fwd_body.push_str(&format!(
                    "      for (size_t i = 0; i < zn; i++) {{ float m = keep(ctx->rng) ? scale : 0.f; buf_dm_{}[i] = m; buf_{}[i] = src[i] * m; }}\n",
                    id, id
                ));
                fwd_body.push_str("    } else {\n");
                fwd_body.push_str(&format!("      for (size_t i = 0; i < zn; i++) buf_{}[i] = src[i];\n", id));
                fwd_body.push_str("    }\n");
                fwd_body.push_str("  }\n");
            }
            MLIROp::CrossEntropy => {
                let preds = instr.operands.get(0).cloned().unwrap_or_default();
                let c = ids.get(&preds).map_or(in_cols, |i| if i.cols > 0 { i.cols } else { in_cols });
                fwd_body.push_str(&format!("  {{ buf_{}.assign(1, 0.f);\n", id));
                fwd_body.push_str(&format!(
                    "    buf_{}[0] = ns_cross_entropy(buf_{}.data(), y, buf_{}.size(), {}); }}\n",
                    id, preds, preds, c
                ));
            }
            MLIROp::LossGrad => {
                let preds = instr.operands.get(0).cloned().unwrap_or_default();
                let c = ids.get(&preds).map_or(in_cols, |i| if i.cols > 0 { i.cols } else { in_cols });
                train_body.push_str(&format!("  {{ size_t zn = buf_{}.size();\n", preds));
                train_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                train_body.push_str(&format!(
                    "    ns_loss_grad(buf_{}.data(), y, buf_{}.data(), zn, {}); }}\n",
                    preds, id, c
                ));
            }
            MLIROp::MatmulGradA => {
                let dc = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let inf = ids.get(&b).copied().unwrap_or_default();
                let ks = inf.rows;
                let ns_ = inf.cols;
                if ks <= 0 || ns_ <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for grad_a '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                train_body.push_str(&format!("  {{ int64_t K = {}, N = {};\n", ks, ns_));
                train_body.push_str(&format!("    int64_t M = (int64_t)(buf_{}.size()) / N;\n", dc));
                train_body.push_str(&format!("    buf_{}.resize((size_t)(M * K));\n", id));
                train_body.push_str(&format!(
                    "    ns_matmul_grad_a(buf_{}.data(), buf_{}.data(), buf_{}.data(), M, K, N); }}\n",
                    dc, b, id
                ));
            }
            MLIROp::MatmulGradW => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let dc = instr.operands.get(1).cloned().unwrap_or_default();
                let b = instr.operands.get(2).cloned().unwrap_or_default();
                let inf = ids.get(&b).copied().unwrap_or_default();
                let ks = inf.rows;
                let ns_ = inf.cols;
                if ks <= 0 || ns_ <= 0 || a.is_empty() || dc.is_empty() {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for grad_w '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                train_body.push_str(&format!("  {{ int64_t K = {}, N = {};\n", ks, ns_));
                train_body.push_str(&format!("    int64_t M = (int64_t)(buf_{}.size()) / N;\n", dc));
                train_body.push_str(&format!("    buf_{}.resize((size_t)(K * N));\n", id));
                train_body.push_str(&format!(
                    "    ns_matmul_grad_w({}, buf_{}.data(), buf_{}.data(), M, K, N); }}\n",
                    src_of(&a),
                    dc,
                    id
                ));
            }
            MLIROp::BinopGrad => {
                let dc = instr.operands.get(0).cloned().unwrap_or_default();
                let a = instr.operands.get(1).cloned().unwrap_or_default();
                let b = instr.operands.get(2).cloned().unwrap_or_default();
                let opcode = if instr.attribute == "-" {
                    1
                } else if instr.attribute == "*" {
                    2
                } else if instr.attribute == "/" {
                    3
                } else {
                    0
                };
                let which = if instr.int_attr != 0 {
                    2 * opcode | 1
                } else {
                    2 * opcode
                };
                if !a.is_empty() {
                    train_body.push_str(&format!("  {{ size_t zn = buf_{}.size();\n", dc));
                    train_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                    train_body.push_str(&format!(
                        "    ns_binop_grad(buf_{}.data(), zn, buf_{}.data(), buf_{}.data(), buf_{}.data(), {}); }}\n",
                        dc, a, b, id, which
                    ));
                }
            }
            MLIROp::EmbeddingGradW => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let idx = instr.operands.get(1).cloned().unwrap_or_default();
                let wt = instr.operands.get(2).cloned().unwrap_or_default();
                let v = ids.get(&wt).map_or(1, |i| if i.rows > 0 { i.rows } else { 1 });
                let d = ids.get(&wt).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                train_body.push_str(&format!("  {{ size_t M = buf_{}.size() / {};\n", dout, d));
                train_body.push_str(&format!("    buf_{}.resize((size_t)({} * {}));\n", id, v, d));
                train_body.push_str(&format!(
                    "    ns_embedding_grad_w(buf_{}.data(), {}, buf_{}.data(), M, {}, {}); }}\n",
                    dout,
                    src_of(&idx),
                    id,
                    d,
                    v
                ));
            }
            MLIROp::MoeGradX => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let xo = instr.operands.get(1).cloned().unwrap_or_default();
                let wg = instr.operands.get(2).cloned().unwrap_or_default();
                let we1 = instr.operands.get(3).cloned().unwrap_or_default();
                let we2 = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                train_body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)(buf_{}.size()) / {};\n",
                    dout, d
                ));
                train_body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, d));
                train_body.push_str(&format!(
                    "    for (size_t q = 0; q < buf_{}.size(); q++) buf_{}[q] = 0.f;\n",
                    id, id
                ));
                train_body.push_str(&format!(
                    "    ns_moe_grad_x(buf_{}.data(), {}, {}, {}, {}, ctx->moe_active.data(), buf_{}.data(), M, {}, {}, {}); }}\n",
                    dout,
                    src_of(&xo),
                    src_of(&wg),
                    src_of(&we1),
                    src_of(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::MoeGradWg => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let xo = instr.operands.get(1).cloned().unwrap_or_default();
                let wg = instr.operands.get(2).cloned().unwrap_or_default();
                let we1 = instr.operands.get(3).cloned().unwrap_or_default();
                let we2 = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                train_body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)(buf_{}.size()) / {};\n",
                    dout, d
                ));
                train_body.push_str(&format!(
                    "    buf_{}.resize((size_t)({} * {}));\n",
                    id, d, e
                ));
                train_body.push_str(&format!(
                    "    ns_moe_grad_wg(buf_{}.data(), {}, {}, {}, {}, ctx->moe_active.data(), buf_{}.data(), M, {}, {}, {}); }}\n",
                    dout,
                    src_of(&xo),
                    src_of(&wg),
                    src_of(&we1),
                    src_of(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::MoeGradWe1 => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let xo = instr.operands.get(1).cloned().unwrap_or_default();
                let wg = instr.operands.get(2).cloned().unwrap_or_default();
                let we1 = instr.operands.get(3).cloned().unwrap_or_default();
                let we2 = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                train_body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)(buf_{}.size()) / {};\n",
                    dout, d
                ));
                train_body.push_str(&format!(
                    "    buf_{}.resize((size_t)({} * {} * {}));\n",
                    id, e, d, h
                ));
                train_body.push_str(&format!(
                    "    ns_moe_grad_we1(buf_{}.data(), {}, {}, {}, {}, ctx->moe_active.data(), buf_{}.data(), M, {}, {}, {}); }}\n",
                    dout,
                    src_of(&xo),
                    src_of(&wg),
                    src_of(&we1),
                    src_of(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::MoeGradWe2 => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let xo = instr.operands.get(1).cloned().unwrap_or_default();
                let wg = instr.operands.get(2).cloned().unwrap_or_default();
                let we1 = instr.operands.get(3).cloned().unwrap_or_default();
                let we2 = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                train_body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)(buf_{}.size()) / {};\n",
                    dout, d
                ));
                train_body.push_str(&format!(
                    "    buf_{}.resize((size_t)({} * {} * {}));\n",
                    id, e, h, d
                ));
                train_body.push_str(&format!(
                    "    ns_moe_grad_we2(buf_{}.data(), {}, {}, {}, {}, ctx->moe_active.data(), buf_{}.data(), M, {}, {}, {}); }}\n",
                    dout,
                    src_of(&xo),
                    src_of(&wg),
                    src_of(&we1),
                    src_of(&we2),
                    id,
                    d,
                    h,
                    e
                ));
            }
            MLIROp::AttentionGradX
            | MLIROp::AttentionGradWq
            | MLIROp::AttentionGradWk
            | MLIROp::AttentionGradWv
            | MLIROp::AttentionGradWo => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let xo = instr.operands.get(1).cloned().unwrap_or_default();
                let wq = instr.operands.get(2).cloned().unwrap_or_default();
                let wk = instr.operands.get(3).cloned().unwrap_or_default();
                let wv = instr.operands.get(4).cloned().unwrap_or_default();
                let wo = instr.operands.get(5).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 { instr.int_attr } else { 1 };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                let dd = d * d;
                let nele = if instr.op == MLIROp::AttentionGradX {
                    format!("(int64_t)(buf_{}.size())", dout)
                } else {
                    dd.to_string()
                };
                let tgt = format!("buf_{}.data()", id);
                let causal = if !instr.ints_attr.is_empty() {
                    instr.ints_attr[0].to_string()
                } else {
                    "0".to_string()
                };
                train_body.push_str(&format!(
                    "  {{ int64_t BS = (int64_t)(buf_{}.size()) / {};\n",
                    dout, d
                ));
                train_body.push_str(&format!("    buf_{}.resize((size_t)({}));\n", id, nele));
                train_body.push_str(&format!(
                    "    ns_attention_bwd(buf_{}.data(), {}, {}, {}, {}, {},\n",
                    dout,
                    src_of(&xo),
                    src_of(&wq),
                    src_of(&wk),
                    src_of(&wv),
                    src_of(&wo)
                ));
                if instr.op == MLIROp::AttentionGradX {
                    train_body.push_str(&format!("        {}, nullptr, nullptr, nullptr, nullptr,\n", tgt));
                } else if instr.op == MLIROp::AttentionGradWq {
                    train_body.push_str(&format!("        nullptr, {}, nullptr, nullptr, nullptr,\n", tgt));
                } else if instr.op == MLIROp::AttentionGradWk {
                    train_body.push_str(&format!("        nullptr, nullptr, {}, nullptr, nullptr,\n", tgt));
                } else if instr.op == MLIROp::AttentionGradWv {
                    train_body.push_str(&format!("        nullptr, nullptr, nullptr, {}, nullptr,\n", tgt));
                } else {
                    train_body.push_str(&format!("        nullptr, nullptr, nullptr, nullptr, {},\n", tgt));
                }
                train_body.push_str(&format!("        BS, {}, {}, BS, {}); }}\n", d, h, causal));
            }
            MLIROp::LayernormGrad => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let act_in = instr.operands.get(1).cloned().unwrap_or_default();
                let mut last = train_row_dims(&act_in);
                if last <= 0 {
                    last = 1;
                }
                train_body.push_str(&format!("  {{ size_t zn = buf_{}.size();\n", dout));
                train_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                train_body.push_str(&format!(
                    "    ns_layernorm_grad(buf_{}.data(), buf_{}.data(), buf_{}.data(), zn, {}); }}\n",
                    dout, act_in, id, last
                ));
            }
            MLIROp::ActivationGrad => {
                let dout = instr.operands.get(0).cloned().unwrap_or_default();
                let act_in = instr.operands.get(1).cloned().unwrap_or_default();
                if instr.attribute == "dropout" {
                    train_body.push_str(&format!("  {{ size_t zn = buf_{}.size();\n", dout));
                    train_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                    train_body.push_str(&format!("    const float* dg = buf_{}.data();\n", dout));
                    train_body.push_str(&format!("    const float* dm = buf_dm_{}.data();\n", act_in));
                    train_body.push_str(&format!(
                        "    for (size_t i = 0; i < zn; i++) buf_{}[i] = dg[i] * dm[i]; }}\n",
                        id
                    ));
                } else {
                    let code = match instr.attribute.as_str() {
                        "relu" => 1,
                        "leaky_relu" => 2,
                        "sigmoid" => 3,
                        "tanh" => 4,
                        "swish" | "silu" => 5,
                        "gelu" => 6,
                        _ => 0,
                    };
                    train_body.push_str(&format!("  {{ size_t zn = buf_{}.size();\n", dout));
                    train_body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                    train_body.push_str(&format!(
                        "    ns_act_grad(buf_{}.data(), buf_{}.data(), buf_{}.data(), zn, {}); }}\n",
                        dout, act_in, id, code
                    ));
                }
            }
            MLIROp::OptStep => {
                let mut i = 0;
                while i + 1 < instr.operands.len() {
                    let wt = instr.operands[i].clone();
                    let g = instr.operands[i + 1].clone();
                    let r = ids.get(&wt).map_or(1, |inf| if inf.rows > 0 { inf.rows } else { 1 });
                    let c = ids.get(&wt).map_or(1, |inf| if inf.cols > 0 { inf.cols } else { 1 });
                    let n = r * c;
                    let use_muon = r > 1 && c > 1 && std::cmp::min(r, c) >= 8;
                    opt_body.push_str(&format!("  {{ size_t n_ = {};\n", n));
                    if use_muon {
                        opt_body.push_str(&format!(
                            "    std::vector<float>& st_m = ctx->st_m_{}; std::vector<float>& st_am = ctx->st_am_{}; std::vector<float>& st_av = ctx->st_av_{};\n",
                            wt, wt, wt
                        ));
                        opt_body.push_str(
                            "    if (st_m.size() != n_) { st_m.assign(n_, 0.f); st_am.assign(n_, 0.f); st_av.assign(n_, 0.f); }\n",
                        );
                        opt_body.push_str(&format!(
                            "    for (size_t i = 0; i < n_; i++) st_m[i] = {}f * st_m[i] + buf_{}[i];\n",
                            fmt_float9(optim::K_MUON_MOMENTUM),
                            g
                        ));
                        opt_body.push_str(&format!(
                            "    ns_orthonom(st_m.data(), (size_t){}, (size_t){});\n",
                            r, c
                        ));
                        opt_body.push_str(&format!(
                            "    float rank = sqrtf((float)std::min((int64_t){}, (int64_t){}));\n",
                            r, c
                        ));
                        opt_body.push_str("    float le = lr * rank;\n");
                        opt_body.push_str(&format!(
                            "    for (size_t i = 0; i < n_; i++) buf_{}[i] -= le * st_m[i] + ({}f * lr) * buf_{}[i];\n",
                            wt,
                            fmt_float9(optim::K_MUON_DECAY),
                            wt
                        ));
                        opt_body.push_str("    std::fill(st_m.begin(), st_m.end(), 0.f);\n");
                    } else {
                        opt_body.push_str(&format!(
                            "    std::vector<float>& st_m = ctx->st_m_{}; std::vector<float>& st_am = ctx->st_am_{}; std::vector<float>& st_av = ctx->st_av_{}; size_t& st_t = ctx->st_t_{};\n",
                            wt, wt, wt, wt
                        ));
                        opt_body.push_str("    size_t t = ++st_t;\n");
                        opt_body.push_str(
                            "    if (st_m.size() != n_) { st_m.assign(n_, 0.f); st_am.assign(n_, 0.f); st_av.assign(n_, 0.f); }\n",
                        );
                        opt_body.push_str(&format!(
                            "    float b1t = 1.f - std::pow({}f, (float)t), b2t = 1.f - std::pow({}f, (float)t);\n",
                            fmt_float9(optim::K_ADAMW_BETA1),
                            fmt_float9(optim::K_ADAMW_BETA2)
                        ));
                        opt_body.push_str("    for (size_t i = 0; i < n_; i++) {\n");
                        opt_body.push_str(&format!("      float gg = buf_{}[i];\n", g));
                        opt_body.push_str(&format!(
                            "      st_am[i] = {}f * st_am[i] + {}f * gg;\n",
                            fmt_float9(optim::K_ADAMW_BETA1),
                            fmt_float9(optim::K_ONE_MINUS_BETA1)
                        ));
                        opt_body.push_str(&format!(
                            "      st_av[i] = {}f * st_av[i] + {}f * gg * gg;\n",
                            fmt_float9(optim::K_ADAMW_BETA2),
                            fmt_float9(optim::K_ONE_MINUS_BETA2)
                        ));
                        opt_body.push_str("      float mh = st_am[i] / b1t, vh = st_av[i] / b2t;\n");
                        opt_body.push_str(&format!(
                            "      buf_{}[i] -= (lr * mh / (sqrtf(vh) + {}f)) + ({}f * lr) * buf_{}[i];\n",
                            wt,
                            fmt_float9(optim::K_ADAMW_EPS),
                            fmt_float9(optim::K_ADAMW_DECAY),
                            wt
                        ));
                        opt_body.push_str("    }\n");
                    }
                    opt_body.push_str("  }\n");
                    i += 2;
                }
            }
            _ => {}
        }
    }

    os.push_str(&fwd_body);
    os.push_str("  if (train_mode) {\n");
    os.push_str(&train_body);
    os.push_str(&opt_body);
    os.push_str("  }\n");

    let mut loss_id = tfn.return_id.clone();
    if loss_id.is_empty() {
        for instr in &tfn.instructions {
            if instr.op == MLIROp::CrossEntropy {
                loss_id = instr.result_id.clone();
            }
        }
    }
    if !loss_id.is_empty() && ids.contains_key(&loss_id) {
        os.push_str(&format!("  if (loss_out) *loss_out = buf_{}[0];\n", loss_id));
    }
    os.push_str("  if (train_mode && wout) {\n");
    {
        let mut off: u64 = 0;
        for wt in &worder_t {
            let n = ids
                .get(wt)
                .map_or(1, |i| if i.static_numel > 0 { i.static_numel } else { 1 });
            os.push_str(&format!(
                "    std::memcpy(wout + {}, buf_{}.data(), {} * sizeof(float));\n",
                off, wt, n
            ));
            off += n as u64;
        }
    }
    os.push_str("  }\n");
    os.push_str("}\n");
    Ok(os)
}

// ---------------------------------------------------------------------------
// gen_cpu — port of `CodeGenerator::gen_cpu` (codegen.cpp 1137-2681).
// ---------------------------------------------------------------------------

fn parse_slice_attr(attr: &str) -> (i64, i64, i64) {
    let b = attr.as_bytes();
    let mut pos = 0usize;
    let read_int = |pos: &mut usize| -> i64 {
        while *pos < b.len() && (b[*pos] as char).is_ascii_whitespace() {
            *pos += 1;
        }
        let mut neg = false;
        if *pos < b.len() && b[*pos] == b'-' {
            neg = true;
            *pos += 1;
        }
        let mut v: i64 = 0;
        let mut any = false;
        while *pos < b.len() && (b[*pos] as char).is_ascii_digit() {
            any = true;
            v = v * 10 + (b[*pos] - b'0') as i64;
            *pos += 1;
        }
        if !any {
            return 0;
        }
        if neg { -v } else { v }
    };
    let axis = read_int(&mut pos);
    if pos < b.len() {
        pos += 1;
    }
    let s = read_int(&mut pos);
    if pos < b.len() {
        pos += 1;
    }
    let e = read_int(&mut pos);
    (axis, s, e)
}

fn gen_cpu(module: &MLIRModule, opts: &CodegenOptions) -> NsResult<String> {
    let (fn_, tfn) = select_functions(module);
    let fn_ = match fn_ {
        Some(f) => f,
        None => {
            let mut o = String::new();
            o.push_str("// Generated by NeuralScript compiler (CPU reference)\n");
            o.push_str("#include <cstddef>\n");
            o.push_str(&format!(
                "extern \"C\" void {}(const float*, const float*, float*, size_t) {{}}\n",
                opts.function_name
            ));
            return Ok(o);
        }
    };

    let have_moe = fn_.instructions.iter().any(|i| i.op == MLIROp::LayerMoe);

    let mut ids: BTreeMap<String, CuIdInfo> = BTreeMap::new();
    fill_cu_ids(fn_, &mut ids);
    {
        let mut marked = false;
        for instr in &fn_.instructions {
            if marked {
                break;
            }
            let op0 = input_op_of(instr);
            if !op0.is_empty() && !ids.get(&op0).map_or(false, |i| i.is_weight) {
                ids.entry(op0.clone()).or_default().is_input = true;
                marked = true;
            }
        }
    }
    {
        let mut in_w: i64 = -1;
        for instr in &fn_.instructions {
            let gemm = instr.op == MLIROp::Matmul || instr.op == MLIROp::Fused;
            if gemm && instr.operands.len() >= 2 && in_w < 0 {
                in_w = ids.get(&instr.operands[1]).map_or(-1, |i| i.rows);
            }
            if instr.op == MLIROp::LayerEmbedding && in_w < 0 {
                in_w = 1;
            }
            if in_w >= 0 {
                break;
            }
        }
        if in_w < 0 {
            in_w = 1;
        }
        for e in ids.values_mut() {
            if e.is_input {
                e.cols = in_w;
            }
        }
    }

    let mut worder: Vec<String> = Vec::new();
    for instr in &fn_.instructions {
        if instr.op == MLIROp::TensorAlloc {
            worder.push(instr.result_id.clone());
        }
    }

    let mut oss = String::new();
    oss.push_str("// Generated by NeuralScript compiler (CPU reference)\n");
    oss.push_str(&format!("// Lowered function: @{}\n", fn_.name));
    oss.push_str("#include <cstddef>\n");
    oss.push_str("#include <cstdint>\n");
    oss.push_str("#include <cmath>\n");
    oss.push_str("#include <cstring>\n");
    oss.push_str("#include <cstdio>\n");
    oss.push_str("#include <vector>\n");
    oss.push_str("#include <random>\n");
    oss.push_str("#include <algorithm>\n\n");

    oss.push_str(cpu_blobs::NS_ACT2);
    oss.push_str(cpu_blobs::NS_MATMUL);
    oss.push_str(cpu_blobs::NS_BINOP);
    oss.push_str(cpu_blobs::NS_BINOP_GRAD);
    oss.push_str(cpu_blobs::NS_LAYERNORM);
    oss.push_str(cpu_blobs::NS_SOFTMAX);
    oss.push_str(cpu_blobs::NS_ACT_DERIV);
    oss.push_str(cpu_blobs::NS_MATMUL_GRAD_A);
    oss.push_str(cpu_blobs::NS_MATMUL_GRAD_W);
    oss.push_str(cpu_blobs::NS_ACT_GRAD);
    oss.push_str(cpu_blobs::NS_LAYERNORM_GRAD);
    oss.push_str(cpu_blobs::NS_EMBEDDING_GRAD_W);
    oss.push_str(cpu_blobs::NS_MOE_GRAD_X);
    oss.push_str(cpu_blobs::NS_MOE_GRAD_WG);
    oss.push_str(cpu_blobs::NS_MOE_GRAD_WE1);
    oss.push_str(cpu_blobs::NS_MOE_GRAD_WE2);
    oss.push_str(cpu_blobs::NS_LOSS_GRAD);
    oss.push_str(cpu_blobs::NS_CROSS_ENTROPY);
    oss.push_str(cpu_blobs::NS_ORTHONOM);
    oss.push_str(cpu_blobs::NS_TRANSPOSE2D);
    oss.push_str(cpu_blobs::NS_CONCAT2);
    oss.push_str(cpu_blobs::NS_CONCAT0);
    oss.push_str(cpu_blobs::NS_SLICE2);
    oss.push_str(cpu_blobs::NS_SLICEROWS);
    oss.push_str(cpu_blobs::NS_INDEX2);
    oss.push_str(cpu_blobs::NS_INDEXROWS);
    oss.push_str(cpu_blobs::NS_SCATTER2);
    oss.push_str(cpu_blobs::NS_SCATTERROWS);
    oss.push_str(cpu_blobs::NS_EMBEDDING);
    oss.push_str(cpu_blobs::NS_ATTENTION_FWD);
    oss.push_str(cpu_blobs::NS_ATTENTION_BWD);
    oss.push_str(cpu_blobs::NS_MOE_FWD);

    for (id, _inf) in &ids {
        oss.push_str(&format!("static std::vector<float> buf_{};\n", id));
    }
    oss.push('\n');

    let numel_expr = |id: &str| -> String {
        if ids.get(id).map_or(false, |i| i.is_input) {
            String::from("n")
        } else {
            format!("buf_{}.size()", id)
        }
    };
    let src_or = |id: &str| -> String {
        if ids.get(id).map_or(false, |i| i.is_input) {
            String::from("input")
        } else {
            format!("buf_{}.data()", id)
        }
    };

    let mut body = String::new();
    for instr in &fn_.instructions {
        let id = instr.result_id.clone();
        match instr.op {
            MLIROp::Matmul => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let inf = ids.get(&b).copied().unwrap_or_default();
                let ks = inf.rows;
                let ns_ = inf.cols;
                if ks <= 0 || ns_ <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for matmul '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                body.push_str(&format!("  {{ int64_t K = {}, N = {};\n", ks, ns_));
                body.push_str(&format!("    int64_t M = (int64_t)({}) / K;\n", numel_expr(&a)));
                body.push_str(&format!("    buf_{}.resize((size_t)(M * N));\n", id));
                body.push_str(&format!(
                    "    ns_matmul({}, buf_{}.data(), buf_{}.data(), M, K, N); }}\n",
                    src_or(&a),
                    b,
                    id
                ));
            }
            MLIROp::Relu
            | MLIROp::LeakyRelu
            | MLIROp::Sigmoid
            | MLIROp::Tanh
            | MLIROp::Swish
            | MLIROp::Gelu
            | MLIROp::Silu
            | MLIROp::Identity => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let code = match instr.op {
                    MLIROp::Relu => 1,
                    MLIROp::LeakyRelu => 2,
                    MLIROp::Sigmoid => 3,
                    MLIROp::Tanh => 4,
                    MLIROp::Swish | MLIROp::Silu => 5,
                    MLIROp::Gelu => 6,
                    _ => 0,
                };
                body.push_str(&format!("  {{ size_t zn = {};\n", numel_expr(&inp)));
                body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                body.push_str(&format!("    const float* src = {};\n", src_or(&inp)));
                body.push_str(&format!(
                    "    for (size_t i = 0; i < zn; i++) buf_{}[i] = ns_act2(src[i], {}); }}\n",
                    id, code
                ));
            }
            MLIROp::Dropout => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                body.push_str(&format!("  {{ size_t zn = {};\n", numel_expr(&inp)));
                body.push_str(&format!("    buf_{}.resize(zn);\n", id));
                body.push_str(&format!("    const float* src = {};\n", src_or(&inp)));
                body.push_str(&format!("    for (size_t i = 0; i < zn; i++) buf_{}[i] = src[i]; }}\n", id));
            }
            MLIROp::ElementwiseBinop => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let code = if instr.attribute == "-" {
                    1
                } else if instr.attribute == "*" {
                    2
                } else if instr.attribute == "/" {
                    3
                } else {
                    0
                };
                body.push_str(&format!("  {{ size_t na = {};\n", numel_expr(&a)));
                body.push_str(&format!("    buf_{}.resize(na);\n", id));
                body.push_str(&format!(
                    "    ns_binop({}, na, buf_{}.data(), buf_{}.size(), buf_{}.data(), {}); }}\n",
                    src_or(&a),
                    b,
                    b,
                    id,
                    code
                ));
            }
            MLIROp::Layernorm => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let last_v = {
                    let c = ids.get(&id).map_or(0, |i| i.cols);
                    if c > 0 {
                        c
                    } else {
                        let c2 = ids.get(&inp).map_or(0, |i| i.cols);
                        if c2 > 0 { c2 } else { 1 }
                    }
                };
                body.push_str(&format!(
                    "  {{ buf_{} = {};\n",
                    id,
                    if ids.get(&inp).map_or(false, |i| i.is_input) {
                        String::from("std::vector<float>(input, input + n)")
                    } else {
                        format!("buf_{}", inp)
                    }
                ));
                body.push_str(&format!(
                    "    ns_layernorm(buf_{}.data(), buf_{}.size(), {}); }}\n",
                    id, id, last_v
                ));
            }
            MLIROp::Softmax => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let last_v = {
                    let c = ids.get(&id).map_or(0, |i| i.cols);
                    if c > 0 {
                        c
                    } else {
                        let c2 = ids.get(&inp).map_or(0, |i| i.cols);
                        if c2 > 0 { c2 } else { 1 }
                    }
                };
                body.push_str(&format!(
                    "  {{ buf_{} = {};\n",
                    id,
                    if ids.get(&inp).map_or(false, |i| i.is_input) {
                        String::from("std::vector<float>(input, input + n)")
                    } else {
                        format!("buf_{}", inp)
                    }
                ));
                body.push_str(&format!(
                    "    ns_softmax(buf_{}.data(), buf_{}.size(), {}); }}\n",
                    id, id, last_v
                ));
            }
            MLIROp::Fused => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let inf = ids.get(&b).copied().unwrap_or_default();
                let ks = inf.rows;
                let ns_ = inf.cols;
                if ks <= 0 || ns_ <= 0 {
                    return Err(ns_error!(
                        "codegen: cannot infer static shape of operand '{}' for fused group '{}' (dynamic dims are not supported here)",
                        b, id
                    ));
                }
                body.push_str(&format!("  {{ int64_t K = {}, N = {};\n", ks, ns_));
                body.push_str(&format!("    int64_t M = (int64_t)({}) / K;\n", numel_expr(&a)));
                body.push_str(&format!("    buf_{}.resize((size_t)(M * N));\n", id));
                body.push_str(&format!(
                    "    ns_matmul({}, buf_{}.data(), buf_{}.data(), M, K, N);\n",
                    src_or(&a),
                    b,
                    id
                ));
                let mut group_idx: Option<usize> = None;
                for (gi, g) in module.fused_groups.iter().enumerate() {
                    if g.result_id == instr.result_id {
                        group_idx = Some(gi);
                    }
                }
                if let Some(gi) = group_idx {
                    let g = &module.fused_groups[gi];
                    let mut extra: usize = 0;
                    for opname in &g.ops {
                        if opname == "+" || opname == "-" || opname == "*" || opname == "/" {
                            let rhs = g
                                .epilogue_operands
                                .get(extra)
                                .cloned()
                                .unwrap_or_default();
                            let code = if opname == "-" {
                                1
                            } else if opname == "*" {
                                2
                            } else if opname == "/" {
                                3
                            } else {
                                0
                            };
                            body.push_str(&format!("    {{ size_t zn = buf_{}.size();\n", id));
                            body.push_str(&format!(
                                "      ns_binop(buf_{}.data(), zn, buf_{}.data(), buf_{}.size(), buf_{}.data(), {}); }}\n",
                                id, rhs, rhs, id, code
                            ));
                            extra += 1;
                        } else if opname == "layernorm" {
                            body.push_str(&format!(
                                "    ns_layernorm(buf_{}.data(), buf_{}.size(), {});\n",
                                id, id, ns_
                            ));
                        } else if opname == "softmax" {
                            body.push_str(&format!(
                                "    ns_softmax(buf_{}.data(), buf_{}.size(), {});\n",
                                id, id, ns_
                            ));
                        } else {
                            let code = match opname.as_str() {
                                "relu" => 1,
                                "leaky_relu" => 2,
                                "sigmoid" => 3,
                                "tanh" => 4,
                                "swish" | "silu" => 5,
                                "gelu" => 6,
                                _ => 0,
                            };
                            body.push_str(&format!("    {{ size_t zn = buf_{}.size();\n", id));
                            body.push_str(&format!(
                                "      for (size_t i = 0; i < zn; i++) buf_{}[i] = ns_act2(buf_{}[i], {}); }}\n",
                                id, id, code
                            ));
                        }
                    }
                }
                body.push_str("  }\n");
            }
            MLIROp::Constant => {
                body.push_str(&format!("  {{ buf_{}.assign(1, (float)({})); }}\n", id, instr.attribute));
            }
            MLIROp::TensorAlloc => {}
            MLIROp::Transpose => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let c = ids.get(&inp).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                let rows = ids.get(&inp).map_or(0, |i| i.rows);
                if rows > 0 {
                    body.push_str(&format!(
                        "  {{ buf_{}.resize((size_t)({} * {}));\n",
                        id, c, rows
                    ));
                    body.push_str(&format!(
                        "    ns_transpose2d({}, buf_{}.data(), {}, {}); }}\n",
                        src_or(&inp),
                        id,
                        rows,
                        c
                    ));
                } else {
                    body.push_str(&format!(
                        "  {{ int64_t R = (int64_t)({}) / {};\n",
                        numel_expr(&inp),
                        c
                    ));
                    body.push_str(&format!("    buf_{}.resize((size_t)(R * {}));\n", id, c));
                    body.push_str(&format!(
                        "    ns_transpose2d({}, buf_{}.data(), R, {}); }}\n",
                        src_or(&inp),
                        id,
                        c
                    ));
                }
            }
            MLIROp::Concat => {
                let a = instr.operands.get(0).cloned().unwrap_or_default();
                let b = instr.operands.get(1).cloned().unwrap_or_default();
                let axis: i64 = if !instr.attribute.is_empty() && instr.attribute == "0" {
                    0
                } else {
                    1
                };
                let src_a = |s: &str| -> String {
                    if ids.get(s).map_or(false, |i| i.is_input) {
                        String::from("input")
                    } else {
                        format!("buf_{}.data()", s)
                    }
                };
                if axis == 1 {
                    let ca = ids.get(&a).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                    let cb = ids.get(&b).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                    body.push_str(&format!(
                        "  {{ int64_t R = (int64_t)({}) / {};\n",
                        numel_expr(&a),
                        ca
                    ));
                    body.push_str(&format!(
                        "    buf_{}.resize((size_t)(R * ({} + {})));\n",
                        id, ca, cb
                    ));
                    body.push_str(&format!(
                        "    ns_concat2({}, {}, buf_{}.data(), R, {}, {}); }}\n",
                        src_a(&a),
                        src_a(&b),
                        id,
                        ca,
                        cb
                    ));
                } else {
                    let d = ids.get(&a).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                    body.push_str(&format!(
                        "  {{ int64_t Ba = (int64_t)({}) / {};\n",
                        numel_expr(&a),
                        d
                    ));
                    body.push_str(&format!(
                        "    int64_t Bb = (int64_t)({}) / {};\n",
                        numel_expr(&b),
                        d
                    ));
                    body.push_str(&format!(
                        "    buf_{}.resize((size_t)((Ba + Bb) * {}));\n",
                        id, d
                    ));
                    body.push_str(&format!(
                        "    ns_concat0({}, {}, buf_{}.data(), Ba, Bb, {}); }}\n",
                        src_a(&a),
                        src_a(&b),
                        id,
                        d
                    ));
                }
            }
            MLIROp::Reshape => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let src = if ids.get(&inp).map_or(false, |i| i.is_input) {
                    String::from("std::vector<float>(input, input+n)")
                } else {
                    format!("buf_{}", inp)
                };
                body.push_str(&format!("  buf_{} = {};\n", id, src));
            }
            MLIROp::Slice => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let (axis, s, e) = parse_slice_attr(&instr.attribute);
                let src = src_or(&inp);
                if axis == 0 {
                    let c = ids.get(&inp).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                    body.push_str(&format!(
                        "  {{ buf_{}.resize((size_t)({}));\n",
                        id,
                        (e - s) * c
                    ));
                    body.push_str(&format!(
                        "    ns_slicerows({}, buf_{}.data(), {}, {}, {}); }}\n",
                        src, id, c, s, e
                    ));
                } else {
                    let c = ids.get(&inp).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                    body.push_str(&format!(
                        "  {{ int64_t M = (int64_t)({}) / {};\n",
                        numel_expr(&inp),
                        c
                    ));
                    body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, e - s));
                    body.push_str(&format!(
                        "    ns_slice2({}, buf_{}.data(), M, {}, {}, {}); }}\n",
                        src, id, c, s, e
                    ));
                }
            }
            MLIROp::Index => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let axis = instr.int_attr as i64;
                let c = ids.get(&inp).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                let l = instr.ints_attr.len() as i64;
                let src = src_or(&inp);
                body.push_str(&format!("  static const int64_t idx_{}[{}] = {{", id, l));
                for (k, v) in instr.ints_attr.iter().enumerate() {
                    body.push_str(&format!("{}", v));
                    if k + 1 < instr.ints_attr.len() {
                        body.push(',');
                    }
                }
                body.push_str("};\n");
                if axis == 0 {
                    body.push_str(&format!("  buf_{}.resize((size_t)({}));\n", id, l * c));
                    body.push_str(&format!(
                        "  ns_indexrows({}, idx_{}, buf_{}.data(), {}, {});\n",
                        src, id, id, l, c
                    ));
                } else {
                    body.push_str(&format!(
                        "  {{ int64_t M = (int64_t)({}) / {};\n",
                        numel_expr(&inp),
                        c
                    ));
                    body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, l));
                    body.push_str(&format!(
                        "    ns_index2({}, idx_{}, buf_{}.data(), M, {}, {}); }}\n",
                        src, id, id, c, l
                    ));
                }
            }
            MLIROp::Scatter => {
                let inp = instr.operands.get(0).cloned().unwrap_or_default();
                let upd = instr.operands.get(1).cloned().unwrap_or_default();
                let axis = instr.int_attr as i64;
                let c = ids.get(&inp).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                let l = instr.ints_attr.len() as i64;
                let src = src_or(&inp);
                let upds = src_or(&upd);
                body.push_str(&format!("  static const int64_t idx_{}[{}] = {{", id, l));
                for (k, v) in instr.ints_attr.iter().enumerate() {
                    body.push_str(&format!("{}", v));
                    if k + 1 < instr.ints_attr.len() {
                        body.push(',');
                    }
                }
                body.push_str("};\n");
                if axis == 0 {
                    let r = ids.get(&inp).map_or(0, |i| i.rows);
                    body.push_str(&format!("  buf_{}.resize((size_t)({}));\n", id, r * c));
                    body.push_str(&format!(
                        "  ns_scatterrows({}, idx_{}, {}, buf_{}.data(), {}, {}, {});\n",
                        src, id, upds, id, r, c, l
                    ));
                } else {
                    body.push_str(&format!(
                        "  {{ int64_t M = (int64_t)({}) / {};\n",
                        numel_expr(&inp),
                        c
                    ));
                    body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, c));
                    body.push_str(&format!(
                        "    ns_scatter2({}, idx_{}, {}, buf_{}.data(), M, {}, {}); }}\n",
                        src, id, upds, id, c, l
                    ));
                }
            }
            MLIROp::LayerEmbedding => {
                let wt = instr.operands.get(0).cloned().unwrap_or_default();
                let idx = instr.operands.get(1).cloned().unwrap_or_default();
                let v = ids.get(&wt).map_or(1, |i| if i.rows > 0 { i.rows } else { 1 });
                let d = ids.get(&wt).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 });
                body.push_str(&format!("  {{ int64_t n_idx = {};\n", numel_expr(&idx)));
                body.push_str(&format!("    buf_{}.resize((size_t)(n_idx * {}));\n", id, d));
                body.push_str(&format!(
                    "    ns_embedding({}, {}, buf_{}.data(), n_idx, {}, {}); }}\n",
                    src_or(&wt),
                    src_or(&idx),
                    id,
                    v,
                    d
                ));
            }
            MLIROp::LayerAttention => {
                let x = instr.operands.get(0).cloned().unwrap_or_default();
                let wq = instr.operands.get(1).cloned().unwrap_or_default();
                let wk = instr.operands.get(2).cloned().unwrap_or_default();
                let wv = instr.operands.get(3).cloned().unwrap_or_default();
                let wo = instr.operands.get(4).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 {
                    instr.int_attr
                } else {
                    ids.get(&x).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 })
                };
                let h: i64 = if instr.attribute.is_empty() {
                    1
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(1)
                };
                let causal = if !instr.ints_attr.is_empty() {
                    instr.ints_attr[0].to_string()
                } else {
                    "0".to_string()
                };
                body.push_str(&format!("  {{ int64_t BS = {} / {};\n", numel_expr(&x), d));
                body.push_str("    int64_t S = BS;\n");
                body.push_str(&format!(
                    "    std::vector<float> q_b(BS*{}), k_b(BS*{}), v_b(BS*{}), proj_b(BS*{}), sc_b(S*S);\n",
                    d, d, d, d
                ));
                body.push_str(&format!("    buf_{}.resize((size_t)(BS*{}));\n", id, d));
                body.push_str(&format!(
                    "    ns_attention_fwd({}, {}, {}, {}, {},\n",
                    src_or(&x),
                    src_or(&wq),
                    src_or(&wk),
                    src_or(&wv),
                    src_or(&wo)
                ));
                body.push_str(&format!(
                    "        q_b.data(), k_b.data(), v_b.data(), sc_b.data(), proj_b.data(), buf_{}.data(),\n",
                    id
                ));
                body.push_str(&format!("        BS, {}, {}, S, {}); }}\n", d, h, causal));
            }
            MLIROp::LayerMoe => {
                let x = instr.operands.get(0).cloned().unwrap_or_default();
                let wg = instr.operands.get(1).cloned().unwrap_or_default();
                let we1 = instr.operands.get(2).cloned().unwrap_or_default();
                let we2 = instr.operands.get(3).cloned().unwrap_or_default();
                let d = if instr.int_attr > 0 {
                    instr.int_attr
                } else {
                    ids.get(&x).map_or(1, |i| if i.cols > 0 { i.cols } else { 1 })
                };
                let e: i64 = if instr.attribute.is_empty() {
                    4
                } else {
                    instr.attribute.parse::<i64>().unwrap_or(4)
                };
                let h: i64 = if instr.float_attr > 0.0 {
                    instr.float_attr as i64
                } else {
                    4 * d
                };
                let active = if have_moe { "moe_active" } else { "0" };
                body.push_str(&format!(
                    "  {{ int64_t M = (int64_t)({}) / {};\n",
                    numel_expr(&x),
                    d
                ));
                body.push_str(&format!("    buf_{}.resize((size_t)(M * {}));\n", id, d));
                body.push_str(&format!(
                    "    ns_moe_fwd({}, {}, {}, {}, {}, buf_{}.data(), M, {}, {}, {}); }}\n",
                    src_or(&x),
                    src_or(&wg),
                    src_or(&we1),
                    src_or(&we2),
                    active,
                    id,
                    d,
                    h,
                    e
                ));
            }
            _ => {}
        }
    }

    oss.push_str(&format!(
        "extern \"C\" void {}(\n    const float* input, const float* weights, float* output, size_t n{}) {{\n",
        opts.function_name,
        if have_moe { ", const uint8_t* moe_active" } else { "" }
    ));
    {
        let mut off: u64 = 0;
        for w in &worder {
            let szn = ids.get(w).map_or(1, |i| if i.static_numel > 0 { i.static_numel } else { 1 });
            oss.push_str(&format!("  // weight {} @ offset {} floats, count {}\n", w, off, szn));
            oss.push_str(&format!("  buf_{}.assign(weights + {}, weights + {});\n", w, off, off + szn as u64));
            off += szn as u64;
        }
    }
    oss.push_str(&body);
    let mut result_id = fn_.return_id.clone();
    if result_id.is_empty() {
        for instr in &fn_.instructions {
            if instr.op == MLIROp::Matmul
                || instr.op == MLIROp::Fused
                || instr.op == MLIROp::Softmax
                || !instr.result_id.is_empty()
            {
                result_id = instr.result_id.clone();
            }
        }
    }
    if result_id.is_empty() {
        oss.push_str("  (void)input; (void)output; (void)n;\n");
    } else {
        oss.push_str(&format!("  {{ size_t rn = buf_{}.size();\n", result_id));
        oss.push_str(&format!("    memcpy(output, buf_{}.data(), rn * sizeof(float)); }}\n", result_id));
    }
    oss.push_str("}\n");

    if opts.emit_runtime_driver {
        let total: u64 = worder
            .iter()
            .map(|w| ids.get(w).map_or(1, |i| if i.static_numel > 0 { i.static_numel as u64 } else { 1 }))
            .sum();

        let mut in_cols: i64 = -1;
        let mut out_cols: i64 = -1;
        for instr in &fn_.instructions {
            if instr.op != MLIROp::TensorAlloc
                && !instr.result_id.is_empty()
                && instr.result_id == fn_.return_id
            {
                out_cols = ids.get(&instr.result_id).map_or(-1, |i| i.cols);
            }
        }
        for instr in &fn_.instructions {
            let gemm = instr.op == MLIROp::Matmul || instr.op == MLIROp::Fused;
            if gemm && instr.operands.len() >= 2 && in_cols < 0 {
                in_cols = ids.get(&instr.operands[1]).map_or(-1, |i| i.rows);
            }
            if gemm && instr.operands.len() >= 2 {
                out_cols = ids.get(&instr.operands[1]).map_or(-1, |i| i.cols);
            }
            if instr.op == MLIROp::LayerEmbedding && in_cols < 0 {
                in_cols = 1;
            }
            if (instr.op == MLIROp::LayerAttention
                || instr.op == MLIROp::Layernorm
                || instr.op == MLIROp::LayerMoe
                || instr.op == MLIROp::Softmax
                || instr.op == MLIROp::Concat
                || instr.op == MLIROp::Slice
                || instr.op == MLIROp::Index
                || instr.op == MLIROp::Scatter
                || instr.op == MLIROp::Transpose
                || instr.op == MLIROp::Reshape)
                && !instr.result_id.is_empty()
            {
                let c = ids.get(&instr.result_id).map_or(0, |i| i.cols);
                if c > 0 {
                    out_cols = c;
                }
            }
        }
        if in_cols < 0 {
            in_cols = 1;
        }
        if out_cols < 0 {
            out_cols = 1;
        }

        oss.push_str("\n// ---- C-ABI (host entry points) ----\n");
        oss.push_str("#include <cstring>\n");

        let tfn_ = tfn;

        let mut moe_e: i64 = 0;
        let mut moe_d: i64 = 0;
        let mut moe_h: i64 = 0;
        let mut moe_k0: i64 = 0;
        let mut moe_off_g: u64 = 0;
        let mut moe_off_e1: u64 = 0;
        let mut moe_off_e2: u64 = 0;
        {
            let probes: [Option<&MLIRFunction>; 2] = [tfn_, Some(fn_)];
            for pf in probes {
                let Some(pf) = pf else { continue };
                for ins in &pf.instructions {
                    if ins.op != MLIROp::LayerMoe || ins.operands.len() < 4 {
                        continue;
                    }
                    moe_e = if ins.attribute.is_empty() {
                        4
                    } else {
                        ins.attribute.parse::<i64>().unwrap_or(4)
                    };
                    moe_d = ins.int_attr;
                    moe_h = if ins.float_attr > 0.0 {
                        ins.float_attr as i64
                    } else {
                        4 * ins.int_attr
                    };
                    moe_k0 = if !ins.ints_attr.is_empty() {
                        ins.ints_attr[0]
                    } else {
                        moe_e
                    };
                    if moe_k0 <= 0 || moe_k0 > moe_e {
                        moe_k0 = moe_e;
                    }
                    break;
                }
                if moe_e > 0 {
                    break;
                }
            }
            if moe_e > 0 {
                let mut off: u64 = 0;
                for w in &worder {
                    let szn = ids.get(w).map_or(1, |i| if i.static_numel > 0 { i.static_numel as u64 } else { 1 });
                    if w.len() > 4 && w.ends_with("_g_w") {
                        moe_off_g = off;
                    } else if w.len() > 5 && w.ends_with("_e1_w") {
                        moe_off_e1 = off;
                    } else if w.len() > 5 && w.ends_with("_e2_w") {
                        moe_off_e2 = off;
                    }
                    off += szn;
                }
            }
        }

        {
            let mut sc: BTreeMap<String, bool> = BTreeMap::new();
            let mut opt_wts: BTreeSet<String> = BTreeSet::new();
            if let Some(tfn_) = tfn_ {
                for ins in &tfn_.instructions {
                    if ins.result_id.is_empty() {
                        continue;
                    }
                    let e = sc.entry(ins.result_id.clone()).or_insert(false);
                    *e = ins.op == MLIROp::TensorAlloc;
                }
                for ins in &tfn_.instructions {
                    for op in &ins.operands {
                        sc.entry(op.clone()).or_insert(false);
                    }
                }
                let mut xid_ = String::new();
                let mut yid_ = String::new();
                for ins in &tfn_.instructions {
                    if ins.op == MLIROp::LayerEmbedding
                        && ins.operands.len() >= 2
                        && !sc.get(&ins.operands[1]).copied().unwrap_or(false)
                        && xid_.is_empty()
                    {
                        xid_ = ins.operands[1].clone();
                    }
                    if ins.op == MLIROp::Matmul
                        && ins.operands.len() >= 2
                        && !sc.get(&ins.operands[0]).copied().unwrap_or(false)
                        && xid_.is_empty()
                    {
                        xid_ = ins.operands[0].clone();
                    }
                    if ins.op == MLIROp::CrossEntropy && ins.operands.len() >= 2 {
                        yid_ = ins.operands[1].clone();
                    }
                }
                oss.push_str("typedef struct ns_cpu_ctx {\n");
                for (id, _ix) in &sc {
                    if id == &xid_ || id == &yid_ {
                        continue;
                    }
                    oss.push_str(&format!("  std::vector<float> buf_{};\n", id));
                }
                for ins in &tfn_.instructions {
                    if ins.op == MLIROp::Dropout {
                        oss.push_str(&format!("  std::vector<float> buf_dm_{};\n", ins.result_id));
                    }
                }
                for ins in &tfn_.instructions {
                    if ins.op != MLIROp::OptStep {
                        continue;
                    }
                    let mut i = 0;
                    while i + 1 < ins.operands.len() {
                        let wt = ins.operands[i].clone();
                        if !opt_wts.contains(&wt) {
                            oss.push_str(&format!(
                                "  std::vector<float> st_m_{}, st_am_{}, st_av_{};\n",
                                wt, wt, wt
                            ));
                            oss.push_str(&format!("  size_t st_t_{} = 0;\n", wt));
                            opt_wts.insert(wt);
                        }
                        i += 2;
                    }
                }
                if moe_e > 0 {
                    oss.push_str(
                        "  std::vector<uint8_t> moe_active; // MoE expert liveness (1 = alive)\n",
                    );
                    oss.push_str("  size_t moe_active_count = 0;     // number of alive experts\n");
                }
                oss.push_str("  std::mt19937 rng;\n");
                oss.push_str("  int64_t lr_step = 0;   // LR schedule step counter (runtime wrapper)\n");
                let ctor_init = if moe_e > 0 {
                    format!(
                        " moe_active.assign({}, 0); for (int e = 0; e < {}; e++) moe_active[(size_t)e] = 1; moe_active_count = (size_t){};",
                        moe_e, moe_k0, moe_k0
                    )
                } else {
                    String::new()
                };
                oss.push_str(&format!("  ns_cpu_ctx() : rng(0x9E3779B9u) {{{} }}\n", ctor_init));
                oss.push_str("} ns_cpu_ctx;\n");
            } else if moe_e > 0 {
                oss.push_str(&format!(
                    "typedef struct ns_cpu_ctx {{ std::mt19937 rng; std::vector<uint8_t> moe_active; size_t moe_active_count = 0; ns_cpu_ctx() : rng(0x9E3779B9u) {{ moe_active.assign({}, 0); for (int e = 0; e < {}; e++) moe_active[(size_t)e] = 1; moe_active_count = (size_t){}; }} }} ns_cpu_ctx;\n",
                    moe_e, moe_k0, moe_k0
                ));
            } else {
                oss.push_str(
                    "typedef struct ns_cpu_ctx { std::mt19937 rng; ns_cpu_ctx() : rng(0x9E3779B9u) {} } ns_cpu_ctx;\n",
                );
            }
        }
        oss.push_str(
            "typedef struct ns_model { float* w; size_t n; ns_cpu_ctx* ctx; } ns_model;\n",
        );
        oss.push_str(
            "typedef struct ns_weight_desc { const char* name; size_t offset; size_t count; } ns_weight_desc;\n",
        );
        oss.push_str("typedef struct ns_weight_layout { size_t num_weights; const ns_weight_desc* desc; } ns_weight_layout;\n\n");

        oss.push_str("namespace { \n");
        if worder.is_empty() {
            oss.push_str("static const ns_weight_desc ns_desc[1] = {};\n");
        } else {
            oss.push_str(&format!(
                "static const ns_weight_desc ns_desc[{}] = {{\n",
                worder.len()
            ));
            let mut off: u64 = 0;
            for (i, w) in worder.iter().enumerate() {
                let szn = ids.get(w).map_or(1, |inf| if inf.static_numel > 0 { inf.static_numel } else { 1 });
                oss.push_str(&format!("    {{\"{}\", {}, {}}}", w, off, szn));
                if i + 1 < worder.len() {
                    oss.push_str(",\n");
                } else {
                    oss.push('\n');
                }
                off += szn as u64;
            }
            oss.push_str("};\n");
        }
        oss.push_str(&format!(
            "static const ns_weight_layout ns_layout = {{ {}, ns_desc }};\n",
            worder.len()
        ));
        oss.push_str(&format!("static const size_t ns_weight_total = {};\n", total));
        oss.push_str(&format!(
            "static const int64_t ns_in_cols = {}, ns_out_cols = {};\n",
            in_cols, out_cols
        ));
        if moe_e > 0 {
            oss.push_str(&format!(
                "static const size_t ns_moe_cap = {}, ns_moe_dim = {}, ns_moe_ffn = {}, ns_moe_k0 = {}, ns_moe_off_g = {}, ns_moe_off_e1 = {}, ns_moe_off_e2 = {};\n",
                moe_e, moe_d, moe_h, moe_k0, moe_off_g, moe_off_e1, moe_off_e2
            ));
        }
        oss.push_str("}\n\n");

        oss.push_str("extern \"C\" ns_model* ns_runtime_init(const float* weights, size_t num_floats) {\n");
        oss.push_str("    if (num_floats != ns_weight_total) return nullptr;\n");
        oss.push_str("    ns_cpu_ctx* ctx = new ns_cpu_ctx();\n");
        oss.push_str("    ns_model* m = new ns_model{ new float[ns_weight_total], ns_weight_total, ctx };\n");
        oss.push_str("    if (!m->w) { delete ctx; delete m; return nullptr; }\n");
        oss.push_str("    std::memcpy(m->w, weights, ns_weight_total * sizeof(float));\n");
        oss.push_str("    return m;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" int ns_eval_infer(ns_model* m, const float* input, float* output, size_t input_numel) {\n");
        oss.push_str("    if (!m || !m->w) return -1;\n");
        oss.push_str(&format!(
            "    {}(input, m->w, output, input_numel{});\n",
            opts.function_name,
            if moe_e > 0 { ", m->ctx->moe_active.data()" } else { "" }
        ));
        oss.push_str("    return 0;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" size_t ns_model_output_numel(const ns_model* m, size_t input_numel) {\n");
        oss.push_str("    (void)m; return (size_t)((int64_t)input_numel / ns_in_cols) * (size_t)ns_out_cols;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" size_t ns_model_weight_count(const ns_model* m) {\n");
        oss.push_str("    (void)m; return ns_weight_total;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" size_t ns_weight_count_static(void) {\n");
        oss.push_str("    return ns_weight_total;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" int ns_model_get_weights(const ns_model* m, float* out, size_t n) {\n");
        oss.push_str("    if (!m || !m->w || !out || n != ns_weight_total) return -1;\n");
        oss.push_str("    std::memcpy(out, m->w, n * sizeof(float));\n");
        oss.push_str("    return 0;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" const ns_weight_layout* ns_model_layout(const ns_model* m) {\n");
        oss.push_str("    (void)m; return &ns_layout;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" void ns_free(ns_model* m) {\n");
        oss.push_str("    if (!m) return; delete m->ctx; delete[] m->w; delete m;\n");
        oss.push_str("}\n\n");
        oss.push_str("// Persist / restore the weight blob + the MoE liveness mask. Format:\n");
        oss.push_str("// 4-byte magic \"NSM2\", size_t float count, raw weights, then (MoE\n");
        oss.push_str("// models) uint32 n_layers, uint32 capacity, <capacity> mask bytes.\n");
        oss.push_str("// Legacy \"NSM1\" files (weights only) load with all experts alive.\n");
        oss.push_str("extern \"C\" int ns_save_checkpoint(const ns_model* m, const char* path) {\n");
        if moe_e > 0 {
            oss.push_str("    if (!m || !m->ctx || !m->w || !path) return -1;\n");
        } else {
            oss.push_str("    if (!m || !m->w || !path) return -1;\n");
        }
        oss.push_str("    FILE* fp = fopen(path, \"wb\");\n");
        oss.push_str("    if (!fp) return -1;\n");
        oss.push_str("    const unsigned magic = 0x4E534D32u; /* \"NSM2\" */\n");
        oss.push_str("    if (fwrite(&magic, sizeof(magic), 1, fp) != 1 ||\n");
        oss.push_str("        fwrite(&ns_weight_total, sizeof(ns_weight_total), 1, fp) != 1 ||\n");
        oss.push_str("        fwrite(m->w, sizeof(float), ns_weight_total, fp) != ns_weight_total) {\n");
        oss.push_str("        fclose(fp); return -1;\n");
        oss.push_str("    }\n");
        if moe_e > 0 {
            oss.push_str("    { const unsigned n_layers = 1U;\n");
            oss.push_str("      if (fwrite(&n_layers, sizeof(n_layers), 1, fp) != 1 ||\n");
            oss.push_str("          fwrite(&ns_moe_cap, sizeof(ns_moe_cap), 1, fp) != 1 ||\n");
            oss.push_str("          fwrite(m->ctx->moe_active.data(), 1, ns_moe_cap, fp) != ns_moe_cap) {\n");
            oss.push_str("          fclose(fp); return -1;\n");
            oss.push_str("      } }\n");
        }
        oss.push_str("    fclose(fp); return 0;\n");
        oss.push_str("}\n\n");
        oss.push_str("extern \"C\" int ns_load_checkpoint(ns_model* m, const char* path) {\n");
        if moe_e > 0 {
            oss.push_str("    if (!m || !m->ctx || !m->w || !path) return -1;\n");
        } else {
            oss.push_str("    if (!m || !m->w || !path) return -1;\n");
        }
        oss.push_str("    FILE* fp = fopen(path, \"rb\");\n");
        oss.push_str("    if (!fp) return -1;\n");
        oss.push_str("    unsigned magic = 0; size_t n = 0;\n");
        oss.push_str("    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||\n");
        oss.push_str("        fread(&n, sizeof(n), 1, fp) != 1 ||\n");
        oss.push_str("        (magic != 0x4E534D31u && magic != 0x4E534D32u) || n != ns_weight_total ||\n");
        oss.push_str("        fread(m->w, sizeof(float), n, fp) != n) {\n");
        oss.push_str("        fclose(fp); return -1;\n");
        oss.push_str("    }\n");
        if moe_e > 0 {
            oss.push_str("    if (magic == 0x4E534D32u) {\n");
            oss.push_str("      unsigned n_layers = 0; size_t cap = 0;\n");
            oss.push_str("      if (fread(&n_layers, sizeof(n_layers), 1, fp) != 1 ||\n");
            oss.push_str("          fread(&cap, sizeof(cap), 1, fp) != 1 ||\n");
            oss.push_str("          n_layers != 1U || cap != ns_moe_cap ||\n");
            oss.push_str("          fread(m->ctx->moe_active.data(), 1, cap, fp) != cap) {\n");
            oss.push_str("          fclose(fp); return -1;\n");
            oss.push_str("      }\n");
            oss.push_str("    } else { /* NSM1 fallback: every capacity slot is alive */\n");
            oss.push_str("      m->ctx->moe_active.assign(ns_moe_cap, 1);\n");
            oss.push_str("    }\n");
            oss.push_str("    m->ctx->moe_active_count = 0;\n");
            oss.push_str("    for (size_t e = 0; e < ns_moe_cap; e++)\n");
            oss.push_str("      m->ctx->moe_active_count += m->ctx->moe_active[e];\n");
        }
        oss.push_str("    fclose(fp); return 0;\n");
        oss.push_str("}\n\n");
        if moe_e > 0 {
            oss.push_str("// ---- MoE expert lifecycle (capacity is compile-time; liveness\n");
            oss.push_str("//      is runtime state in the context) ----\n");
            oss.push_str("extern \"C\" size_t ns_expert_count(const ns_model* m) {\n");
            oss.push_str("    if (!m || !m->ctx) return 0;\n");
            oss.push_str("    return m->ctx->moe_active_count;\n");
            oss.push_str("}\n\n");
            oss.push_str("extern \"C\" size_t ns_expert_birth(ns_model* m, int n) {\n");
            oss.push_str("    if (!m || !m->ctx || !m->w || n <= 0) return m && m->ctx ? m->ctx->moe_active_count : 0;\n");
            oss.push_str("    if (m->ctx->moe_active_count >= ns_moe_cap) return m->ctx->moe_active_count;\n");
            oss.push_str("    // Source: a random live expert (weights copied + perturbed).\n");
            oss.push_str("    int src = (int)(m->ctx->rng() % ns_moe_cap);\n");
            oss.push_str("    size_t tries = 0;\n");
            oss.push_str("    while (tries++ < ns_moe_cap && !m->ctx->moe_active[(size_t)src])\n");
            oss.push_str("      src = (int)(m->ctx->rng() % ns_moe_cap);\n");
            oss.push_str("    const size_t s1 = (size_t)src * ns_moe_dim * ns_moe_ffn;\n");
            oss.push_str("    const size_t s2 = (size_t)src * ns_moe_ffn * ns_moe_dim;\n");
            oss.push_str("    for (int e = 0; e < (int)ns_moe_cap && n > 0; e++) {\n");
            oss.push_str("      if (m->ctx->moe_active[(size_t)e]) continue;\n");
            oss.push_str("      const size_t g1 = (size_t)e * ns_moe_dim, g2 = (size_t)e * ns_moe_dim * ns_moe_ffn;\n");
            oss.push_str("      const size_t g3 = (size_t)e * ns_moe_ffn * ns_moe_dim;\n");
            oss.push_str("      for (size_t k = 0; k < ns_moe_dim; k++)\n");
            oss.push_str("        m->w[ns_moe_off_g + k * ns_moe_cap + (size_t)e] =\n");
            oss.push_str("          m->w[ns_moe_off_g + k * ns_moe_cap + (size_t)src] +\n");
            oss.push_str("          ((float)(m->ctx->rng() % 1001) / 1000.f - 0.5f) * 1e-2f;\n");
            oss.push_str("      for (size_t q = 0; q < ns_moe_dim * ns_moe_ffn; q++)\n");
            oss.push_str("        m->w[ns_moe_off_e1 + g2 + q] = m->w[ns_moe_off_e1 + s1 + q] +\n");
            oss.push_str("          ((float)(m->ctx->rng() % 1001) / 1000.f - 0.5f) * 1e-2f;\n");
            oss.push_str("      for (size_t q = 0; q < ns_moe_ffn * ns_moe_dim; q++)\n");
            oss.push_str("        m->w[ns_moe_off_e2 + g3 + q] = m->w[ns_moe_off_e2 + s2 + q] +\n");
            oss.push_str("          ((float)(m->ctx->rng() % 1001) / 1000.f - 0.5f) * 1e-2f;\n");
            oss.push_str("      m->ctx->moe_active[(size_t)e] = 1;\n");
            oss.push_str("      m->ctx->moe_active_count++;\n");
            oss.push_str("      n--;\n");
            oss.push_str("    }\n");
            oss.push_str("    return m->ctx->moe_active_count;\n");
            oss.push_str("}\n\n");
            oss.push_str("extern \"C\" size_t ns_expert_merge(ns_model* m, int a, int b) {\n");
            oss.push_str("    if (!m || !m->ctx || !m->w || a < 0 || b < 0 || a == b ||\n");
            oss.push_str("        a >= (int)ns_moe_cap || b >= (int)ns_moe_cap ||\n");
            oss.push_str("        !m->ctx->moe_active[(size_t)a] || !m->ctx->moe_active[(size_t)b])\n");
            oss.push_str("        return m && m->ctx ? m->ctx->moe_active_count : 0;\n");
            oss.push_str("    const size_t s1 = (size_t)a * ns_moe_dim * ns_moe_ffn;\n");
            oss.push_str("    const size_t s1b = (size_t)b * ns_moe_dim * ns_moe_ffn;\n");
            oss.push_str("    const size_t t2 = (size_t)a * ns_moe_ffn * ns_moe_dim;\n");
            oss.push_str("    const size_t t2b = (size_t)b * ns_moe_ffn * ns_moe_dim;\n");
            oss.push_str("    for (size_t k = 0; k < ns_moe_dim; k++)\n");
            oss.push_str("      m->w[ns_moe_off_g + k * ns_moe_cap + (size_t)a] =\n");
            oss.push_str("        0.5f * (m->w[ns_moe_off_g + k * ns_moe_cap + (size_t)a] +\n");
            oss.push_str("                m->w[ns_moe_off_g + k * ns_moe_cap + (size_t)b]);\n");
            oss.push_str("    for (size_t q = 0; q < ns_moe_dim * ns_moe_ffn; q++)\n");
            oss.push_str("      m->w[ns_moe_off_e1 + s1 + q] =\n");
            oss.push_str("        0.5f * (m->w[ns_moe_off_e1 + s1 + q] + m->w[ns_moe_off_e1 + s1b + q]);\n");
            oss.push_str("    for (size_t q = 0; q < ns_moe_ffn * ns_moe_dim; q++)\n");
            oss.push_str("      m->w[ns_moe_off_e2 + t2 + q] =\n");
            oss.push_str("        0.5f * (m->w[ns_moe_off_e2 + t2 + q] + m->w[ns_moe_off_e2 + t2b + q]);\n");
            oss.push_str("    m->ctx->moe_active[(size_t)b] = 0;\n");
            oss.push_str("    m->ctx->moe_active_count--;\n");
            oss.push_str("    return m->ctx->moe_active_count;\n");
            oss.push_str("}\n\n");
            oss.push_str("extern \"C\" size_t ns_expert_kill(ns_model* m, int k) {\n");
            oss.push_str("    if (!m || !m->ctx || k < 0 || k >= (int)ns_moe_cap ||\n");
            oss.push_str("        !m->ctx->moe_active[(size_t)k])\n");
            oss.push_str("        return m && m->ctx ? m->ctx->moe_active_count : 0;\n");
            oss.push_str("    m->ctx->moe_active[(size_t)k] = 0;\n");
            oss.push_str("    m->ctx->moe_active_count--;\n");
            oss.push_str("    return m->ctx->moe_active_count;\n");
            oss.push_str("}\n\n");
        }
        for f in &module.functions {
            if !f.is_train {
                continue;
            }
            let t = f;
            oss.push_str("\n// ---- Training core (forward + backward + optimizer) ----\n");
            oss.push_str(&emit_train_core(t, in_cols)?);
            oss.push_str("\n");
            oss.push_str(&lr_schedule_source());
            oss.push_str("\nextern \"C\" int ns_runtime_train_step(ns_model* m, const float* input,\n");
            oss.push_str("                                        const float* labels, size_t input_numel,\n");
            oss.push_str("                                        float* loss_out, float lr) {\n");
            oss.push_str("    if (!m || !m->w) return -1;\n");
            oss.push_str("    int64_t st = m->ctx->lr_step;\n");
            oss.push_str("    if (st < 9223372036854775807LL) m->ctx->lr_step = st + 1;\n");
            oss.push_str("    const float lr_eff = lr * ns_lr_schedule(st);\n");
            oss.push_str("    ns_train_core(m->ctx, input, input_numel, labels, m->w, m->w, loss_out, lr_eff, 1);\n");
            oss.push_str("    return 0;\n");
            oss.push_str("}\n\n");
            oss.push_str("extern \"C\" int ns_objective_loss(ns_model* m, const float* input,\n");
            oss.push_str("                                   const float* labels, size_t input_numel,\n");
            oss.push_str("                                   float* loss_out) {\n");
            oss.push_str("    if (!m || !m->w) return -1;\n");
            oss.push_str("    ns_train_core(m->ctx, input, input_numel, labels, m->w, (float*)0, loss_out, 0.f, 0);\n");
            oss.push_str("    return 0;\n");
            oss.push_str("}\n");
            break;
        }
    }

    Ok(oss)
}