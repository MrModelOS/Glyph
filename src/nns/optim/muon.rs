//! Muon + AdamW optimizer reference — port of `ns/optim/muon.cpp`.
//!
//! Muon: for 2D weight matrices, run SGD-style momentum on the gradient, then
//! orthonormalize the momentum buffer via (modified) Gram-Schmidt before
//! applying. Non-matrix parameters fall back to AdamW.

use super::optim_params::*;

/// Low-precision storage formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageFormat {
    FP32,
    Fp8E4m3, // e4m3: 1 sign, 4 exponent, 3 mantissa
    Fp8E5m2, // e5m2: 1 sign, 5 exponent, 2 mantissa
    Fp4,     // 1 sign, 2 exponent, 1 mantissa (E2M1-ish value set)
}

/// C `frexp` equivalent: x = m * 2^e with m in [0.5, 1).
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let e = x.abs().log2().floor() as i32 + 1;
    let m = x / 2f64.powi(e);
    (m, e)
}

/// C `ldexp` equivalent: x * 2^e.
fn ldexp(x: f64, e: i32) -> f64 {
    x * 2f64.powi(e)
}

pub struct LowPrecision;

impl LowPrecision {
    pub fn pack(v: f64, fmt: StorageFormat) -> u8 {
        let neg = v < 0.0;
        let a = v.abs();
        match fmt {
            StorageFormat::Fp4 => {
                // E2M1 value set (magnitude): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
                let tab: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                let mut best = 0usize;
                let mut best_d = a;
                for (i, t) in tab.iter().enumerate() {
                    let d = (a - t).abs();
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                (if neg { 0x8u8 } else { 0x0 }) | best as u8
            }
            StorageFormat::Fp8E4m3 => {
                if a == 0.0 {
                    return if neg { 0x80 } else { 0x00 };
                }
                let (m, mut e) = frexp(a); // a = m*2^e, m in [0.5,1)
                e -= 1; // mantissa*2^e with m2 in [1,2)
                let mut e4 = e + 7; // e4m3 exponent bias = 7
                e4 = e4.clamp(0, 15);
                let mut m2 = m * 2.0;
                if e4 >= 15 {
                    m2 = 2.0; // saturate near max
                }
                let mut mant = ((m2 - 1.0) * 8.0).round() as i32;
                mant = mant.clamp(0, 7);
                (if neg { 0x80u8 } else { 0x0 }) | ((e4 as u8) << 3) | mant as u8
            }
            StorageFormat::Fp8E5m2 => {
                if a == 0.0 {
                    return if neg { 0x80 } else { 0x00 };
                }
                let (m, mut e) = frexp(a);
                e -= 1;
                let mut e5 = e + 15; // e5m2 bias = 15
                e5 = e5.clamp(0, 30);
                let mut m2 = m * 2.0;
                if e5 >= 30 {
                    m2 = 2.0;
                }
                let mut mant = ((m2 - 1.0) * 4.0).round() as i32;
                mant = mant.clamp(0, 3);
                (if neg { 0x80u8 } else { 0x0 }) | ((e5 as u8) << 2) | mant as u8
            }
            StorageFormat::FP32 => 0,
        }
    }

    pub fn unpack(bits: u8, fmt: StorageFormat) -> f64 {
        match fmt {
            StorageFormat::Fp4 => {
                let neg = (bits & 0x8) != 0;
                let tab: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                let v = tab[(bits & 0x7) as usize];
                if neg {
                    -v
                } else {
                    v
                }
            }
            StorageFormat::Fp8E4m3 => {
                let neg = (bits & 0x80) != 0;
                let e4 = (bits >> 3) & 0xF;
                let mant = bits & 0x7;
                let v = if e4 == 0 {
                    mant as f64 / 8.0 * ldexp(1.0, -6)
                } else {
                    let m = 1.0 + mant as f64 / 8.0;
                    m * ldexp(1.0, e4 as i32 - 7)
                };
                if neg {
                    -v
                } else {
                    v
                }
            }
            StorageFormat::Fp8E5m2 => {
                let neg = (bits & 0x80) != 0;
                let e5 = (bits >> 2) & 0x1F;
                let mant = bits & 0x3;
                let v = if e5 == 0 {
                    mant as f64 / 4.0 * ldexp(1.0, -14)
                } else {
                    let m = 1.0 + mant as f64 / 4.0;
                    m * ldexp(1.0, e5 as i32 - 15)
                };
                if neg {
                    -v
                } else {
                    v
                }
            }
            StorageFormat::FP32 => 0.0,
        }
    }

    pub fn pack_buf(input: &[f64], out: &mut [u8], fmt: StorageFormat) {
        for (i, v) in input.iter().enumerate() {
            out[i] = Self::pack(*v, fmt);
        }
    }

    pub fn unpack_buf(input: &[u8], out: &mut [f64], fmt: StorageFormat) {
        for (i, b) in input.iter().enumerate() {
            out[i] = Self::unpack(*b, fmt);
        }
    }
}

/// Newton-Schulz variant using modified Gram-Schmidt orthonormalization,
/// producing an orthonormal-stable matrix with the same shape as G.
pub struct NewtonSchulz;

impl NewtonSchulz {
    pub fn orthonormalize(g: &mut [f64], m: usize, n: usize, _steps: usize) {
        if m * n == 0 {
            return;
        }
        // Work in the orientation where we orthonormalize the *columns*: for
        // M >= N operate on G directly; for M < N operate on G^T (rows of G).
        let mut work: Vec<f64>;
        if m >= n {
            work = g.to_vec();
        } else {
            work = vec![0.0; n * m];
            for i in 0..m {
                for j in 0..n {
                    work[j * m + i] = g[i * n + j];
                }
            }
        }
        let r = m.max(n); // rows of the working matrix
        let c = m.min(n); // cols of the working matrix

        // Modified Gram-Schmidt on the columns.
        for j in 0..c {
            for i in 0..j {
                let mut dot = 0.0;
                for r0 in 0..r {
                    dot += work[r0 * c + j] * work[r0 * c + i];
                }
                for r0 in 0..r {
                    work[r0 * c + j] -= dot * work[r0 * c + i];
                }
            }
            let mut nrm = 0.0;
            for r0 in 0..r {
                nrm += work[r0 * c + j] * work[r0 * c + j];
            }
            nrm = nrm.sqrt();
            if nrm > 1e-12 {
                for r0 in 0..r {
                    work[r0 * c + j] /= nrm;
                }
            }
        }

        // Write back into G.
        if m >= n {
            g.copy_from_slice(&work);
        } else {
            for i in 0..m {
                for j in 0..n {
                    g[i * n + j] = work[j * m + i];
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MuonParams {
    pub lr: f64,                 // base learning rate
    pub weight_decay: f64,
    pub muon_momentum: f64,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_eps: f64,
}

impl Default for MuonParams {
    fn default() -> Self {
        MuonParams {
            lr: 0.02,
            weight_decay: K_MUON_DECAY as f64,
            muon_momentum: K_MUON_MOMENTUM as f64,
            adam_beta1: K_ADAMW_BETA1 as f64,
            adam_beta2: K_ADAMW_BETA2 as f64,
            adam_eps: K_ADAMW_EPS as f64,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MuonState {
    pub muon_m: Vec<f64>, // persistent momentum buffer (matrix params)
    pub adam_m: Vec<f64>,
    pub adam_v: Vec<f64>,
    pub is_matrix: bool,
}

struct ParamSlot {
    data: Vec<f64>, // owned copy; written back after stepping
    rows: usize,
    cols: usize,
    n: usize,
}

/// Muon optimizer: applies Muon for 2D matrix params, AdamW otherwise.
///
/// NOTE: the C++ original registers raw pointers into the caller's buffers.
/// In Rust the optimizer owns a copy of each parameter vector; after
/// `step_param` the caller retrieves it with `param_data(i)`.
pub struct MuonOptimizer {
    p: MuonParams,
    params: Vec<ParamSlot>,
    states: Vec<MuonState>,
    step: usize,
}

impl MuonOptimizer {
    pub fn new(p: MuonParams) -> Self {
        MuonOptimizer {
            p,
            params: Vec::new(),
            states: Vec::new(),
            step: 0,
        }
    }

    pub fn add_param(&mut self, data: Vec<f64>, rows: usize, cols: usize) {
        let n = rows * cols;
        let mut st = MuonState::default();
        st.is_matrix = rows > 1 && cols > 1;
        st.muon_m = vec![0.0; n];
        if !st.is_matrix {
            st.adam_m = vec![0.0; n];
            st.adam_v = vec![0.0; n];
        }
        self.params.push(ParamSlot { data, rows, cols, n });
        self.states.push(st);
    }

    pub fn zero_state(&mut self) {
        for s in self.states.iter_mut() {
            s.muon_m.fill(0.0);
            s.adam_m.fill(0.0);
            s.adam_v.fill(0.0);
        }
    }

    /// Apply one step to a single registered parameter index.
    pub fn step_param(&mut self, index: usize, grad: &[f64]) {
        if index >= self.params.len() || grad.is_empty() {
            return;
        }
        let n = self.params[index].n;
        let t = self.step + 1; // 1-based step for bias correction

        let is_matrix = self.states[index].is_matrix;
        if is_matrix {
            // --- Muon path ---
            let momentum = self.p.muon_momentum;
            for i in 0..n {
                self.states[index].muon_m[i] = momentum * self.states[index].muon_m[i] + grad[i];
            }
            let (rows, cols) = (self.params[index].rows, self.params[index].cols);
            let mdata: &mut [f64] = &mut self.states[index].muon_m;
            NewtonSchulz::orthonormalize(mdata, rows, cols, 10);

            let rank = (self.params[index].rows.min(self.params[index].cols) as f64).sqrt();
            let lr_eff = self.p.lr * rank;
            let wd = self.p.weight_decay;
            for i in 0..n {
                let decay = wd * self.params[index].data[i];
                self.params[index].data[i] -= lr_eff * self.states[index].muon_m[i] + decay;
            }
            // reset momentum after apply
            self.states[index].muon_m.fill(0.0);
        } else {
            // --- AdamW path (bias / vector / scalar) ---
            let (b1, b2) = (self.p.adam_beta1, self.p.adam_beta2);
            let t_f = t as f64;
            for i in 0..n {
                let g = grad[i];
                self.states[index].adam_m[i] = b1 * self.states[index].adam_m[i] + (1.0 - b1) * g;
                self.states[index].adam_v[i] =
                    b2 * self.states[index].adam_v[i] + (1.0 - b2) * g * g;
                let mhat = self.states[index].adam_m[i] / (1.0 - b1.powf(t_f));
                let vhat = self.states[index].adam_v[i] / (1.0 - b2.powf(t_f));
                let step_sz = self.p.lr / (vhat.sqrt() + self.p.adam_eps);
                let decay = self.p.weight_decay * self.params[index].data[i];
                self.params[index].data[i] -= mhat * step_sz + decay;
            }
        }
    }

    /// Owned-data deviation: retrieve the current parameter values, so the
    /// caller can sync its own buffer after a step.
    pub fn param_data(&self, index: usize) -> &[f64] {
        &self.params[index].data
    }

    pub fn set_step(&mut self, s: usize) {
        self.step = s;
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.p.lr = lr;
    }

    pub fn params(&self) -> &MuonParams {
        &self.p
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    pub fn state(&self, i: usize) -> &MuonState {
        &self.states[i]
    }
}

impl Default for MuonOptimizer {
    fn default() -> Self {
        Self::new(MuonParams::default())
    }
}