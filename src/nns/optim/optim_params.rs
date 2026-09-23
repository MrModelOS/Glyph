#![allow(dead_code)]
// NNS port: public API preserved for parity with C++ nsc; not all items are used in current pipeline — intentional, not tech debt
//! Canonical optimizer hyperparameters — port of `ns/optim/optim_params.hpp`.
//!
//! SINGLE source of truth for the generated AOT training cores (CPU and CUDA
//! emitters) and the double-precision reference optimizer (MuonOptimizer).

use std::f32::consts::PI;

/// Muon exp-decay on the momentum buffer.
pub const K_MUON_MOMENTUM: f32 = 0.95;
/// Muon coupled weight-decay coefficient.
pub const K_MUON_DECAY: f32 = 0.01;
/// AdamW first-moment decay.
pub const K_ADAMW_BETA1: f32 = 0.9;
/// AdamW second-moment decay.
pub const K_ADAMW_BETA2: f32 = 0.999;
/// AdamW epsilon (denominator floor).
pub const K_ADAMW_EPS: f32 = 1e-8;
/// AdamW decoupled weight-decay coefficient.
pub const K_ADAMW_DECAY: f32 = 0.01;

/// Derived counters: (1 - beta) as EXACT float literals.
pub const K_ONE_MINUS_BETA1: f32 = 0.1;
pub const K_ONE_MINUS_BETA2: f32 = 0.001;

/// Learning-rate schedules (compiled into the AOT runtime driver).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LRSchedule {
    Constant = 0,         // multiplier always 1.0
    CosineWithWarmup = 1, // linear warmup then cosine decay to kLrMinFactor
}

/// Global schedule chosen at compile time.
pub const K_LR_SCHEDULE: LRSchedule = LRSchedule::Constant;

/// Linear ramp 0 -> 1 over this many steps.
pub const K_LR_WARMUP_STEPS: i64 = 100;
/// Cosine decay length (per-process).
pub const K_LR_TOTAL_STEPS: i64 = 10000;
/// Floor for the cosine tail.
pub const K_LR_MIN_FACTOR: f32 = 0.05;

/// Multiplier applied by the runtime wrapper.
pub fn lr_scale(step: i64) -> f32 {
    if K_LR_SCHEDULE == LRSchedule::Constant {
        return 1.0;
    }
    if step < K_LR_WARMUP_STEPS {
        return if K_LR_WARMUP_STEPS > 0 {
            step as f32 / K_LR_WARMUP_STEPS as f32
        } else {
            1.0
        };
    }
    let end = if K_LR_TOTAL_STEPS > K_LR_WARMUP_STEPS {
        K_LR_TOTAL_STEPS
    } else {
        K_LR_WARMUP_STEPS + 1
    };
    let s = if step >= end { end } else { step };
    let t = (s - K_LR_WARMUP_STEPS) as f32 / (end - K_LR_WARMUP_STEPS) as f32;
    K_LR_MIN_FACTOR + 0.5 * (1.0 - K_LR_MIN_FACTOR) * (1.0 + (PI * t).cos())
}
