# NeuralScript (.ns) — Tensor DSL inside glyphc

NeuralScript is the `.ns` DSL for neural networks, integrated as `glyphc nns`.
It type-checks tensor shapes at compile time and emits a self-contained C++,
CPU SIMD-reference, or CUDA translation unit plus an optional C-ABI runtime
header.

Source files: `examples/nns/*.ns` (mlp, transformer, dense, static, moe, neumoe).

---

## 1. Quick start

```bash
glyphc nns examples/nns/mlp.ns --check              # shape/type check only
glyphc nns examples/nns/mlp.ns --cpp                # C++ reference to stdout
glyphc nns examples/nns/mlp.ns --cpp --runtime      # C++ + ns_runtime.h header
glyphc nns examples/nns/mlp.ns --cuda --runtime     # CUDA backend + header
glyphc nns examples/nns/mlp.ns --mlir               # MLIR IR dump (debug)
glyphc nns examples/nns/mlp.ns --cuda --fp16 --runtime  # experimental fp16 header/ABI
glyphc nns examples/nns/mlp.ns --cpp --runtime -o model.cpp  # write output directly

# Compile the emitted driver with the generic host:
glyphc nns examples/nns/mlp.ns --cpp --runtime > /tmp/model.cpp
g++ /tmp/model.cpp examples/nns/host.cpp -I . -o /tmp/driver && /tmp/driver
```

All diagnostics use `Static shape/type errors:` with `line:col` locations;
codegen shape errors (e.g. dynamic dims not lowered) surface the same way.

---

## 2. Grammar

### 2.1 Type aliases and dimensions

```ns
type Batch    = Dynamic     // symbolic batch dim, resolved at call-site
type Features = 784         // constant dim
type Classes  = 10
type Seq      = Dynamic
type D        = 512
type H        = 8

// Shorthand forms:
type Hidden = 2048
```

`DimExpr` is one of:
- `Dynamic` — unknown until runtime (batch / sequence length)
- `Constant` — integer literal (`784`, `10`, `512`)
- `Symbolic` — alias name that resolves to Dynamic or Constant via `type` table

Tensor type syntax (shape checker normalises alias names):

```ns
Tensor[Batch, Features] float32
Tensor[S, C] float32
Tensor[Seq, D] float32
```

Supported dtypes (`src/nns/ast.rs:Dtype`):

| Token | Dtype |
|-------|-------|
| `float16` | Parsed as Float16; codegen currently rejects native half kernels |
| `float32` | Float32 |
| `float64` | Float64 |
| `int8` / `int16` / `int32` / `int64` | Int* |
| `fp8` / `fp4` | Parsed as Fp8 / Fp4; codegen currently rejects these types |
| `bool` | Bool |

Scalars use the same dtype tokens without `Tensor[...]`.

### 2.2 Network, layers, forward

```ns
network MLPClassifier {
    input:  Tensor[Batch, Features] float32
    output: Tensor[Batch, Classes] float32

    layer fc1  = Dense(in: Features, out: 512, activation: ReLU)
    layer drop = Dropout(rate: 0.1)
    layer fc2  = Dense(in: 512, out: Classes, activation: Identity)

    forward(x) {
        return x -> fc1 -> drop -> fc2
    }
}
```

Grammar elements:

- `network <Name> { ... }` — one network per file is typical; multiple allowed.
- `input:` / `output:` — declared I/O tensor types; used to type `forward` params
  and verify the return shape.
- `layer <name> = <LayerType>(params...)` — layer declarations.

Layer types and params (`src/nns/shape_checker.rs:LayerRule`):

| Layer | Params | Shape rule |
|-------|--------|------------|
| `Dense` / `Linear` | `in:`, `out:`/`out_features:`, `activation:` | `[..., in] -> [..., out]` |
| `Dropout` | `rate:` | shape-preserving |
| `LayerNorm` | (none / `d_model:`) | shape-preserving |
| `Attention` / `MultiHeadAttention` | `d_model:`/`dim:`, `num_heads:`/`heads:`, `causal:` | shape-preserving (`[B, S, D] -> [B, S, D]`) |
| `Embedding` | `vocab_size:`, `d_model:` | `[..., S] -> [..., S, D]` (appends dim) |
| `MoE` / `MixtureOfExperts` | `experts:`/`num_experts:`, `ffn_dim:`, `initial_experts:` | shape-preserving |

`activation:` values are free-form strings; codegen handles `ReLU`, `GELU`,
`Identity`, `Softmax`, etc. Param values may be literals or symbolic alias names.

Forward:

```ns
forward(x) {
    var h = x -> emb
    var a = h -> attn
    var n = a -> ln1
    var g = n -> mlp1 -> mlp2   // pipeline chain desugars left-to-right
    var s = g + n               // elementwise add (same shape)
    return s -> ln2 -> fc       // return expression must match declared output rank+dims
}
```

- `->` is the pipeline operator (`PipelineOp`): `x -> fc1 -> drop` applies
  each layer's shape rule sequentially; unknown stages are passthrough.
- `+` is elementwise tensor addition (rank+dim must unify).
- `var` / `let` bindings inside `forward` are supported.
- `forward` params with no explicit type default to the network's `input` type.

Larger example — 7-block transformer with residual adds
(`examples/nns/dense.ns`):

```ns
network DenseWide {
    input:  Tensor[S] float32
    output: Tensor[S, C] float32
    layer emb = Embedding(vocab_size: V, d_model: D)
    layer b0_attn = Attention(d_model: D, num_heads: H, causal: true)
    layer b0_f1 = Dense(in: D, out: FF, activation: GELU)
    layer b0_f2 = Dense(in: FF, out: D, activation: Identity)
    // ... b1..b6 identical blocks ...
    layer head = Dense(in: D, out: C, activation: Identity)

    forward(x) {
        var h = x -> emb
        var t0a = h -> b0_attn
        var t0b = h + t0a
        var t0c = t0b -> b0_f1
        var t0d = t0c -> b0_f2
        var t0  = t0b + t0d
        // ... repeat ...
        return t6 -> head
    }
}
```

### 2.3 `train { grad {} }` — AOT autodiff

```ns
train(batch_x: Tensor[Batch, Features],
      batch_y: Tensor[Batch, Classes]) -> float32 {
    grad {
        var hidden = batch_x @ fc1       // matmul: [B,784] @ [784,512] -> [B,512]
        var act    = relu(hidden)         // elementwise activation
        var preds  = act @ fc2             // [B,512] @ [512,10] -> [B,10]
        var loss   = cross_entropy(preds, batch_y)
    }
    return loss
}
```

`train` declares the optimisation step. Inside `grad { ... }`:

- `@` is matrix multiply (`MatmulOp`); shape checker enforces 2-D, inner dims unify.
- Activations: `relu`, `gelu`, `sigmoid`, `tanh`, `silu`, `swish`, `softmax`,
  `leaky_relu`, `dropout`, `identity` — shape-preserving.
- `cross_entropy(preds, labels)` expects `Tensor[B, C]` both; returns scalar `float32`.
- Data-movement builtins: `slice`, `index`, `scatter`, `concat`, `transpose`,
  `reshape` with special shape inference (see `shape_checker.rs`).
- Layer names (`fc1`, `fc2`) bind to their weight tensors `[in, out]` for autodiff.
  No duplicate `w1/w2` — the network owns one weight per layer.

The compiler lowers `grad {}` to a reverse-mode tape, fuses kernels
(`mlir/fusion`), then appends the optimizer. Scalars promote via the same
backend as inference. Return value of `train` must be the scalar `loss`.

In a full model the same `forward` body is usually replayed inside `grad`
(`examples/nns/transformer.ns`, `dense.ns`):

```ns
train(x: Tensor[S], labels: Tensor[S, C]) -> float32 {
    grad {
        var h = x -> emb
        var a = h -> attn
        var n1 = a -> ln1
        var g  = n1 -> mlp1 -> mlp2
        var r  = g + n1
        var n2 = r -> ln2
        var logits = n2 -> fc
        var loss   = cross_entropy(logits, labels)
    }
    return loss
}
```

---

## 3. Shape and type checking

The checker (`src/nns/shape_checker.rs`) runs in phases:

1. Collect `type` aliases into a dim table.
2. Register `layer` inference rules.
3. Walk `network` → `forward` / `train` blocks, inferring `TensorType` per
   `Expr.inferred_type` and unifying dims.

Unification rules:

- `Dynamic` unifies with anything.
- `Const == Const` must match value-wise, else `Dimension mismatch`.
- `Symbolic == Symbolic` with different names merges symbolic constraints.
- `Symbolic == Const` binds the symbol to that constant (see `unify_dim`,
  `merge_symbols`, `record_symbol_const`).
- Rank must match for elementwise ops and `cross_entropy`; `@` requires
  both operands 2-D with equal inner dims.

The declared `output` rank is verified against `forward` return expression.
Failure emits:

```
Static shape/type errors:
  4:12  Dimension mismatch: constant 784 vs 512
  7:5   Network forward() produces rank-2 but declared output has rank-3
```

Only after a clean check does the compiler proceed to MLIR lowering; `--check`
stops here with `Shape checking passed.`.

---

## 4. Backends and flags

| Flag | Effect |
|------|--------|
| `--check` | Only run shape/type checking, no codegen. |
| `--mlir` | Lower to MLIR and dump IR to stdout (`MLIRCompiler::dump`). No C++ emit. |
| `--cpp` | Emit CPU C++ reference backend (`TargetBackend::CpuCxx`). |
| `--simd` | Select the CPU SIMD target; currently emits the same scalar reference with an explicit marker. |
| `--cuda` | Emit CUDA backend (`TargetBackend::Cuda`). Default when neither `--cpp` nor `--cuda` is set; `--cuda` wins if both are set. |
| `--runtime` | Prepend the C-ABI runtime header (`ns_runtime.h`, from `src/nns/runtime/ns_runtime.rs`) to the emitted source. Required when linking against `host.cpp`. Without it the output is just the graph kernels. |
| `--fp16` | Experimental CUDA fp16 header/ABI mode. The current kernels still use the float reference path; native `half` arithmetic and tuned half2 launches are not enabled yet. |
| `-o <file>` / `--output <file>` | Write generated source to a file instead of stdout. |

The CLI stages are `Lex -> Parse -> ShapeCheck -> MLIR -> Fusion -> Codegen`.
Fusion (`FusionPass::run`) fuses `matmul + activation / layernorm / bias`
groups and reports `Fusion: N groups fused` on stderr for codegen runs.

`Rocm` and `Metal` backend variants are API-level placeholders that delegate to
the CUDA emitter with a backend marker; they are not native HIP or MSL backends
yet. The CLI currently exposes the stable CPU and CUDA targets.

Example — inspect all stages for `mlp.ns`:

```bash
glyphc nns examples/nns/mlp.ns --check          # 1) types only
glyphc nns examples/nns/mlp.ns --mlir            # 2) MLIR dump
glyphc nns examples/nns/mlp.ns --cpp              # 3) kernels only (no header)
glyphc nns examples/nns/mlp.ns --cpp --runtime    # 4) kernels + header (host-linkable)
```

---

## 5. `--runtime` C-ABI — symbol list

When `--runtime` is set the emitted file starts with `ns_runtime.h` and then
defines the following `extern "C"` interface (see `src/nns/runtime/ns_runtime.rs`
for the exact header text, version `1.2.0`):

### Weight layout

```c
typedef struct ns_weight_desc {
    const char* name;   // e.g. "fc1_w", "emb_w", "moe_g_w"
    size_t      offset; // float offset in the blob
    size_t      count;  // number of floats
} ns_weight_desc;

typedef struct ns_weight_layout {
    size_t                 num_weights;
    const ns_weight_desc*  desc;
} ns_weight_layout;
```

Weights are one contiguous host-owned `float` blob `[w0 | w1 | ... | wN-1]`
in model-definition order.

### Lifecycle / inference

```c
ns_model* ns_runtime_init(const float* weights, size_t num_floats);
void      ns_free(ns_model* m);
int       ns_eval_infer(ns_model* m, const float* input, float* output,
                        size_t input_numel);
size_t    ns_model_output_numel(const ns_model* m, size_t input_numel);
size_t    ns_model_weight_count(const ns_model* m);
size_t    ns_weight_count_static(void);  // no model needed, compile-time constant
int       ns_model_get_weights(const ns_model* m, float* out, size_t n);
const ns_weight_layout* ns_model_layout(const ns_model* m);
```

- `ns_weight_count_static()` lets the host size its buffer before init.
- `input_numel` is `batch * in_cols` (row-major, batch-leading).
- `ns_model_output_numel` tells the host how large the output buffer must be.

### Checkpoints

```c
int ns_save_checkpoint(const ns_model* m, const char* path);
int ns_load_checkpoint(ns_model* m, const char* path);
```

Format: `NSM1` (weights only) or `NSM2` (weights + MoE mask `{n_layers, capacity, mask}`).
`NSM1` loads with all experts alive.

### MoE expert lifecycle (only if graph has a MoE layer)

```c
size_t ns_expert_count(const ns_model* m);
size_t ns_expert_birth(ns_model* m, int n);       // activate n dead slots (perturbed copy)
size_t ns_expert_merge(ns_model* m, int a, int b); // average b into a, deactivate b
size_t ns_expert_kill(ns_model* m, int k);        // deactivate k
```

Capacity (`ns_moe_cap`) is compile-time constant (gate row `[D,cap]` + per-expert
FFN weights); only liveness is runtime. Exactly one MoE layer per model.

### AOT training (only if `train` is defined)

```c
int ns_runtime_train_step(ns_model* m, const float* input, const float* labels,
                          size_t input_numel, float* loss_out, float lr);
int ns_objective_loss(ns_model* m, const float* input, const float* labels,
                      size_t input_numel, float* loss_out);
```

Training and inference share the same weight blob; `ns_runtime_train_step`
runs forward + backward + one optimizer step (Muon for matrices, AdamW
otherwise) and writes updated weights back into the model. `ns_objective_loss`
is forward-only.

---

## 6. Minimal `host.cpp` walkthrough

Reference driver: `examples/nns/host.cpp` (also `examples/nns/neumoe/host.cpp`
for the corpus/MoE study). It is generic over any compiled `.ns` — the weight
layout and MoE lifecycle are queried at runtime.

```cpp
#include "ns/runtime/ns_runtime.h"
#include <vector>
#include <cstdio>

int main() {
    // 1. Size the blob before creating a model.
    const size_t nw = ns_weight_count_static();
    std::printf("weight_count=%zu (%.2fM)\n", nw, nw / 1e6f);
    std::vector<float> w(nw);

    // 2. Need a layout to init correctly — build a temp model to fetch it.
    //    (The real host rolls Xavier / N(0,0.02) per desc.name — see host.cpp:init_weights)
    ns_model* tmp = ns_runtime_init(w.data(), nw);
    const ns_weight_layout* lay = ns_model_layout(tmp);
    // init_weights(w, lay) — per-weight-name strategy:
    //   emb_w/head_w -> N(0,0.02), _attn_ -> N(0,0.02)*0.88,
    //   moe_g_w -> N(0,0.02), f1/f2 -> Xavier uniform
    ns_free(tmp);

    // 3. Create the real model with the rolled blob.
    ns_model* m = ns_runtime_init(w.data(), nw);
    std::printf("experts_initial=%zu\n", ns_expert_count(m));

    // 4. Training loop (mlp-style: batch of features + one-hot labels).
    //    For the LM study (dense/neumoe) seq_len = L, vocab = 10240,
    //    xs is float-encoded token ids [L-1], ys is one-hot [L-1, V].
    std::vector<float> xs(L - 1), ys((L - 1) * 10240), out((L - 1) * 10240);
    for (int step = 0; step < steps; ++step) {
        float loss = 0.f;
        float lr = /* cosine with warmup, see host.cpp:lr_at */;
        ns_runtime_train_step(m, xs.data(), ys.data(), L - 1, &loss, lr);

        // Optional: grow MoE gradually
        if (step % 50 == 0 && ns_expert_count(m) < 9)
            ns_expert_birth(m, 1);

        // Validation: forward-only path
        float vloss = 0.f;
        ns_objective_loss(m, xs.data(), ys.data(), L - 1, &vloss);
        ns_eval_infer(m, xs.data(), out.data(), L - 1); // logits
    }

    // 5. Persist. NSM2 captures MoE mask.
    ns_save_checkpoint(m, "model.nsm2");
    ns_free(m);
}
```

Build (CPU):

```bash
glyphc nns examples/nns/mlp.ns --cpp --runtime > /tmp/model.cpp
g++ /tmp/model.cpp examples/nns/host.cpp -I . -o /tmp/driver && /tmp/driver data/ 1000 out/
```

Build (CUDA, as in `examples/nns/neumoe/run.sh`):

```bash
glyphc nns examples/nns/neumoe/neumoe.ns --cuda --runtime > /tmp/model.cu
nvcc /tmp/model.cu examples/nns/neumoe/host.cpp -I . -arch=sm_75 -o /tmp/driver
```

The same `host.cpp` drives NeuMoE (growing `ns_expert_birth`), Static-MoE
(K=9 from step 0), and Dense (no MoE) by probing `ns_expert_count`.

---

## 7. See also

- `src/nns/` — lexer, parser, `shape_checker.rs`, `mlir/`, `codegen/`, `runtime/ns_runtime.rs`
- `examples/nns/mlp.ns` — minimal classifier with `train{grad{}}`
- `examples/nns/transformer.ns` — embedding + attention + layernorm + MLP
- `examples/nns/dense.ns` — 7-block transformer with wide final FFN
- `docs/language_en.md` / `docs/language.md` — Glyph (.glyph) reference
