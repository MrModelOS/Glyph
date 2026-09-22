//! Verbatim CPU-reference preamble snippets emitted by `gen_cpu`, extracted
//! byte-for-byte from the literal chains in
//! `NeuralScript/src/codegen/codegen.cpp`.
//!
//! GENERATED FILE — do not edit by hand; regenerate with
//! `python3 /tmp/opencode/gen_cpu_blobs.py`.

pub const NS_ACT2: &str = r#"static inline float ns_act2(float x, int code) {
  switch (code) {
    case 0: return x;                                   // identity
    case 1: return x > 0 ? x : 0.f;                     // relu
    case 2: return x > 0 ? x : 0.01f * x;               // leaky_relu
    case 3: return 1.f / (1.f + expf(-x));              // sigmoid
    case 4: return tanhf(x);                            // tanh
    case 5: return x / (1.f + expf(-x));                // swish/silu
    case 6: return 0.5f * x * (1.f + erff(x / 1.41421356f)); // gelu
    default: return x;
  }
}

"#;

pub const NS_MATMUL: &str = r#"static void ns_matmul(const float* A, const float* B, float* C,
                     int64_t M, int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; i++)
    for (int64_t j = 0; j < N; j++) {
      float acc = 0.f;
      for (int64_t k = 0; k < K; k++) acc += A[(size_t)i*K + k] * B[(size_t)k*N + j];
      C[(size_t)i*N + j] = acc;
    }
}

"#;

pub const NS_BINOP: &str = r#"static void ns_binop(const float* A, size_t na, const float* B, size_t nb,
                     float* C, int op) {
  for (size_t i = 0; i < na; i++) {
    float b = nb > 0 ? B[i % nb] : 0.f;
    switch (op) {
      case 0: C[i] = A[i] + b; break;
      case 1: C[i] = A[i] - b; break;
      case 2: C[i] = A[i] * b; break;
      case 3: C[i] = A[i] / b; break;
    }
  }
}

"#;

pub const NS_BINOP_GRAD: &str = r#"// Elementwise binop backward: one grad output at a time.  A/C store the
// lhs/rhs forward inputs, dC the upstream grad; which=0 computes dA,
// which=1 computes dB.  Bits 1-2 of `which` select the operator
// (0=+,1=-,2=*,3=/), bit 0 is the lhs/rhs selector.
static void ns_binop_grad(const float* dC, size_t n, const float* A, const float* B,
                         float* C, int which) {
  int op = which >> 1;
  int side = which & 1;
  for (size_t i = 0; i < n; i++) {
    float a = A[i], b = B[i], dc = dC[i];
    switch (op) {
      case 0: C[i] = dc; break;                                  // +
      case 1: C[i] = side ? -dc : dc; break;                    // -
      case 2: C[i] = side ? dc * a : dc * b; break;             // *
      default: C[i] = side ? -dc * a / (b * b) : dc / b; break; // /
    }
  }
}

"#;

pub const NS_LAYERNORM: &str = r#"static void ns_layernorm(float* m, size_t numel, int64_t last) {
  int64_t rows = (int64_t)numel / last;
  for (int64_t r = 0; r < rows; r++) {
    float mean = 0.f, var = 0.f;
    for (int64_t j = 0; j < last; j++) mean += m[(size_t)r*last + j];
    mean /= last;
    for (int64_t j = 0; j < last; j++) { float d = m[(size_t)r*last + j] - mean; var += d*d; }
    var /= last;
    float inv = 1.f / sqrtf(var + 1e-5f);
    for (int64_t j = 0; j < last; j++) m[(size_t)r*last + j] = (m[(size_t)r*last + j] - mean) * inv;
  }
}

"#;

pub const NS_SOFTMAX: &str = r#"static void ns_softmax(float* m, size_t numel, int64_t last) {
  int64_t rows = (int64_t)numel / last;
  for (int64_t r = 0; r < rows; r++) {
    float mx = m[(size_t)r*last];
    for (int64_t j = 1; j < last; j++) mx = std::max(mx, m[(size_t)r*last + j]);
    float s = 0.f;
    for (int64_t j = 0; j < last; j++) { m[(size_t)r*last + j] = expf(m[(size_t)r*last + j] - mx); s += m[(size_t)r*last + j]; }
    for (int64_t j = 0; j < last; j++) m[(size_t)r*last + j] /= s;
  }
}

"#;

pub const NS_ACT_DERIV: &str = r#"static float ns_act_deriv(float x, int code) {
  switch (code) {
    case 1: return x > 0.f ? 1.f : 0.f;                    // relu
    case 2: return x > 0.f ? 1.f : 0.01f;                  // leaky_relu
    case 3: { float s = 1.f / (1.f + expf(-x)); return s * (1.f - s); } // sigmoid
    case 4: { float t = tanhf(x); return 1.f - t * t; }    // tanh
    case 5: { float s = x / (1.f + expf(-x)); return s + x * (1.f - s); } // swish/silu
    case 6: { float u = x / 1.41421356f; float p = 0.5f * (1.f + erff(u)); return p + x * expf(-u*u) / 2.50662827f; } // gelu
    default: return 1.f;
  }
}

"#;

pub const NS_MATMUL_GRAD_A: &str = r#"// dA[i,k] += dC[i,j] * B[k,j]   (forward: C = A @ B)
static void ns_matmul_grad_a(const float* dC, const float* B, float* dA,
                             int64_t M, int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; i++)
    for (int64_t k = 0; k < K; k++) {
      float acc = 0.f;
      for (int64_t j = 0; j < N; j++) acc += dC[(size_t)i*N + j] * B[(size_t)k*N + j];
      dA[(size_t)i*K + k] = acc;
    }
}

"#;

pub const NS_MATMUL_GRAD_W: &str = r#"// dB[k,j] += A[i,k] * dC[i,j]   (forward: C = A @ B)
static void ns_matmul_grad_w(const float* A, const float* dC, float* dB,
                             int64_t M, int64_t K, int64_t N) {
  for (int64_t k = 0; k < K; k++)
    for (int64_t j = 0; j < N; j++) {
      float acc = 0.f;
      for (int64_t i = 0; i < M; i++) acc += A[(size_t)i*K + k] * dC[(size_t)i*N + j];
      dB[(size_t)k*N + j] = acc;
    }
}

"#;

pub const NS_ACT_GRAD: &str = r#"// dOut = dIn * act'(actInput)   (elementwise)
static void ns_act_grad(const float* dIn, const float* actInput, float* dOut,
                       size_t n, int code) {
  for (size_t i = 0; i < n; i++) dOut[i] = dIn[i] * ns_act_deriv(actInput[i], code);
}

"#;

pub const NS_LAYERNORM_GRAD: &str = r#"// LayerNorm backward: dx = (dout - mean(dout) - (x - mean_x) * mean((x-mean_x)*dout)) * inv
static void ns_layernorm_grad(const float* dout, const float* x, float* dx,
                               size_t numel, int64_t last) {
  int64_t rows = (int64_t)numel / last;
  for (int64_t r = 0; r < rows; r++) {
    float mean_x = 0.f, mean_dout = 0.f;
    for (int64_t j = 0; j < last; j++) {
      mean_x   += x[(size_t)r*last + j];
      mean_dout += dout[(size_t)r*last + j];
    }
    mean_x   /= last;
    mean_dout /= last;
    float var_x = 0.f, gamma = 0.f;
    for (int64_t j = 0; j < last; j++) {
      float xj = x[(size_t)r*last + j] - mean_x;
      var_x += xj * xj;
      gamma += xj * dout[(size_t)r*last + j];
    }
    var_x /= last;
    gamma /= last;
    float inv = 1.f / sqrtf(var_x + 1e-5f);
    for (int64_t j = 0; j < last; j++) {
      float xj = x[(size_t)r*last + j] - mean_x;
      dx[(size_t)r*last + j] = (dout[(size_t)r*last + j] - mean_dout - xj * gamma) * inv;
    }
  }
}

"#;

pub const NS_EMBEDDING_GRAD_W: &str = r#"// Embedding weight gradient: scatter-add dOut rows into dW[indices[i]].
static void ns_embedding_grad_w(const float* dout, const float* idx, float* dW,
                                size_t M, int64_t emb_dim, int64_t vocab_size) {
  for (int64_t k = 0; k < vocab_size * emb_dim; k++) dW[k] = 0.f;
  for (size_t i = 0; i < M; i++) {
    int64_t row = (int64_t)idx[i];
    if (row < 0 || row >= vocab_size) continue;
    for (int64_t j = 0; j < emb_dim; j++)
      dW[(size_t)row * emb_dim + j] += dout[(size_t)i * emb_dim + j];
  }
}

"#;

pub const NS_MOE_GRAD_X: &str = r#"// MoE backward (FFN experts): recompute router/liveness and route
// gradients into dx. gate [D,E], expert1 [E,D,H], expert2 [E,H,D].
// `active` may be null (all live); else active[e]!=0 means live.
static void ns_moe_grad_x(const float* dout, const float* x,
                         const float* Wg, const float* We1, const float* We2,
                         const uint8_t* active, float* dx,
                         int64_t M, int64_t D, int64_t H, int64_t E) {
  std::vector<float> lg((size_t)E), p((size_t)E), h((size_t)H), dpre2((size_t)H), dH((size_t)H);
  auto live = [&](int64_t e)->bool { return !active || active[(size_t)e] != 0; };
  for (int64_t i = 0; i < M; i++) {
    float mx = -1.0e30f, sum = 0.f, pb = 0.f;
    int64_t best = -1;
    for (int64_t e = 0; e < E; e++) {
      float a = 0.f;
      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
      lg[(size_t)e] = a;
      if (live(e) && a > mx) mx = a;
    }
    for (int64_t e = 0; e < E; e++) {
      if (!live(e)) { p[(size_t)e] = 0.f; continue; }
      lg[(size_t)e] = expf(lg[(size_t)e] - mx);
      sum += lg[(size_t)e];
      if (best < 0 || lg[(size_t)e] > lg[(size_t)best]) best = e;
    }
    if (sum > 0.f) {
      for (int64_t e = 0; e < E; e++) p[(size_t)e] = live(e) ? lg[(size_t)e]/sum : 0.f;
      pb = p[(size_t)best];
    }
    for (int64_t a = 0; a < H; a++) {
      float acc = 0.f;
      for (int64_t k = 0; k < D; k++) acc += x[i*D+k] * We1[best*D*H + k*H + a];
      h[(size_t)a] = acc;
    }
    for (int64_t a = 0; a < H; a++) {
      float acc = 0.f;
      for (int64_t j = 0; j < D; j++) acc += dout[i*D+j] * We2[best*H*D + a*D + j];
      dpre2[(size_t)a] = acc;
    }
    float dpb = 0.f;
    for (int64_t a = 0; a < H; a++) {
      const float u = h[(size_t)a];
      const float ga = 0.5f*u*(1.f + erff(u*0.7071067811865476f));       /* gelu */
      const float gd = 0.5f*(1.f + erff(u*0.7071067811865476f))
                        + u*expf(-u*u*0.5f)*0.3989422804014327f;         /* gelu' */
      dpb += ga * dpre2[(size_t)a];
      dH[(size_t)a] = dpre2[(size_t)a] * pb * gd;
    }
    const float sd = dpb * pb;
    for (int64_t e = 0; e < E; e++) {
      const float dg = p[(size_t)e] * ((e == best ? dpb : 0.f) - sd);
      for (int64_t k = 0; k < D; k++) dx[i*D+k] += dg * Wg[k*E+e];
    }
    for (int64_t k = 0; k < D; k++) {
      float acc = 0.f;
      for (int64_t a = 0; a < H; a++) acc += dH[(size_t)a] * We1[best*D*H + k*H + a];
      dx[i*D+k] += acc;
    }
  }
}

"#;

pub const NS_MOE_GRAD_WG: &str = r#"// MoE router-weight gradient dWg[D,E]: dWg[k,e] += x[i,k] * dg[i,e].
static void ns_moe_grad_wg(const float* dout, const float* x,
                           const float* Wg, const float* We1, const float* We2,
                           const uint8_t* active, float* dWg,
                           int64_t M, int64_t D, int64_t H, int64_t E) {
  for (int64_t k = 0; k < D*E; k++) dWg[k] = 0.f;
  std::vector<float> lg((size_t)E), p((size_t)E), h((size_t)H), dpre2((size_t)H);
  auto live = [&](int64_t e)->bool { return !active || active[(size_t)e] != 0; };
  for (int64_t i = 0; i < M; i++) {
    float mx = -1.0e30f, sum = 0.f, pb = 0.f;
    int64_t best = -1;
    for (int64_t e = 0; e < E; e++) {
      float a = 0.f;
      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
      lg[(size_t)e] = a;
      if (live(e) && a > mx) mx = a;
    }
    for (int64_t e = 0; e < E; e++) {
      if (!live(e)) { p[(size_t)e] = 0.f; continue; }
      lg[(size_t)e] = expf(lg[(size_t)e] - mx);
      sum += lg[(size_t)e];
      if (best < 0 || lg[(size_t)e] > lg[(size_t)best]) best = e;
    }
    if (sum > 0.f)
      for (int64_t e = 0; e < E; e++) p[(size_t)e] = live(e) ? lg[(size_t)e]/sum : 0.f;
    pb = best >= 0 ? p[(size_t)best] : 0.f;
    if (best >= 0) {
      for (int64_t a = 0; a < H; a++) {
        float acc = 0.f;
        for (int64_t k = 0; k < D; k++) acc += x[i*D+k] * We1[best*D*H + k*H + a];
        h[(size_t)a] = 0.5f*acc*(1.f + erff(acc*0.7071067811865476f));
      }
      for (int64_t a = 0; a < H; a++) {
        float acc = 0.f;
        for (int64_t j = 0; j < D; j++) acc += dout[i*D+j] * We2[best*H*D + a*D + j];
        dpre2[(size_t)a] = acc;
      }
    }
    float dpb = 0.f;
    for (int64_t a = 0; a < H; a++) dpb += h[(size_t)a] * dpre2[(size_t)a];
    const float sd = dpb * pb;
    for (int64_t e = 0; e < E; e++) {
      const float dg = p[(size_t)e] * ((e == best ? dpb : 0.f) - sd);
      for (int64_t k = 0; k < D; k++) dWg[k*E+e] += x[i*D+k] * dg;
    }
  }
}

"#;

pub const NS_MOE_GRAD_WE1: &str = r#"// MoE expert-1 gradient dWe1[E,D,H] += (x^T @ dH) on the routed expert.
static void ns_moe_grad_we1(const float* dout, const float* x,
                            const float* Wg, const float* We1, const float* We2,
                            const uint8_t* active, float* dWe1,
                            int64_t M, int64_t D, int64_t H, int64_t E) {
  for (int64_t k = 0; k < (int64_t)E*D*H; k++) dWe1[k] = 0.f;
  std::vector<float> lg((size_t)E), p((size_t)E), h((size_t)H), dpre2((size_t)H);
  auto live = [&](int64_t e)->bool { return !active || active[(size_t)e] != 0; };
  for (int64_t i = 0; i < M; i++) {
    float mx = -1.0e30f, sum = 0.f, pb = 0.f;
    int64_t best = -1;
    for (int64_t e = 0; e < E; e++) {
      float a = 0.f;
      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
      lg[(size_t)e] = a;
      if (live(e) && a > mx) mx = a;
    }
    for (int64_t e = 0; e < E; e++) {
      if (!live(e)) { p[(size_t)e] = 0.f; continue; }
      lg[(size_t)e] = expf(lg[(size_t)e] - mx);
      sum += lg[(size_t)e];
      if (best < 0 || lg[(size_t)e] > lg[(size_t)best]) best = e;
    }
    if (sum > 0.f)
      for (int64_t e = 0; e < E; e++) p[(size_t)e] = live(e) ? lg[(size_t)e]/sum : 0.f;
    pb = best >= 0 ? p[(size_t)best] : 0.f;
    if (best >= 0) {
      for (int64_t a = 0; a < H; a++) {
        float acc = 0.f;
        for (int64_t k = 0; k < D; k++) acc += x[i*D+k] * We1[best*D*H + k*H + a];
        h[(size_t)a] = acc;
      }
      for (int64_t a = 0; a < H; a++) {
        float acc = 0.f;
        for (int64_t j = 0; j < D; j++) acc += dout[i*D+j] * We2[best*H*D + a*D + j];
        dpre2[(size_t)a] = acc;
      }
      for (int64_t a = 0; a < H; a++) {
        const float u = h[(size_t)a];
        const float gd = 0.5f*(1.f + erff(u*0.7071067811865476f))
                          + u*expf(-u*u*0.5f)*0.3989422804014327f;
        const float dH = dpre2[(size_t)a] * pb * gd;
        for (int64_t k = 0; k < D; k++)
          dWe1[best*D*H + k*H + a] += x[i*D+k] * dH;
      }
    }
  }
}

"#;

pub const NS_MOE_GRAD_WE2: &str = r#"// MoE expert-2 gradient dWe2[E,H,D] += (h^T @ dout) * pb on routed expert.
static void ns_moe_grad_we2(const float* dout, const float* x,
                            const float* Wg, const float* We1, const float* We2,
                            const uint8_t* active, float* dWe2,
                            int64_t M, int64_t D, int64_t H, int64_t E) {
  for (int64_t k = 0; k < (int64_t)E*H*D; k++) dWe2[k] = 0.f;
  std::vector<float> lg((size_t)E), p((size_t)E), h((size_t)H);
  auto live = [&](int64_t e)->bool { return !active || active[(size_t)e] != 0; };
  for (int64_t i = 0; i < M; i++) {
    float mx = -1.0e30f, sum = 0.f, pb = 0.f;
    int64_t best = -1;
    for (int64_t e = 0; e < E; e++) {
      float a = 0.f;
      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
      lg[(size_t)e] = a;
      if (live(e) && a > mx) mx = a;
    }
    for (int64_t e = 0; e < E; e++) {
      if (!live(e)) { p[(size_t)e] = 0.f; continue; }
      lg[(size_t)e] = expf(lg[(size_t)e] - mx);
      sum += lg[(size_t)e];
      if (best < 0 || lg[(size_t)e] > lg[(size_t)best]) best = e;
    }
    if (sum > 0.f)
      for (int64_t e = 0; e < E; e++) p[(size_t)e] = live(e) ? lg[(size_t)e]/sum : 0.f;
    pb = best >= 0 ? p[(size_t)best] : 0.f;
    if (best >= 0) {
      for (int64_t a = 0; a < H; a++) {
        float acc = 0.f;
        for (int64_t k = 0; k < D; k++) acc += x[i*D+k] * We1[best*D*H + k*H + a];
        h[(size_t)a] = 0.5f*acc*(1.f + erff(acc*0.7071067811865476f));
      }
      for (int64_t a = 0; a < H; a++)
        for (int64_t j = 0; j < D; j++)
          dWe2[best*H*D + a*D + j] += h[(size_t)a] * pb * dout[i*D+j];
    }
  }
}

"#;

pub const NS_LOSS_GRAD: &str = r#"// dL/d(preds) = (softmax(preds) - labels) / B   (cross-entropy seed)
static void ns_loss_grad(const float* preds, const float* labels, float* d,
                         size_t numel, int64_t C) {
  int64_t rows = (int64_t)numel / C;
  for (int64_t r = 0; r < rows; r++) {
    float mx = preds[(size_t)r*C];
    for (int64_t j = 1; j < C; j++) mx = std::max(mx, preds[(size_t)r*C + j]);
    float s = 0.f;
    for (int64_t j = 0; j < C; j++) s += expf(preds[(size_t)r*C + j] - mx);
    for (int64_t j = 0; j < C; j++) {
      float p = expf(preds[(size_t)r*C + j] - mx) / s;
      d[(size_t)r*C + j] = (p - labels[(size_t)r*C + j]) / (float)rows;
    }
  }
}

"#;

pub const NS_CROSS_ENTROPY: &str = r#"// Cross-entropy loss over one-hot labels: -sum(y * log(p)).
static float ns_cross_entropy(const float* preds, const float* labels,
                              size_t numel, int64_t C) {
  int64_t rows = (int64_t)numel / C;
  float loss = 0.f;
  for (int64_t r = 0; r < rows; r++) {
    float mx = preds[(size_t)r*C];
    for (int64_t j = 1; j < C; j++) mx = std::max(mx, preds[(size_t)r*C + j]);
    float s = 0.f;
    for (int64_t j = 0; j < C; j++) s += expf(preds[(size_t)r*C + j] - mx);
    for (int64_t j = 0; j < C; j++) {
      float p = expf(preds[(size_t)r*C + j] - mx) / s;
      if (labels[(size_t)r*C + j] > 0.f) loss -= logf(p);
    }
  }
  return loss / (float)rows;
}

"#;

pub const NS_ORTHONOM: &str = r#"// Modified Gram-Schmidt: orthonormalize the columns of X (M>=N),
// or the rows of X (M<N). Matches the CPU-reference Muon semantics.
static void ns_orthonom(float* G, size_t M, size_t N) {
  bool transposed = M < N;
  size_t R = transposed ? N : M;
  size_t C = transposed ? M : N;
  auto at = [&](size_t k, size_t j) -> float& { return transposed ? G[j*R + k] : G[k*N + j]; };
  for (size_t j = 0; j < C; j++) {
    for (size_t i = 0; i < j; i++) {
      float dot = 0.f;
      for (size_t k = 0; k < R; k++) dot += at(k, i) * at(k, j);
      for (size_t k = 0; k < R; k++) at(k, j) -= dot * at(k, i);
    }
    float nrm = 0.f;
    for (size_t k = 0; k < R; k++) nrm += at(k, j) * at(k, j);
    nrm = sqrtf(nrm) + 1e-30f;
    for (size_t k = 0; k < R; k++) at(k, j) /= nrm;
  }
}

"#;

pub const NS_TRANSPOSE2D: &str = r#"static void ns_transpose2d(const float* in, float* out,
                           int64_t rows, int64_t cols) {
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++)
      out[j*rows + i] = in[i*cols + j];
}

"#;

pub const NS_CONCAT2: &str = r#"static void ns_concat2(const float* a, const float* b, float* out,
                       int64_t rows, int64_t ca, int64_t cb) {
  int64_t co = ca + cb;
  for (int64_t r = 0; r < rows; r++) {
    memcpy(out + r*co, a + r*ca, ca*sizeof(float));
    memcpy(out + r*co + ca, b + r*cb, cb*sizeof(float));
  }
}

"#;

pub const NS_CONCAT0: &str = r#"static void ns_concat0(const float* a, const float* b, float* out,
                       int64_t ba, int64_t bb, int64_t d) {
  memcpy(out, a, ba*d*sizeof(float));
  memcpy(out + ba*d, b, bb*d*sizeof(float));
}

"#;

pub const NS_SLICE2: &str = r#"static void ns_slice2(const float* in, float* out,
                      int64_t M, int64_t C, int64_t cs, int64_t ce) {
  int64_t ow = ce - cs;
  for (int64_t r = 0; r < M; r++)
    memcpy(out + r*ow, in + r*C + cs, ow*sizeof(float));
}

"#;

pub const NS_SLICEROWS: &str = r#"static void ns_slicerows(const float* in, float* out,
                         int64_t C, int64_t rs, int64_t re) {
  for (int64_t r = rs; r < re; r++)
    memcpy(out + (r-rs)*C, in + r*C, C*sizeof(float));
}

"#;

pub const NS_INDEX2: &str = r#"static void ns_index2(const float* in, const int64_t* idx, float* out,
                      int64_t M, int64_t C, int64_t L) {
  for (int64_t r = 0; r < M; r++)
    for (int64_t k = 0; k < L; k++)
      out[r*L + k] = in[r*C + idx[k]];
}

"#;

pub const NS_INDEXROWS: &str = r#"static void ns_indexrows(const float* in, const int64_t* idx, float* out,
                         int64_t L, int64_t C) {
  for (int64_t k = 0; k < L; k++)
    memcpy(out + k*C, in + idx[k]*C, C*sizeof(float));
}

"#;

pub const NS_SCATTER2: &str = r#"static void ns_scatter2(const float* in, const int64_t* idx, const float* upd,
                        float* out, int64_t M, int64_t C, int64_t L) {
  memcpy(out, in, M*C*sizeof(float));
  for (int64_t r = 0; r < M; r++)
    for (int64_t k = 0; k < L; k++)
      out[r*C + idx[k]] = upd[r*L + k];
}

"#;

pub const NS_SCATTERROWS: &str = r#"static void ns_scatterrows(const float* in, const int64_t* idx, const float* upd,
                           float* out, int64_t R, int64_t C, int64_t L) {
  memcpy(out, in, R*C*sizeof(float));
  for (int64_t k = 0; k < L; k++)
    memcpy(out + idx[k]*C, upd + k*C, C*sizeof(float));
}

"#;

pub const NS_EMBEDDING: &str = r#"static void ns_embedding(const float* w, const float* idx, float* out,
                        int64_t n, int64_t V, int64_t D) {
  for (int64_t i = 0; i < n; i++) {
    int64_t k = (int64_t)idx[i];
    if (k < 0 || k >= V) k = 0;
    memcpy(out + i*D, w + k*D, D*sizeof(float));
  }
}

"#;

pub const NS_ATTENTION_FWD: &str = r#"static void ns_attention_fwd(const float* x,
    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
    float* Q, float* K, float* V, float* sc, float* proj, float* out,
    int64_t BS, int64_t D, int64_t H, int64_t S, int causal) {
  int64_t Dk = D / H;
  float scale = 1.0f / sqrtf((float)Dk);
  // Q = x @ Wq,  K = x @ Wk,  V = x @ Wv  (all [BS, D])
  auto gemm = [](const float* A, const float* B, float* C,
                 int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
      for (int64_t j = 0; j < N; j++) {
        float a = 0.f;
        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*B[k*N+j];
        C[i*N+j] = a;
      }
  };
  gemm(x, Wq, Q, BS, D, D);
  gemm(x, Wk, K, BS, D, D);
  gemm(x, Wv, V, BS, D, D);
  int64_t B = BS / S;
  // Per-head attention, then scatter into out.
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      // sc[i,j] = Q[b*S+i, h*Dk..] · K[b*S+j, h*Dk..] * scale
      for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
          float a = 0.f;
          for (int64_t d = 0; d < Dk; d++)
            a += Q[((b*S+i)*D)+h*Dk+d] * K[((b*S+j)*D)+h*Dk+d];
          sc[i*S+j] = (causal && j > i) ? -1e30f : a * scale;
        }
      }
      // softmax over rows of sc
      for (int64_t i = 0; i < S; i++) {
        float mx = sc[i*S];
        for (int64_t j = 1; j < S; j++) mx = std::max(mx, sc[i*S+j]);
        float s = 0.f;
        for (int64_t j = 0; j < S; j++) { sc[i*S+j] = expf(sc[i*S+j]-mx); s += sc[i*S+j]; }
        for (int64_t j = 0; j < S; j++) sc[i*S+j] /= s;
      }
      // out_head[i,d] = sc[i,:] · V[b*S+:, h*Dk+d]
      for (int64_t i = 0; i < S; i++)
        for (int64_t d = 0; d < Dk; d++) {
          float a = 0.f;
          for (int64_t j = 0; j < S; j++)
            a += sc[i*S+j] * V[((b*S+j)*D)+h*Dk+d];
          out[((b*S+i)*D)+h*Dk+d] = a;
        }
    }
  }
  // output projection: proj = out @ Wo
  gemm(out, Wo, proj, BS, D, D);
  memcpy(out, proj, BS*D*sizeof(float));
}

"#;

pub const NS_ATTENTION_BWD: &str = r#"// Multi-head attention backward: recompute Q/K/V and the per-head
// softmax, then push the upstream dL/dout through O -> head-concat ->
// softmax -> Q/K/V -> the four DxD projections.  The caller selects a
// SINGLE target output (dX or one of dWq/dWk/dWv/dWo) by passing a
// non-null pointer; the layer input gradient accumulates the gate + 
// expert paths in the recomputed stack.
static void ns_attention_bwd(const float* dout, const float* x,
                            const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                            float* dX, float* dWq, float* dWk, float* dWv, float* dWo,
                            int64_t BS, int64_t D, int64_t H, int64_t S, int causal) {
  if (D <= 0 || H <= 0 || S <= 0) return;
  int64_t Dk = D / H;
  float scale = 1.0f / sqrtf((float)Dk);
  int64_t B = BS / S;
  if (B <= 0) B = 1;
  auto gemm = [](const float* A, const float* Bw, float* C,
                  int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; i++)
      for (int64_t j = 0; j < N; j++) {
        float a = 0.f;
        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*Bw[k*N+j];
        C[i*N+j] = a;
      }
  };
  auto gtx = [](const float* A, const float* Bw, float* C,
                int64_t M, int64_t K, int64_t N) {  // C = A^T @ Bw
    for (int64_t k = 0; k < K; k++)
      for (int64_t n = 0; n < N; n++) {
        float a = 0.f;
        for (int64_t i = 0; i < M; i++) a += A[i*K+k]*Bw[i*N+n];
        C[k*N+n] = a;
      }
  };
  auto gmt = [](const float* A, const float* Bw, float* C,
                int64_t M, int64_t K, int64_t N) {  // C = A @ Bw^T
    for (int64_t i = 0; i < M; i++)
      for (int64_t j = 0; j < N; j++) {
        float a = 0.f;
        for (int64_t k = 0; k < K; k++) a += A[i*K+k]*Bw[j*K+k];
        C[i*N+j] = a;
      }
  };
  std::vector<float> Qq(BS*D), Kk(BS*D), Vv(BS*D);
  std::vector<float> dQ2(BS*D), dK2(BS*D), dV2(BS*D), dPre(BS*D), oh(BS*D), sc((size_t)S*S);
  gemm(x, Wq, Qq.data(), BS, D, D);
  gemm(x, Wk, Kk.data(), BS, D, D);
  gemm(x, Wv, Vv.data(), BS, D, D);
  for (int64_t i = 0; i < BS; i++)
    for (int64_t k = 0; k < D; k++) {
      float a = 0.f;
      for (int64_t n = 0; n < D; n++) a += dout[i*D+n] * Wo[k*D+n];
      dPre[i*D+k] = a;
    }
  std::fill(dQ2.begin(), dQ2.end(), 0.f);
  std::fill(dK2.begin(), dK2.end(), 0.f);
  std::fill(dV2.begin(), dV2.end(), 0.f);
  std::fill(oh.begin(), oh.end(), 0.f);
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      // scores + softmax
      for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
          float a = 0.f;
          for (int64_t d = 0; d < Dk; d++)
            a += Qq[((b*S+i)*D)+h*Dk+d] * Kk[((b*S+j)*D)+h*Dk+d];
          sc[i*S+j] = (causal && j > i) ? -1e30f : a * scale;
        }
        float mx = sc[i*S];
        for (int64_t j = 1; j < S; j++) mx = std::max(mx, sc[i*S+j]);
        float s = 0.f;
        for (int64_t j = 0; j < S; j++) { sc[i*S+j] = expf(sc[i*S+j]-mx); s += sc[i*S+j]; }
        for (int64_t j = 0; j < S; j++) sc[i*S+j] /= s;
      }
      // out_head (for dWo) and dP
      for (int64_t i = 0; i < S; i++)
        for (int64_t d = 0; d < Dk; d++) {
          float a = 0.f;
          for (int64_t j = 0; j < S; j++)
            a += sc[i*S+j] * Vv[((b*S+j)*D)+h*Dk+d];
          oh[((b*S+i)*D)+h*Dk+d] = a;
        }
      // V grad and P grad (dPre is the head output grad)
      for (int64_t j = 0; j < S; j++)
        for (int64_t d = 0; d < Dk; d++) {
          float a = 0.f;
          for (int64_t i = 0; i < S; i++)
            a += sc[i*S+j] * dPre[((b*S+i)*D)+h*Dk+d];
          dV2[((b*S+j)*D)+h*Dk+d] += a;
        }
      std::vector<float> dp((size_t)S*S), ds((size_t)S*S);
      for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
          float a = 0.f;
          for (int64_t d = 0; d < Dk; d++)
            a += Vv[((b*S+j)*D)+h*Dk+d] * dPre[((b*S+i)*D)+h*Dk+d];
          dp[i*S+j] = a;
        }
        float dot = 0.f;
        for (int64_t j = 0; j < S; j++) dot += sc[i*S+j]*dp[i*S+j];
        for (int64_t j = 0; j < S; j++) ds[i*S+j] = sc[i*S+j]*(dp[i*S+j]-dot);
      }
      // Q, K grads
      for (int64_t i = 0; i < S; i++)
        for (int64_t d = 0; d < Dk; d++) {
          float a = 0.f;
          for (int64_t j = 0; j < S; j++) a += ds[i*S+j]*Kk[((b*S+j)*D)+h*Dk+d];
          dQ2[((b*S+i)*D)+h*Dk+d] += a * scale;
        }
      for (int64_t j = 0; j < S; j++)
        for (int64_t d = 0; d < Dk; d++) {
          float a = 0.f;
          for (int64_t i = 0; i < S; i++) a += ds[i*S+j]*Qq[((b*S+i)*D)+h*Dk+d];
          dK2[((b*S+j)*D)+h*Dk+d] += a * scale;
        }
    }
  }
  // projection grads and input grad
  if (dWq) { for (int64_t n = 0; n < D*D; n++) dWq[n] = 0.f; gtx(x, dQ2.data(), dWq, BS, D, D); }
  if (dWk) { for (int64_t n = 0; n < D*D; n++) dWk[n] = 0.f; gtx(x, dK2.data(), dWk, BS, D, D); }
  if (dWv) { for (int64_t n = 0; n < D*D; n++) dWv[n] = 0.f; gtx(x, dV2.data(), dWv, BS, D, D); }
  if (dWo) { for (int64_t n = 0; n < D*D; n++) dWo[n] = 0.f; gtx(oh.data(), dout, dWo, BS, D, D); }
  if (dX) {
    std::vector<float> gq(BS*D), gk(BS*D), gv(BS*D);
    gmt(dQ2.data(), Wq, gq.data(), BS, D, D);  // dQ2 @ Wq^T
    gmt(dK2.data(), Wk, gk.data(), BS, D, D);
    gmt(dV2.data(), Wv, gv.data(), BS, D, D);
    for (int64_t i = 0; i < BS*D; i++) dX[i] = gq[i] + gk[i] + gv[i];
  }
}

"#;

pub const NS_MOE_FWD: &str = r#"// MIXTURE-OF-EXPERTS (fused router + top-1 dispatch + weighted combine).
// x[M,D]  Wg[D,E]  We1[E,D,H]  We2[E,H,D]  →  out[M,D]
// Router logits = x @ Wg, softmax over LIVE experts only, per token the
// top-1 live expert computes y = GELU(x @ We1) @ We2, scaled by the
// (softmaxed) routing probability. `active` may be null (all live);
// otherwise active[e]!=0 means expert e is live.
static void ns_moe_fwd(const float* x, const float* Wg, const float* We1, const float* We2,
                      const uint8_t* active, float* out,
                      int64_t M, int64_t D, int64_t H, int64_t E) {
  std::vector<float> lg((size_t)E), h((size_t)H);
  auto live = [&](int64_t e)->bool { return !active || active[(size_t)e] != 0; };
  for (int64_t i = 0; i < M; i++) {
    float mx = -1.0e30f, sum = 0.f;
    for (int64_t e = 0; e < E; e++) {
      float a = 0.f;
      for (int64_t k = 0; k < D; k++) a += x[i*D+k] * Wg[k*E+e];
      lg[(size_t)e] = a;
      if (live(e) && a > mx) mx = a;
    }
    for (int64_t e = 0; e < E; e++)
      if (live(e)) { lg[(size_t)e] = expf(lg[(size_t)e] - mx); sum += lg[(size_t)e]; }
    int64_t best = -1;
    for (int64_t e = 0; e < E; e++)
      if (live(e) && (best < 0 || lg[(size_t)e] > lg[(size_t)best])) best = e;
    const float p = (sum > 0.f && best >= 0) ? lg[(size_t)best] / sum : 0.f;
    for (int64_t a = 0; a < H; a++) {
      float acc = 0.f;
      for (int64_t k = 0; k < D; k++) acc += x[i*D+k] * We1[best*D*H + k*H + a];
      h[(size_t)a] = 0.5f*acc*(1.f + erff(acc*0.7071067811865476f));
    }
    for (int64_t j = 0; j < D; j++) {
      float a = 0.f;
      for (int64_t b = 0; b < H; b++) a += h[(size_t)b] * We2[best*H*D + b*D + j];
      out[i*D+j] = p * a;
    }
  }
}

"#;
