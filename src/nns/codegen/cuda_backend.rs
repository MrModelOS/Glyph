#![allow(dead_code)]
// NNS port: public API preserved for parity with C++ nsc; not all items are used in current pipeline — intentional, not tech debt
//! CUDA backend layer: device source snippets spliced into emitted
//! `.cu` units, ported 1:1 from `NeuralScript/src/codegen/cuda_backend.cpp`.

// FP16 support header: included when CodegenOptions::enable_fp16 is true.
// Defines NS_DTYPE as __half and enables half2 vectorization comment for
// NS_LAUNCH_BLOCKS tuning. Used by gen_cuda when fp16 is enabled.
pub const CUDA_FP16_HEADER: &str =
    "#include <cuda_fp16.h>\n#define NS_DTYPE __half\n#define NS_FP16 1\n// fp16: using half2\n";

// ROCm/HIP header placeholder: hipified CUDA. Emitted by gen_rocm to make
// the output distinguishable from pure CUDA.
pub const ROCM_HEADER: &str = "// ROCm backend: hipified CUDA (placeholder, delegates to CUDA)\n#ifdef __HIP_PLATFORM_AMD__\n#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n#endif\n";

// Metal header placeholder: MSL/MPS not yet implemented. Emitted by gen_metal.
pub const METAL_HEADER: &str = "// Metal backend: placeholder, delegates to CUDA\n// TODO(Metal): MSL/MPS kernels not yet implemented\n";

// Launch grid for flat 1D kernels (256 threads/block).
pub const K_LAUNCH_MACROS: &str = r#"
#define NS_CU_GRID(n) (((unsigned)(n) + 255u) / 256u)
#define NS_LAUNCH1(kern, n, ...) do { kern<<<NS_CU_GRID(n), 256>>>(__VA_ARGS__); } while (0)
// Block-per-tile launch for tiled kernels: grid.x == number of blocks (each
// block owns one output tile), blockDim.x == block size in threads.
#define NS_LAUNCH_BLOCKS(kern, nblocks, ...) do { kern<<<((nblocks) > 0 ? (nblocks) : 1), 256>>>(__VA_ARGS__); } while (0)
"#;

pub const K_DEVICE_HELPERS: &str = r#"
__device__ float ns_act2_d(float x, int code) {
  switch (code) {
    case 0: return x;
    case 1: return x > 0.f ? x : 0.f;
    case 2: return x > 0.f ? x : 0.01f * x;
    case 3: return 1.f / (1.f + expf(-x));
    case 4: return tanhf(x);
    case 5: return x / (1.f + expf(-x));
    case 6: { float u = x / 1.41421356f; return 0.5f * x * (1.f + erff(u)); }
    default: return x;
  }
}
__device__ float ns_act_deriv_d(float x, int code) {
  switch (code) {
    case 0: return 1.f;
    case 1: return x > 0.f ? 1.f : 0.f;
    case 2: return x > 0.f ? 1.f : 0.01f;
    case 3: { float s = 1.f / (1.f + expf(-x)); return s * (1.f - s); }
    case 4: { float t = tanhf(x); return 1.f - t * t; }
    case 5: { float s = x / (1.f + expf(-x)); return s + x * (1.f - s); }
    case 6: { float u = x / 1.41421356f; float p = 0.5f * (1.f + erff(u));
              return p + x * expf(-u * u) / 2.50662827f; }
    default: return 1.f;
  }
}
"#;

// Uniform forward kernels.
pub const K_FORWARD_KERNELS: &str = r#"
// Tiled GEMM: 16x16 output tiles, 16-wide K strip in shared memory (padded to
// avoid bank conflicts). Every load/store is bounds-checked, so the kernel is
// safe for arbitrary M/K/N (including tiny M<16 or K<16) and for M that is not
// a multiple of the tile width. Launched via NS_LAUNCH_BLOCKS with one block
// per output tile: grid.x = ceil(M/16) * ceil(N/16).
__global__ void ns_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int K, int N) {
  __shared__ float As[16][17];
  __shared__ float Bs[16][17];
  int nb = (N + 15) >> 4;
  int blk = blockIdx.x;
  int by = blk / nb, bx = blk - by * nb;
  int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
  int row = (by << 4) + ty, col = (bx << 4) + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    int ak = k0 + tx, ai = row;
    As[ty][tx] = (ai < M && ak < K) ? A[ai * K + ak] : 0.f;
    int bk = k0 + ty, bj = col;
    Bs[ty][tx] = (bk < K && bj < N) ? B[bk * N + bj] : 0.f;
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; kk++) acc += As[ty][kk] * Bs[kk][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}
__global__ void ns_act_kernel(const float* __restrict__ in, float* __restrict__ out,
                              int n, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = ns_act2_d(in[idx], code);
}
__global__ void ns_copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx];
}
__global__ void ns_fill_kernel(float* __restrict__ out, int n, float v) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = v;
}
__global__ void ns_transpose2d_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int i = idx / cols, j = idx - i * cols;
  out[j * rows + i] = in[i * cols + j];
}
__global__ void ns_concat2_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int rows, int ca, int co) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * co;
  if (idx >= total) return;
  int r = idx / co, c = idx - r * co;
  out[idx] = (c < ca) ? A[r * ca + c] : B[r * (co - ca) + (c - ca)];
}
__global__ void ns_concat0_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int na, int nb, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (na + nb) * D;
  if (idx >= total) return;
  int r = idx / D, c = idx - r * D;
  out[idx] = (r < na) ? A[idx] : B[(r - na) * D + c];
}
__global__ void ns_slice2_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 int M, int C, int cs, int ce) {
  int ow = ce - cs;
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * ow;
  if (idx_i >= total) return;
  int r = idx_i / ow, c = idx_i - r * ow;
  out[idx_i] = in[r * C + cs + c];
}
__global__ void ns_slicerows_kernel(const float* __restrict__ in, float* __restrict__ out,
                                    int C, int rs, int re) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = (re - rs) * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[(rs + r) * C + c];
}
__global__ void ns_index_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                float* __restrict__ out, int M, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * L;
  if (idx_i >= total) return;
  int r = idx_i / L, k = idx_i - r * L;
  out[idx_i] = in[r * C + idx[k]];
}
__global__ void ns_indexrows_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                    float* __restrict__ out, int L, int C) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = L * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[(int)idx[r] * C + c];
}
__global__ void ns_scatter_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                  const float* __restrict__ upd, float* __restrict__ out,
                                  int M, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[idx_i];
  for (int k = 0; k < L; k++)
    if (idx[k] == c) { out[idx_i] = upd[r * L + k]; break; }
}
__global__ void ns_scatterrows_kernel(const float* __restrict__ in, const int64_t* __restrict__ idx,
                                      const float* __restrict__ upd, float* __restrict__ out,
                                      int R, int C, int L) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = R * C;
  if (idx_i >= total) return;
  int r = idx_i / C, c = idx_i - r * C;
  out[idx_i] = in[idx_i];
  for (int k = 0; k < L; k++)
    if (idx[k] == r) { out[idx_i] = upd[k * C + c]; break; }
}
__global__ void ns_embedding_kernel(const float* __restrict__ W, const float* __restrict__ idx,
                                    float* __restrict__ out, int n, int V, int D) {
  int idx_i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * D;
  if (idx_i >= total) return;
  int r = idx_i / D, c = idx_i - r * D;
  int k = (int)idx[r];
  if (k < 0 || k >= V) k = 0;
  out[idx_i] = W[k * D + c];
}
// Multi-head attention core: one block per (batch, head). Grid = B*H blocks.
// Shared memory holds the S*S score matrix (dynamic shared memory). Q, K, V
// are already projected on device (each [B*S, D]); Out accumulates the per-head
// context [B*S, D]. Head dimension Dk = D/H, sequence length S, scale=1/sqrt(Dk).
extern "C" __global__ void ns_attention_core_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ Out,
    int S, int D, int H, float scale, int causal) {
  extern __shared__ float sc[];   // [S*S]
  int b = blockIdx.x / H, h = blockIdx.x - b * H;
  int Dk = D / H;
  const float* Qb = Q + b * S * D + h * Dk;
  const float* Kb = K + b * S * D + h * Dk;
  const float* Vb = V + b * S * D + h * Dk;
  // scores[i,j] = Qb[i*D + d] . Kb[j*D + d] * scale
  for (int t = threadIdx.x; t < S * S; t += blockDim.x) {
    int i = t / S, j = t - i * S;
    float a = 0.f;
    for (int d = 0; d < Dk; d++)
      a += Qb[i * D + d] * Kb[j * D + d];
    sc[t] = (causal && j > i) ? -1e30f : a * scale;
  }
  __syncthreads();
  // row softmax in-place
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float mx = sc[i * S];
    for (int j = 1; j < S; j++) mx = fmaxf(mx, sc[i * S + j]);
    float s = 0.f;
    for (int j = 0; j < S; j++) { sc[i * S + j] = expf(sc[i * S + j] - mx); s += sc[i * S + j]; }
    for (int j = 0; j < S; j++) sc[i * S + j] /= s;
  }
  __syncthreads();
  // context[i,d] = sum_j sc[i,j] * Vb[j*D+d]
  float* Ob = Out + b * S * D + h * Dk;
  for (int i = threadIdx.x; i < S; i += blockDim.x)
    for (int d = 0; d < Dk; d++) {
      float a = 0.f;
      for (int j = 0; j < S; j++) a += sc[i * S + j] * Vb[j * D + d];
      Ob[i * D + d] = a;
    }
}
// Multi-head attention backward: one block per (head). Grid = B*H blocks.
// Each block recomputes the head-local forward (Q/K/V from x@W, scores, softmax)
// and the head-local output, then produces its full contribution to every target:
// dX is written head-disjoint (no atomics); the four DxD weight grads accumulate
// with atomicAdd. `which` selects the single output the caller wants:
// 0=dX, 1=dWq, 2=dWk, 3=dWv, 4=dWo. weight targets must be zeroed by the caller.
// The head-local working set (sc[S*S] + 8 arrays of S*Dk) lives in GLOBAL scratch
// (one per block) so shared-memory usage stays 0 — the full set at S=128/D=512/H=8
// is 320KB, far beyond the 64KB per-block shared limit even with the opt-in
// attribute. sc is per-block: scr + blockIdx.x * perblk.
extern "C" __global__ void ns_attention_grad_kernel(
    const float* __restrict__ dout, const float* __restrict__ x,
    const float* __restrict__ Wq, const float* __restrict__ Wk,
    const float* __restrict__ Wv, const float* __restrict__ Wo,
    float* __restrict__ dX, float* __restrict__ dWq, float* __restrict__ dWk,
    float* __restrict__ dWv, float* __restrict__ dWo,
    float* __restrict__ scr,
    int BS, int D, int H, int S, int which, int causal) {
  int perblk = S * S + 8 * S * (D / H);
  float* sc = scr + blockIdx.x * perblk;
  float* qh = sc + S * S;
  float* kh = qh + S * (D / H);
  float* vh = kh + S * (D / H);
  float* dQ = vh + S * (D / H);
  float* dK = dQ + S * (D / H);
  float* dV = dK + S * (D / H);
  float* ohv = dV + S * (D / H);
  float* dPr = ohv + S * (D / H);
  int Dk = D / H;
  float scale = 1.0f / sqrtf((float)Dk);
  int b = blockIdx.x / H, h = blockIdx.x - b * H;
  int h0 = h * Dk;
  const float* xb = x + b * S * D;
  const float* db = dout + b * S * D;
  for (int t = threadIdx.x; t < S * Dk; t += blockDim.x) {
    int i = t / Dk, d = t % Dk;
    float aq = 0.f, ak = 0.f, av = 0.f;
    for (int k = 0; k < D; k++) {
      float xv = xb[i * D + k];
      aq += xv * Wq[k * D + h0 + d];
      ak += xv * Wk[k * D + h0 + d];
      av += xv * Wv[k * D + h0 + d];
    }
    qh[t] = aq; kh[t] = ak; vh[t] = av;
  }
  __syncthreads();
  for (int t = threadIdx.x; t < S * S; t += blockDim.x) {
    int i = t / S, j = t - i * S;
    float a = 0.f;
    for (int d = 0; d < Dk; d++) a += qh[i * Dk + d] * kh[j * Dk + d];
    sc[t] = (causal && j > i) ? -1e30f : a * scale;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float mx = sc[i * S];
    for (int j = 1; j < S; j++) mx = fmaxf(mx, sc[i * S + j]);
    float s = 0.f;
    for (int j = 0; j < S; j++) { sc[i * S + j] = expf(sc[i * S + j] - mx); s += sc[i * S + j]; }
    for (int j = 0; j < S; j++) sc[i * S + j] /= s;
  }
  __syncthreads();
  // head output oh and head output grad dPr = dout @ Wo^T
  for (int t = threadIdx.x; t < S * Dk; t += blockDim.x) {
    int i = t / Dk, d = t % Dk;
    float a = 0.f;
    for (int j = 0; j < S; j++) a += sc[i * S + j] * vh[j * Dk + d];
    ohv[t] = a;
    float b2 = 0.f;
    for (int n = 0; n < D; n++) b2 += db[i * D + n] * Wo[(h0 + d) * D + n];
    dPr[t] = b2;
  }
  __syncthreads();
  // dV[j,d] = sum_i sc[i,j] * dPr[i,d]
  for (int t = threadIdx.x; t < S * Dk; t += blockDim.x) {
    int j = t / Dk, d = t % Dk;
    float a = 0.f;
    for (int i = 0; i < S; i++) a += sc[i * S + j] * dPr[i * Dk + d];
    dV[t] = a;
  }
  __syncthreads();
  // dp[i,j], then row-blend into ds (reuses sc)
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float dot = 0.f;
    float dp[256];
    for (int j = 0; j < S; j++) {
      float a = 0.f;
      for (int d = 0; d < Dk; d++) a += vh[j * Dk + d] * dPr[i * Dk + d];
      dp[j] = a;
      dot += sc[i * S + j] * dp[j];
    }
    for (int j = 0; j < S; j++) sc[i * S + j] = sc[i * S + j] * (dp[j] - dot);
  }
  __syncthreads();
  // dQ[i,d] = scale * sum_j ds[i,j] * kh[j,d];  dK[j,d] = scale * sum_i ds[i,j] * qh[i,d]
  for (int t = threadIdx.x; t < S * Dk; t += blockDim.x) {
    int i = t / Dk, d = t % Dk;
    float aq = 0.f, ak = 0.f;
    for (int j = 0; j < S; j++) {
      float ds = sc[i * S + j];
      aq += ds * kh[j * Dk + d];
      ak += ds * qh[j * Dk + d];
    }
    dQ[t] = aq * scale;
    dK[t] = ak * scale;
  }
  __syncthreads();
  if (which == 0) {  // dX = dQ@Wq^T + dK@Wk^T + dV@Wv^T  (head-owned dims, disjoint)
    for (int i = threadIdx.x; i < S; i += blockDim.x)
      for (int d = 0; d < Dk; d++) {
        float gq = 0.f, gk = 0.f, gv = 0.f;
        for (int k = 0; k < D; k++) {
          gq += dQ[i * Dk + d] * Wq[k * D + h0 + d];
          gk += dK[i * Dk + d] * Wk[k * D + h0 + d];
          gv += dV[i * Dk + d] * Wv[k * D + h0 + d];
        }
        dX[(b * S + i) * D + h0 + d] = gq + gk + gv;
      }
  } else if (which == 1) {  // dWq = x^T @ dQ
    for (int k = 0; k < D; k++)
      for (int n = h0; n < h0 + Dk; n++) {
        float a = 0.f;
        for (int i = 0; i < S; i++) a += xb[i * D + k] * dQ[i * Dk + (n - h0)];
        atomicAdd(&dWq[k * D + n], a);
      }
  } else if (which == 2) {  // dWk = x^T @ dK
    for (int k = 0; k < D; k++)
      for (int n = h0; n < h0 + Dk; n++) {
        float a = 0.f;
        for (int i = 0; i < S; i++) a += xb[i * D + k] * dK[i * Dk + (n - h0)];
        atomicAdd(&dWk[k * D + n], a);
      }
  } else if (which == 3) {  // dWv = x^T @ dV
    for (int k = 0; k < D; k++)
      for (int n = h0; n < h0 + Dk; n++) {
        float a = 0.f;
        for (int i = 0; i < S; i++) a += xb[i * D + k] * dV[i * Dk + (n - h0)];
        atomicAdd(&dWv[k * D + n], a);
      }
  } else {  // dWo = oh^T @ dout
    for (int k = h0; k < h0 + Dk; k++)
      for (int n = 0; n < D; n++) {
        float a = 0.f;
        for (int i = 0; i < S; i++) a += ohv[i * Dk + (k - h0)] * db[i * D + n];
        atomicAdd(&dWo[k * D + n], a);
      }
  }
}
// Mixture-of-experts router: one thread per token. Router logits = x @ Wg with
// a softmax over LIVE experts (active null = all live, active[e]!=0 = live);
// the top-1 live expert computes y = GELU(x @ We1) @ We2 and the output is
// scaled by the routing weight. We1[e] is [D,H], We2[e] is [H,D].
__global__ void ns_moe_kernel(const float* __restrict__ x, const float* __restrict__ Wg,
                              const float* __restrict__ We1, const float* __restrict__ We2,
                              const uint8_t* __restrict__ active, float* __restrict__ out,
                              int M, int D, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float lg[256], h[2048];
  int Ee = E < 256 ? E : 256;
  int He = H < 2048 ? H : 2048;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { lg[e] = -1.0e30f; continue; }
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    lg[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { lg[e] = 0.f; continue; }
    lg[e] = expf(lg[e] - mx); sum += lg[e];
  }
  int best = -1;
  for (int e = 0; e < Ee; e++)
    if (!(active && !active[e]) && (best < 0 || lg[e] > lg[best])) best = e;
  float p = (sum > 0.f && best >= 0) ? lg[best] / sum : 0.f;
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int k = 0; k < D; k++) acc += x[i * D + k] * We1[best * D * H + k * H + a];
    h[a] = 0.5f * acc * (1.f + erff(acc * 0.7071067811865476f));
  }
  for (int j = 0; j < D; j++) {
    float a = 0.f;
    for (int b = 0; b < He; b++) a += h[b] * We2[best * H * D + b * D + j];
    out[i * D + j] = p * a;
  }
}
__global__ void ns_binop_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int n, int nb, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float b = nb > 0 ? B[idx % nb] : 0.f;
  switch (code) {
    case 0: C[idx] = A[idx] + b; break;
    case 1: C[idx] = A[idx] - b; break;
    case 2: C[idx] = A[idx] * b; break;
    case 3: C[idx] = A[idx] / b; break;
  }
}
// Elementwise binop backward; `which` bit0 = rhs(1)/lhs(0), bits1-2 = op
// (0=+,1=-,2=*,3=/). A/B are the forward operands, dc the upstream grad.
__global__ void ns_binop_grad_kernel(const float* __restrict__ dc,
                                     const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int n, int which) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int op = which >> 1;
  int side = which & 1;
  float a = A[idx], b = B[idx], g = dc[idx];
  switch (op) {
    case 0: C[idx] = g; break;                          // +
    case 1: C[idx] = side ? -g : g; break;             // -
    case 2: C[idx] = side ? g * a : g * b; break;      // *
    default: C[idx] = side ? -g * a / (b * b) : g / b; // /
  }
}
__global__ void ns_layernorm_kernel(float* __restrict__ m, int n, int last) {
  int rows = n / last;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mean = 0.f, var = 0.f;
  for (int j = 0; j < last; j++) mean += m[r * last + j];
  mean /= last;
  for (int j = 0; j < last; j++) { float d = m[r * last + j] - mean; var += d * d; }
  var /= last;
  float inv = rsqrtf(var + 1e-5f);
  for (int j = 0; j < last; j++) m[r * last + j] = (m[r * last + j] - mean) * inv;
}
__global__ void ns_softmax_kernel(float* __restrict__ m, int n, int last) {
  int rows = n / last;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = m[r * last];
  for (int j = 1; j < last; j++) mx = fmaxf(mx, m[r * last + j]);
  float s = 0.f;
  for (int j = 0; j < last; j++) { m[r * last + j] = expf(m[r * last + j] - mx); s += m[r * last + j]; }
  for (int j = 0; j < last; j++) m[r * last + j] /= s;
}
"#;

// Reverse-mode + optimizer kernels (AOT training).
pub const K_TRAIN_KERNELS: &str = r#"
__global__ void ns_grad_a_kernel(const float* __restrict__ dC, const float* __restrict__ B,
                                 float* __restrict__ dA, int M, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * K;
  for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
    int i = t / K, k = t % K;
    float acc = 0.f;
    for (int j = 0; j < N; j++) acc += dC[i * N + j] * B[k * N + j];
    dA[t] = acc;
  }
}
__global__ void ns_grad_w_kernel(const float* __restrict__ A, const float* __restrict__ dC,
                                 float* __restrict__ dB, int M, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = K * N;
  for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
    int k = t / N, j = t % N;
    float acc = 0.f;
    for (int i = 0; i < M; i++) acc += A[i * K + k] * dC[i * N + j];
    dB[t] = acc;
  }
}
__global__ void ns_act_grad_kernel(const float* __restrict__ dIn, const float* __restrict__ actIn,
                                   float* __restrict__ dOut, int n, int code) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dOut[idx] = dIn[idx] * ns_act_deriv_d(actIn[idx], code);
}
__global__ void ns_layernorm_grad_kernel(const float* __restrict__ dout,
                                         const float* __restrict__ x,
                                         float* __restrict__ dx, int n, int last) {
  int rows = n / last;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mean_x = 0.f, mean_dout = 0.f;
  for (int j = 0; j < last; j++) {
    mean_x    += x[r * last + j];
    mean_dout += dout[r * last + j];
  }
  mean_x    /= last;
  mean_dout /= last;
  float var_x = 0.f, gamma = 0.f;
  for (int j = 0; j < last; j++) {
    float xj = x[r * last + j] - mean_x;
    var_x += xj * xj;
    gamma += xj * dout[r * last + j];
  }
  var_x /= last;
  gamma /= last;
  float inv = rsqrtf(var_x + 1e-5f);
  for (int j = 0; j < last; j++) {
    float xj = x[r * last + j] - mean_x;
    dx[r * last + j] = (dout[r * last + j] - mean_dout - xj * gamma) * inv;
  }
}
__global__ void ns_embedding_grad_w_kernel(const float* __restrict__ dout,
                                           const float* __restrict__ idx,
                                           float* __restrict__ dW,
                                           int M, int emb_dim, int vocab_size) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < M * emb_dim) {
    int row = t / emb_dim;
    int col = t % emb_dim;
    int tok = (int)idx[row];
    if (tok >= 0 && tok < vocab_size)
      atomicAdd(&dW[tok * emb_dim + col], dout[t]);
  }
}
__global__ void ns_moe_grad_x_kernel(const float* __restrict__ dout, const float* __restrict__ x,
                                     const float* __restrict__ Wg, const float* __restrict__ We1,
                                     const float* __restrict__ We2, const uint8_t* __restrict__ active,
                                     float* __restrict__ dx, int M, int D, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float g[256], u[2048], s[2048], dpre[2048], dH[2048];
  int Ee = E < 256 ? E : 256;
  int He = H < 2048 ? H : 2048;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = -1.0e30f; continue; }
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    g[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = 0.f; continue; }
    g[e] = expf(g[e] - mx); sum += g[e];
  }
  int best = -1;
  for (int e = 0; e < Ee; e++)
    if (!(active && !active[e]) && (best < 0 || g[e] > g[best])) best = e;
  float pb = (sum > 0.f && best >= 0) ? g[best] / sum : 0.f;
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int k = 0; k < D; k++) acc += x[i * D + k] * We1[best * D * H + k * H + a];
    u[a] = acc;                                    // pre-activation
    s[a] = 0.5f * acc * (1.f + erff(acc * 0.7071067811865476f)); // gelu
  }
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int j = 0; j < D; j++) acc += dout[i * D + j] * We2[best * H * D + a * D + j];
    dpre[a] = acc;
  }
  float dpb = 0.f;
  for (int a = 0; a < He; a++) {
    float uu = u[a];
    dH[a] = dpre[a] * pb * (0.5f * (1.f + erff(uu * 0.7071067811865476f)) +
                            uu * expf(-uu * uu * 0.5f) * 0.3989422804014327f);
    dpb += s[a] * dpre[a];
  }
  float sd = dpb * pb;
  for (int k = 0; k < D; k++) {
    float gacc = 0.f, eacc = 0.f;
    for (int e = 0; e < Ee; e++) {
      float pe = (active && !active[e]) ? 0.f : g[e] / sum;
      float dg = pe * ((e == best ? dpb : 0.f) - sd);
      gacc += dg * Wg[k * E + e];
    }
    for (int a = 0; a < He; a++)
      eacc += dH[a] * We1[best * D * H + k * H + a];
    dx[i * D + k] = gacc + eacc;
  }
}
__global__ void ns_moe_grad_wg_kernel(const float* __restrict__ dout, const float* __restrict__ x,
                                      const float* __restrict__ Wg, const float* __restrict__ We1,
                                      const float* __restrict__ We2, const uint8_t* __restrict__ active,
                                      float* __restrict__ dWg, int M, int D, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float g[256], h[2048], dpre[2048];
  int Ee = E < 256 ? E : 256;
  int He = H < 2048 ? H : 2048;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = -1.0e30f; continue; }
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    g[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = 0.f; continue; }
    g[e] = expf(g[e] - mx); sum += g[e];
  }
  int best = -1;
  for (int e = 0; e < Ee; e++)
    if (!(active && !active[e]) && (best < 0 || g[e] > g[best])) best = e;
  float pb = (sum > 0.f && best >= 0) ? g[best] / sum : 0.f;
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int k = 0; k < D; k++) acc += x[i * D + k] * We1[best * D * H + k * H + a];
    h[a] = 0.5f * acc * (1.f + erff(acc * 0.7071067811865476f));
  }
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int j = 0; j < D; j++) acc += dout[i * D + j] * We2[best * H * D + a * D + j];
    dpre[a] = acc;
  }
  float dpb = 0.f;
  for (int a = 0; a < He; a++) dpb += h[a] * dpre[a];
  float sd = dpb * pb;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) continue;
    float pe = g[e] / sum;
    float dg = pe * ((e == best ? dpb : 0.f) - sd);
    for (int k = 0; k < D; k++) atomicAdd(&dWg[k * E + e], x[i * D + k] * dg);
  }
}
__global__ void ns_moe_grad_we1_kernel(const float* __restrict__ dout, const float* __restrict__ x,
                                       const float* __restrict__ Wg, const float* __restrict__ We1,
                                       const float* __restrict__ We2, const uint8_t* __restrict__ active,
                                        float* __restrict__ dWe1, int M, int D, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float g[256], dpre[2048];
  int Ee = E < 256 ? E : 256;
  int He = H < 2048 ? H : 2048;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = -1.0e30f; continue; }
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    g[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = 0.f; continue; }
    g[e] = expf(g[e] - mx); sum += g[e];
  }
  int best = -1;
  for (int e = 0; e < Ee; e++)
    if (!(active && !active[e]) && (best < 0 || g[e] > g[best])) best = e;
  float pb = (sum > 0.f && best >= 0) ? g[best] / sum : 0.f;
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int k = 0; k < D; k++) acc += x[i * D + k] * We1[best * D * H + k * H + a];
    float u = acc;
    acc = 0.f;
    for (int j = 0; j < D; j++) acc += dout[i * D + j] * We2[best * H * D + a * D + j];
    dpre[a] = acc;
    float dH = dpre[a] * pb * (0.5f * (1.f + erff(u * 0.7071067811865476f)) +
                               u * expf(-u * u * 0.5f) * 0.3989422804014327f);
    for (int k = 0; k < D; k++)
      atomicAdd(&dWe1[best * D * H + k * H + a], x[i * D + k] * dH);
  }
}
__global__ void ns_moe_grad_we2_kernel(const float* __restrict__ dout, const float* __restrict__ x,
                                       const float* __restrict__ Wg, const float* __restrict__ We1,
                                       const float* __restrict__ We2, const uint8_t* __restrict__ active,
                                        float* __restrict__ dWe2, int M, int D, int H, int E) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float g[256], h[2048];
  int Ee = E < 256 ? E : 256;
  int He = H < 2048 ? H : 2048;
  float mx = -1.0e30f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = -1.0e30f; continue; }
    float a = 0.f;
    for (int k = 0; k < D; k++) a += x[i * D + k] * Wg[k * E + e];
    g[e] = a;
    mx = fmaxf(mx, a);
  }
  float sum = 0.f;
  for (int e = 0; e < Ee; e++) {
    if (active && !active[e]) { g[e] = 0.f; continue; }
    g[e] = expf(g[e] - mx); sum += g[e];
  }
  int best = -1;
  for (int e = 0; e < Ee; e++)
    if (!(active && !active[e]) && (best < 0 || g[e] > g[best])) best = e;
  float pb = (sum > 0.f && best >= 0) ? g[best] / sum : 0.f;
  for (int a = 0; a < He; a++) {
    float acc = 0.f;
    for (int k = 0; k < D; k++) acc += x[i * D + k] * We1[best * D * H + k * H + a];
    h[a] = 0.5f * acc * (1.f + erff(acc * 0.7071067811865476f));
  }
  for (int a = 0; a < He; a++)
    for (int j = 0; j < D; j++)
      atomicAdd(&dWe2[best * H * D + a * D + j], h[a] * pb * dout[i * D + j]);
}
__global__ void ns_loss_grad_kernel(const float* __restrict__ preds, const float* __restrict__ labels,
                                    float* __restrict__ d, int numel, int C) {
  int rows = numel / C;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = preds[r * C];
  for (int j = 1; j < C; j++) mx = fmaxf(mx, preds[r * C + j]);
  float s = 0.f;
  for (int j = 0; j < C; j++) s += expf(preds[r * C + j] - mx);
  for (int j = 0; j < C; j++) {
    float p = expf(preds[r * C + j] - mx) / s;
    d[r * C + j] = (p - labels[r * C + j]) / (float)rows;
  }
}
__global__ void ns_ce_row_kernel(const float* __restrict__ preds, const float* __restrict__ labels,
                                 float* __restrict__ rowloss, int numel, int C) {
  int rows = numel / C;
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float mx = preds[r * C];
  for (int j = 1; j < C; j++) mx = fmaxf(mx, preds[r * C + j]);
  float s = 0.f;
  for (int j = 0; j < C; j++) s += expf(preds[r * C + j] - mx);
  float loss = 0.f;
  for (int j = 0; j < C; j++) {
    float p = expf(preds[r * C + j] - mx) / s;
    if (labels[r * C + j] > 0.f) loss -= logf(p);
  }
  rowloss[r] = loss;
}
__global__ void ns_adamw_kernel(float* __restrict__ w, const float* __restrict__ g,
                                float* __restrict__ am, float* __restrict__ av,
                                int n, float lr, float b1t, float b2t,
                                float beta1, float beta2, float omb1, float omb2,
                                float eps, float decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float gg = g[idx];
  float a = beta1 * am[idx] + omb1 * gg;
  float v = beta2 * av[idx] + omb2 * gg * gg;
  float mh = a / b1t, vh = v / b2t;
  am[idx] = a; av[idx] = v;
  w[idx] -= (lr * mh / (sqrtf(vh) + eps)) + decay * w[idx];
}
__global__ void ns_muon_ema_kernel(float* __restrict__ m, const float* __restrict__ g,
                                   int n, float momentum) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) m[idx] = momentum * m[idx] + g[idx];
}
__global__ void ns_muon_step_kernel(float* __restrict__ w, float* __restrict__ m,
                                    int n, float le, float decay) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { w[idx] -= le * m[idx] + decay * w[idx]; m[idx] = 0.f; }
}
// Modified Gram-Schmidt on G (columns if M>=N, rows otherwise), single block,
// thread 0 only: byte-for-byte mirrors the CPU-reference ns_orthonom, so the
// Muon update is bit-identical regardless of device.
__global__ void ns_orthonom_kernel(float* __restrict__ G, int M, int N) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  bool transposed = M < N;
  int R = transposed ? N : M;
  int C = transposed ? M : N;
  for (int j = 0; j < C; j++) {
    for (int i = 0; i < j; i++) {
      float dot = 0.f;
      for (int k = 0; k < R; k++) {
        float av = transposed ? G[i * R + k] : G[k * N + i];
        float bv = transposed ? G[j * R + k] : G[k * N + j];
        dot += av * bv;
      }
      for (int k = 0; k < R; k++) {
        float av = transposed ? G[i * R + k] : G[k * N + i];
        float v = transposed ? G[j * R + k] : G[k * N + j];
        if (transposed) G[j * R + k] = v - dot * av; else G[k * N + j] = v - dot * av;
      }
    }
    float nrm = 0.f;
    for (int k = 0; k < R; k++) { float v = transposed ? G[j * R + k] : G[k * N + j]; nrm += v * v; }
    nrm = sqrtf(nrm) + 1e-30f;
    for (int k = 0; k < R; k++) {
      if (transposed) G[j * R + k] /= nrm; else G[k * N + j] /= nrm;
    }
  }
}
// Inverted dropout forward (train only): draws per-element uniform samples via
// curand, keeps with probability (1 - rate), scales kept units by 1/(1 - rate).
// The scaled mask is stored so the backward pass can apply the SAME mask
// (dOut[i] = dIn[i] * mask[i]).
__global__ void ns_dropout_fwd_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      float* __restrict__ mask, int n, float rate, unsigned seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  curandState st;
  curand_init(seed, idx, 0, &st);
  float u = curand_uniform(&st);
  float keep = (u >= rate) ? (1.0f / (1.0f - rate)) : 0.0f;
  mask[idx] = keep;
  out[idx] = in[idx] * keep;
}
__global__ void ns_dropout_mul_kernel(const float* __restrict__ dIn, const float* __restrict__ mask,
                                      float* __restrict__ dOut, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dOut[idx] = dIn[idx] * mask[idx];
}
"#;

// Runtime utilities: launcher macros and a lazy grow-only device allocator.
pub const K_RUNTIME_UTILS: &str = r#"
static int ns_cu_reserve(float** pp, size_t* pc, size_t bytes, int zero) {
  if (!*pp || *pc < bytes) {
    if (*pp) cudaFree(*pp);
    if (bytes == 0) bytes = 1;
    if (cudaMalloc((void**)pp, bytes) != cudaSuccess) return -1;
    *pc = bytes;
    if (zero) cudaMemset(*pp, 0, bytes);
  }
  return 0;
}
"#;

// The byte-mask allocator is only used by MoE models.
pub const K_RUNTIME_UTILS_U8: &str = r#"
static int ns_cu_reserve_u8(uint8_t** pp, size_t* pc, size_t bytes, int zero) {
  if (!*pp || *pc < bytes) {
    if (*pp) cudaFree(*pp);
    if (bytes == 0) bytes = 1;
    if (cudaMalloc((void**)pp, bytes) != cudaSuccess) return -1;
    *pc = bytes;
    if (zero) cudaMemset(*pp, 0, bytes);
  }
  return 0;
}
"#;

pub fn device_helpers_source() -> String {
    format!("{}{}", K_LAUNCH_MACROS, K_DEVICE_HELPERS)
}
pub fn forward_kernels_source() -> &'static str {
    K_FORWARD_KERNELS
}
pub fn train_kernels_source() -> &'static str {
    K_TRAIN_KERNELS
}
pub fn runtime_utils_source() -> &'static str {
    K_RUNTIME_UTILS
}
pub fn runtime_utils_u8_source() -> &'static str {
    K_RUNTIME_UTILS_U8
}

/// Return the fp16 header snippet (conditional include for half precision).
pub fn fp16_header_source() -> &'static str {
    CUDA_FP16_HEADER
}
/// Return the ROCm header snippet.
pub fn rocm_header_source() -> &'static str {
    ROCM_HEADER
}
/// Return the Metal header snippet.
pub fn metal_header_source() -> &'static str {
    METAL_HEADER
}
