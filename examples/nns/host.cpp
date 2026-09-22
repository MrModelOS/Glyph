// Host driver for the NeuMoE/Static-MoE/Dense MX450 study.
//
// Build sequence (run.sh):
//   nsc  examples/neumoe/<net>.ns --cuda --runtime > <net>_driver.cu
//   nvcc <net>_driver.cu host.cpp -I <repo> -o <net>_driver   (+ -arch=sm_75)
//
// The driver is written generically against the ns_runtime.h C-ABI and the
// generated ns_weight_layout, so the SAME host.cpp drives NeuMoE (growing
// lifecycle via ns_expert_birth), Static-MoE (K=9 from step 0) and Dense
// (no MoE layer). Config is passed through argv:
//   <net>_driver <data_dir> <steps> <out_dir> [<eval_every>]
//
// Corpus format (gen_corpus.py): u32 magic 'NNM1', u32 n, u32 seq_len, then
// n*seq_len u16 tokens row-major.
//
// Metrics CSV (per config):
//   step,train_loss,val_prose,val_code,val_qa,val_mean,k,active_params,
//   total_params,tok_s

#include "ns/runtime/ns_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// corpus
// ---------------------------------------------------------------------------
struct Corpus {
    std::vector<uint16_t> seqs;  // n * seq_len tokens, row-major
    uint32_t n = 0;
    uint32_t seq_len = 0;
};

static Corpus load_corpus(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { std::printf("FATAL: cannot open %s\n", path.c_str()); std::exit(1); }
    uint32_t magic, n, seq_len;
    if (fread(&magic, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 || fread(&seq_len, 4, 1, f) != 1 ||
        magic != 0x4E4E4D31) {
        std::printf("FATAL: bad header in %s\n", path.c_str());
        std::exit(1);
    }
    Corpus c;
    c.n = n;
    c.seq_len = seq_len;
    c.seqs.resize((size_t)n * seq_len);
    size_t got = fread(c.seqs.data(), 2, c.seqs.size(), f);
    fclose(f);
    if (got != c.seqs.size()) { std::printf("FATAL: truncated %s\n", path.c_str()); std::exit(1); }
    return c;
}

// ---------------------------------------------------------------------------
// random init over the generated weight layout
// ---------------------------------------------------------------------------
static void init_weights(std::vector<float>& w, const ns_weight_layout* lay) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    for (size_t i = 0; i < lay->num_weights; i++) {
        const ns_weight_desc& d = lay->desc[i];
        const size_t off = d.offset, cnt = d.count;
        const std::string nm = d.name;
        float* wp = w.data() + off;
        if (nm == "emb_w" || nm == "head_w") {
            for (size_t j = 0; j < cnt; j++) wp[j] = nd(rng);          // ~N(0,0.02)
        } else if (nm.find("_attn_") != std::string::npos) {
            const float scale = 0.88f * 0.02f;                          // torch ref 0.02 init
            for (size_t j = 0; j < cnt; j++) wp[j] = nd(rng) * scale;
        } else if (nm == "moe_g_w") {
            for (size_t j = 0; j < cnt; j++) wp[j] = nd(rng);          // router weight
        } else {
            // Dense FFN (f1/f2, bf1/bf2, df1/df2): Xavier from fan_in.
            // fan_in is not exposed; approximate by sqrt(mean element) of the
            // count: for z->FF 512->N, count = N; in-col = 512 (z). For
            // FF->z, count = z. Use shape approx: in = (size_t)round(sqrt(cnt
            // * (512*512<squared? ... ))). Instead: derive from common layouts:
            // f1/bf1/df1 in=512, f2/bf2/df2 in=cnt/512shape. We approximate
            // doubly by assuming the small dim is always 512 for these layers.
            size_t in = (nm[0] == 'f' && nm[1] == 'f') || nm == "bf1_w" || nm == "df1_w"
                        ? 512 : (cnt / 512);
            if (cnt == 512UL*2048 || cnt == 512UL*20484) in = 512;    // widen f1
            const float limit = std::sqrt(6.f / (float)(in + cnt / in));
            for (size_t j = 0; j < cnt; j++) wp[j] = (ud(rng) * 2.f - 1.f) * limit;
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
static float lr_at(int step, int total_steps, float max_lr) {
    const int warmup = std::min(200, total_steps / 10);
    if (step < warmup) return max_lr * (float)step / (float)warmup;
    float t = (float)(step - warmup) / (float)std::max(1, total_steps - warmup);
    return max_lr * 0.5f * (1.f + std::cos((float)M_PI * std::min(t, 1.f)));
}

static double time_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <data_dir> <steps> <out_dir> [eval_every]\n", argv[0]);
        return 2;
    }
    const std::string data_dir = argv[1];
    const int total_steps = atoi(argv[2]);
    const std::string out_dir = argv[3];
    const int eval_every = argc >= 5 ? atoi(argv[4]) : 50;
    const float max_lr = 3e-4f;
    const float weight_decay = 0.01f;  // informational: AOT optimizer carries it

    const size_t nw = ns_weight_count_static();
    const size_t k_max = total_steps > 0;
    (void)weight_decay; (void)k_max;
    std::printf("weight_count=%zu (%.2fM floats, %.1f MB)\n", nw, nw / 1e6f, nw * 4.f / 1e6f);

    std::vector<float> w(nw);
    init_weights(w, nullptr);  // layout available only through a model; see below

    // The layout needs a model. We instead init from the FIRST model, then
    // re-roll against the layout descs (cheap, done once).
    // Simplest correct path: build a temporary model to fetch the layout.
    ns_model* tmp = ns_runtime_init(w.data(), nw);
    if (!tmp) { std::printf("FATAL: model init\n"); return 1; }
    const ns_weight_layout* lay = ns_model_layout(tmp);
    init_weights(w, lay);
    ns_free(tmp);

    // Copy the rolled blob into the real model.
    ns_model* m = ns_runtime_init(w.data(), nw);
    if (!m) { std::printf("FATAL: model init (2)\n"); return 1; }

    const size_t K0 = ns_expert_count(m);
    const size_t d_exp = 2UL * 512 * 2048;                       // expert ffn pair
    std::printf("experts_initial=%zu total=%.2fM active(K=%zu)=%.2fM\n",
                K0, nw / 1e6f, K0, (nw - (size_t)(nw > d_exp ? (nw - 0) / nw : 0)) / 1e6f);

    // load corpus
    Corpus tr = load_corpus(data_dir + "/train.bin");
    Corpus vp = load_corpus(data_dir + "/val_prose.bin");
    Corpus vc = load_corpus(data_dir + "/val_code.bin");
    Corpus vq = load_corpus(data_dir + "/val_qa.bin");
    const int L = (int)tr.seq_len;
    const int NV = (int)vp.n;
    std::printf("train=%u x %u  val=512/dom  S=%d\n", tr.n, tr.seq_len, L - 1);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, (int)tr.n - 1);

    // buffers
    std::vector<float> xs(L - 1);
    std::vector<float> ys((size_t)(L - 1) * 10240);
    std::vector<float> out((size_t)(L - 1) * 10240);

    const int maxK = 9;
    float best = 1e30f;
    int best_step = -1;
    double t0 = time_s(), t_last = t0;
    double toks = 0.0;

    char csv_path[512];
    std::snprintf(csv_path, sizeof(csv_path), "%s/metrics.csv", out_dir.c_str());
    std::FILE* csv = fopen(csv_path, "w");
    if (!csv) { std::printf("FATAL: cannot write %s\n", csv_path); return 1; }
    std::fprintf(csv, "step,train_loss,val_prose,val_code,val_qa,val_mean,k,active_params,total_params,tok_s\n");

    for (int step = 0; step < total_steps; step++) {
        int r = pick(rng);
        const uint16_t* seq = tr.seqs.data() + (size_t)r * L;
        for (int i = 0; i < L - 1; i++) xs[i] = (float)seq[i];          // src = seq[0..L-2]
        std::memset(ys.data(), 0, sizeof(float) * ys.size());
        for (int i = 1; i < L; i++) ys[(size_t)(i - 1) * 10240 + seq[i]] = 1.f;  // tgt = seq[1..L-1]

        float loss = 0.f;
        const float lr = lr_at(step, total_steps, max_lr);
        if (ns_runtime_train_step(m, xs.data(), ys.data(), (size_t)(L - 1), &loss, lr) != 0) {
            std::printf("FATAL: train_step %d\n", step);
            return 1;
        }
        toks += (double)(L - 1);

        // linear birth schedule: +1 expert every 50 steps until k_max=9
        if (step > 0 && step % 50 == 0) {
            const size_t k = ns_expert_count(m);
            if (k < maxK) ns_expert_birth(m, 1);
        }

        if (step % eval_every == 0 || step == total_steps - 1) {
            // validation: mean CE + accuracy per domain, single sequence at a time
            double losses[3] = {0, 0, 0}, accs[3] = {0, 0, 0};
            int nacc[3] = {0, 0, 0};
            const Corpus* vals[3] = {&vp, &vc, &vq};
            for (int d = 0; d < 3; d++) {
                const Corpus& v = *vals[d];
                for (int s = 0; s < NV; s++) {
                    const uint16_t* sq = v.seqs.data() + (size_t)s * L;
                    for (int i = 0; i < L - 1; i++) xs[i] = (float)sq[i];
                    std::memset(ys.data(), 0, sizeof(float) * ys.size());
                    for (int i = 1; i < L; i++) ys[(size_t)(i - 1) * 10240 + sq[i]] = 1.f;
                    float l = 0.f;
                    if (ns_objective_loss(m, xs.data(), ys.data(), (size_t)(L - 1), &l) != 0)
                        { std::printf("FATAL: objective\n"); return 1; }
                    losses[d] += l;
                    if (ns_eval_infer(m, xs.data(), out.data(), (size_t)(L - 1)) != 0)
                        { std::printf("FATAL: eval_infer\n"); return 1; }
                    int corr = 0;
                    for (int i = 0; i < L - 1; i++) {
                        int a = 0;
                        for (int t = 1; t < 10240; t++)
                            if (out[(size_t)i * 10240 + t] > out[(size_t)i * 10240 + a]) a = t;
                        if (a == (int)sq[i + 1]) corr++;
                    }
                    accs[d] += (double)corr / (L - 1);
                    nacc[d]++;
                }
                losses[d] /= NV;
                accs[d] /= NV;
            }
            const float vmean = (float)((losses[0] + losses[1] + losses[2]) / 3.0);
            const size_t k = ns_expert_count(m);
            const size_t active = nw - (k < maxK ? (size_t)(maxK - k) * (2UL * 512 * 2048) : 0UL);
            const double dt = time_s() - t_last;
            t_last = time_s();
            const float tok_s = (float)(toks / dt);
            toks = 0;
            std::printf("[EVAL] step=%d loss=%.4f val=%.4f (p %.4f c %.4f q %.4f) "
                        "acc=%.3f/%.3f/%.3f K=%zu active=%.2fM %.1f tok/s\n",
                        step, loss, vmean, losses[0], losses[1], losses[2],
                        accs[0], accs[1], accs[2], k, active / 1e6f, tok_s);
            std::fflush(stdout);
            std::fprintf(csv, "%d,%.4f,%.4f,%.4f,%.4f,%.4f,%zu,%.0f,%.0f,%.2f\n",
                         step, loss, losses[0], losses[1], losses[2], vmean,
                         k, (double)active, (double)nw, tok_s);
            std::fflush(csv);
            if (vmean < best) { best = vmean; best_step = step; }
        }
        if (step % 10 == 0)
            std::printf("[%s] step=%d loss=%.4f lr=%.1e t+%.0fs\n",
                        out_dir.c_str(), step, loss, lr, time_s() - t0);
    }

    std::fclose(csv);
    std::printf("DONE %s: best_val=%.4f @ step %d  total_time=%.1fs  K=%zu\n",
                out_dir.c_str(), best, best_step, time_s() - t0, ns_expert_count(m));

    char ck[512];
    std::snprintf(ck, sizeof(ck), "%s/model.nsm2", out_dir.c_str());
    ns_save_checkpoint(m, ck);
    ns_free(m);
    return 0;
}