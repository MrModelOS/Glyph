#include "ns/runtime/ns_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

struct Corpus {
    std::vector<uint16_t> seqs;
    uint32_t n = 0, seq_len = 0;
};

static Corpus load(const char* path) {
    FILE* f = fopen(path, "rb");
    Corpus c;
    if (!f) { std::printf("FATAL: open %s\n", path); return c; }
    uint32_t magic, n, sl;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x4E4E4D31) {
        std::printf("FATAL: bad magic in %s\n", path); fclose(f); return c;
    }
    fread(&n, 4, 1, f); fread(&sl, 4, 1, f);
    c.n = n; c.seq_len = sl;
    c.seqs.resize((size_t)n * sl);
    fread(c.seqs.data(), 2, (size_t)n * sl, f);
    fclose(f);
    return c;
}

static size_t count_nonfinite(const std::vector<float>& w, size_t off, size_t n) {
    size_t c = 0;
    for (size_t i = off; i < off + n; i++) if (!std::isfinite(w[i])) c++;
    return c;
}

int main(int argc, char** argv) {
    const char* data_dir = argc > 1 ? argv[1] : "data";
    std::string d (data_dir);

    std::vector<float> w(ns_weight_count_static());
    ns_model* m = ns_runtime_init(w.data(), w.size());
    if (!m) { std::printf("FATAL: init\n"); return 1; }

    const size_t nw = ns_model_weight_count(m);
    const ns_weight_layout* L = ns_model_layout(m);
    std::printf("nw=%zu desc=%zu\n", nw, L->num_weights);

    Corpus tr = load((d + "/train.bin").c_str());
    Corpus vp = load((d + "/val_prose.bin").c_str());
    int Ls = (int)tr.seq_len;
    std::vector<float> xs(Ls - 1), ys((size_t)(Ls - 1) * 10240), out((size_t)(Ls - 1) * 10240);

    // first training sequence
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, (int)tr.n - 1);
    int r = pick(rng);
    const uint16_t* seq = tr.seqs.data() + (size_t)r * Ls;
    for (int i = 0; i < Ls - 1; i++) xs[i] = (float)seq[i];
    std::memset(ys.data(), 0, sizeof(float) * ys.size());
    for (int i = 1; i < Ls; i++) ys[(size_t)(i - 1) * 10240 + seq[i]] = 1.f;

    float l0 = -1.f;
    ns_objective_loss(m, xs.data(), ys.data(), (size_t)(Ls - 1), &l0);
    std::printf("PRE-TRAIN objective loss = %.6f\n", l0);

    float l1 = -1.f, l2 = -1.f, l3 = -1.f;
    ns_runtime_train_step(m, xs.data(), ys.data(), (size_t)(Ls - 1), &l1, 3e-4f);
    ns_runtime_train_step(m, xs.data(), ys.data(), (size_t)(Ls - 1), &l2, 3e-4f);
    ns_runtime_train_step(m, xs.data(), ys.data(), (size_t)(Ls - 1), &l3, 3e-4f);
    std::printf("TRAIN step losses = %.6f %.6f %.6f\n", l1, l2, l3);

    std::vector<float> wd(nw);
    ns_model_get_weights(m, wd.data(), nw);
    size_t all = 0;
    for (size_t i = 0; i < nw; i++) if (!std::isfinite(wd[i])) all++;
    std::printf("nonfinite(ALL)=%zu / %zu\n", all, nw);
    // per major region
    const char* names[53];
    for (size_t i = 0; i < L->num_weights; i++) names[i] = L->desc[i].name;
    size_t off = 0;
    for (size_t i = 0; i < L->num_weights; i++) {
        size_t n = L->desc[i].num;
        size_t c = count_nonfinite(wd, off, n);
        if (c) std::printf("  [%02zu] %s off=%zu n=%zu nonfinite=%zu\n",
                           i, names[i] ? names[i] : "?", off, n, c);
        off += n;
    }

    ns_free(m);
    return 0;
}