/* MiniAlexNet in C: training (forward + backward + SGD-momentum) and inference.
 *
 * Same network, same init weights, same data order and same file/log formats
 * as the Python implementations in ../python (see python/common.py).
 *
 * Two convolution back-ends, chosen at compile time:
 *   default      direct 6-deep loops, line-for-line identical to pure_python.py
 *   -DUSE_BLAS   im2col + cblas_sgemm (Apple Accelerate / OpenBLAS), the same
 *                strategy numpy_impl.py uses
 *
 * The Makefile builds three binaries from this one file:
 *   alexnet_O0    direct loops, no compiler optimisation
 *   alexnet_O3    direct loops, -O3 -mcpu=native (auto-vectorised NEON/SIMD)
 *   alexnet_blas  im2col + BLAS
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ---- network spec (must match python/common.py) ---- */
#define IMG 32
#define NCONV 5
#define NFC 3
#define NPARAM (2 * (NCONV + NFC))
typedef struct { int cin, cout, k, pad, pool; } ConvSpec;
typedef struct { int fin, fout, relu; } FcSpec;
static const ConvSpec CONVS[NCONV] = {
    {3, 16, 5, 2, 1}, {16, 32, 5, 2, 1}, {32, 48, 3, 1, 0}, {48, 48, 3, 1, 0}, {48, 32, 3, 1, 1}};
static const FcSpec FCS[NFC] = {{512, 128, 1}, {128, 128, 1}, {128, 10, 0}};

static float *P[NPARAM], *V[NPARAM], *G[NPARAM];
static long N[NPARAM];

/* ---- per-layer activation caches for one image ---- */
static int cH[NCONV], cW[NCONV];          /* spatial size at conv input */
static float *cXP[NCONV];                 /* padded input */
static float *cA[NCONV];                  /* post-ReLU output */
static int *cArg[NCONV];                  /* max-pool argmax */
static float *cPool[NCONV];               /* pooled output */
static float *fX[NFC], *fA[NFC];
#ifdef USE_BLAS
static float *cCols[NCONV];               /* im2col matrix [cin*k*k, H*W] */
#endif
static float *bufD1, *bufD2, *bufDXP;     /* gradient scratch */

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *xmalloc(size_t n) {
    void *p = calloc(1, n);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

static void setup(void) {
    int H = IMG, W = IMG;
    for (int l = 0; l < NCONV; l++) {
        const ConvSpec *s = &CONVS[l];
        N[2 * l] = (long)s->cout * s->cin * s->k * s->k;
        N[2 * l + 1] = s->cout;
        cH[l] = H; cW[l] = W;
        cXP[l] = xmalloc(sizeof(float) * s->cin * (H + 2 * s->pad) * (W + 2 * s->pad));
        cA[l] = xmalloc(sizeof(float) * s->cout * H * W);
#ifdef USE_BLAS
        cCols[l] = xmalloc(sizeof(float) * s->cin * s->k * s->k * H * W);
#endif
        if (s->pool) {
            cArg[l] = xmalloc(sizeof(int) * s->cout * (H / 2) * (W / 2));
            cPool[l] = xmalloc(sizeof(float) * s->cout * (H / 2) * (W / 2));
            H /= 2; W /= 2;
        }
    }
    for (int f = 0; f < NFC; f++) {
        N[2 * NCONV + 2 * f] = (long)FCS[f].fout * FCS[f].fin;
        N[2 * NCONV + 2 * f + 1] = FCS[f].fout;
        fX[f] = xmalloc(sizeof(float) * FCS[f].fin);
        fA[f] = xmalloc(sizeof(float) * FCS[f].fout);
    }
    for (int i = 0; i < NPARAM; i++) {
        P[i] = xmalloc(sizeof(float) * N[i]);
        V[i] = xmalloc(sizeof(float) * N[i]);
        G[i] = xmalloc(sizeof(float) * N[i]);
    }
    size_t big = 16 * IMG * IMG * 4;  /* >= any activation size */
    bufD1 = xmalloc(sizeof(float) * big);
    bufD2 = xmalloc(sizeof(float) * big);
    bufDXP = xmalloc(sizeof(float) * big * 2);
}

/* ---------------- convolution ---------------- */
static void pad_input(const float *x, float *xp, int C, int H, int W, int pad) {
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    memset(xp, 0, sizeof(float) * C * Hp * Wp);
    for (int c = 0; c < C; c++)
        for (int i = 0; i < H; i++)
            memcpy(&xp[(c * Hp + i + pad) * Wp + pad], &x[(c * H + i) * W], sizeof(float) * W);
}

static void unpad_grad(const float *dxp, float *dx, int C, int H, int W, int pad) {
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    for (int c = 0; c < C; c++)
        for (int i = 0; i < H; i++)
            memcpy(&dx[(c * H + i) * W], &dxp[(c * Hp + i + pad) * Wp + pad], sizeof(float) * W);
}

#ifndef USE_BLAS
/* Direct convolution: the same loop nest as pure_python.py. */
static void conv_forward(int l, const float *x) {
    const ConvSpec *s = &CONVS[l];
    int C = s->cin, H = cH[l], W = cW[l], k = s->k, pad = s->pad;
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    const float *w = P[2 * l], *b = P[2 * l + 1];
    float *xp = cXP[l], *y = cA[l];
    pad_input(x, xp, C, H, W, pad);
    for (int co = 0; co < s->cout; co++) {
        const float *wc = w + (long)co * C * k * k;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                float acc = b[co];
                for (int ci = 0; ci < C; ci++)
                    for (int ki = 0; ki < k; ki++) {
                        const float *xr = &xp[(ci * Hp + i + ki) * Wp + j];
                        const float *wr = &wc[(ci * k + ki) * k];
                        for (int kj = 0; kj < k; kj++) acc += wr[kj] * xr[kj];
                    }
                y[(co * H + i) * W + j] = acc > 0.f ? acc : 0.f;
            }
    }
}

static void conv_backward(int l, const float *dy, float *dx) {
    const ConvSpec *s = &CONVS[l];
    int C = s->cin, H = cH[l], W = cW[l], k = s->k, pad = s->pad;
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    const float *w = P[2 * l], *a = cA[l], *xp = cXP[l];
    float *dw = G[2 * l], *db = G[2 * l + 1];
    int need_dx = dx != NULL;
    if (need_dx) memset(bufDXP, 0, sizeof(float) * C * Hp * Wp);
    for (int co = 0; co < s->cout; co++) {
        const float *wc = w + (long)co * C * k * k;
        float *dwc = dw + (long)co * C * k * k;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                int idx = (co * H + i) * W + j;
                if (a[idx] <= 0.f) continue;
                float g = dy[idx];
                db[co] += g;
                for (int ci = 0; ci < C; ci++)
                    for (int ki = 0; ki < k; ki++) {
                        int xo = (ci * Hp + i + ki) * Wp + j;
                        int wo = (ci * k + ki) * k;
                        for (int kj = 0; kj < k; kj++) {
                            dwc[wo + kj] += g * xp[xo + kj];
                            if (need_dx) bufDXP[xo + kj] += g * wc[wo + kj];
                        }
                    }
            }
    }
    if (need_dx) unpad_grad(bufDXP, dx, C, H, W, pad);
}

#else
/* im2col + GEMM: turn the convolution into one matrix multiply. */
static void im2col(const float *xp, float *cols, int C, int H, int W, int k, int pad) {
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    for (int c = 0; c < C; c++)
        for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++) {
                float *row = &cols[((c * k + ki) * k + kj) * H * W];
                for (int i = 0; i < H; i++)
                    memcpy(&row[i * W], &xp[(c * Hp + i + ki) * Wp + kj], sizeof(float) * W);
            }
}

static void conv_forward(int l, const float *x) {
    const ConvSpec *s = &CONVS[l];
    int C = s->cin, H = cH[l], W = cW[l], k = s->k, pad = s->pad, HW = H * W, K = C * k * k;
    float *y = cA[l];
    const float *b = P[2 * l + 1];
    pad_input(x, cXP[l], C, H, W, pad);
    im2col(cXP[l], cCols[l], C, H, W, k, pad);
    for (int co = 0; co < s->cout; co++)
        for (int p = 0; p < HW; p++) y[co * HW + p] = b[co];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, s->cout, HW, K,
                1.f, P[2 * l], K, cCols[l], HW, 1.f, y, HW);
    for (int i = 0; i < s->cout * HW; i++) y[i] = y[i] > 0.f ? y[i] : 0.f;
}

static void conv_backward(int l, const float *dy_in, float *dx) {
    const ConvSpec *s = &CONVS[l];
    int C = s->cin, H = cH[l], W = cW[l], k = s->k, pad = s->pad, HW = H * W, K = C * k * k;
    int Hp = H + 2 * pad, Wp = W + 2 * pad;
    const float *a = cA[l];
    float *dy = bufD2;  /* ReLU-masked gradient */
    for (int co = 0; co < s->cout; co++) {
        float sum = 0.f;
        for (int p = 0; p < HW; p++) {
            int i = co * HW + p;
            float g = a[i] > 0.f ? dy_in[i] : 0.f;
            dy[i] = g; sum += g;
        }
        G[2 * l + 1][co] += sum;
    }
    /* dW += dy[cout,HW] * cols^T[HW,K] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, s->cout, K, HW,
                1.f, dy, HW, cCols[l], HW, 1.f, G[2 * l], K);
    if (!dx) return;
    /* dcols[K,HW] = W^T[K,cout] * dy[cout,HW] ; then col2im */
    float *dcols = cCols[l];  /* cols no longer needed: reuse */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, HW, s->cout,
                1.f, P[2 * l], K, dy, HW, 0.f, dcols, HW);
    memset(bufDXP, 0, sizeof(float) * C * Hp * Wp);
    for (int c = 0; c < C; c++)
        for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++) {
                const float *row = &dcols[((c * k + ki) * k + kj) * HW];
                for (int i = 0; i < H; i++) {
                    float *dst = &bufDXP[(c * Hp + i + ki) * Wp + kj];
                    for (int j = 0; j < W; j++) dst[j] += row[i * W + j];
                }
            }
    unpad_grad(bufDXP, dx, C, H, W, pad);
}
#endif

/* ---------------- pooling / fully connected ---------------- */
static void pool_forward(int l) {
    int C = CONVS[l].cout, H = cH[l], W = cW[l], Ho = H / 2, Wo = W / 2;
    const float *x = cA[l];
    for (int c = 0; c < C; c++)
        for (int i = 0; i < Ho; i++)
            for (int j = 0; j < Wo; j++) {
                float best = -1e30f; int bi = 0;
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++) {
                        int idx = (c * H + 2 * i + di) * W + 2 * j + dj;
                        if (x[idx] > best) { best = x[idx]; bi = idx; }
                    }
                int o = (c * Ho + i) * Wo + j;
                cPool[l][o] = best; cArg[l][o] = bi;
            }
}

static void fc_forward(int f, const float *x) {
    const FcSpec *s = &FCS[f];
    const float *w = P[2 * NCONV + 2 * f], *b = P[2 * NCONV + 2 * f + 1];
    memcpy(fX[f], x, sizeof(float) * s->fin);
    for (int o = 0; o < s->fout; o++) {
        float acc = b[o];
        const float *wr = w + (long)o * s->fin;
        for (int i = 0; i < s->fin; i++) acc += wr[i] * x[i];
        fA[f][o] = s->relu ? (acc > 0.f ? acc : 0.f) : acc;
    }
}

static void fc_backward(int f, const float *dy, float *dx) {
    const FcSpec *s = &FCS[f];
    const float *w = P[2 * NCONV + 2 * f], *x = fX[f];
    float *dw = G[2 * NCONV + 2 * f], *db = G[2 * NCONV + 2 * f + 1];
    memset(dx, 0, sizeof(float) * s->fin);
    for (int o = 0; o < s->fout; o++) {
        if (s->relu && fA[f][o] <= 0.f) continue;
        float g = dy[o];
        db[o] += g;
        float *dwr = dw + (long)o * s->fin;
        const float *wr = w + (long)o * s->fin;
        for (int i = 0; i < s->fin; i++) { dwr[i] += g * x[i]; dx[i] += g * wr[i]; }
    }
}

static const float *forward(const float *img) {
    const float *h = img;
    for (int l = 0; l < NCONV; l++) {
        conv_forward(l, h);
        h = cA[l];
        if (CONVS[l].pool) { pool_forward(l); h = cPool[l]; }
    }
    for (int f = 0; f < NFC; f++) { fc_forward(f, h); h = fA[f]; }
    return h;
}

/* Forward + backward for one image, accumulating into G. Returns its loss. */
static float backprop(const float *img, int label, int B) {
    const float *logits = forward(img);
    float m = logits[0], e[10], sum = 0.f;
    for (int i = 1; i < 10; i++) if (logits[i] > m) m = logits[i];
    for (int i = 0; i < 10; i++) { e[i] = expf(logits[i] - m); sum += e[i]; }
    float *d = bufD1, *dn = bufD2;
    for (int i = 0; i < 10; i++) d[i] = e[i] / sum / B;
    float loss = -logf(e[label] / sum + 1e-12f);
    d[label] -= 1.f / B;

    for (int f = NFC - 1; f >= 0; f--) {
        fc_backward(f, d, dn);
        float *t = d; d = dn; dn = t;
    }
    for (int l = NCONV - 1; l >= 0; l--) {
        const ConvSpec *s = &CONVS[l];
        if (s->pool) {
            int n_in = s->cout * cH[l] * cW[l], n_out = n_in / 4;
            memset(dn, 0, sizeof(float) * n_in);
            for (int o = 0; o < n_out; o++) dn[cArg[l][o]] += d[o];
            float *t = d; d = dn; dn = t;
        }
#ifdef USE_BLAS
        /* the BLAS backward uses bufD2 as scratch, so stage dy in bufD1 */
        if (d != bufD1) { memcpy(bufD1, d, sizeof(float) * s->cout * cH[l] * cW[l]); d = bufD1; }
        float *dx = l > 0 ? bufD1 : NULL;  /* safe: dy fully consumed into bufD2 first */
        conv_backward(l, d, dx);
        d = bufD1; dn = bufD2;
#else
        conv_backward(l, d, l > 0 ? dn : NULL);
        float *t = d; d = dn; dn = t;
#endif
    }
    return loss;
}

static float train_step(const float *xs, const int *ys, int B, float lr, float mu) {
    for (int i = 0; i < NPARAM; i++) memset(G[i], 0, sizeof(float) * N[i]);
    float total = 0.f;
    for (int b = 0; b < B; b++) total += backprop(xs + (long)b * 3 * IMG * IMG, ys[b], B);
    for (int i = 0; i < NPARAM; i++)
        for (long j = 0; j < N[i]; j++) {
            V[i][j] = mu * V[i][j] + G[i][j];
            P[i][j] -= lr * V[i][j];
        }
    return total / B;
}

/* ---------------- I/O ---------------- */
static void *read_file(const char *path, long *bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *bytes = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = xmalloc(*bytes);
    if (fread(buf, 1, *bytes, f) != (size_t)*bytes) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return buf;
}

static void load_params(const char *path) {
    long bytes; float *flat = read_file(path, &bytes), *p = flat;
    for (int i = 0; i < NPARAM; i++) { memcpy(P[i], p, sizeof(float) * N[i]); p += N[i]; }
    if ((p - flat) * 4 != bytes) { fprintf(stderr, "weight file size mismatch\n"); exit(1); }
    free(flat);
}

static void save_params(const char *path) {
    FILE *f = fopen(path, "wb");
    for (int i = 0; i < NPARAM; i++) fwrite(P[i], sizeof(float), N[i], f);
    fclose(f);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *mode = "train", *name = NULL, *root = ".";
    double budget = 60; int steps = 3, B = 16, nsnaps = 20; float lr = 0.005f, mu = 0.9f;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--mode")) mode = argv[i + 1];
        else if (!strcmp(argv[i], "--budget")) budget = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--batch")) B = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--lr")) lr = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--momentum")) mu = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--snaps")) nsnaps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--name")) name = argv[i + 1];
        else if (!strcmp(argv[i], "--root")) root = argv[i + 1];
    }
    if (!name) { fprintf(stderr, "--name required\n"); return 1; }

    setup();
    char path[1024], outdir[1024];
    snprintf(path, sizeof path, "%s/data/init_weights.bin", root);
    load_params(path);
    long bytes;
    snprintf(path, sizeof path, "%s/data/train_x.bin", root);
    float *train_x = read_file(path, &bytes);
    int n_train = bytes / (sizeof(float) * 3 * IMG * IMG);
    snprintf(path, sizeof path, "%s/data/train_y.bin", root);
    int *train_y = read_file(path, &bytes);
    snprintf(path, sizeof path, "%s/data/test_x.bin", root);
    float *test_x = read_file(path, &bytes);
    int n_test = bytes / (sizeof(float) * 3 * IMG * IMG);

    snprintf(outdir, sizeof outdir, "%s/results", root); mkdir(outdir, 0755);
    snprintf(outdir, sizeof outdir, "%s/results/%s", root, name); mkdir(outdir, 0755);
    const long IMGSZ = 3 * IMG * IMG;

    if (!strcmp(mode, "steps")) {
        snprintf(path, sizeof path, "%s/parity.json", outdir);
        FILE *f = fopen(path, "w");
        fprintf(f, "{\"batch\": %d, \"losses\": [", B);
        printf("{\"name\": \"%s\", \"losses\": [", name);
        for (int s = 0; s < steps; s++) {
            float loss = train_step(train_x + (long)s * B * IMGSZ, train_y + s * B, B, lr, mu);
            fprintf(f, "%s%.9g", s ? ", " : "", loss);
            printf("%s%.9g", s ? ", " : "", loss);
        }
        fprintf(f, "]}"); fclose(f);
        printf("]}\n");
        return 0;
    }

    if (!strcmp(mode, "infer")) {
        int nx = n_test < 200 ? n_test : 200, cap = 20000, n = 0;
        double *times = xmalloc(sizeof(double) * cap), sum = 0;
        forward(test_x);  /* warm-up */
        double t_start = now();
        while (now() - t_start < budget && n < cap) {
            double t0 = now();
            const float *out = forward(test_x + (long)(n % nx) * IMGSZ);
            times[n] = now() - t0;
            volatile float sink = out[0]; (void)sink;
            sum += times[n++];
        }
        qsort(times, n, sizeof(double), cmp_double);
        snprintf(path, sizeof path, "%s/infer.json", outdir);
        FILE *f = fopen(path, "w");
        fprintf(f, "{\"name\": \"%s\", \"n\": %d, \"median_ms\": %.6f, \"mean_ms\": %.6f}",
                name, n, times[n / 2] * 1e3, sum / n * 1e3);
        fclose(f);
        printf("{\"name\": \"%s\", \"n\": %d, \"median_ms\": %.6f, \"mean_ms\": %.6f}\n",
               name, n, times[n / 2] * 1e3, sum / n * 1e3);
        return 0;
    }

    /* ---- timed training race (same protocol as python/common.py:run) ---- */
    int nb = n_train / B, step = 0, snap_idx = 0;
    double elapsed = 0, snap_every = budget / nsnaps, next_snap = 0;
    snprintf(path, sizeof path, "%s/train_log.csv", outdir);
    FILE *log = fopen(path, "w");
    fprintf(log, "step,elapsed,images,loss\n");
    snprintf(path, sizeof path, "%s/snaps.csv", outdir);
    FILE *snaps = fopen(path, "w");
    fprintf(snaps, "snap,elapsed,step,images\n");
#define SNAPSHOT() do { \
        snprintf(path, sizeof path, "%s/snap_%03d.bin", outdir, snap_idx); save_params(path); \
        fprintf(snaps, "%d,%.6f,%d,%d\n", snap_idx, elapsed, step, step * B); fflush(snaps); \
        snap_idx++; } while (0)

    while (elapsed < budget) {
        if (elapsed >= next_snap) { SNAPSHOT(); next_snap += snap_every; }
        int b = step % nb;
        double t0 = now();
        float loss = train_step(train_x + (long)b * B * IMGSZ, train_y + b * B, B, lr, mu);
        elapsed += now() - t0;
        step++;
        fprintf(log, "%d,%.6f,%d,%.6f\n", step, elapsed, step * B, loss);
        if (step == 1 || step % 200 == 0)
            fprintf(stderr, "[%s] step %6d  t=%7.2fs  loss=%.4f\n", name, step, elapsed, loss);
    }
    SNAPSHOT();
    fclose(log); fclose(snaps);
    printf("{\"name\": \"%s\", \"steps\": %d, \"elapsed\": %f, \"images_per_sec\": %f}\n",
           name, step, elapsed, step * B / elapsed);
    return 0;
}
