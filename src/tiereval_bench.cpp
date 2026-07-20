// tiereval_bench.cpp — Benchmark TierEval (composed-weight fast eval) vs full NNUE eval.
// Tests: (1) speed of fast path, (2) correlation with full eval.
// Compile: g++ -O3 -mavx512f -mavx512bw -mavx512vnni -o tiereval tiereval_bench.cpp
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <x86intrin.h>

#define L1 512
#define L2 16
#define L3 32
#define ITER 5000000  // 5M evals

// ---- Network data (random, realistic shapes) ----
static float acc[2][L1];                    // float accumulator (current Luminex)
static int8_t  l2_w[L2][2*L1];             // L2 weights (16 x 1024)
static float   l2_b[L2];                    // L2 bias
static int8_t  l3_w[L3][L2];               // L3 weights (32 x 16)
static float   l3_b[L3];
static int8_t  out_w[L3];                   // out weights (1 x 32)
static float   out_b;

// ---- TierEval: composed weights (1024 → 1) ----
// W_composed = out_w × L3_w × L2_w  (compose the 3 linear layers)
// b_composed = out_b + out_w × (l3_b + L3_w × l2_b)
static float composed_w[2*L1];              // 1024 composed weights
static float composed_b;                    // composed bias

// ---- SCReLU activation ----
static inline float screlu(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x * x;
}

// ---- Full NNUE eval (SCReLU + L2 + SCReLU + L3 + SCReLU + out) ----
float full_eval() {
    // Concatenate perspectives (stm first)
    float x[2*L1];
    for (int i = 0; i < L1; i++) x[i] = screlu(acc[0][i]);
    for (int i = 0; i < L1; i++) x[L1+i] = screlu(acc[1][i]);

    // L2: 1024 → 16
    float y2[L2];
    for (int o = 0; o < L2; o++) {
        float s = l2_b[o];
        for (int i = 0; i < 2*L1; i++) s += l2_w[o][i] * x[i];
        y2[o] = screlu(s);
    }
    // L3: 16 → 32
    float y3[L3];
    for (int o = 0; o < L3; o++) {
        float s = l3_b[o];
        for (int i = 0; i < L2; i++) s += l3_w[o][i] * y2[i];
        y3[o] = screlu(s);
    }
    // out: 32 → 1
    float result = out_b;
    for (int i = 0; i < L3; i++) result += out_w[i] * y3[i];
    return result;
}

// ---- TierEval fast path: ReLU(acc) · composed_w + composed_b ----
float fast_eval() {
    float s = composed_b;
    // ReLU + dot for perspective 0
    for (int i = 0; i < L1; i++) {
        float v = acc[0][i];
        if (v > 0) s += v * composed_w[i];
    }
    // ReLU + dot for perspective 1
    for (int i = 0; i < L1; i++) {
        float v = acc[1][i];
        if (v > 0) s += v * composed_w[L1 + i];
    }
    return s;
}

// ---- TierEval with SIMD (VPDPBUSD on int8) ----
// Quantize acc to uint8 [0,127], then VPDPBUSD with int8 composed weights
static uint8_t acc_q[2*L1];                 // quantized activations
static int8_t  composed_w_i8[2*L1];         // int8 composed weights

float fast_eval_simd() {
    // Quantize accumulator: clip to [0, 127], convert to uint8
    // (approximate SCReLU with just ReLU+quant — no square)
    for (int i = 0; i < 2*L1; i++) {
        float v = (i < L1) ? acc[0][i] : acc[1][i-L1];
        if (v < 0) v = 0;
        if (v > 127) v = 127;
        acc_q[i] = (uint8_t)v;
    }
    // VPDPBUSD dot: 1024 uint8 × int8 → int32
    __m512i acc32 = _mm512_setzero_si512();
    for (int i = 0; i < 2*L1; i += 64) {
        __m512i w = _mm512_loadu_si512(composed_w_i8 + i);
        __m512i x = _mm512_loadu_si512((int8_t*)acc_q + i);
        acc32 = _mm512_dpbusd_epi32(acc32, w, x);
    }
    // Horizontal sum (16 int32 lanes → 1)
    int32_t sum = _mm512_reduce_add_epi32(acc32);
    return (float)sum + composed_b;
}

void init_network() {
    srand(42);
    for (int o = 0; o < L2; o++)
        for (int i = 0; i < 2*L1; i++)
            l2_w[o][i] = (rand() % 255) - 127;
    for (int o = 0; o < L2; o++) l2_b[o] = (rand() % 100 - 50) * 0.01f;
    for (int o = 0; o < L3; o++)
        for (int i = 0; i < L2; i++)
            l3_w[o][i] = (rand() % 255) - 127;
    for (int o = 0; o < L3; o++) l3_b[o] = (rand() % 100 - 50) * 0.01f;
    for (int i = 0; i < L3; i++) out_w[i] = (rand() % 255) - 127;
    out_b = (rand() % 100 - 50) * 0.01f;

    // Random accumulator (realistic: mostly small values, some large)
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < L1; i++)
            acc[p][i] = (rand() % 200 - 100) * 0.01f;

    // ---- Compose linear layers: W_composed = out_w × L3_w × L2_w ----
    // Step 1: L3_w × L2_w → temp[32][1024]
    static float temp[L3][2*L1];
    for (int o = 0; o < L3; o++)
        for (int i = 0; i < 2*L1; i++) {
            float s = 0;
            for (int k = 0; k < L2; k++)
                s += (float)l3_w[o][k] * (float)l2_w[k][i];
            temp[o][i] = s;
        }
    // Step 2: out_w × temp → composed_w[1024]
    for (int i = 0; i < 2*L1; i++) {
        float s = 0;
        for (int k = 0; k < L3; k++)
            s += (float)out_w[k] * temp[k][i];
        composed_w[i] = s;
    }
    // Step 3: composed bias = out_b + out_w × (l3_b + L3_w × l2_b)
    float b1[L3];
    for (int o = 0; o < L3; o++) {
        b1[o] = l3_b[o];
        for (int k = 0; k < L2; k++)
            b1[o] += (float)l3_w[o][k] * l2_b[k];
    }
    composed_b = out_b;
    for (int k = 0; k < L3; k++)
        composed_b += (float)out_w[k] * b1[k];

    // Quantize composed weights to int8 for SIMD
    float max_abs = 0;
    for (int i = 0; i < 2*L1; i++)
        if (fabsf(composed_w[i]) > max_abs) max_abs = fabsf(composed_w[i]);
    float scale = 127.0f / (max_abs > 1e-8f ? max_abs : 1.0f);
    for (int i = 0; i < 2*L1; i++)
        composed_w_i8[i] = (int8_t)roundf(composed_w[i] * scale);
    // Adjust bias for the int8 scale
    composed_b /= scale;  // approximate (not exact, but for cycle benchmarking this is fine)
}

int main() {
    init_network();

    // ---- Correctness: compare fast_eval vs full_eval ----
    printf("=== Accuracy comparison (10 random positions) ===\n");
    float corr_sum = 0, full_var = 0, fast_var = 0;
    for (int t = 0; t < 1000; t++) {
        // Random accumulator
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < L1; i++)
                acc[p][i] = (rand() % 200 - 100) * 0.01f;
        float full = full_eval();
        float fast = fast_eval();
        float fast_s = fast_eval_simd();
        if (t < 5)
            printf("  full=%.1f  fast=%.1f  fast_simd=%.1f  err=%.1f%%\n",
                   full, fast, fast_s, fabsf(fast - full) / (fabsf(full) + 1.0f) * 100);
        corr_sum += full * fast;
        full_var += full * full;
        fast_var += fast * fast;
    }
    float correlation = corr_sum / (sqrtf(full_var) * sqrtf(fast_var) + 1e-8f);
    printf("  Correlation (full vs fast): %.4f\n", correlation);
    printf("  (>0.9 = fast eval tracks full eval well enough for pruning decisions)\n");

    // ---- Benchmark ----
    printf("\n=== Benchmark (%dM iterations) ===\n", ITER / 1000000);
    volatile float sink;
    uint64_t t0, t1;

    // Full eval
    t0 = __rdtsc();
    for (int i = 0; i < ITER; i++) {
        sink = full_eval();
        acc[0][0] += 0.0001f * (i & 1);  // perturb to prevent CSE
    }
    t1 = __rdtsc();
    double full_cyc = (double)(t1 - t0) / ITER;
    printf("Full NNUE eval (SCReLU+L2+L3+out):  %8.1f cyc  (1.00x)\n", full_cyc);

    // Fast eval (scalar)
    t0 = __rdtsc();
    for (int i = 0; i < ITER; i++) {
        sink = fast_eval();
        acc[0][0] += 0.0001f * (i & 1);
    }
    t1 = __rdtsc();
    double fast_cyc = (double)(t1 - t0) / ITER;
    printf("TierEval fast (scalar ReLU+dot):     %8.1f cyc  (%.2fx)\n", fast_cyc, full_cyc / fast_cyc);

    // Fast eval SIMD (VPDPBUSD)
    t0 = __rdtscsc();
    for (int i = 0; i < ITER; i++) {
        sink = fast_eval_simd();
        acc[0][0] += 0.0001f * (i & 1);
    }
    t1 = __rdtsc();
    double simd_cyc = (double)(t1 - t0) / ITER;
    printf("TierEval fast (VPDPBUSD SIMD):       %8.1f cyc  (%.2fx)\n", simd_cyc, full_cyc / simd_cyc);

    printf("\n=== Speedup potential ===\n");
    printf("If 70%% of evals use fast path:\n");
    double avg_cyc = 0.7 * simd_cyc + 0.3 * full_cyc;
    printf("  Average eval: %.1f cyc (vs %.1f full = %.2fx speedup)\n", avg_cyc, full_cyc, full_cyc / avg_cyc);
    printf("  Eval cost reduction: %.0f cyc saved per node\n", full_cyc - avg_cyc);
    printf("\nFull eval ~1362 cyc in engine (includes VPDPBUSD L2+SCReLU). Fast eval ~%.0f cyc.\n", simd_cyc);
    printf("At 70%% Tier-1 routing: eval avg ~%.0f cyc → projected NPS gain ~%.0f%%\n",
           avg_cyc, (full_cyc / avg_cyc - 1) * 100);

    return 0;
}
