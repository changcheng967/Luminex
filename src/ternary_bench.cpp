// ternary_bench.cpp — Microbenchmark: int8 VPDPBUSD dot vs ternary addition-based dot.
// Tests the TernaryForge hypothesis: does ternary L2 hit ~120 cyc vs int8's ~500 cyc?
// Compile: g++ -O3 -mavx512f -mavx512bw -mavx512vnni -o ternary_bench ternary_bench.cpp
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <x86intrin.h>

#define L1 512       // NNUE L1 width (per perspective)
#define L2 16        // NNUE L2 width
#define ITERATIONS 10000000  // 10M evals for stable timing

// ---- INT8 dot product using VPDPBUSD (current NNUE L2) ----
// Computes: y[o] = sum_i W[o][i] * x[i]  for o=0..15, i=0..1023 (2*L1 perspectives concatenated)
static alignas(64) int8_t  int8_weights[L2][2*L1];   // 16KB
static alignas(64) uint8_t activations[2*L1];          // 1024 int8

void int8_dot(int32_t* out) {
    memset(out, 0, L2 * sizeof(int32_t));
    for (int o = 0; o < L2; o++) {
        __m512i acc = _mm512_setzero_si512();
        const int8_t* w = int8_weights[o];
        for (int i = 0; i < 2*L1; i += 64) {
            __m512i w_vec = _mm512_loadu_si512(w + i);
            __m512i x_vec = _mm512_loadu_si512((const int8_t*)activations + i);
            // VPDPBUSD: 4 int8 dots per lane, accumulate into int32
            acc = _mm512_dpbusd_epi32(acc, w_vec, x_vec);
        }
        // horizontal sum the 16 int32 lanes
        out[o] = _mm512_reduce_add_epi32(acc);
    }
}

// ---- Ternary dot product (addition-based, no multiply) ----
// Weights are {-1, 0, +1}. Packed: 2 bits per weight.
// Encoding: +1=0b01, 0=0b00, -1=0b11. 4 weights per byte.
// For the benchmark, we store ternary weights as sign + magnitude masks.
static alignas(64) uint8_t ternary_packed[L2][(2*L1)/4];  // 4KB total!
// Pre-computed masks for fast lookup
static alignas(64) uint8_t pos_mask[L2][2*L1];  // 1 where weight=+1, 0 elsewhere (for bench only)
static alignas(64) uint8_t neg_mask[L2][2*L1];  // 1 where weight=-1, 0 elsewhere

void ternary_dot_naive(int32_t* out) {
    // Naive scalar version (for correctness check)
    memset(out, 0, L2 * sizeof(int32_t));
    for (int o = 0; o < L2; o++) {
        for (int i = 0; i < 2*L1; i++) {
            uint8_t pm = pos_mask[o][i];
            uint8_t nm = neg_mask[o][i];
            out[o] += (int32_t)activations[i] * pm - (int32_t)activations[i] * nm;
        }
    }
}

void ternary_dot_avx512(int32_t* out) {
    // AVX-512 version: use mask-based conditional add/subtract
    // For each output neuron, load 64 activations at a time.
    // Use VPMOVB2M to create k-mask from pos_mask/neg_mask bytes.
    // Then: VPADDB with mask (add only where weight=+1), VPSUBB with mask (sub where weight=-1).
    // But VPADDB operates on int8, and we need int32 accumulation.
    // Strategy: use VPCMPB to create masks, then masked add to int32 accumulator.
    for (int o = 0; o < L2; o++) {
        __m512i acc = _mm512_setzero_si512();
        for (int i = 0; i < 2*L1; i += 64) {
            __m512i x = _mm512_loadu_si512(activations + i);
            __m512i pm = _mm512_loadu_si512(pos_mask[o] + i);
            __m512i nm = _mm512_loadu_si512(neg_mask[o] + i);
            // Create masks: pos = (byte != 0), neg = (byte != 0)
            __mmask64 kpos = _mm512_cmpneq_epi8_mask(pm, _mm512_setzero_si512());
            __mmask64 kneg = _mm512_cmpneq_epi8_mask(nm, _mm512_setzero_si512());
            // Convert activations to int32 (4 at a time, but we need all 64...)
            // Actually: use int8 addition directly, then expand.
            // Approach: masked int8 add/sub, then horizontal sum via dpbusd on the mask.
            // Better approach: treat as: result = sum(x[i] * (pm[i] - nm[i]))
            // = sum(x[i] * ternary[i]) where ternary = pm - nm
            __m512i ternary = _mm512_sub_epi8(pm, nm);  // +1 where pos, -1 where neg, 0 elsewhere
            // Now do: acc += x * ternary using VPDPBUSD (which does int8 MAC!)
            // Wait — this IS VPDPBUSD. But ternary weights are {-1,0,+1} in int8.
            // VPDPBUSD needs UNSIGNED activations + SIGNED weights.
            // activations are uint8 [0,127]. ternary as int8 = {-1,0,+1}.
            // VPDPBUSD(acc, ternary_as_uint8, x) — but VPDPBUSD does unsigned*unsigned.
            // We need signed*unsigned. Use VPDPBUSD with a trick, or just use the masks.

            // Simplest correct approach: expand int8 to int32 in groups of 16, then masked add.
            // For 64 bytes: 4 groups of 16.
            for (int g = 0; g < 4; g++) {
                __m512i x32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(activations + i + g*16)));
                // Extract 16 ternary signs
                int8_t signs[16];
                for (int j = 0; j < 16; j++) {
                    signs[j] = (int8_t)pos_mask[o][i+g*16+j] - (int8_t)neg_mask[o][i+g*16+j];
                }
                __m512i w32 = _mm512_loadu_si512(signs);
                // Multiply-accumulate (this uses VPMULLD — not VPDPBUSD, but it's a fair
                // comparison since ternary means the multiply is trivial)
                __m512i prod = _mm512_mullo_epi32(x32, w32);
                acc = _mm512_add_epi32(acc, prod);
            }
        }
        out[o] = _mm512_reduce_add_epi32(acc);
    }
}

// Better ternary approach: use VPSUBBD + VPADDB directly on int8, then sum.
void ternary_dot_int8_addsub(int32_t* out) {
    // For each output: accumulate via masked int8 add/sub into an int8 buffer.
    // Then sum the int8 buffer to int32.
    // Problem: int8 overflow after ~127 additions. With 1024 inputs, need int16 accumulator.
    // Use int16 accumulation via VPMADDUBSW or manual widening.
    //
    // Most practical approach for benchmarking: pre-expand ternary to int8 {-1,0,+1},
    // then use VPDPBUSD (unsigned x * signed — but VPDPBUSD is unsigned*unsigned).
    // The actual fastest ternary kernel on Zen 4 would be:
    //   1. Pack ternary as int8 {-1,0,+1} — same size as int8 weights!
    //   2. Use the SAME VPDPBUSD but with sign trick: store activations as uint8,
    //      weights as int8. VPDPBUSD does uint8*int8? NO — it does uint8*uint8.
    //   3. Alternative: VPMADDUBSW (int8*int8→int16 multiply-add). This IS signed.
    //      VPMADDUBSW computes: for each pair (a,b): a[2i]*b[2i] + a[2i+1]*b[2i+1] → int16.
    //      With ternary b ∈ {-1,0,+1}, this is just conditional add/subtract!
    //      And VPMADDUBSW has HIGHER throughput than VPDPBUSD on some architectures.
    //
    // For this benchmark, let's use the VPMADDUBSW approach.
    static alignas(64) int8_t ternary_i8[L2][2*L1];  // {-1,0,+1} as int8 — same 16KB
    // Initialize once
    static bool init = false;
    if (!init) {
        for (int o = 0; o < L2; o++)
            for (int i = 0; i < 2*L1; i++)
                ternary_i8[o][i] = (int8_t)pos_mask[o][i] - (int8_t)neg_mask[o][i];
        init = true;
    }
    for (int o = 0; o < L2; o++) {
        __m512i acc = _mm512_setzero_si512();
        const int8_t* w = ternary_i8[o];
        for (int i = 0; i < 2*L1; i += 32) {
            // VPMADDUBSW: signed int8 × signed int8 → int16, pairs summed
            __m256i w_vec = _mm256_loadu_si256((__m256i*)(w + i));
            __m256i x_vec = _mm256_loadu_si256((__m256i*)((int8_t*)activations + i));
            __m256i mad = _mm256_maddubs_epi16(x_vec, w_vec);  // 16 int16 results
            // Widen to int32 and accumulate
            __m512i mad32 = _mm512_cvtepi16_epi32(mad);
            acc = _mm512_add_epi32(acc, mad32);
        }
        out[o] = _mm512_reduce_add_epi32(acc);
    }
}

// ---- Packed ternary (2-bit, 4KB) + bandwidth measurement ----
// The KEY advantage of ternary is BANDWIDTH (4KB vs 16KB → L1 fit).
// This version reads from the packed 2-bit format, unpacks on the fly.
void ternary_dot_packed(int32_t* out) {
    // Unpack 2-bit weights on the fly. 4 weights per byte.
    // Encoding: bits[1:0] of each 2-bit field: 00=0, 01=+1, 11=-1.
    // For benchmarking: use a LUT to unpack a byte to 4 int8 values.
    static alignas(64) int8_t lut[256][4];
    static bool lut_init = false;
    if (!lut_init) {
        for (int b = 0; b < 256; b++) {
            for (int j = 0; j < 4; j++) {
                int val = (b >> (j*2)) & 3;
                lut[b][j] = (val == 1) ? 1 : (val == 3) ? -1 : 0;
            }
        }
        lut_init = true;
    }
    for (int o = 0; o < L2; o++) {
        __m512i acc = _mm512_setzero_si512();
        const uint8_t* pk = ternary_packed[o];
        for (int i = 0; i < (2*L1)/4; i += 8) {
            // Unpack 8 packed bytes → 32 ternary int8 values
            int8_t unpacked[32];
            for (int j = 0; j < 8; j++) {
                memcpy(unpacked + j*4, lut[pk[i+j]], 4);
            }
            __m256i w_vec = _mm256_loadu_si256((__m256i*)unpacked);
            __m256i x_vec = _mm256_loadu_si256((__m256i*)((int8_t*)activations + i*4));
            __m256i mad = _mm256_maddubs_epi16(x_vec, w_vec);
            __m512i mad32 = _mm512_cvtepi16_epi32(mad);
            acc = _mm512_add_epi32(acc, mad32);
        }
        out[o] = _mm512_reduce_add_epi32(acc);
    }
}

int main() {
    // Initialize random data
    srand(42);
    for (int o = 0; o < L2; o++)
        for (int i = 0; i < 2*L1; i++) {
            int8_weights[o][i] = (rand() % 255) - 127;
            activations[i] = (uint8_t)(rand() % 128);
        }
    // Create ternary weights (~33% each: -1, 0, +1)
    for (int o = 0; o < L2; o++)
        for (int i = 0; i < 2*L1; i++) {
            int t = rand() % 3 - 1;  // -1, 0, or +1
            pos_mask[o][i] = (t == 1) ? 1 : 0;
            neg_mask[o][i] = (t == -1) ? 1 : 0;
        }
    // Pack ternary to 2-bit
    for (int o = 0; o < L2; o++)
        for (int i = 0; i < (2*L1)/4; i++) {
            uint8_t byte = 0;
            for (int j = 0; j < 4; j++) {
                int t = (int)pos_mask[o][i*4+j] - (int)neg_mask[o][i*4+j];
                int code = (t == 1) ? 1 : (t == -1) ? 3 : 0;
                byte |= (code << (j*2));
            }
            ternary_packed[o][i] = byte;
        }

    int32_t out_int8[L2], out_tern[L2], out_naive[L2], out_packed[L2];

    // Correctness check
    int8_dot(out_int8);
    ternary_dot_naive(out_naive);
    ternary_dot_int8_addsub(out_tern);
    ternary_dot_packed(out_packed);
    printf("=== Correctness ===\n");
    printf("int8 dot[0] = %d\n", out_int8[0]);
    printf("ternary naive[0] = %d (expected different from int8 — different weights)\n", out_naive[0]);
    printf("ternary AVX int8_addsub[0] = %d (should match naive: %s)\n",
           out_tern[0], out_tern[0] == out_naive[0] ? "YES" : "NO!");
    printf("ternary packed[0] = %d (should match naive: %s)\n",
           out_packed[0], out_packed[0] == out_naive[0] ? "YES" : "NO!");

    // Benchmark
    printf("\n=== Benchmark (%dM iterations) ===\n", ITERATIONS/1000000);
    printf("Format                                     cyc/eval   relative\n");
    printf("                                          --------   --------\n");

    // Warm cache
    for (int i = 0; i < 1000; i++) int8_dot(out_int8);

    uint64_t t0, t1;
    // INT8 VPDPBUSD (baseline)
    t0 = __rdtsc();
    for (int i = 0; i < ITERATIONS; i++) int8_dot(out_int8);
    t1 = __rdtsc();
    double int8_cyc = (double)(t1-t0) / ITERATIONS;
    printf("%-42s %8.1f    1.00x\n", "INT8 VPDPBUSD (16KB weights)", int8_cyc);

    // Ternary int8_addsub (VPMADDUBSW, same 16KB but {-1,0,+1})
    t0 = __rdtsc();
    for (int i = 0; i < ITERATIONS; i++) ternary_dot_int8_addsub(out_tern);
    t1 = __rdtsc();
    double tern_i8_cyc = (double)(t1-t0) / ITERATIONS;
    printf("%-42s %8.1f    %.2fx\n", "Ternary VPMADDUBSW (16KB, int8 ternary)", tern_i8_cyc, int8_cyc/tern_i8_cyc);

    // Ternary packed (4KB, unpack on the fly)
    t0 = __rdtsc();
    for (int i = 0; i < ITERATIONS; i++) ternary_dot_packed(out_packed);
    t1 = __rdtsc();
    double tern_pk_cyc = (double)(t1-t0) / ITERATIONS;
    printf("%-42s %8.1f    %.2fx\n", "Ternary packed 2-bit (4KB, L1 fit!)", tern_pk_cyc, int8_cyc/tern_pk_cyc);

    printf("\n=== Bandwidth comparison ===\n");
    printf("INT8 weights: %zu bytes (L2 cache resident)\n", sizeof(int8_weights));
    printf("Ternary packed: %zu bytes (L1 cache resident!)\n", sizeof(ternary_packed));
    printf("Ratio: %.1fx less bandwidth\n", (double)sizeof(int8_weights)/sizeof(ternary_packed));

    printf("\n=== Key result ===\n");
    printf("If ternary is FASTER: TernaryForge hypothesis CONFIRMED (bandwidth win > compute cost)\n");
    printf("If ternary is SLOWER: bandwidth win insufficient at L2=16 (too small for L2 miss penalty)\n");
    printf("Speedup needed for 2M NPS: eval 1362 -> 680 cyc (2.0x). L2 is ~500 cyc.\n");
    printf("  If ternary L2 hits ~120 cyc: eval -> ~980 cyc -> 1.38M NPS (need ShiftReLU too)\n");
    printf("  If ternary L2 hits ~250 cyc: eval -> ~1110 cyc -> 1.22M NPS\n");
    printf("  If ternary L2 hits ~500 cyc: eval -> 1362 cyc -> 990K NPS (no improvement)\n");

    return 0;
}
