// nnue.cpp — NNUE evaluation with an incremental accumulator.
//
// A full feature-transformer rebuild per eval is ~15x slower than this design
// (measured 135K nps on the first prototype). The accumulator lives in a
// per-thread stack (acc_stack[], indexed by Position::st_ply()) and is updated
// incrementally on make_move, so evaluate() just reads it. The debug self-check
// validates it against the full-refresh reference exactly.
//
// The accumulator lives in a thread_local global (NOT in StateInfo) — putting it in
// StateInfo bloated state_stack[2048] to ~8MB and overflowed the stack (Position is
// stack-local). thread_local also makes it safe across Luminex's helper threads.
//
// HalfKAv2_hm feature indexer mirrored EXACTLY from nnue/luminex_nnue_train.py.
#include "nnue.h"
#include "board.h"
#include "embedded_net.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include <fstream>
#include <istream>
#include <streambuf>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>   // _mm_prefetch needs xmmintrin even without AVX2 (Debug builds)
#endif

namespace luminex::nnue {

#if defined(__AVX2__)
// Horizontal sum of 8 floats -> scalar.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
// SCReLU on 8 floats: clamp to [0,1], then square.
static inline __m256 screlu8(__m256 x, __m256 z, __m256 one) {
    __m256 c = _mm256_min_ps(_mm256_max_ps(x, z), one);
    return _mm256_mul_ps(c, c);
}
// Quantize 8 floats ([0,1]) -> 8 uint8 ([0,127]), stored contiguously at dst.
static inline void quant8(const float* h, uint8_t* dst, __m256 scale127) {
    __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(h), scale127));
    __m128i lo = _mm256_castsi256_si128(i32);
    __m128i hi = _mm256_extracti128_si256(i32, 1);
    __m128i i16 = _mm_packs_epi32(lo, hi);        // 8 int16 (in-order)
    __m128i i8 = _mm_packus_epi16(i16, i16);      // 16 uint8 (low 8 = ours)
    _mm_storel_epi64((__m128i*)dst, i8);          // store low 8 bytes
}
// int8 dot product over n bytes: sum(w_signed[i] * a_unsigned[i]) -> int32. Handles
// any n (SIMD for multiples of 32, scalar tail for the remainder, e.g. L2=16 inputs).
static inline int32_t dot_i8(const int8_t* w, const uint8_t* a, int n) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i wv = _mm256_loadu_si256((const __m256i*)(w + i));
        __m256i av = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i p16 = _mm256_maddubs_epi16(av, wv);   // a(uint8) × w(int8) -> 16 int16
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));  // -> 8 int32, accumulate
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_hadd_epi32(_mm_add_epi32(lo, hi), _mm_setzero_si128());
    int32_t sum = _mm_cvtsi128_si32(_mm_hadd_epi32(s, s));
    for (; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
#endif
#if defined(__AVX512VNNI__)
// VPDPBUSD: 64 uint8 × int8 → 16 int32 accumulate in ONE instruction.
// ~6x throughput vs AVX2 (2x width + 3x fewer instructions per MAC chunk).
static inline int32_t dot_i8_vnni(const int8_t* w, const uint8_t* a, int n) {
    // 4 independent accumulator chains: a single chain serializes on vpdpbusd
    // latency (~4-5 cyc), so a 1024-byte L2 dot alone costs ~70+ cycles of pure
    // stalls. Integer adds are associative -> the result is bit-identical.
    __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512(),
            acc2 = _mm512_setzero_si512(), acc3 = _mm512_setzero_si512();
    int i = 0;
    for (; i + 256 <= n; i += 256) {
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(a + i)),       _mm512_loadu_si512((const __m512i*)(w + i)));
        acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(a + i + 64)),  _mm512_loadu_si512((const __m512i*)(w + i + 64)));
        acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(a + i + 128)), _mm512_loadu_si512((const __m512i*)(w + i + 128)));
        acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(a + i + 192)), _mm512_loadu_si512((const __m512i*)(w + i + 192)));
    }
    for (; i + 64 <= n; i += 64)
        acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(a + i)), _mm512_loadu_si512((const __m512i*)(w + i)));
    __m512i acc = _mm512_add_epi32(_mm512_add_epi32(acc0, acc1), _mm512_add_epi32(acc2, acc3));
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
// Small-n dot (n <= 64): one VPDPBUSD over zero-extended inputs, then reduce.
// For L3 (n=L2=16) and out (n=L3=32) this replaces ~512 scalar int muls with a
// single VNNI instruction — the dominant eval cost after L2.
static inline int32_t dot_i8_vnni_small(const int8_t* w, const uint8_t* a, int n) {
    __m512i av, wv;
    if (n <= 16) {
        av = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)a));
        wv = _mm512_zextsi128_si512(_mm_loadu_si128((const __m128i*)w));
    } else if (n <= 32) {
        av = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)a));
        wv = _mm512_zextsi256_si512(_mm256_loadu_si256((const __m256i*)w));
    } else {  // 32 < n <= 64: load full 64 bytes (caller guarantees bounds).
        av = _mm512_loadu_si512((const void*)a);
        wv = _mm512_loadu_si512((const void*)w);
    }
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), av, wv);
    return _mm512_reduce_add_epi32(acc);
}
#endif

#if !defined(__AVX2__)
static inline int32_t dot_i8(const int8_t* w, const uint8_t* a, int n) {
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) sum += (int32_t)w[i] * (int32_t)a[i];
    return sum;
}
#endif

// ---- profiling counters (data-driven: find the NPS bottleneck from numbers) ----
namespace { struct Stat { long long n = 0, cyc = 0; }; }

static Stat st_eval, st_update, st_refresh, st_king_refresh, st_incremental;
static long long g_act_nz = 0, g_act_tot = 0;   // activation sparsity (non-zero L1 outputs after SCReLU)
static long long g_sat_events = 0;              // exact accumulator leaving int16 range (self-check only)
static bool g_profile = false;
static bool g_profile_inited = false;
static inline bool profile_on() {
    if (!g_profile_inited) { g_profile = (std::getenv("NNUE_PROFILE") != nullptr); g_profile_inited = true; }
    return g_profile;
}
static inline long long rdtsc() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned a, d; __asm__ __volatile__("rdtsc" : "=a"(a), "=d"(d));
    return ((long long)d << 32) | a;
#else
    return 0;
#endif
}
void print_stats() {
    if (!profile_on()) return;
    std::fprintf(stderr, "NNUE profile: evals=%lld (%.1fM cyc, %.0f cyc/eval) | updates=%lld (%.1fM cyc, %.0f cyc/upd) | king-refreshes=%lld (%.0f cyc/kr) | incrementals=%lld (%.0f cyc/inc) | refreshes=%lld (%.1fM cyc) | L1-act density=%.1f%% | sat_events=%lld\n",
                 st_eval.n, st_eval.cyc/1000000.0, st_eval.n ? double(st_eval.cyc)/st_eval.n : 0.0,
                 st_update.n, st_update.cyc/1000000.0, st_update.n ? double(st_update.cyc)/st_update.n : 0.0,
                 st_king_refresh.n, st_king_refresh.n ? double(st_king_refresh.cyc)/st_king_refresh.n : 0.0,
                 st_incremental.n, st_incremental.n ? double(st_incremental.cyc)/st_incremental.n : 0.0,
                 st_refresh.n, st_refresh.cyc/1000000.0,
                 g_act_tot ? 100.0*g_act_nz/g_act_tot : 0.0,
                 g_sat_events);
}

constexpr int NUM_SQ = 64;
constexpr int NUM_PLANES = 768;              // 64 sq * 12 piece types
constexpr int NUM_INPUTS = NUM_PLANES * 32;  // 24576 per half (32 king buckets)
constexpr int OUT_SCALE = 300;               // matches trainer's output *300
constexpr int NNUE_MAX_PLY = 2048;           // MUST match Position::MAX_STATES (board.h)

static const int KingBuckets[NUM_SQ] = {
    -1, -1, -1, -1, 31, 30, 29, 28,
    -1, -1, -1, -1, 27, 26, 25, 24,
    -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16,
    -1, -1, -1, -1, 15, 14, 13, 12,
    -1, -1, -1, -1, 11, 10,  9,  8,
    -1, -1, -1, -1,  7,  6,  5,  4,
    -1, -1, -1, -1,  3,  2,  1,  0,
};

static inline int orient(bool white_pov, int sq, int ksq) {
    int kfile = ksq & 7;
    return ((kfile < 4) ? 7 : 0) ^ (white_pov ? 0 : 56) ^ sq;
}

static inline int halfka_idx(bool white_pov, int king_sq, int sq, Piece p) {
    int pt = static_cast<int>(piece_type_of(p));               // 0..5 (PAWN..KING)
    bool is_white_piece = (color_of_piece(p) == WHITE);
    int p_idx = pt * 2 + (is_white_piece != white_pov ? 1 : 0);
    int o_ksq = orient(white_pov, king_sq, king_sq);
    return orient(white_pov, sq, king_sq) + p_idx * NUM_SQ + KingBuckets[o_ksq] * NUM_PLANES;
}

static int g_L1 = 0, g_L2 = 0, g_L3 = 0;
// FT weights stored as int16 (half the memory traffic vs float → fewer L3 misses on
// feature delta). The accumulator stays full float32 (no SCReLU precision loss).
static constexpr float FT_WSCALE = 8192.0f;
static constexpr float FT_WINV = 1.0f / FT_WSCALE;
static std::vector<int16_t> ft_w;
static std::vector<int32_t> ft_b_i32;   // integer-domain FT bias (b*FT_WSCALE)
// DOSL linear head (Step-0): per-feature scalar weights over the SAME HalfKAv2_hg
// indices; accumulator keeps one int32 per perspective (O(1) qsearch stand-pat).
// Weights stored in 1/32 cp units: a queen feature (~900cp) still fits int16.

static std::vector<float> ft_b;   // (L1,)
static std::vector<float> l2_w, l2_b, l3_w, l3_b, out_w;
static float out_b = 0.0f;
// int8 path (LNI8): L2/L3/out weights as int8 + per-layer scales. FT stays float.
static bool g_int8 = false;
static std::vector<int8_t> l2_w_i8, l3_w_i8, out_w_i8;
static float g_s2 = 1.0f, g_s3 = 1.0f, g_so = 1.0f;  // weight quant scales
static bool g_loaded = false;
static bool g_l3_batched = false;   // l3_w_i8 interleaved for batched VNNI L3
static bool g_l2_by_input = false;  // l2_w_i8 in 64-byte input-chunk blocks for VNNI L2

static bool g_enabled = true;
static uint32_t g_net_id = 0;   // incremented by every successful load(); keys the refresh cache

// Per-thread accumulator stack, indexed by Position::st_ply(). v[perspective][neuron].
// Cache-line aligned for clean loads during the SCReLU/quant8 pass.
// int16 saturating accumulator: exact integer sums, clip window [0, 8192]
// (1.0x FT_WSCALE); int16 saturation at +-32767 sits 4x above the window, so
// saturating lanes were already fully clipped by SCReLU.
// that saturates was already fully clipped. Simulated on 6000 real perspectives:
// 2 single-lane activation-quant diffs out of 3.07M lanes (0.03% of perspectives).
// Halves copy/update/eval traffic vs the int32 accumulator.
struct alignas(64) Accumulator { int16_t v[2][NNUE_L1_MAX]; int32_t lin[2]; };

// Heap-allocated per thread, NOT a thread_local C-array: an 8MB thread_local array
// reserves 8MB of *static TLS* for every thread and overflows the stack at creation.
// The vector object is ~24 bytes of TLS; its 8MB buffer lives on the heap and is freed
// when the thread exits (RAII).
static inline Accumulator* nnue_acc_stack() {
    thread_local std::vector<Accumulator> store(NNUE_MAX_PLY);
    return store.data();
}

// at startpos (beyond int16's +-4.0 = 32767/8192) — the soft NNUE_ACC_REG did
// not pull the extremes into int16 range, so saturating int16 adds corrupt
// lanes that leave and re-enter the clip window (spec §3 fallback stands).

bool loaded() { return g_loaded; }
bool enabled() { return g_loaded && g_enabled; }
void set_enabled(bool on) { g_enabled = on; }
int l1() { return g_L1; }

static bool read_tensor(std::istream& f, std::vector<float>& v, int expected) {
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (n != expected) { std::fprintf(stderr, "nnue: size mismatch %d != %d\n", n, expected); return false; }
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n);
    return f.good();
}

// l2_w(i8), l2_b(i32), l3_w(i8), l3_b(i32), out_w(i8), s_out(f32), out_b(i32).

// Read a network from f, a file or the copy built into the binary; name is
// what the report calls it.
static bool load_stream(std::istream& f, const std::string& name) {
    char magic[4];
    f.read(magic, 4);
    std::string mg(magic, 4);
    if (mg != "LNN1" && mg != "LNI8") { std::fprintf(stderr, "nnue: bad magic\n"); return false; }
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) { std::fprintf(stderr, "nnue: truncated header\n"); return false; }
    int L1 = hdr[0], L2 = hdr[1], L3 = hdr[2], NI = hdr[3];
    if (NI != NUM_INPUTS) { std::fprintf(stderr, "nnue: NUM_INPUTS mismatch %d\n", NI); return false; }
    if (L1 <= 0 || L1 > NNUE_L1_MAX) { std::fprintf(stderr, "nnue: L1=%d out of range (max %d)\n", L1, NNUE_L1_MAX); return false; }
    // The SIMD loops step L1 in 32-lane blocks and L2/L3 in 8-lane blocks, and
    // the small VNNI dots read at most 64 bytes.
    if (L1 % 32 != 0 || L2 <= 0 || L2 > 64 || L2 % 8 != 0 || L3 <= 0 || L3 > 64 || L3 % 8 != 0) {
        std::fprintf(stderr, "nnue: unsupported layer sizes L1=%d L2=%d L3=%d\n", L1, L2, L3);
        return false;
    }
    // From here on the globals are overwritten; a failure part-way must not
    // leave a half-loaded net marked usable.
    g_loaded = false;
    g_l3_batched = false;
    g_l2_by_input = false;
    g_int8 = (mg == "LNI8");
    g_L1 = L1; g_L2 = L2; g_L3 = L3;
    std::vector<float> ft_w_raw;
    if (!read_tensor(f, ft_w_raw, L1 * NUM_INPUTS)) return false;
    ft_w.resize(static_cast<size_t>(NUM_INPUTS) * L1);
    // int16 at x8192 holds |w| < 4.0; clamp instead of letting the cast wrap.
    long long clamped = 0;
    for (int i = 0; i < NUM_INPUTS; ++i)
        for (int l = 0; l < L1; ++l) {
            long long q = std::llround(ft_w_raw[l * NUM_INPUTS + i] * FT_WSCALE);
            if (q > 32767 || q < -32767) { q = q > 0 ? 32767 : -32767; ++clamped; }
            ft_w[i * L1 + l] = (int16_t)q;
        }
    if (!read_tensor(f, ft_b, L1)) return false;
    ft_b_i32.resize(L1);
    for (int l = 0; l < L1; ++l) ft_b_i32[l] = std::lround(ft_b[l] * FT_WSCALE);
    if (g_int8) {
        // LNI8: int8 weights + float bias + float scale, per layer.
        auto read_i8 = [&](std::vector<int8_t>& v, int expected) -> bool {
            int32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
            if (n != expected) { std::fprintf(stderr, "nnue: i8 size %d != %d\n", n, expected); return false; }
            v.resize(n); f.read(reinterpret_cast<char*>(v.data()), n); return f.good(); };
        if (!read_i8(l2_w_i8, L2 * 2 * L1)) return false;
        if (!read_tensor(f, l2_b, L2)) return false;
        f.read(reinterpret_cast<char*>(&g_s2), 4);
        if (!read_i8(l3_w_i8, L3 * L2)) return false;
#if defined(__AVX512VNNI__)
        // Batched-L3 layout: dpbusd lane (o*4+j) accumulates h2-slice j against
        // l3_w[o] slice j — one interleaved row per 4 outputs, 2 dpbusd for all 32.
        if (L2 == 16 && (L3 % 4) == 0) {
            std::vector<int8_t> tmp(static_cast<size_t>(L3) * L2);
            for (int o = 0; o < L3; ++o)
                for (int j = 0; j < 4; ++j)
                    std::memcpy(&tmp[(static_cast<size_t>(o) * 4 + j) * 4], &l3_w_i8[static_cast<size_t>(o) * L2 + j * 4], 4);
            l3_w_i8 = std::move(tmp);
            g_l3_batched = true;
        }
        // Input-chunk L2 layout: block c (64 bytes) holds, for each of the 16
        // outputs, its 4 weights for inputs 4c..4c+3, so one VPDPBUSD of a
        // broadcast 4-byte input chunk updates all 16 output sums at once.
        if (L2 == 16 && (2 * L1) % 4 == 0) {
            const int total_in = 2 * L1;
            std::vector<int8_t> tmp(static_cast<size_t>(L2) * total_in);
            for (int c = 0; c < total_in / 4; ++c)
                for (int o = 0; o < L2; ++o)
                    std::memcpy(&tmp[(static_cast<size_t>(c) * L2 + o) * 4],
                                &l2_w_i8[static_cast<size_t>(o) * total_in + c * 4], 4);
            l2_w_i8 = std::move(tmp);
            g_l2_by_input = true;
        }
#endif
        if (!read_tensor(f, l3_b, L3)) return false;
        f.read(reinterpret_cast<char*>(&g_s3), 4);
        if (!read_i8(out_w_i8, L3)) return false;
        std::vector<float> ob;
        if (!read_tensor(f, ob, 1)) return false;
        out_b = ob[0];
        f.read(reinterpret_cast<char*>(&g_so), 4);
        // Reorder L2 weights for AVX2 SF-style chunk access (not needed for AVX-512 VNNI).
#if defined(__AVX2__) && !defined(__AVX512VNNI__)
        // Old: [output][chunk]  →  New: [chunk][output]  (each chunk's 16 weight rows contiguous)
        // Old: [output][chunk]  →  New: [chunk][output]  (each chunk's 16 weight rows contiguous)
        {
            int total_in = 2 * L1;
            int chunks = total_in / 32;
            std::vector<int8_t> tmp(static_cast<size_t>(L2) * total_in);
            for (int c = 0; c < chunks; ++c)
                for (int o = 0; o < L2; ++o)
                    std::memcpy(&tmp[(static_cast<size_t>(c) * L2 + o) * 32],
                                &l2_w_i8[static_cast<size_t>(o) * total_in + c * 32], 32);
            l2_w_i8 = std::move(tmp);
        }
#endif // !__AVX512VNNI__
    } else {
        if (!read_tensor(f, l2_w, L2 * 2 * L1)) return false;
        if (!read_tensor(f, l2_b, L2)) return false;
        if (!read_tensor(f, l3_w, L3 * L2)) return false;
        if (!read_tensor(f, l3_b, L3)) return false;
        std::vector<float> out_w_raw;
        if (!read_tensor(f, out_w_raw, L3)) return false;
        out_w = out_w_raw;
        int32_t ob_n;
        f.read(reinterpret_cast<char*>(&ob_n), 4);
        if (ob_n != 1) { std::fprintf(stderr, "nnue: out.bias size %d\n", ob_n); return false; }
        f.read(reinterpret_cast<char*>(&out_b), sizeof(float));
    }
    // Optional DOSL linear head (older nets simply end here).
    {
        char tag[4] = {0,0,0,0};
        std::streampos save = f.tellg();
        f.read(tag, 4);
        if (f.gcount() == 4 && tag[0]=='L' && tag[1]=='I' && tag[2]=='N' && tag[3]=='H') {
            // Skip the LINH section (removed feature): read size, skip data
            int32_t n = 0; f.read(reinterpret_cast<char*>(&n), 4);
            if (n > 0 && n <= NUM_INPUTS) {
                f.seekg(static_cast<long>(n) * 4, std::ios::cur); // skip float weights
            }
        } else {
            f.seekg(save);   // not a LINH section — rewind (older net format)
        }
    }
    g_loaded = true;
    ++g_net_id;   // refresh-cache entries of the previous net are stale
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    // THP for the big weight tables: the incremental update path is
    // memory-latency bound (random row access into ft_w); huge pages cut
    // TLB misses. Best effort — silently ignored where disallowed.
    madvise(ft_w.data(), (ft_w.size() * sizeof(int16_t) + 0xFFFFF) & ~0xFFFFF, MADV_HUGEPAGE);
#endif
    // stdout carries UCI traffic, so the report is an info string.
    std::printf("info string NNUE evaluation using %s (%s L1=%d L2=%d L3=%d)\n",
                name.c_str(), g_int8 ? "int8" : "float", L1, L2, L3);
    if (clamped > 0)
        std::printf("info string NNUE warning: %lld feature weights exceed the int16 range and were clamped\n",
                    clamped);
    return true;
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "nnue: cannot open %s\n", path.c_str()); return false; }
    return load_stream(f, path);
}

namespace {
// Read-only stream buffer over bytes already in memory, seekable because the
// loader probes for an optional trailing section and rewinds when it is absent.
class MemoryBuffer : public std::streambuf {
public:
    MemoryBuffer(const unsigned char* data, std::size_t size) {
        char* p = reinterpret_cast<char*>(const_cast<unsigned char*>(data));
        setg(p, p, p + size);
    }

protected:
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (!(which & std::ios_base::in)) return pos_type(off_type(-1));
        char* from = dir == std::ios_base::beg ? eback() : dir == std::ios_base::cur ? gptr() : egptr();
        if (off < eback() - from || off > egptr() - from) return pos_type(off_type(-1));
        setg(eback(), from + off, egptr());
        return pos_type(off_type(gptr() - eback()));
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
    }
};
} // namespace

bool has_embedded() { return embedded_net().size > 0; }

bool load_embedded() {
    const EmbeddedNet net = embedded_net();
    if (net.size == 0) return false;
    MemoryBuffer buf(net.data, net.size);
    std::istream f(&buf);
    return load_stream(f, std::string("built-in ") + DEFAULT_NET);
}

static inline float clip01(float x) { float c = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); return c * c; }

// ClippedReLU on a x8192-domain lane, requantized to the u8 dense input
// (min(in>>6, 127) — 8192>>6 = 128 caps at 127).
static inline uint8_t crelu_q8(int v) {
    v = v < 0 ? 0 : (v > 8192 ? 8192 : v);
    return uint8_t(std::min(v >> 6, 127));
}

// Saturating int16 on both rows; equivalent fused or sequential per sign.


static inline const int16_t* feature_row(bool white_pov, int ksq, int sq, Piece piece) {
    return &ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, piece)) * g_L1];
}

// Add / subtract one feature row to a perspective's lanes (saturating int16).
static inline void add_row(int16_t* acc, const int16_t* w) {
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 32)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_adds_epi16(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_loadu_si512((const __m512i*)(w + l))));
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 16)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_adds_epi16(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_loadu_si256((const __m256i*)(w + l))));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] = int16_t(std::max(-32768, std::min(32767, int(acc[l]) + int(w[l]))));
#endif
}
static inline void sub_row(int16_t* acc, const int16_t* w) {
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 32)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_subs_epi16(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_loadu_si512((const __m512i*)(w + l))));
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 16)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_subs_epi16(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_loadu_si256((const __m256i*)(w + l))));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] = int16_t(std::max(-32768, std::min(32767, int(acc[l]) - int(w[l]))));
#endif
}

static inline void add_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    add_row(a.v[p], feature_row(p == 0, ksq, sq, piece));
}
static inline void remove_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    sub_row(a.v[p], feature_row(p == 0, ksq, sq, piece));
}

// FUSED single-pass delta for the dominant quiet-move pattern remove(from)+add(to):
// one accumulator read-modify-write instead of two. Net lane delta is (w_to - w_from).
// Exactly the sequential remove+add pair whenever no lane saturates; saturating
// add/sub do NOT commute, so on saturation the fused path can differ from a
// sequential int16 pair (the self-check counts those events in g_sat_events —
// range-constrained training must keep it at zero).
static inline void move_feature(Accumulator& a, int p, int ksq, Square from, Square to, Piece pf, Piece pt) {
    bool white_pov = (p == 0);
    int f_idx = halfka_idx(white_pov, ksq, from, pf);
    int t_idx = halfka_idx(white_pov, ksq, to, pt);
    size_t fi = static_cast<size_t>(f_idx) * g_L1;
    size_t ti = static_cast<size_t>(t_idx) * g_L1;
    const int16_t* wf = &ft_w[fi];
    const int16_t* wt = &ft_w[ti];
    int16_t* acc = a.v[p];
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 32) {
        __m512i av = _mm512_loadu_si512((const __m512i*)(acc + l));
        av = _mm512_adds_epi16(av, _mm512_loadu_si512((const __m512i*)(wt + l)));
        av = _mm512_subs_epi16(av, _mm512_loadu_si512((const __m512i*)(wf + l)));
        _mm512_storeu_si512((__m512i*)(acc + l), av);
    }
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 16) {
        __m256i av = _mm256_loadu_si256((const __m256i*)(acc + l));
        av = _mm256_adds_epi16(av, _mm256_loadu_si256((const __m256i*)(wt + l)));
        av = _mm256_subs_epi16(av, _mm256_loadu_si256((const __m256i*)(wf + l)));
        _mm256_storeu_si256((__m256i*)(acc + l), av);
    }
#else
    for (int l = 0; l < g_L1; ++l) {
        int v = int(acc[l]) + int(wt[l]) - int(wf[l]);
        acc[l] = int16_t(std::max(-32768, std::min(32767, v)));
    }
#endif
}

// Lanes of an empty board: the FT bias, clamped to int16.
static void init_bias(int16_t* acc) {
#if defined(__AVX512F__)
    {   // vectorized clamped bias init (was 512 scalar stores per refresh);
        // inputs pre-clamped to int16 range so signed saturation == exact
        const __m512i lo32 = _mm512_set1_epi32(-32768), hi32 = _mm512_set1_epi32(32767);
        for (int l = 0; l < g_L1; l += 32) {
            __m512i b0 = _mm512_min_epi32(hi32, _mm512_max_epi32(lo32, _mm512_loadu_si512((const __m512i*)&ft_b_i32[l])));
            __m512i b1 = _mm512_min_epi32(hi32, _mm512_max_epi32(lo32, _mm512_loadu_si512((const __m512i*)&ft_b_i32[l + 16])));
            _mm256_storeu_si256((__m256i*)(acc + l),      _mm512_cvtsepi32_epi16(b0));
            _mm256_storeu_si256((__m256i*)(acc + l + 16), _mm512_cvtsepi32_epi16(b1));
        }
    }
#else
    for (int l = 0; l < g_L1; ++l)
        acc[l] = int16_t(std::max(-32768, std::min(32767, ft_b_i32[l])));
#endif
}

static void refresh_perspective(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    init_bias(a.v[p]);
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        add_feature(a, p, ksq, Square(sq), pc);
    }
}

// Refresh cache: per perspective and own-king square, the lanes of the board
// that king square last saw. A king move then applies only the pieces that
// differ from that board instead of rebuilding from every piece. The cached
// sums are exact int32, saturated only on the copy into the int16
// accumulator, so an entry reused for a whole game never drifts. Entries
// built under a different net (net_id) start over from the bias.
struct alignas(64) RefreshEntry {
    int32_t acc[NNUE_L1_MAX];
    Bitboard pieces[2][6];   // [color][piece type] of the board acc describes
    uint32_t net_id = 0;
};

static inline void add_row_i32(int32_t* acc, const int16_t* w) {
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 16)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_add_epi32(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(w + l)))));
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 8)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(w + l)))));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
#endif
}
static inline void sub_row_i32(int32_t* acc, const int16_t* w) {
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 16)
        _mm512_storeu_si512((__m512i*)(acc + l), _mm512_sub_epi32(
            _mm512_loadu_si512((const __m512i*)(acc + l)),
            _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(w + l)))));
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 8)
        _mm256_storeu_si256((__m256i*)(acc + l), _mm256_sub_epi32(
            _mm256_loadu_si256((const __m256i*)(acc + l)),
            _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(w + l)))));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] -= w[l];
#endif
}
// int32 lanes -> int16 lanes with signed saturation.
static inline void store_saturated(int16_t* dst, const int32_t* src) {
#if defined(__AVX512F__)
    for (int l = 0; l < g_L1; l += 16)
        _mm256_storeu_si256((__m256i*)(dst + l),
            _mm512_cvtsepi32_epi16(_mm512_loadu_si512((const __m512i*)(src + l))));
#elif defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 16) {
        __m256i lo = _mm256_loadu_si256((const __m256i*)(src + l));
        __m256i hi = _mm256_loadu_si256((const __m256i*)(src + l + 8));
        // packs works per 128-bit half: reorder the 64-bit quarters back
        __m256i p = _mm256_permute4x64_epi64(_mm256_packs_epi32(lo, hi), 0xD8);
        _mm256_storeu_si256((__m256i*)(dst + l), p);
    }
#else
    for (int l = 0; l < g_L1; ++l) dst[l] = int16_t(std::max(-32768, std::min(32767, src[l])));
#endif
}

static inline RefreshEntry* refresh_cache() {
    thread_local std::vector<RefreshEntry> store(2 * NUM_SQ);
    return store.data();
}

static void refresh_perspective_cached(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    RefreshEntry& e = refresh_cache()[p * NUM_SQ + ksq];
    if (e.net_id != g_net_id) {
        std::memcpy(e.acc, ft_b_i32.data(), sizeof(int32_t) * size_t(g_L1));
        std::memset(e.pieces, 0, sizeof(e.pieces));
        e.net_id = g_net_id;
    }
    for (int c = 0; c < 2; ++c)
        for (int pt = PAWN; pt <= KING; ++pt) {
            Bitboard now = pos.pieces(Color(c), PieceType(pt));
            Bitboard was = e.pieces[c][pt];
            if (now == was) continue;
            Piece pc = make_piece(Color(c), PieceType(pt));
            for (Bitboard gone = was & ~now; gone; )
                sub_row_i32(e.acc, feature_row(white_pov, ksq, pop_lsb(gone), pc));
            for (Bitboard added = now & ~was; added; )
                add_row_i32(e.acc, feature_row(white_pov, ksq, pop_lsb(added), pc));
            e.pieces[c][pt] = now;
        }
    store_saturated(a.v[p], e.acc);
}

void refresh(Position& pos) {
    if (!g_loaded) return;
    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
    Accumulator& a = acc_stack[pos.state_ply()];
    refresh_perspective(pos, a, 0);
    refresh_perspective(pos, a, 1);
    if (pr) { st_refresh.cyc += rdtsc() - t0; st_refresh.n++; }
}

void copy_for_null(Position& pos) {
    if (!g_loaded) return;
    Accumulator* const acc_stack = nnue_acc_stack();
    int ply = pos.state_ply();
    acc_stack[ply] = acc_stack[ply - 1];   // board unchanged → child = parent
}

void update(Position& pos, Move m, Piece moved, PieceType captured) {
    if (!g_loaded) return;

    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
    int ply = pos.state_ply();
    Accumulator& a = acc_stack[ply];

    Square from = m.from(), to = m.to();
    Color us = color_of_piece(moved);
    Color them = ~us;
    PieceType moved_pt = piece_type_of(moved);
    bool king_moved = (moved_pt == KING);
    bool promo = m.is_promotion();
    bool ep = m.is_en_passant();
    bool castling = m.is_castling();
    PieceType from_pt = promo ? PAWN : moved_pt;
    PieceType to_pt   = promo ? PieceType(m.promotion_type()) : moved_pt;

    // Prefetch the feature weight rows that will be needed for the delta, OVERLAPPING
    // with the 4KB copy below. This hides the L3→L1 latency (~100 cycles) behind the
    // memcpy latency (~200 cycles). No engine does this — SF's flat layout doesn't need it,
    // but our transposed layout has poor spatial locality per feature access.
    {
        int wk = static_cast<int>(pos.king_sq(WHITE));
        int bk = static_cast<int>(pos.king_sq(BLACK));
        Piece dep_piece = make_piece(us, from_pt);
        Piece arr_piece = make_piece(us, to_pt);
        for (int i = 0; i < 4; ++i) {  // up to 4 prefetches per perspective
            int idx = -1;
            if (i == 0) idx = halfka_idx(true,  wk, from, dep_piece);
            else if (i == 1) idx = halfka_idx(true,  wk, to, arr_piece);
            else if (i == 2) idx = halfka_idx(false, bk, from, dep_piece);
            else idx = halfka_idx(false, bk, to, arr_piece);
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
            _mm_prefetch((const char*)&ft_w[static_cast<size_t>(idx) * g_L1], _MM_HINT_T0);
#endif
        }
    }

    acc_stack[ply] = acc_stack[ply - 1];   // copy parent → child (overlaps with prefetch)

    for (int p = 0; p < 2; ++p) {
        bool white_pov = (p == 0);
        bool our_king_moved = king_moved && ((us == WHITE) == white_pov);
        if (our_king_moved) {
            long long kr_t0 = pr ? rdtsc() : 0;   // #51: independent king-refresh cost (not derived)
            refresh_perspective_cached(pos, a, p);   // king_sq changed: rebuild via the refresh cache
            if (pr) { st_king_refresh.cyc += rdtsc() - kr_t0; st_king_refresh.n++; }
            continue;
        }
        long long inc_t0 = pr ? rdtsc() : 0;   // #51: independent incremental cost (closes last derived number)
        int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
        move_feature(a, p, ksq, from, to, make_piece(us, from_pt), make_piece(us, to_pt));
        if (captured != PT_NONE && !ep)
            remove_feature(a, p, ksq, to, make_piece(them, captured));
        if (ep) {
            Square cap_sq = Square(to - (us == WHITE ? 8 : -8));
            remove_feature(a, p, ksq, cap_sq, make_piece(them, PAWN));
        }
        if (castling) {
            Square rfrom, rto;
            if (to == (us == WHITE ? Square(G1) : Square(G8))) {
                rfrom = us == WHITE ? Square(H1) : Square(H8);
                rto   = us == WHITE ? Square(F1) : Square(F8);
            } else {
                rfrom = us == WHITE ? Square(A1) : Square(A8);
                rto   = us == WHITE ? Square(D1) : Square(D8);
            }
            remove_feature(a, p, ksq, rfrom, make_piece(us, ROOK));
            add_feature(a, p, ksq, rto,   make_piece(us, ROOK));
        }
        if (pr) { st_incremental.cyc += rdtsc() - inc_t0; st_incremental.n++; }
    }
    if (pr) { st_update.cyc += rdtsc() - t0; st_update.n++; }
}


Value evaluate(const Position& pos) {
    bool pr = profile_on();
    long long t0 = pr ? rdtsc() : 0;
    Accumulator* const acc_stack = nnue_acc_stack();
#if !defined(NDEBUG) || defined(NNUE_FORCE_SELFCHECK)
    // Self-check (debug builds, or release with -DNNUE_FORCE_SELFCHECK): recompute the
    // accumulator from scratch and compare to the incrementally-maintained one. Catches
    // any bug in update()'s feature deltas immediately.
    if (g_loaded) {
        const Accumulator& inc = acc_stack[pos.state_ply()];
        for (int p = 0; p < 2; ++p) {
            bool white_pov = (p == 0);
            int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
            for (int l = 0; l < g_L1; ++l) {
                int32_t ref = ft_b_i32[l];
                for (int sq = 0; sq < NUM_SQ; ++sq) {
                    Piece pc = pos.piece_on(Square(sq));
                    if (pc == NO_PIECE) continue;
                    int idx = halfka_idx(white_pov, ksq, sq, pc);
                    // the loop computed idx and discarded it — the "reference"
                    // was the bias alone, so any real position aborted here
                    ref += ft_w[(size_t)idx * g_L1 + l];
                }
                // int16 saturating acc: saturated lanes differ from the exact
                // reference in raw value but are IDENTICAL after the SCReLU clip
                // (window [0,8192] sits 4x below saturation). Compare activations.
                // g_sat_events tracks how often the exact value even reaches the
                // int16 boundary — it should stay at zero under range-constrained
                // training; if it grows, the fused/sequential divergence above
                // becomes live and the training constraint has failed.
                if (std::fabs(ref) >= 32768.0f) ++g_sat_events;
                float ref_a = std::min(1.0f, std::max(0.0f, ref * FT_WINV));
                float inc_a = std::min(1.0f, std::max(0.0f, inc.v[p][l] * FT_WINV));
                if (std::fabs(ref_a - inc_a) > 1.0f/127.0f + 1e-6) {
                    std::fprintf(stderr, "NNUE ACC MISMATCH ply=%d p=%d l=%d ref=%.4f inc=%.4f\n",
                                 pos.state_ply(), p, l, ref, inc.v[p][l]);
                    std::abort();
                }
            }
        }
    }
#endif
    const Accumulator& a = acc_stack[pos.state_ply()];
    int L1 = g_L1, L2 = g_L2, L3 = g_L3;
    bool stm_white = (pos.side_to_move() == WHITE);
    const int16_t* acc_stm  = stm_white ? a.v[0] : a.v[1];
    const int16_t* acc_nstm = stm_white ? a.v[1] : a.v[0];


#if defined(__AVX2__)
    // ---- int8 path: FUSED SCReLU+quant8 (no intermediate h[] — saves 8KB L1 traffic/eval) ----
    if (g_int8) {
        uint8_t h_i8[2 * NNUE_L1_MAX];
        const __m256 sc127 = _mm256_set1_ps(127.0f);  // shared for quant8 (L3/out)
#if defined(__AVX512F__)
        // AVX-512, 64 lanes per pass. Clamping to the SCReLU window [0, 8192]
        // in int16 and then rounding fl(fl(a*a) * 127*2^-26) gives the same
        // bytes as the float clamp-square-scale below: scaling by a power of
        // two never rounds, so both round the same product (all 8193 window
        // values checked). Packing 4 vectors at a time replaces two narrowing
        // converts per 16 lanes.
        if (L1 % 64 == 0) {
            const __m512i lo16 = _mm512_setzero_si512(), hi16 = _mm512_set1_epi16(8192);
            const __m512 k = _mm512_set1_ps(127.0f / 67108864.0f);   // 127 * 2^-26, exact
            // packs/packus work inside 128-bit lanes; this dword permute (a 4x4
            // transpose of 4-byte groups) restores input order
            const __m512i order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
            auto act16 = [&](__m256i v) {   // 16 clamped int16 -> 16 int32 in [0, 127]
                const __m512 f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v));
                return _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_mul_ps(f, f), k));
            };
            for (int side = 0; side < 2; ++side) {
                const int16_t* in = side == 0 ? acc_stm : acc_nstm;
                uint8_t* out = h_i8 + side * L1;
                for (int l = 0; l < L1; l += 64) {
                    const __m512i x0 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(in + l), lo16), hi16);
                    const __m512i x1 = _mm512_min_epi16(_mm512_max_epi16(_mm512_loadu_si512(in + l + 32), lo16), hi16);
                    const __m512i q0 = act16(_mm512_castsi512_si256(x0)), q1 = act16(_mm512_extracti64x4_epi64(x0, 1));
                    const __m512i q2 = act16(_mm512_castsi512_si256(x1)), q3 = act16(_mm512_extracti64x4_epi64(x1, 1));
                    const __m512i b = _mm512_packus_epi16(_mm512_packs_epi32(q0, q1), _mm512_packs_epi32(q2, q3));
                    _mm512_storeu_si512(out + l, _mm512_permutexvar_epi32(order, b));
                }
            }
        } else {
        const __m512 z16 = _mm512_setzero_ps(), one16 = _mm512_set1_ps(1.0f), sc16 = _mm512_set1_ps(127.0f);
        const __m512 winv16 = _mm512_set1_ps(FT_WINV);
        for (int l = 0; l < L1; l += 16) {
            __m512 s = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(acc_stm + l)))), winv16);
            __m512 c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
            __m512 sq = _mm512_mul_ps(c, c);
            __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
            // Pack 16 int32 → 16 uint8: int32→int16→uint8
            __m256i i16 = _mm512_cvtepi32_epi16(i32);  // AVX512F: 16 int32 → 16 int16
            __m128i i8 = _mm256_cvtepi16_epi8(i16);     // AVX512F: 16 int16 → 16 int8
            // c=clip(s,0,1) -> c² in [0,1] -> ×127 -> [0,127] exactly: no clamp needed.
            _mm_storeu_si128((__m128i*)(h_i8 + l), i8);
            // nstm
            s = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i*)(acc_nstm + l)))), winv16);
            c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
            sq = _mm512_mul_ps(c, c);
            i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
            i16 = _mm512_cvtepi32_epi16(i32);
            i8 = _mm256_cvtepi16_epi8(i16);
            _mm_storeu_si128((__m128i*)(h_i8 + L1 + l), i8);
        }
        }
#else
        const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), winv8 = _mm256_set1_ps(FT_WINV);
        for (int l = 0; l < L1; l += 8) {
            __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_stm + l)))), winv8);
            __m256 c = _mm256_min_ps(_mm256_max_ps(s, z), one);
            __m256 sq = _mm256_mul_ps(c, c);
            __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
            __m128i lo = _mm256_castsi256_si128(i32), hi = _mm256_extracti128_si256(i32, 1);
            _mm_storel_epi64((__m128i*)(h_i8 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
            s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_nstm + l)))), winv8);
            c = _mm256_min_ps(_mm256_max_ps(s, z), one);
            sq = _mm256_mul_ps(c, c);
            i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
            lo = _mm256_castsi256_si128(i32); hi = _mm256_extracti128_si256(i32, 1);
            _mm_storel_epi64((__m128i*)(h_i8 + L1 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
        }
#endif
        if (pr) { for (int i = 0; i < 2*L1; ++i) g_act_nz += (h_i8[i] != 0); g_act_tot += 2*L1; }
        const float cs2 = g_s2 * 127.0f, cs3 = g_s3 * 127.0f, cso = g_so * 127.0f;
        // 49 float divisions per eval -> 3 + 46 multiplies (sub-ULP diff, far below
        // the 1/127 activation quantization step; fingerprint-verified).
        const float ics2 = 1.0f / cs2, ics3 = 1.0f / cs3, icso = 1.0f / cso;
        float h2[NNUE_L1_MAX];
#if defined(__AVX512VNNI__)
        if (g_l2_by_input) {
            // One VPDPBUSD per 4 inputs adds their products to all 16 output sums
            // (lane o = output o), so no per-output horizontal sum is needed.
            // Integer adds associate: each sum equals the per-output dot exactly.
            const int n4 = 2 * L1 / 4;
            const int8_t* w = l2_w_i8.data();
            auto chunk = [&](int c) {
                int32_t v; std::memcpy(&v, h_i8 + 4 * c, 4);
                return _mm512_set1_epi32(v);
            };
            __m512i s0 = _mm512_setzero_si512(), s1 = s0, s2 = s0, s3 = s0;
            int c = 0;
            for (; c + 4 <= n4; c += 4) {
                s0 = _mm512_dpbusd_epi32(s0, chunk(c + 0), _mm512_loadu_si512(w + 64 * (c + 0)));
                s1 = _mm512_dpbusd_epi32(s1, chunk(c + 1), _mm512_loadu_si512(w + 64 * (c + 1)));
                s2 = _mm512_dpbusd_epi32(s2, chunk(c + 2), _mm512_loadu_si512(w + 64 * (c + 2)));
                s3 = _mm512_dpbusd_epi32(s3, chunk(c + 3), _mm512_loadu_si512(w + 64 * (c + 3)));
            }
            for (; c < n4; ++c)
                s0 = _mm512_dpbusd_epi32(s0, chunk(c), _mm512_loadu_si512(w + 64 * c));
            alignas(64) int32_t d[16];
            _mm512_store_si512(d, _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3)));
            for (int o = 0; o < 16; ++o) h2[o] = clip01(l2_b[o] + d[o] * ics2);
        } else
        // VPDPBUSD: 64 MACs/instruction. Per-output dot (output-major weights, no reorder).
        for (int o = 0; o < L2; ++o)
            h2[o] = clip01(l2_b[o] + dot_i8_vnni(&l2_w_i8[static_cast<size_t>(o) * 2 * L1], h_i8, 2 * L1) * ics2);
#else
        // AVX2 SF-style chunk L2 (chunk-major weights, reordered at load).
        const __m256i ones = _mm256_set1_epi16(1);
        const int total_in = 2 * L1;
        const int num_chunks = total_in / 32;
        for (int pass = 0; pass < (L2 + 7) / 8; ++pass) {
            int base_o = pass * 8;
            int n_out = std::min(8, L2 - base_o);
            __m256i acc8[8];
            for (int i = 0; i < 8; ++i) acc8[i] = _mm256_setzero_si256();
            for (int c = 0; c < num_chunks; ++c) {
                __m256i hv = _mm256_loadu_si256((const __m256i*)(h_i8 + c * 32));
                const int8_t* wp = &l2_w_i8[(static_cast<size_t>(c) * L2 + base_o) * 32];
                for (int o = 0; o < n_out; ++o) {
                    __m256i wv = _mm256_loadu_si256((const __m256i*)(wp + o * 32));
                    __m256i p16 = _mm256_maddubs_epi16(hv, wv);
                    acc8[o] = _mm256_add_epi32(acc8[o], _mm256_madd_epi16(p16, ones));
                }
            }
            for (int o = 0; o < n_out; ++o) {
                __m128i lo = _mm256_castsi256_si128(acc8[o]);
                __m128i hi = _mm256_extracti128_si256(acc8[o], 1);
                __m128i sv = _mm_hadd_epi32(_mm_add_epi32(lo, hi), _mm_setzero_si128());
                int32_t d = _mm_cvtsi128_si32(_mm_hadd_epi32(sv, sv));
                h2[base_o + o] = clip01(l2_b[base_o + o] + d * ics2);
            }
        }
#endif // AVX2 SF-L2 vs AVX512 VNNI L2
        // L3/out (shared — uses VNNI when available, AVX2 otherwise)
        uint8_t h2_i8[NNUE_L1_MAX];
        for (int l = 0; l < L2; l += 8) quant8(h2 + l, h2_i8 + l, sc127);
        float h3[NNUE_L1_MAX];
#if defined(__AVX512VNNI__)
        if (g_l3_batched) {
            // All L3 dots via 8 dpbusd + lane-group reduction (bit-exact: integer
            // adds associate). a512: every 4-lane group holds all 16 h2 bytes.
            const __m128i h2_128 = _mm_loadu_si128((const __m128i*)h2_i8);
            const __m512i av = _mm512_broadcast_i32x4(h2_128);   // replicate 16B into each lane group
            const __m512i pairswap = _mm512_setr_epi32(1,0,3,2, 5,4,7,6, 9,8,11,10, 13,12,15,14);
            const __m512i groupswap = _mm512_setr_epi32(2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13);
            // Butterfly leaves each group's sum in ALL its lanes; the four output
            // dots sit at lanes {0,4,8,12} — gather them into 0..3 before extract.
            const __m512i pick0408 = _mm512_setr_epi32(0,4,8,12, 0,4,8,12, 0,4,8,12, 0,4,8,12);
            int o = 0;
            for (int v = 0; v < L3 / 4; ++v, o += 4) {
                __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), av,
                                _mm512_loadu_si512((const __m512i*)&l3_w_i8[v * 64]));
                __m512i t1 = _mm512_add_epi32(acc, _mm512_permutexvar_epi32(pairswap, acc));
                __m512i t2 = _mm512_add_epi32(t1, _mm512_permutexvar_epi32(groupswap, t1));
                // t2 lane 4k = dot for output (v*4 + k)
                __m128i sums = _mm512_castsi512_si128(_mm512_permutexvar_epi32(pick0408, t2));
                h3[o+0] = clip01(l3_b[o+0] + _mm_extract_epi32(sums, 0) * ics3);
                h3[o+1] = clip01(l3_b[o+1] + _mm_extract_epi32(sums, 1) * ics3);
                h3[o+2] = clip01(l3_b[o+2] + _mm_extract_epi32(sums, 2) * ics3);
                h3[o+3] = clip01(l3_b[o+3] + _mm_extract_epi32(sums, 3) * ics3);
            }
        } else
        for (int o = 0; o < L3; ++o)
            h3[o] = clip01(l3_b[o] + dot_i8_vnni_small(&l3_w_i8[static_cast<size_t>(o) * L2], h2_i8, L2) * ics3);
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
        float ov = out_b + dot_i8_vnni_small(out_w_i8.data(), h3_i8, L3) * icso;
#else
        for (int o = 0; o < L3; ++o)
            h3[o] = clip01(l3_b[o] + dot_i8(&l3_w_i8[static_cast<size_t>(o) * L2], h2_i8, L2) * ics3);
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
        float ov = out_b + dot_i8(out_w_i8.data(), h3_i8, L3) * icso;
#endif
        Value v = static_cast<Value>(std::llround(ov * OUT_SCALE));
        if (pr) { st_eval.cyc += rdtsc() - t0; st_eval.n++; }
        return v;
    }
    // ---- float L2/L3/out path (LNN1 nets) ----
    float h[2 * NNUE_L1_MAX];
    const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f), winv8 = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < L1; l += 8) {
        _mm256_storeu_ps(h + l,      screlu8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_stm + l)))), winv8), z, one));
        _mm256_storeu_ps(h + L1 + l, screlu8(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(acc_nstm + l)))), winv8), z, one));
    }
    float h2[NNUE_L1_MAX];
    for (int o = 0; o < L2; ++o) {
        const float* w = &l2_w[o * 2 * L1];
        __m256 accv = _mm256_setzero_ps();
        for (int i = 0; i < 2 * L1; i += 8)
            accv = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(h + i), accv);
        h2[o] = clip01(hsum256(accv) + l2_b[o]);
    }
    // L3: SIMD dot products over L2 inputs.
    float h3[NNUE_L1_MAX];
    for (int o = 0; o < L3; ++o) {
        const float* w = &l3_w[o * L2];
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= L2; i += 8)
            accv = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(h2 + i), accv);
        float s = hsum256(accv) + l3_b[o];
        for (; i < L2; ++i) s += w[i] * h2[i];
        h3[o] = clip01(s);
    }
#else
    // ---- non-AVX2 path (ARM etc.) ----
#if defined(__ARM_FEATURE_DOTPROD)
    if (g_int8) {
        uint8_t h_i8[2 * NNUE_L1_MAX];
        // SCReLU + quantize: acc int16 → float → clip01 → square → ×127 → uint8
        // 8 elements per iteration via NEON
        for (int l = 0; l < L1; l += 8) {
            // stm half
            int16x8_t v = vld1q_s16(acc_stm + l);
            int32x4_t lo = vmovl_s16(vget_low_s16(v));
            int32x4_t hi = vmovl_s16(vget_high_s16(v));
            float32x4_t flo = vcvtq_f32_s32(lo);
            float32x4_t fhi = vcvtq_f32_s32(hi);
            flo = vmulq_n_f32(flo, FT_WINV);
            fhi = vmulq_n_f32(fhi, FT_WINV);
            flo = vminq_f32(vmaxq_f32(flo, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
            fhi = vminq_f32(vmaxq_f32(fhi, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
            flo = vmulq_f32(flo, flo);
            fhi = vmulq_f32(fhi, fhi);
            flo = vmulq_n_f32(flo, 127.0f);
            fhi = vmulq_n_f32(fhi, 127.0f);
            // vcvtnq = round-to-nearest-even, matching _mm*_cvtps_epi32 on x86;
            // vcvtq truncates and drifts up to one quantization step per lane.
            int32x4_t ilo = vcvtnq_s32_f32(flo);
            int32x4_t ihi = vcvtnq_s32_f32(fhi);
            int16x8_t i16 = vcombine_s16(vqmovn_s32(ilo), vqmovn_s32(ihi));
            uint8x8_t u8 = vqmovun_s16(i16);
            vst1_u8(h_i8 + l, u8);
            // nstm half
            v = vld1q_s16(acc_nstm + l);
            lo = vmovl_s16(vget_low_s16(v));
            hi = vmovl_s16(vget_high_s16(v));
            flo = vcvtq_f32_s32(lo);
            fhi = vcvtq_f32_s32(hi);
            flo = vmulq_n_f32(flo, FT_WINV);
            fhi = vmulq_n_f32(fhi, FT_WINV);
            flo = vminq_f32(vmaxq_f32(flo, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
            fhi = vminq_f32(vmaxq_f32(fhi, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f));
            flo = vmulq_f32(flo, flo);
            fhi = vmulq_f32(fhi, fhi);
            flo = vmulq_n_f32(flo, 127.0f);
            fhi = vmulq_n_f32(fhi, 127.0f);
            ilo = vcvtnq_s32_f32(flo);
            ihi = vcvtnq_s32_f32(fhi);
            i16 = vcombine_s16(vqmovn_s32(ilo), vqmovn_s32(ihi));
            u8 = vqmovun_s16(i16);
            vst1_u8(h_i8 + L1 + l, u8);
        }
        const float cs2 = g_s2 * 127.0f, cs3 = g_s3 * 127.0f, cso = g_so * 127.0f;
        const float ics2 = 1.0f / cs2, ics3 = 1.0f / cs3, icso = 1.0f / cso;
        const int total_in = 2 * L1;
        float h2[NNUE_L1_MAX];
        // L2: output-major dots via SDOT (int8 weights × uint8 activations)
        for (int o = 0; o < L2; ++o) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t* w = &l2_w_i8[static_cast<size_t>(o) * total_in];
            for (int i = 0; i < total_in; i += 16) {
                int8x16_t wv = vld1q_s8(w + i);
                uint8x16_t hv = vld1q_u8(h_i8 + i);
                acc = vdotq_s32(acc, wv, hv);
            }
            int32_t dot = vaddvq_s32(acc);
            float s = l2_b[o] + dot * ics2;
            float c = s < 0 ? 0 : (s > 1 ? 1 : s);
            h2[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
        }
        uint8_t h2_i8[NNUE_L1_MAX];
        for (int o = 0; o < L2; ++o) h2_i8[o] = (uint8_t)std::lrintf(h2[o] * 127.0f);
        float h3[NNUE_L1_MAX];
        // L3: same SDOT pattern over L2 inputs
        for (int o = 0; o < L3; ++o) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t* w = &l3_w_i8[static_cast<size_t>(o) * L2];
            for (int i = 0; i < L2; i += 16) {
                int8x16_t wv = vld1q_s8(w + i);
                uint8x16_t hv = vld1q_u8(h2_i8 + i);
                acc = vdotq_s32(acc, wv, hv);
            }
            int32_t dot = vaddvq_s32(acc);
            float s = l3_b[o] + dot * ics3;
            float c = s < 0 ? 0 : (s > 1 ? 1 : s);
            h3[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
        }
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int o = 0; o < L3; ++o) h3_i8[o] = (uint8_t)std::lrintf(h3[o] * 127.0f);
        // output layer: L3=32 is a multiple of 16, no tail handling needed
        int32x4_t acc = vdupq_n_s32(0);
        for (int i = 0; i < L3; i += 16)
            acc = vdotq_s32(acc, vld1q_s8(out_w_i8.data() + i), vld1q_u8(h3_i8 + i));
        int32_t dot = vaddvq_s32(acc);
        float ov = out_b + dot * icso;
        return static_cast<Value>(std::llround(ov * OUT_SCALE));
    }
#else
    // Scalar fallback for ARM without dotprod (older ARMv8.0)
    if (g_int8) {
        uint8_t h_i8[2 * NNUE_L1_MAX];
        for (int l = 0; l < L1; ++l) {
            float s  = acc_stm[l]  * FT_WINV; float c  = s  < 0 ? 0 : (s  > 1 ? 1 : s);
            h_i8[l]        = (uint8_t)std::lrintf(c * c * 127.0f);
            float s2 = acc_nstm[l] * FT_WINV; float c2 = s2 < 0 ? 0 : (s2 > 1 ? 1 : s2);
            h_i8[L1 + l]   = (uint8_t)std::lrintf(c2 * c2 * 127.0f);
        }
        const float cs2 = g_s2 * 127.0f, cs3 = g_s3 * 127.0f, cso = g_so * 127.0f;
        const float ics2 = 1.0f / cs2, ics3 = 1.0f / cs3, icso = 1.0f / cso;
        const int total_in = 2 * L1;
        float h2[NNUE_L1_MAX];
        for (int o = 0; o < L2; ++o) {
            int32_t acc = 0;
            const int8_t* w = &l2_w_i8[static_cast<size_t>(o) * total_in];
            for (int i = 0; i < total_in; ++i) acc += w[i] * h_i8[i];
            float s = l2_b[o] + acc * ics2;
            float c = s < 0 ? 0 : (s > 1 ? 1 : s);
            h2[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
        }
        uint8_t h2_i8[NNUE_L1_MAX];
        for (int o = 0; o < L2; ++o) h2_i8[o] = (uint8_t)std::lrintf(h2[o] * 127.0f);
        float h3[NNUE_L1_MAX];
        for (int o = 0; o < L3; ++o) {
            int32_t acc = 0;
            const int8_t* w = &l3_w_i8[static_cast<size_t>(o) * L2];
            for (int i = 0; i < L2; ++i) acc += w[i] * h2_i8[i];
            float s = l3_b[o] + acc * ics3;
            float c = s < 0 ? 0 : (s > 1 ? 1 : s);
            h3[o] = c * c;  // SCReLU: square after clamp, matching the x86 path
        }
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int o = 0; o < L3; ++o) h3_i8[o] = (uint8_t)std::lrintf(h3[o] * 127.0f);
        int32_t acc = 0;
        for (int i = 0; i < L3; ++i) acc += out_w_i8[i] * h3_i8[i];
        float ov = out_b + acc * icso;
        return static_cast<Value>(std::llround(ov * OUT_SCALE));
    }
#endif // __ARM_FEATURE_DOTPROD
    float h[2 * NNUE_L1_MAX];
    float h2[NNUE_L1_MAX];
    for (int l = 0; l < L1; ++l) { h[l] = clip01(acc_stm[l] * FT_WINV); h[L1 + l] = clip01(acc_nstm[l] * FT_WINV); }
    for (int o = 0; o < L2; ++o) {
        float s = l2_b[o]; const float* w = &l2_w[o * 2 * L1];
        for (int i = 0; i < 2 * L1; ++i) s += w[i] * h[i];
        h2[o] = clip01(s);
    }
    float h3[NNUE_L1_MAX];
    for (int o = 0; o < L3; ++o) {
        float s = l3_b[o]; const float* w = &l3_w[o * L2];
        for (int i = 0; i < L2; ++i) s += w[i] * h2[i];
        h3[o] = clip01(s);
    }
#endif
    float ov = out_b;
    for (int i = 0; i < L3; ++i) ov += out_w[i] * h3[i];
    Value v = static_cast<Value>(std::llround(ov * OUT_SCALE));
    if (pr) { st_eval.cyc += rdtsc() - t0; st_eval.n++; }
    return v;
}

} // namespace luminex::nnue
