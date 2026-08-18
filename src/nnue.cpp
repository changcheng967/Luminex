// nnue.cpp — NNUE evaluation with an INCREMENTAL accumulator (float, stage 1).
//
// v1 rebuilt the feature transformer from scratch every eval (135K NPS — 15x slower
// than HCE). This version keeps the accumulator in a per-thread stack (acc_stack[],
// indexed by Position::st_ply()) and updates it incrementally on make_move, so
// evaluate() just reads it. Validated to match the full-refresh reference exactly
// (debug self-check) before int8/SIMD (stage 2).
//
// The accumulator lives in a thread_local global (NOT in StateInfo) — putting it in
// StateInfo bloated state_stack[2048] to ~8MB and overflowed the stack (Position is
// stack-local). thread_local also makes it safe across Luminex's helper threads.
//
// HalfKAv2_hm feature indexer mirrored EXACTLY from nnue/luminex_nnue_train.py.
#include "nnue.h"
#include "board.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
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
    __m512i acc = _mm512_setzero_si512();
    int i = 0;
    for (; i + 64 <= n; i += 64)
        acc = _mm512_dpbusd_epi32(acc,
            _mm512_loadu_si512((const __m512i*)(a + i)),
            _mm512_loadu_si512((const __m512i*)(w + i)));
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

// ---- profiling counters (data-driven: find the NPS bottleneck from numbers) ----
namespace { struct Stat { long long n = 0, cyc = 0; }; }
static Stat st_eval, st_update, st_refresh, st_king_refresh, st_incremental;
static long long g_act_nz = 0, g_act_tot = 0;   // activation sparsity (non-zero L1 outputs after SCReLU)
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
    std::fprintf(stderr, "NNUE profile: evals=%lld (%.1fM cyc, %.0f cyc/eval) | updates=%lld (%.1fM cyc, %.0f cyc/upd) | king-refreshes=%lld (%.0f cyc/kr) | incrementals=%lld (%.0f cyc/inc) | refreshes=%lld (%.1fM cyc) | L1-act density=%.1f%%\n",
                 st_eval.n, st_eval.cyc/1000000.0, st_eval.n ? double(st_eval.cyc)/st_eval.n : 0.0,
                 st_update.n, st_update.cyc/1000000.0, st_update.n ? double(st_update.cyc)/st_update.n : 0.0,
                 st_king_refresh.n, st_king_refresh.n ? double(st_king_refresh.cyc)/st_king_refresh.n : 0.0,
                 st_incremental.n, st_incremental.n ? double(st_incremental.cyc)/st_incremental.n : 0.0,
                 st_refresh.n, st_refresh.cyc/1000000.0,
                 g_act_tot ? 100.0*g_act_nz/g_act_tot : 0.0);
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
static std::vector<int16_t> ft_w;   // (NUM_INPUTS, L1) int16, transposed
static std::vector<float> ft_b;   // (L1,)
static std::vector<float> l2_w, l2_b, l3_w, l3_b, out_w;
static float out_b = 0.0f;
// int8 path (LNI8): L2/L3/out weights as int8 + per-layer scales. FT stays float.
static bool g_int8 = false;
static std::vector<int8_t> l2_w_i8, l3_w_i8, out_w_i8;
static float g_s2 = 1.0f, g_s3 = 1.0f, g_so = 1.0f;  // weight quant scales
static bool g_loaded = false;
static bool g_enabled = false;

// Per-thread accumulator stack, indexed by Position::st_ply(). v[perspective][neuron].
// Cache-line aligned so each 4KB accumulator starts on a 64B boundary (cleaner loads
// during the SCReLU/quant8 pass and the feature-delta add).
struct alignas(64) Accumulator { float v[2][NNUE_L1_MAX]; };
// Heap-allocated per thread, NOT a thread_local C-array: an 8MB thread_local array
// reserves 8MB of *static TLS* for every thread and overflows the stack at creation.
// The vector object is ~24 bytes of TLS; its 8MB buffer lives on the heap and is freed
// when the thread exits (RAII).
static inline Accumulator* nnue_acc_stack() {
    thread_local std::vector<Accumulator> store(NNUE_MAX_PLY);
    return store.data();
}

bool loaded() { return g_loaded; }
bool enabled() { return g_loaded && g_enabled; }
void set_enabled(bool on) { g_enabled = on; }
int l1() { return g_L1; }

static bool read_tensor(std::ifstream& f, std::vector<float>& v, int expected) {
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (n != expected) { std::fprintf(stderr, "nnue: size mismatch %d != %d\n", n, expected); return false; }
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), sizeof(float) * n);
    return f.good();
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "nnue: cannot open %s\n", path.c_str()); return false; }
    char magic[4];
    f.read(magic, 4);
    std::string mg(magic, 4);
    if (mg != "LNN1" && mg != "LNI8") { std::fprintf(stderr, "nnue: bad magic\n"); return false; }
    g_int8 = (mg == "LNI8");
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    int L1 = hdr[0], L2 = hdr[1], L3 = hdr[2], NI = hdr[3];
    if (NI != NUM_INPUTS) { std::fprintf(stderr, "nnue: NUM_INPUTS mismatch %d\n", NI); return false; }
    if (L1 <= 0 || L1 > NNUE_L1_MAX) { std::fprintf(stderr, "nnue: L1=%d out of range (max %d)\n", L1, NNUE_L1_MAX); return false; }
    g_L1 = L1; g_L2 = L2; g_L3 = L3;
    std::vector<float> ft_w_raw;
    if (!read_tensor(f, ft_w_raw, L1 * NUM_INPUTS)) return false;
    ft_w.resize(static_cast<size_t>(NUM_INPUTS) * L1);
    for (int i = 0; i < NUM_INPUTS; ++i)
        for (int l = 0; l < L1; ++l)
            ft_w[i * L1 + l] = (int16_t)std::lround(ft_w_raw[l * NUM_INPUTS + i] * FT_WSCALE);
    if (!read_tensor(f, ft_b, L1)) return false;
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
        if (!read_tensor(f, l3_b, L3)) return false;
        f.read(reinterpret_cast<char*>(&g_s3), 4);
        if (!read_i8(out_w_i8, L3)) return false;
        std::vector<float> ob;
        if (!read_tensor(f, ob, 1)) return false;
        out_b = ob[0];
        f.read(reinterpret_cast<char*>(&g_so), 4);
        // Reorder L2 weights for AVX2 SF-style chunk access (not needed for AVX-512 VNNI).
#if !defined(__AVX512VNNI__)
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
    g_loaded = true;
    std::printf("nnue: loaded %s (%s L1=%d L2=%d L3=%d)\n", path.c_str(), g_int8 ? "int8" : "float", L1, L2, L3);
    return true;
}

static inline float clip01(float x) { float c = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); return c * c; }

static inline void add_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    int idx = halfka_idx(p == 0, ksq, sq, piece);
    const int16_t* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
#if defined(__AVX2__)
    const __m256 winv = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < g_L1; l += 8) {
        __m128i w8 = _mm_loadu_si128((const __m128i*)(w + l));   // 8 int16
        __m256i w32 = _mm256_cvtepi16_epi32(w8);                  // 8 int32
        __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(w32), winv); // 8 float (scaled)
        _mm256_storeu_ps(acc + l, _mm256_add_ps(_mm256_loadu_ps(acc + l), wf));
    }
#else
    for (int l = 0; l < g_L1; ++l) acc[l] += w[l] * FT_WINV;
#endif
}
static inline void remove_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    int idx = halfka_idx(p == 0, ksq, sq, piece);
    const int16_t* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
#if defined(__AVX2__)
    const __m256 winv = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < g_L1; l += 8) {
        __m128i w8 = _mm_loadu_si128((const __m128i*)(w + l));
        __m256i w32 = _mm256_cvtepi16_epi32(w8);
        __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(w32), winv);
        _mm256_storeu_ps(acc + l, _mm256_sub_ps(_mm256_loadu_ps(acc + l), wf));
    }
#else
    for (int l = 0; l < g_L1; ++l) acc[l] -= w[l] * FT_WINV;
#endif
}

static void refresh_perspective(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    float* acc = a.v[p];
#if defined(__AVX2__)
    const __m256 winv = _mm256_set1_ps(FT_WINV);
    for (int l = 0; l < g_L1; l += 8) _mm256_storeu_ps(acc + l, _mm256_loadu_ps(&ft_b[l]));
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        const int16_t* w = &ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, pc)) * g_L1];
        for (int l = 0; l < g_L1; l += 8) {
            __m128i w8 = _mm_loadu_si128((const __m128i*)(w + l));
            __m256 wf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(w8)), winv);
            _mm256_storeu_ps(acc + l, _mm256_add_ps(_mm256_loadu_ps(acc + l), wf));
        }
    }
#else
    for (int l = 0; l < g_L1; ++l) acc[l] = ft_b[l];
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        int idx = halfka_idx(white_pov, ksq, sq, pc);
        const int16_t* w = &ft_w[static_cast<size_t>(idx) * g_L1];
        for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
    }
#endif
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
            refresh_perspective(pos, a, p);   // king_sq changed → full refresh of this perspective
            if (pr) { st_king_refresh.cyc += rdtsc() - kr_t0; st_king_refresh.n++; }
            continue;
        }
        long long inc_t0 = pr ? rdtsc() : 0;   // #51: independent incremental cost (closes last derived number)
        int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
        remove_feature(a, p, ksq, from, make_piece(us, from_pt));
        add_feature(a, p, ksq, to,   make_piece(us, to_pt));
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
#ifndef NDEBUG
    // Self-check (debug only): recompute the accumulator from scratch and compare to the
    // incrementally-maintained one. Catches any bug in update()'s feature deltas immediately.
    if (g_loaded) {
        const Accumulator& inc = acc_stack[pos.state_ply()];
        for (int p = 0; p < 2; ++p) {
            bool white_pov = (p == 0);
            int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
            for (int l = 0; l < g_L1; ++l) {
                float ref = ft_b[l];
                for (int sq = 0; sq < NUM_SQ; ++sq) {
                    Piece pc = pos.piece_on(Square(sq));
                    if (pc == NO_PIECE) continue;
                    ref += ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, pc)) * g_L1 + l] * FT_WINV;
                }
                if (std::fabs(ref - inc.v[p][l]) > 1e-3) {
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
    const float* acc_stm  = stm_white ? a.v[0] : a.v[1];
    const float* acc_nstm = stm_white ? a.v[1] : a.v[0];

#if defined(__AVX2__)
    // ---- int8 path: FUSED SCReLU+quant8 (no intermediate h[] — saves 8KB L1 traffic/eval) ----
    if (g_int8) {
        uint8_t h_i8[2 * NNUE_L1_MAX];
        const __m256 sc127 = _mm256_set1_ps(127.0f);  // shared for quant8 (L3/out)
#if defined(__AVX512F__)
        // AVX-512: 16 floats/instruction → 2x throughput over AVX2
        const __m512 z16 = _mm512_setzero_ps(), one16 = _mm512_set1_ps(1.0f), sc16 = _mm512_set1_ps(127.0f);
        for (int l = 0; l < L1; l += 16) {
            __m512 s = _mm512_loadu_ps(acc_stm + l);
            __m512 c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
            __m512 sq = _mm512_mul_ps(c, c);
            __m512i i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
            // Pack 16 int32 → 16 uint8: int32→int16→uint8
            __m256i i16 = _mm512_cvtepi32_epi16(i32);  // AVX512F: 16 int32 → 16 int16
            __m128i i8 = _mm256_cvtepi16_epi8(i16);     // AVX512F: 16 int16 → 16 int8
            // c=clip(s,0,1) -> c² in [0,1] -> ×127 -> [0,127] exactly: no clamp needed.
            _mm_storeu_si128((__m128i*)(h_i8 + l), i8);
            // nstm
            s = _mm512_loadu_ps(acc_nstm + l);
            c = _mm512_min_ps(_mm512_max_ps(s, z16), one16);
            sq = _mm512_mul_ps(c, c);
            i32 = _mm512_cvtps_epi32(_mm512_mul_ps(sq, sc16));
            i16 = _mm512_cvtepi32_epi16(i32);
            i8 = _mm256_cvtepi16_epi8(i16);
            _mm_storeu_si128((__m128i*)(h_i8 + L1 + l), i8);
        }
#else
        const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
        for (int l = 0; l < L1; l += 8) {
            __m256 s = _mm256_loadu_ps(acc_stm + l);
            __m256 c = _mm256_min_ps(_mm256_max_ps(s, z), one);
            __m256 sq = _mm256_mul_ps(c, c);
            __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
            __m128i lo = _mm256_castsi256_si128(i32), hi = _mm256_extracti128_si256(i32, 1);
            _mm_storel_epi64((__m128i*)(h_i8 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
            s = _mm256_loadu_ps(acc_nstm + l);
            c = _mm256_min_ps(_mm256_max_ps(s, z), one);
            sq = _mm256_mul_ps(c, c);
            i32 = _mm256_cvtps_epi32(_mm256_mul_ps(sq, sc127));
            lo = _mm256_castsi256_si128(i32); hi = _mm256_extracti128_si256(i32, 1);
            _mm_storel_epi64((__m128i*)(h_i8 + L1 + l), _mm_packus_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()));
        }
#endif
        if (pr) { for (int i = 0; i < 2*L1; ++i) g_act_nz += (h_i8[i] != 0); g_act_tot += 2*L1; }
        const float cs2 = g_s2 * 127.0f, cs3 = g_s3 * 127.0f, cso = g_so * 127.0f;
        float h2[NNUE_L1_MAX];
#if defined(__AVX512VNNI__)
        // VPDPBUSD: 64 MACs/instruction. Per-output dot (output-major weights, no reorder).
        for (int o = 0; o < L2; ++o)
            h2[o] = clip01(l2_b[o] + dot_i8_vnni(&l2_w_i8[static_cast<size_t>(o) * 2 * L1], h_i8, 2 * L1) / cs2);
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
                h2[base_o + o] = clip01(l2_b[base_o + o] + d / cs2);
            }
        }
#endif // AVX2 SF-L2 vs AVX512 VNNI L2
        // L3/out (shared — uses VNNI when available, AVX2 otherwise)
        uint8_t h2_i8[NNUE_L1_MAX];
        for (int l = 0; l < L2; l += 8) quant8(h2 + l, h2_i8 + l, sc127);
        float h3[NNUE_L1_MAX];
#if defined(__AVX512VNNI__)
        for (int o = 0; o < L3; ++o)
            h3[o] = clip01(l3_b[o] + dot_i8_vnni_small(&l3_w_i8[static_cast<size_t>(o) * L2], h2_i8, L2) / cs3);
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
        float ov = out_b + dot_i8_vnni_small(out_w_i8.data(), h3_i8, L3) / cso;
#else
        for (int o = 0; o < L3; ++o)
            h3[o] = clip01(l3_b[o] + dot_i8(&l3_w_i8[static_cast<size_t>(o) * L2], h2_i8, L2) / cs3);
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc127);
        float ov = out_b + dot_i8(out_w_i8.data(), h3_i8, L3) / cso;
#endif
        Value v = static_cast<Value>(std::llround(ov * OUT_SCALE));
        if (pr) { st_eval.cyc += rdtsc() - t0; st_eval.n++; }
        return v;
    }
    // ---- float L2/L3/out path (LNN1 nets) ----
    float h[2 * NNUE_L1_MAX];
    const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (int l = 0; l < L1; l += 8) {
        _mm256_storeu_ps(h + l,      screlu8(_mm256_loadu_ps(acc_stm + l), z, one));
        _mm256_storeu_ps(h + L1 + l, screlu8(_mm256_loadu_ps(acc_nstm + l), z, one));
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
    float h[2 * NNUE_L1_MAX];
    float h2[NNUE_L1_MAX];
    for (int l = 0; l < L1; ++l) { h[l] = clip01(acc_stm[l]); h[L1 + l] = clip01(acc_nstm[l]); }
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
