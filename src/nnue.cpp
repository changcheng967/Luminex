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

// ---- profiling counters (data-driven: find the NPS bottleneck from numbers) ----
namespace { struct Stat { long long n = 0, us = 0; }; }
static Stat st_eval, st_update, st_refresh;
static inline long long now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
void print_stats() {
    std::fprintf(stderr, "NNUE stats: evals=%lld (%lldms, %.2fus/eval) | updates=%lld (%lldms, %.2fus/upd) | refreshes=%lld (%lldms)\n",
                 st_eval.n, st_eval.us/1000, st_eval.n ? double(st_eval.us)/st_eval.n : 0.0,
                 st_update.n, st_update.us/1000, st_update.n ? double(st_update.us)/st_update.n : 0.0,
                 st_refresh.n, st_refresh.us/1000);
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
static std::vector<float> ft_w;   // (NUM_INPUTS, L1) transposed
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
struct Accumulator { float v[2][NNUE_L1_MAX]; };
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
            ft_w[i * L1 + l] = ft_w_raw[l * NUM_INPUTS + i];
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
    const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
#if defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 8)
        _mm256_storeu_ps(acc + l, _mm256_add_ps(_mm256_loadu_ps(acc + l), _mm256_loadu_ps(w + l)));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
#endif
}
static inline void remove_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    int idx = halfka_idx(p == 0, ksq, sq, piece);
    const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
#if defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 8)
        _mm256_storeu_ps(acc + l, _mm256_sub_ps(_mm256_loadu_ps(acc + l), _mm256_loadu_ps(w + l)));
#else
    for (int l = 0; l < g_L1; ++l) acc[l] -= w[l];
#endif
}

static void refresh_perspective(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    float* acc = a.v[p];
#if defined(__AVX2__)
    for (int l = 0; l < g_L1; l += 8) _mm256_storeu_ps(acc + l, _mm256_loadu_ps(&ft_b[l]));
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        const float* w = &ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, pc)) * g_L1];
        for (int l = 0; l < g_L1; l += 8)
            _mm256_storeu_ps(acc + l, _mm256_add_ps(_mm256_loadu_ps(acc + l), _mm256_loadu_ps(w + l)));
    }
#else
    for (int l = 0; l < g_L1; ++l) acc[l] = ft_b[l];
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        int idx = halfka_idx(white_pov, ksq, sq, pc);
        const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
        for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
    }
#endif
}

void refresh(Position& pos) {
    if (!g_loaded) return;
    long long _t = now_us();
    Accumulator* const acc_stack = nnue_acc_stack();
    Accumulator& a = acc_stack[pos.state_ply()];
    refresh_perspective(pos, a, 0);
    refresh_perspective(pos, a, 1);
    st_refresh.us += now_us() - _t; ++st_refresh.n;
}

void copy_for_null(Position& pos) {
    if (!g_loaded) return;
    Accumulator* const acc_stack = nnue_acc_stack();
    int ply = pos.state_ply();
    acc_stack[ply] = acc_stack[ply - 1];   // board unchanged → child = parent
}

void update(Position& pos, Move m, Piece moved, PieceType captured) {
    if (!g_loaded) return;
    long long _t = now_us();
    Accumulator* const acc_stack = nnue_acc_stack();
    int ply = pos.state_ply();
    Accumulator& a = acc_stack[ply];
    acc_stack[ply] = acc_stack[ply - 1];   // copy parent → child, then apply deltas below

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

    for (int p = 0; p < 2; ++p) {
        bool white_pov = (p == 0);
        bool our_king_moved = king_moved && ((us == WHITE) == white_pov);
        if (our_king_moved) {
            refresh_perspective(pos, a, p);   // king_sq changed → full refresh of this perspective
            continue;
        }
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
    }
    st_update.us += now_us() - _t; ++st_update.n;
}

Value evaluate(const Position& pos) {
    long long _t = now_us();
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
                    ref += ft_w[static_cast<size_t>(halfka_idx(white_pov, ksq, sq, pc)) * g_L1 + l];
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

    float h[2 * NNUE_L1_MAX];
#if defined(__AVX2__)
    // SCReLU on the accumulator → h, 8 floats at a time.
    const __m256 z = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (int l = 0; l < L1; l += 8) {
        __m256 s1 = _mm256_loadu_ps(acc_stm + l);
        __m256 s2 = _mm256_loadu_ps(acc_nstm + l);
        _mm256_storeu_ps(h + l,      screlu8(s1, z, one));
        _mm256_storeu_ps(h + L1 + l, screlu8(s2, z, one));
    }
    // ---- int8 path (LNI8): quantize activations, int8 SIMD matmuls. ~3.5x over float ----
    if (g_int8) {
        uint8_t h_i8[2 * NNUE_L1_MAX];
        const __m256 sc = _mm256_set1_ps(127.0f);
        for (int l = 0; l < 2 * L1; l += 8) quant8(h + l, h_i8 + l, sc);
        const float cs2 = g_s2 * 127.0f, cs3 = g_s3 * 127.0f, cso = g_so * 127.0f;
        float h2[NNUE_L1_MAX];
        for (int o = 0; o < L2; ++o)
            h2[o] = clip01(l2_b[o] + dot_i8(&l2_w_i8[static_cast<size_t>(o) * 2 * L1], h_i8, 2 * L1) / cs2);
        uint8_t h2_i8[NNUE_L1_MAX];
        for (int l = 0; l < L2; l += 8) quant8(h2 + l, h2_i8 + l, sc);
        float h3[NNUE_L1_MAX];
        for (int o = 0; o < L3; ++o)
            h3[o] = clip01(l3_b[o] + dot_i8(&l3_w_i8[static_cast<size_t>(o) * L2], h2_i8, L2) / cs3);
        uint8_t h3_i8[NNUE_L1_MAX];
        for (int l = 0; l < L3; l += 8) quant8(h3 + l, h3_i8 + l, sc);
        float ov = out_b + dot_i8(out_w_i8.data(), h3_i8, L3) / cso;
        Value v = static_cast<Value>(std::llround(ov * OUT_SCALE));
        st_eval.us += now_us() - _t; ++st_eval.n;
        return v;
    }
    // L2: 16 SIMD dot products over 2*L1 inputs (the dominant cost — was scalar).
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
    st_eval.us += now_us() - _t; ++st_eval.n;
    return v;
}

} // namespace luminex::nnue
