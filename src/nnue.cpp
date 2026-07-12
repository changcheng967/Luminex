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
#include <fstream>
#include <string>
#include <vector>

namespace luminex::nnue {

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
    if (std::string(magic, 4) != "LNN1") { std::fprintf(stderr, "nnue: bad magic\n"); return false; }
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
    g_loaded = true;
    std::printf("nnue: loaded %s (L1=%d L2=%d L3=%d)\n", path.c_str(), L1, L2, L3);
    return true;
}

static inline float clip01(float x) { float c = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); return c * c; }

static inline void add_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    int idx = halfka_idx(p == 0, ksq, sq, piece);
    const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
    for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
}
static inline void remove_feature(Accumulator& a, int p, int ksq, int sq, Piece piece) {
    int idx = halfka_idx(p == 0, ksq, sq, piece);
    const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
    float* acc = a.v[p];
    for (int l = 0; l < g_L1; ++l) acc[l] -= w[l];
}

static void refresh_perspective(const Position& pos, Accumulator& a, int p) {
    bool white_pov = (p == 0);
    int ksq = static_cast<int>(pos.king_sq(white_pov ? WHITE : BLACK));
    float* acc = a.v[p];
    for (int l = 0; l < g_L1; ++l) acc[l] = ft_b[l];
    for (int sq = 0; sq < NUM_SQ; ++sq) {
        Piece pc = pos.piece_on(Square(sq));
        if (pc == NO_PIECE) continue;
        int idx = halfka_idx(white_pov, ksq, sq, pc);
        const float* w = &ft_w[static_cast<size_t>(idx) * g_L1];
        for (int l = 0; l < g_L1; ++l) acc[l] += w[l];
    }
}

void refresh(Position& pos) {
    if (!g_loaded) return;
    Accumulator* const acc_stack = nnue_acc_stack();
    Accumulator& a = acc_stack[pos.state_ply()];
    refresh_perspective(pos, a, 0);
    refresh_perspective(pos, a, 1);
}

void copy_for_null(Position& pos) {
    if (!g_loaded) return;
    Accumulator* const acc_stack = nnue_acc_stack();
    int ply = pos.state_ply();
    acc_stack[ply] = acc_stack[ply - 1];   // board unchanged → child = parent
}

void update(Position& pos, Move m, Piece moved, PieceType captured) {
    if (!g_loaded) return;
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
}

Value evaluate(const Position& pos) {
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
    for (int l = 0; l < L1; ++l) {
        h[l]      = clip01(acc_stm[l]);
        h[L1 + l] = clip01(acc_nstm[l]);
    }
    float h2[NNUE_L1_MAX];
    for (int o = 0; o < L2; ++o) {
        float s = l2_b[o];
        const float* w = &l2_w[o * 2 * L1];
        for (int i = 0; i < 2 * L1; ++i) s += w[i] * h[i];
        h2[o] = clip01(s);
    }
    float h3[NNUE_L1_MAX];
    for (int o = 0; o < L3; ++o) {
        float s = l3_b[o];
        const float* w = &l3_w[o * L2];
        for (int i = 0; i < L2; ++i) s += w[i] * h2[i];
        h3[o] = clip01(s);
    }
    float o = out_b;
    for (int i = 0; i < L3; ++i) o += out_w[i] * h3[i];
    return static_cast<Value>(std::llround(o * OUT_SCALE));
}

} // namespace luminex::nnue
