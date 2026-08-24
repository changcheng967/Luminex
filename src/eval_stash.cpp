// eval_stash.cpp — port of Morgan Houppin's Stash hand-crafted evaluation
// (github.com/mhouppin/stash-bot, GPL-3.0, Copyright 2019-2025 Morgan Houppin).
// Translated to Luminex's Position API; all tuned constants verbatim from the
// v37-era source. Enabled via UCI "UseStashEval" (default off).
//
// Scale note: Stash EG piece scores live at ~2x the MG scale (pawn MG 103 /
// EG 207); their search constants were tuned to this. We port the scale
// verbatim — our search margins may need follow-up tuning if the eval wins.
#include "luminex.h"
#include "eval_feat.h"
#include <cstring>
#include <algorithm>

namespace luminex {

// Local rank/file bitmasks (our bitboard.h exposes rank_bb()/file_bb() only)
static const Bitboard RANK_1_BB = 0x00000000000000FFULL;
static const Bitboard RANK_2_BB = 0x000000000000FF00ULL;
static const Bitboard RANK_3_BB = 0x0000000000FF0000ULL;
static const Bitboard RANK_4_BB = 0x00000000FF000000ULL;
static const Bitboard RANK_5_BB = 0x000000FF00000000ULL;
static const Bitboard RANK_6_BB = 0x0000FF0000000000ULL;
static const Bitboard RANK_7_BB = 0x00FF000000000000ULL;
static const Bitboard RANK_8_BB = 0xFF00000000000000ULL;
static const Bitboard FILE_A_BB = 0x0101010101010101ULL;
static const Bitboard FILE_H_BB = 0x8080808080808080ULL;

// Full line through two squares (for pin mobility trims)
static Bitboard line_through(Square a, Square b) {
    if (a == b) return square_bb(a);
    int fa = file_of(a), ra = rank_of(a), fb = file_of(b), rb2 = rank_of(b);
    int df = (fb > fa) - (fb < fa), dr = (rb2 > ra) - (rb2 < ra);
    if (fa != fb && ra != rb2 && std::abs(fb - fa) != std::abs(rb2 - ra)) return square_bb(b);
    Bitboard out = 0;
    int f = fa + df, r = ra + dr;
    while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        out |= square_bb(make_square(File(f), Rank(r)));
        if (f == fb && r == rb2) break;
        f += df; r += dr;
    }
    f = fa - df; r = ra - dr;
    while (f >= 0 && f <= 7 && r >= 0 && r <= 7) {
        out |= square_bb(make_square(File(f), Rank(r)));
        f -= df; r -= dr;
    }
    return out;
}

namespace stash_eval {

// ============================================================
// Scorepair: (mg, eg) pair arithmetic, Stash convention
// ============================================================
struct SP {
    int mg, eg;
    SP(int m = 0, int e = 0) : mg(m), eg(e) {}
};
inline SP operator+(SP a, SP b) { return SP(a.mg + b.mg, a.eg + b.eg); }
inline SP operator-(SP a, SP b) { return SP(a.mg - b.mg, a.eg - b.eg); }
inline SP operator-(SP a) { return SP(-a.mg, -a.eg); }
inline SP& operator+=(SP& a, SP b) { a.mg += b.mg; a.eg += b.eg; return a; }
inline SP& operator-=(SP& a, SP b) { a.mg -= b.mg; a.eg -= b.eg; return a; }
inline SP operator*(SP a, int n) { return SP(a.mg * n, a.eg * n); }
#define SPAIR(m, e) SP(m, e)

// ============================================================
// Piece values (Stash psq_table.h)
// ============================================================
enum : int {
    PAWN_MG = 103, KNIGHT_MG = 386, BISHOP_MG = 402, ROOK_MG = 528, QUEEN_MG = 1096,
    PAWN_EG = 207, KNIGHT_EG = 664, BISHOP_EG = 727, ROOK_EG = 1158, QUEEN_EG = 2191,
    MIDGAME_COUNT = 24, ENDGAME_COUNT = 4,
    SCALE_NORMAL = 256, SCALE_DRAW = 0, VICTORY = 10000,
    BISHOP_MG_SCORE = BISHOP_MG, ROOK_MG_SCORE = ROOK_MG
};

// ============================================================
// Bitboard helpers (Stash shift semantics on our Bitboard)
// ============================================================
inline Bitboard shift_up(Bitboard b)         { return b << 8; }
inline Bitboard shift_down(Bitboard b)       { return b >> 8; }
inline Bitboard shift_left(Bitboard b)       { return (b & ~FILE_H_BB) << 1; }
inline Bitboard shift_right(Bitboard b)      { return (b & ~FILE_A_BB) >> 1; }
inline Bitboard shift_up_left(Bitboard b)    { return (b & ~FILE_H_BB) << 9; }
inline Bitboard shift_up_right(Bitboard b)   { return (b & ~FILE_A_BB) << 7; }
inline Bitboard shift_down_left(Bitboard b)  { return (b & ~FILE_A_BB) >> 9; }
inline Bitboard shift_down_right(Bitboard b) { return (b & ~FILE_H_BB) >> 7; }
inline Bitboard shift_up_rel(Bitboard b, Color c)    { return c == WHITE ? b << 8 : b >> 8; }
inline Bitboard shift_down_rel(Bitboard b, Color c)  { return c == WHITE ? b >> 8 : b << 8; }
inline Bitboard pawn_attacks(Bitboard p, Color c) {
    return c == WHITE ? (shift_up_left(p) | shift_up_right(p))
                      : (shift_down_left(p) | shift_down_right(p));
}
inline Bitboard pawn_attacks2(Bitboard p, Color c) {
    return c == WHITE ? (shift_up_left(p) & shift_up_right(p))
                      : (shift_down_left(p) & shift_down_right(p));
}
inline Bitboard forward_file_bb(Square sq, Color c) {
    Bitboard out = 0;
    if (c == WHITE) for (int r = rank_of(sq) + 1; r <= 7; ++r) out |= rank_bb(Rank(r));
    else            for (int r = rank_of(sq) - 1; r >= 0; --r) out |= rank_bb(Rank(r));
    return out & file_bb(file_of(sq));
}
inline Bitboard pawn_attack_span_bb(Square sq, Color c) {
    Bitboard out = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int rr = (c == WHITE ? r + 1 : r - 1); (c == WHITE ? rr <= 7 : rr >= 0); (c == WHITE ? ++rr : --rr)) {
        for (int ff = f - 1; ff <= f + 1; ++ff)
            if (ff >= 0 && ff <= 7) out |= square_bb(make_square(File(ff), Rank(rr)));
    }
    return out;
}
// Static bitboards (built once; the naive per-call loops cost ~36% NPS)
static const Bitboard g_dark = [] {
    Bitboard b = 0;
    for (int s = 0; s < 64; ++s)
        if (((s / 8) + (s % 8)) % 2 == 1) b |= square_bb(Square(s));
    return b;
}();
static const Bitboard g_center = square_bb(Square(D4)) | square_bb(Square(E4))
                               | square_bb(Square(D5)) | square_bb(Square(E5));
static const Bitboard g_queenside = [] {
    Bitboard b = 0;
    for (int r = 0; r < 8; ++r)
        for (int f = 0; f < 4; ++f) b |= square_bb(make_square(File(f), Rank(r)));
    return b;
}();
static const Bitboard g_kingside = ~g_queenside;
inline Bitboard dark_squares() { return g_dark; }
inline Bitboard center_bb() { return g_center; }
inline Bitboard queenside_bb() { return g_queenside; }
inline Bitboard kingside_bb() { return g_kingside; }

// King blockers: squares between our king and enemy sliders that would
// expose a check if the occupant moved (SF-style, used for pin mobility).
inline Bitboard king_blockers(const Position& pos, Color c) {
    Color them = Color(c ^ 1);
    Square ksq = pos.king_sq(c);
    Bitboard their_snipers = pos.pieces(them, ROOK, QUEEN) | pos.pieces(them, BISHOP, QUEEN);
    Bitboard occ = pos.pieces();
    Bitboard snipers = (rook_attacks_bb(ksq, their_snipers & (pos.pieces(ROOK) | pos.pieces(QUEEN)))
                        & pos.pieces(them, ROOK, QUEEN))
                     | (bishop_attacks_bb(ksq, their_snipers & (pos.pieces(BISHOP) | pos.pieces(QUEEN)))
                        & pos.pieces(them, BISHOP, QUEEN));
    Bitboard blockers = square_bb(ksq);
    Bitboard s = snipers;
    while (s) {
        Square sq = pop_lsb(s);
        // squares strictly between
        Bitboard between = 0;
        int kf = file_of(ksq), kr = rank_of(ksq), sf = file_of(sq), sr = rank_of(sq);
        int df = (sf > kf) - (sf < kf), dr = (sr > kr) - (sr < kr);
        int f = kf + df, r = kr + dr;
        while (f != sf || r != sr) { between |= square_bb(make_square(File(f), Rank(r))); f += df; r += dr; }
        blockers |= between & occ & ~square_bb(sq);
    }
    return blockers;
}

// ============================================================
// Stash eval constants — verbatim (evaluate.c)
// ============================================================
const SP Initiative = SPAIR(24, 32);
const SP KnightShielded = SPAIR(4, 23);
const SP KnightOutpost  = SPAIR(31, 30);
const SP ClosedPosKnight[5] = {
    SPAIR(6, -18), SPAIR(6, 3), SPAIR(8, 20), SPAIR(13, 29), SPAIR(14, 46)
};
const SP BishopPairBonus    = SPAIR(21, 93);
const SP BishopShielded     = SPAIR(1, 3);
const SP BishopOutpost      = SPAIR(47, 24);
const SP BishopLongDiagonal = SPAIR(13, 22);
const SP BishopPawnsSameColor[7] = {
    SPAIR(15, 36), SPAIR(15, 26), SPAIR(13, 17), SPAIR(9, 11),
    SPAIR(6, 3), SPAIR(3, -2), SPAIR(-3, -10)
};
const SP RookOnSemiOpenFile = SPAIR(19, 13);
const SP RookOnOpenFile     = SPAIR(35, 9);
const SP RookOnBlockedFile  = SPAIR(-8, -8);
const SP RookXrayQueen      = SPAIR(15, 4);
const SP RookTrapped        = SPAIR(-8, -16);
const SP RookBuried         = SPAIR(-69, -33);
const SP KnightMobility[9] = {
    SPAIR(-56, 20), SPAIR(-45, -28), SPAIR(-35, 36), SPAIR(-26, 58),
    SPAIR(-19, 76), SPAIR(-13, 95), SPAIR(-8, 104), SPAIR(0, 109), SPAIR(4, 103)
};
const SP BishopMobility[14] = {
    SPAIR(-56, -44), SPAIR(-44, -39), SPAIR(-27, -15), SPAIR(-25, 15),
    SPAIR(-17, 32), SPAIR(-13, 46), SPAIR(-10, 56), SPAIR(-8, 61),
    SPAIR(-7, 64), SPAIR(-5, 67), SPAIR(-2, 61), SPAIR(7, 56),
    SPAIR(9, 57), SPAIR(35, 48)
};
const SP RookMobility[15] = {
    SPAIR(-88, -47), SPAIR(-39, 40), SPAIR(-27, 76), SPAIR(-31, 92),
    SPAIR(-29, 103), SPAIR(-33, 117), SPAIR(-34, 125), SPAIR(-29, 131),
    SPAIR(-26, 140), SPAIR(-18, 149), SPAIR(-17, 156), SPAIR(-13, 163),
    SPAIR(-6, 167), SPAIR(4, 169), SPAIR(23, 161)
};
const SP QueenMobility[28] = {
    SPAIR(-14, -93), SPAIR(19, 216), SPAIR(0, 147), SPAIR(-6, 95),
    SPAIR(1, 65), SPAIR(-2, 118), SPAIR(-4, 151), SPAIR(-3, 175),
    SPAIR(-2, 185), SPAIR(-2, 205), SPAIR(-1, 214), SPAIR(2, 219),
    SPAIR(2, 228), SPAIR(5, 232), SPAIR(5, 237), SPAIR(5, 243),
    SPAIR(6, 244), SPAIR(9, 241), SPAIR(14, 232), SPAIR(12, 234),
    SPAIR(36, 204), SPAIR(39, 198), SPAIR(48, 183), SPAIR(34, 163),
    SPAIR(59, 156), SPAIR(11, 164), SPAIR(11, 170), SPAIR(36, 141)
};
const SP PassedBlocked[4] = {
    SPAIR(-2, -29), SPAIR(2, -52), SPAIR(3, -89), SPAIR(-45, -132)
};
const SP PassedSafeAdvance[4] = {
    SPAIR(-3, 26), SPAIR(-10, 45), SPAIR(8, 70), SPAIR(56, 61)
};
const SP FarKnight = SPAIR(-22, -13);
const SP FarBishop = SPAIR(-8, -10);
const SP FarRook   = SPAIR(-10, 5);
const SP FarQueen  = SPAIR(-8, 15);
const SP KnightWeight    = SPAIR(46, 72);
const SP BishopWeight    = SPAIR(31, 116);
const SP RookWeight      = SPAIR(33, -46);
const SP QueenWeight     = SPAIR(10, 2);
const SP AttackWeight    = SPAIR(9, 36);
const SP WeakKingZone    = SPAIR(27, -76);
const SP SafeKnightCheck = SPAIR(74, 41);
const SP SafeBishopCheck = SPAIR(36, 170);
const SP SafeRookCheck   = SPAIR(89, 183);
const SP SafeQueenCheck  = SPAIR(45, 231);
const SP UnsafeCheck     = SPAIR(16, 123);
const SP QueenlessAttack = SPAIR(-88, -81);
const SP SafetyOffset    = SPAIR(18, 53);
const SP KingStorm[24] = {
    SPAIR(2, -10), SPAIR(-36, -6), SPAIR(24, 26), SPAIR(7, 20),
    SPAIR(-11, 25), SPAIR(-13, 8), SPAIR(-33, -14), SPAIR(-4, -36),
    SPAIR(0, 0), SPAIR(2, -7), SPAIR(33, 44), SPAIR(3, -13),
    SPAIR(-6, -18), SPAIR(-7, 18), SPAIR(3, 55), SPAIR(15, -14),
    SPAIR(8, 2), SPAIR(12, -1), SPAIR(34, 17), SPAIR(21, -11),
    SPAIR(-8, -24), SPAIR(-15, 48), SPAIR(-6, 112), SPAIR(-2, -81)
};
const SP KingShelter[24] = {
    SPAIR(-38, 20), SPAIR(-27, 145), SPAIR(-26, -49), SPAIR(-12, 14),
    SPAIR(20, -13), SPAIR(21, -16), SPAIR(-16, -3), SPAIR(6, -102),
    SPAIR(0, 0), SPAIR(-8, -35), SPAIR(-8, 82), SPAIR(0, 66),
    SPAIR(9, 36), SPAIR(35, -4), SPAIR(4, -1), SPAIR(13, -74),
    SPAIR(-37, -59), SPAIR(12, -173), SPAIR(-2, 111), SPAIR(7, 160),
    SPAIR(10, 45), SPAIR(20, 1), SPAIR(16, -1), SPAIR(13, -3)
};
const SP PawnThreats[6] = {
    SPAIR(-2, -34), SPAIR(71, 65), SPAIR(67, 111), SPAIR(61, 65), SPAIR(62, 24), SP(0, 0)
};
const SP KnightThreats[6] = {
    SPAIR(-9, 8), SPAIR(4, 49), SPAIR(42, 56), SPAIR(93, 46), SPAIR(49, 31), SP(0, 0)
};
const SP BishopThreats[6] = {
    SPAIR(-3, 5), SPAIR(15, 50), SPAIR(1, 58), SPAIR(55, 75), SPAIR(50, 152), SP(0, 0)
};
const SP RookThreats[6] = {
    SPAIR(-10, 13), SPAIR(7, 29), SPAIR(24, 22), SPAIR(11, 27), SPAIR(50, 63), SP(0, 0)
};
const SP QueenThreats[6] = {
    SPAIR(0, 9), SPAIR(1, 15), SPAIR(-4, 39), SPAIR(-2, -4), SPAIR(9, 2), SP(0, 0)
};
const SP HangingPawn = SPAIR(13, 52);

// kp_eval.c constants — verbatim
const SP BackwardPenalty = SPAIR(-20, -26);
const SP DoubledPenalty  = SPAIR(-14, -46);
const SP IsolatedPenalty = SPAIR(-5, -11);
const SP PassedBonus[8] = {
    SP(0, 0), SPAIR(-10, 7), SPAIR(-14, 15), SPAIR(-27, 41),
    SPAIR(16, 99), SPAIR(67, 189), SPAIR(107, 350), SP(0, 0)
};
const SP PassedOurKingDistance[24] = {
    SPAIR(33, 96), SPAIR(49, 5), SPAIR(-29, -87), SP(0, 0), SP(0, 0), SP(0, 0),
    SPAIR(37, 72), SPAIR(43, 17), SPAIR(16, -38), SPAIR(-60, -61), SP(0, 0), SP(0, 0),
    SPAIR(6, 69), SPAIR(-28, 36), SPAIR(-13, -12), SPAIR(-5, -43), SPAIR(40, -53), SP(0, 0),
    SPAIR(-40, 55), SPAIR(-27, 31), SPAIR(-12, -10), SPAIR(0, -14), SPAIR(28, -23), SPAIR(43, -28)
};
const SP PassedTheirKingDistance[24] = {
    SPAIR(13, -151), SPAIR(7, 4), SPAIR(4, 162), SP(0, 0), SP(0, 0), SP(0, 0),
    SPAIR(-24, -119), SPAIR(26, -46), SPAIR(10, 44), SPAIR(6, 115), SP(0, 0), SP(0, 0),
    SPAIR(-13, -78), SPAIR(30, -35), SPAIR(15, -10), SPAIR(-4, 46), SPAIR(-32, 77), SP(0, 0),
    SPAIR(-53, -35), SPAIR(-9, -8), SPAIR(17, -10), SPAIR(29, -6), SPAIR(-4, 36), SPAIR(-3, 35)
};
const SP PhalanxBonus[8] = {
    SP(0, 0), SPAIR(5, -2), SPAIR(18, 11), SPAIR(21, 28),
    SPAIR(42, 70), SPAIR(177, 282), SPAIR(182, 288), SP(0, 0)
};
const SP DefenderBonus[8] = {
    SP(0, 0), SPAIR(18, 22), SPAIR(14, 22), SPAIR(23, 34),
    SPAIR(61, 97), SPAIR(189, 173), SP(0, 0), SP(0, 0)
};

// PSQT tables — verbatim (psq_table.c). Pawns: 48 entries ranks 2-7 relative.
// Pieces: 32 entries, queenside-folded (rank*4 + file_to_queenside).
const SP PawnSQT[48] = {
    SPAIR(-40, 6), SPAIR(-18, 5), SPAIR(-36, 6), SPAIR(-24, -6), SPAIR(-29, 14), SPAIR(11, 15), SPAIR(18, 5), SPAIR(-25, -22),
    SPAIR(-38, -4), SPAIR(-39, -2), SPAIR(-22, -7), SPAIR(-23, -10), SPAIR(-13, -2), SPAIR(-24, 4), SPAIR(1, -18), SPAIR(-22, -18),
    SPAIR(-31, 11), SPAIR(-31, 4), SPAIR(-16, -17), SPAIR(-2, -23), SPAIR(-1, -22), SPAIR(0, -12), SPAIR(-17, -7), SPAIR(-25, -14),
    SPAIR(-17, 32), SPAIR(-24, 10), SPAIR(-12, -4), SPAIR(-2, -26), SPAIR(3, -16), SPAIR(24, -13), SPAIR(-9, 1), SPAIR(-15, 12),
    SPAIR(8, 47), SPAIR(-12, 26), SPAIR(9, 2), SPAIR(21, -12), SPAIR(37, 5), SPAIR(77, 20), SPAIR(32, 34), SPAIR(19, 40),
    SPAIR(97, -5), SPAIR(32, -8), SPAIR(52, -8), SPAIR(81, -24), SPAIR(81, -4), SPAIR(35, 11), SPAIR(-24, 14), SPAIR(-37, 19)
};
const SP KnightSQT[32] = {
    SPAIR(-49, -43), SPAIR(-14, -39), SPAIR(-12, -17), SPAIR(-7, 0),
    SPAIR(-8, -24), SPAIR(-2, -6), SPAIR(7, -17), SPAIR(6, -4),
    SPAIR(9, -34), SPAIR(14, -8), SPAIR(25, -7), SPAIR(20, 18),
    SPAIR(18, 0), SPAIR(19, 17), SPAIR(45, 21), SPAIR(31, 35),
    SPAIR(47, 16), SPAIR(47, 13), SPAIR(61, 22), SPAIR(57, 40),
    SPAIR(-5, 8), SPAIR(29, 11), SPAIR(59, 27), SPAIR(55, 26),
    SPAIR(-8, -2), SPAIR(-40, 17), SPAIR(33, 7), SPAIR(33, 32),
    SPAIR(-155, -102), SPAIR(-116, 11), SPAIR(-185, 22), SPAIR(7, 10)
};
const SP BishopSQT[32] = {
    SPAIR(32, -38), SPAIR(28, -17), SPAIR(-2, -7), SPAIR(4, -10),
    SPAIR(39, -43), SPAIR(41, -27), SPAIR(38, -15), SPAIR(16, 1),
    SPAIR(28, -11), SPAIR(39, -4), SPAIR(25, -7), SPAIR(22, 24),
    SPAIR(12, -24), SPAIR(22, 4), SPAIR(20, 17), SPAIR(25, 25),
    SPAIR(9, 1), SPAIR(15, 16), SPAIR(27, 19), SPAIR(27, 33),
    SPAIR(42, 7), SPAIR(20, 29), SPAIR(18, 13), SPAIR(35, 11),
    SPAIR(-49, 6), SPAIR(-55, 0), SPAIR(-2, 16), SPAIR(-16, 15),
    SPAIR(-46, -12), SPAIR(-72, 21), SPAIR(-157, 19), SPAIR(-152, 14)
};
const SP RookSQT[32] = {
    SPAIR(-6, -33), SPAIR(-6, -23), SPAIR(-6, -14), SPAIR(0, -22),
    SPAIR(-30, -29), SPAIR(-16, -28), SPAIR(-7, -14), SPAIR(-8, -15),
    SPAIR(-27, -22), SPAIR(-3, -17), SPAIR(-26, -4), SPAIR(-16, -4),
    SPAIR(-21, -7), SPAIR(-18, 4), SPAIR(-26, 8), SPAIR(-7, 0),
    SPAIR(-5, 14), SPAIR(5, 18), SPAIR(15, 11), SPAIR(29, 5),
    SPAIR(-13, 24), SPAIR(25, 12), SPAIR(23, 13), SPAIR(52, 4),
    SPAIR(11, 24), SPAIR(-5, 24), SPAIR(31, 24), SPAIR(45, 26),
    SPAIR(8, 33), SPAIR(1, 37), SPAIR(1, 37), SPAIR(5, 34)
};
const SP QueenSQT[32] = {
    SPAIR(14, -67), SPAIR(-1, -61), SPAIR(17, -85), SPAIR(25, -61),
    SPAIR(21, -63), SPAIR(21, -58), SPAIR(34, -55), SPAIR(21, -7),
    SPAIR(16, -42), SPAIR(24, -16), SPAIR(13, 24), SPAIR(11, 19),
    SPAIR(15, -4), SPAIR(19, 27), SPAIR(1, 33), SPAIR(-9, 65),
    SPAIR(21, 13), SPAIR(-5, 50), SPAIR(7, 33), SPAIR(-14, 67),
    SPAIR(10, 5), SPAIR(-7, 36), SPAIR(-13, 53), SPAIR(-3, 48),
    SPAIR(-9, 2), SPAIR(-39, 7), SPAIR(-31, 58), SPAIR(-45, 82),
    SPAIR(-54, 27), SPAIR(-10, 24), SPAIR(-21, 47), SPAIR(-10, 52)
};
const SP KingSQT[32] = {
    SPAIR(67, -117), SPAIR(66, -60), SPAIR(-20, -44), SPAIR(-26, -59),
    SPAIR(66, -52), SPAIR(21, -22), SPAIR(-10, -7), SPAIR(-48, -2),
    SPAIR(-39, -39), SPAIR(14, -16), SPAIR(-22, 10), SPAIR(-25, 21),
    SPAIR(-162, -20), SPAIR(-79, 7), SPAIR(-50, 30), SPAIR(-20, 40),
    SPAIR(-107, 10), SPAIR(-20, 49), SPAIR(23, 60), SPAIR(24, 60),
    SPAIR(-47, 29), SPAIR(57, 74), SPAIR(60, 83), SPAIR(65, 68),
    SPAIR(-55, -10), SPAIR(3, 59), SPAIR(38, 71), SPAIR(39, 54),
    SPAIR(22, -238), SPAIR(87, -38), SPAIR(64, 0), SPAIR(4, 11)
};

inline int file_to_qs(File f) { return f > FILE_D ? FILE_H - f : f; }  // fold to 0..3
inline SP psqt_piece(const SP* table, int mg_val, int eg_val, Square rel) {
    // rel = relative square (white perspective); index = rank*4 + queenside-folded file
    return SP(mg_val, eg_val) + table[rank_of(rel) * 4 + file_to_qs(file_of(rel))];
}

// ============================================================
// Eval data + KingPawn entry
// ============================================================
struct KPEntry {
    SP value;
    Bitboard attack_span[2];
    Bitboard passed[2];
};

struct EvalData {
    Bitboard attacked[2] = {0, 0};
    Bitboard attacked2[2] = {0, 0};
    Bitboard attacked_by[2][8] = {};
    Bitboard king_zone[2] = {0, 0};
    Bitboard mobility_zone[2] = {0, 0};
    int safety_attackers[2] = {0, 0};
    int safety_attacks[2] = {0, 0};
    SP safety_value[2];
    int position_closed = 0;
};

static int material_mg(const Position& pos, Color c) {
    return popcount(pos.pieces(c, PAWN)) * PAWN_MG + popcount(pos.pieces(c, KNIGHT)) * KNIGHT_MG
         + popcount(pos.pieces(c, BISHOP)) * BISHOP_MG + popcount(pos.pieces(c, ROOK)) * ROOK_MG
         + popcount(pos.pieces(c, QUEEN)) * QUEEN_MG;
}

static void evaldata_init(EvalData& d, const Position& pos, Color us) {
    Color them = Color(us ^ 1);
    Square our_king = pos.king_sq(us);
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard pa = pawn_attacks(our_pawns, us);

    d.attacked[us] = d.attacked_by[us][KING] = d.king_zone[them] = king_attacks_bb(our_king);
    d.king_zone[them] |= shift_up_rel(d.king_zone[them], us);
    if (file_of(our_king) == FILE_A) d.king_zone[them] |= shift_right(d.king_zone[them]);
    if (file_of(our_king) == FILE_H) d.king_zone[them] |= shift_left(d.king_zone[them]);

    d.attacked_by[us][PAWN] = pa;
    d.attacked2[us] |= d.attacked[us] & pa;
    d.attacked2[us] |= pawn_attacks2(our_pawns, us);
    d.attacked[us] |= pa;

    d.mobility_zone[them] = ~pa;
    d.king_zone[them] &= ~pa;
}

static void evaldata_init_next(EvalData& d, const Position& pos, Color us) {
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard occ = pos.pieces();
    Bitboard low_ranks = (us == WHITE) ? (RANK_2_BB | RANK_3_BB) : (RANK_6_BB | RANK_7_BB);
    d.mobility_zone[us] &= ~(our_pawns & (shift_down_rel(occ, us) | low_ranks));
    d.mobility_zone[us] &= ~pos.pieces(us, KING);
}

static void evaldata_set_closed(EvalData& d, const Position& pos) {
    Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
    Bitboard occ = pos.pieces();
    Bitboard fixed = (wp & shift_down(occ | d.attacked_by[BLACK][PAWN]))
                   | (bp & shift_up(occ | d.attacked_by[WHITE][PAWN]));
    d.position_closed = std::min(4, popcount(fixed) / 2);
}

// ---- King-pawn structure (kp_eval.c, no hash: computed per call) ----
static SP kp_backward(KPEntry& kpe, Bitboard our_pawns, Bitboard their_pawns,
                      Bitboard their_attacks, Bitboard our_span, Color us) {
    Color them = Color(us ^ 1);
    Bitboard stop = shift_up_rel(our_pawns, us);
    Bitboard backward = shift_down_rel(stop & their_attacks & ~our_span, us);
    if (!backward) return SP(0, 0);
    Bitboard closed_files = 0;
    Bitboard tp = their_pawns;
    while (tp) closed_files |= forward_file_bb(pop_lsb(tp), them);
    backward &= ~closed_files;
    if (!backward) return SP(0, 0);
    return BackwardPenalty * popcount(backward);
}

static SP kp_connected(Bitboard our_pawns, Color us) {
    SP ret(0, 0);
    Bitboard phalanx = our_pawns & shift_left(our_pawns);
    Bitboard defenders = our_pawns & pawn_attacks(our_pawns, Color(us ^ 1));
    while (phalanx) {
        Square sq = pop_lsb(phalanx);
        ret += PhalanxBonus[relative_rank(us, rank_of(sq))];
    }
    while (defenders) {
        Square sq = pop_lsb(defenders);
        ret += DefenderBonus[relative_rank(us, rank_of(sq))];
    }
    return ret;
}

static SP kp_passed(KPEntry& kpe, Bitboard our_pawns, Bitboard their_pawns,
                    Bitboard their_attacks, Bitboard our_attacks,
                    Bitboard their_attacks2, Bitboard our_attacks2, Color us) {
    SP ret(0, 0);
    Bitboard p = our_pawns;
    while (p) {
        Square sq = pop_lsb(p);
        Bitboard path = forward_file_bb(sq, us);
        if (!(path & their_pawns)
            && !(path & their_attacks & ~our_attacks)
            && !(path & their_attacks2 & ~our_attacks2)) {
            ret += PassedBonus[relative_rank(us, rank_of(sq))];
            kpe.passed[us] |= square_bb(sq);
        }
    }
    return ret;
}

static int pkd_index(int qdist, int kdist) {
    return (qdist - 1) * 6 + std::min(qdist + 2, kdist) - 1;
}

static SP kp_passed_pos(const KPEntry& kpe, const Position& pos, Color us) {
    SP ret(0, 0);
    Square our_king = pos.king_sq(us), their_king = pos.king_sq(Color(us ^ 1));
    Bitboard bb = kpe.passed[us];
    while (bb) {
        Square sq = pop_lsb(bb);
        int qdist = 7 - relative_rank(us, rank_of(sq));
        if (qdist <= 4) {
            int od = distance(our_king, sq), td = distance(their_king, sq);
            ret += PassedOurKingDistance[pkd_index(qdist, od)];
            ret += PassedTheirKingDistance[pkd_index(qdist, td)];
        }
    }
    return ret;
}

static SP kp_doubled_isolated(Bitboard our_pawns) {
    SP ret(0, 0);
    for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
        Bitboard fp = our_pawns & file_bb(f);
        if (fp) {
            if (popcount(fp) > 1) ret += DoubledPenalty;
            if (!((shift_left(file_bb(f)) | shift_right(file_bb(f))) & our_pawns))
                ret += IsolatedPenalty;
        }
    }
    return ret;
}

// Pawn-structure hash (keyed by pawn_key; king-dependent passed-pos scoring
// computed fresh — it needs king squares, which the pawn key doesn't cover).
struct KPHashEntry {
    uint64_t key = 0;
    SP core;
    Bitboard attack_span[2] = {0, 0};
    Bitboard passed[2] = {0, 0};
};
static KPHashEntry kpt[1 << 14];

static KPEntry king_pawn_probe(const Position& pos) {
    uint64_t key = pos.pawn_key();
    KPHashEntry& e = kpt[key & ((1 << 14) - 1)];
    KPEntry kpe;
    if (e.key == key) {
        kpe.value = e.core;
        kpe.attack_span[WHITE] = e.attack_span[WHITE];
        kpe.attack_span[BLACK] = e.attack_span[BLACK];
        kpe.passed[WHITE] = e.passed[WHITE];
        kpe.passed[BLACK] = e.passed[BLACK];
    } else {
        Bitboard wp = pos.pieces(WHITE, PAWN), bp = pos.pieces(BLACK, PAWN);
        Bitboard wa = pawn_attacks(wp, WHITE), ba = pawn_attacks(bp, BLACK);
        Bitboard wa2 = pawn_attacks2(wp, WHITE), ba2 = pawn_attacks2(bp, BLACK);

        auto span = [](Bitboard pawns, Color c) {
            Bitboard s = 0;
            while (pawns) s |= pawn_attack_span_bb(pop_lsb(pawns), c);
            return s;
        };
        e.key = key;
        e.attack_span[WHITE] = kpe.attack_span[WHITE] = span(wp, WHITE);
        e.attack_span[BLACK] = kpe.attack_span[BLACK] = span(bp, BLACK);
        e.passed[WHITE] = kpe.passed[WHITE] = 0;
        e.passed[BLACK] = kpe.passed[BLACK] = 0;

        SP core(0, 0);
        // PSQT pawns (relative, ranks 2..7)
        auto pawn_psq = [&](Bitboard pawns, Color c) {
            SP r(0, 0);
            while (pawns) {
                Square sq = pop_lsb(pawns);
                Square rel = relative_square(c, sq);
                r += PawnSQT[(rank_of(rel) - 1) * 8 + file_of(rel)];
            }
            return r;
        };
        core += pawn_psq(wp, WHITE);
        core -= pawn_psq(bp, BLACK);
        core += kp_backward(kpe, wp, bp, ba, e.attack_span[WHITE], WHITE);
        core -= kp_backward(kpe, bp, wp, wa, e.attack_span[BLACK], BLACK);
        core += kp_connected(wp, WHITE);
        core -= kp_connected(bp, BLACK);
        core += kp_passed(kpe, wp, bp, ba, wa, ba2, wa2, WHITE);
        core -= kp_passed(kpe, bp, wp, wa, ba, wa2, ba2, BLACK);
        core += kp_doubled_isolated(wp);
        core -= kp_doubled_isolated(bp);
        e.core = kpe.value = core;
        e.passed[WHITE] = kpe.passed[WHITE];
        e.passed[BLACK] = kpe.passed[BLACK];
    }
    // King-dependent scoring: fresh every call (cheap — passed pawns only)
    kpe.value += kp_passed_pos(kpe, pos, WHITE);
    kpe.value -= kp_passed_pos(kpe, pos, BLACK);
    return kpe;
}

// ============================================================
// Piece evaluations
// ============================================================
static SP eval_knights(const Position& pos, EvalData& d, const KPEntry& kpe,
                       Bitboard kblockers, Color us) {
    Color them = Color(us ^ 1);
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard outpost_ranks = RANK_4_BB | RANK_5_BB | (us == WHITE ? RANK_6_BB : RANK_3_BB);
    SP ret(0, 0);
    Bitboard bb = pos.pieces(us, KNIGHT);
    while (bb) {
        Square sq = pop_lsb(bb);
        Bitboard b = knight_attacks_bb(sq);
        ret += psqt_piece(KnightSQT, KNIGHT_MG, KNIGHT_EG, relative_square(us, sq));
        ret += ClosedPosKnight[d.position_closed];
        if (kblockers & square_bb(sq)) b = 0;
        d.attacked_by[us][KNIGHT] |= b;
        d.attacked2[us] |= d.attacked[us] & b;
        d.attacked[us] |= b;
        ret += KnightMobility[popcount(b & d.mobility_zone[us])];
        if (shift_down_rel(our_pawns, us) & square_bb(sq)) ret += KnightShielded;
        if (outpost_ranks & d.attacked_by[us][PAWN] & ~kpe.attack_span[them] & square_bb(sq))
            ret += KnightOutpost;
        if (distance(sq, pos.king_sq(us)) > 3) ret += FarKnight;
        if (b & d.king_zone[us]) {
            d.safety_attackers[us] += 1;
            d.safety_attacks[us] += popcount(b & d.king_zone[us]);
            d.safety_value[us] += KnightWeight;
        }
    }
    return ret;
}

static SP eval_bishops(const Position& pos, EvalData& d, const KPEntry& kpe,
                       Bitboard kblockers, Color us, Bitboard dark) {
    Color them = Color(us ^ 1);
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard outpost_ranks = RANK_4_BB | RANK_5_BB | (us == WHITE ? RANK_6_BB : RANK_3_BB);
    Bitboard occ_xq = pos.pieces() ^ pos.pieces(us, QUEEN);
    SP ret(0, 0);
    Bitboard bb = pos.pieces(us, BISHOP);
    if (popcount(bb) > 1) ret += BishopPairBonus;
    while (bb) {
        Square sq = pop_lsb(bb);
        Bitboard b = bishop_attacks_bb(sq, occ_xq);
        ret += psqt_piece(BishopSQT, BISHOP_MG, BISHOP_EG, relative_square(us, sq));
        {
            Bitboard sqm = (dark & square_bb(sq)) ? dark : ~dark;
            ret += BishopPawnsSameColor[std::min(popcount(sqm & our_pawns), 6)];
        }
        if (kblockers & square_bb(sq)) {
            // trim to king line (approximate: full line through king)
            b &= line_through(pos.king_sq(us), sq);
        }
        d.attacked_by[us][BISHOP] |= b;
        d.attacked2[us] |= d.attacked[us] & b;
        d.attacked[us] |= b;
        ret += BishopMobility[popcount(b & d.mobility_zone[us])];
        if (shift_down_rel(our_pawns, us) & square_bb(sq)) ret += BishopShielded;
        if (outpost_ranks & d.attacked_by[us][PAWN] & ~kpe.attack_span[them] & square_bb(sq))
            ret += BishopOutpost;
        if (popcount(b & center_bb()) > 1) ret += BishopLongDiagonal;
        if (distance(sq, pos.king_sq(us)) > 3) ret += FarBishop;
        if (b & d.king_zone[us]) {
            d.safety_attackers[us] += 1;
            d.safety_attacks[us] += popcount(b & d.king_zone[us]);
            d.safety_value[us] += BishopWeight;
        }
    }
    return ret;
}

static SP eval_rooks(const Position& pos, EvalData& d, const KPEntry& kpe,
                     Bitboard kblockers, Color us) {
    Color them = Color(us ^ 1);
    Bitboard occ_xrq = pos.pieces() ^ pos.pieces(us, ROOK) ^ pos.pieces(us, QUEEN);
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard their_pawns = pos.pieces(them, PAWN);
    SP ret(0, 0);
    Bitboard bb = pos.pieces(us, ROOK);
    while (bb) {
        Square sq = pop_lsb(bb);
        Bitboard rfile = file_bb(file_of(sq));
        Bitboard b = rook_attacks_bb(sq, occ_xrq);
        ret += psqt_piece(RookSQT, ROOK_MG, ROOK_EG, relative_square(us, sq));
        if (kblockers & square_bb(sq)) b &= line_through(pos.king_sq(us), sq);
        d.attacked_by[us][ROOK] |= b;
        d.attacked2[us] |= d.attacked[us] & b;
        d.attacked[us] |= b;
        if (!(rfile & our_pawns)) {
            ret += (rfile & their_pawns) ? RookOnSemiOpenFile : RookOnOpenFile;
        } else if (shift_up_rel(rfile & our_pawns, us) & pos.pieces()) {
            ret += RookOnBlockedFile;
        }
        if (rfile & pos.pieces(them, QUEEN)) ret += RookXrayQueen;
        int mob = popcount(b & d.mobility_zone[us]);
        ret += RookMobility[mob];
        if (mob <= 4 && relative_rank(us, rank_of(sq)) <= RANK_2) {
            File kf = file_of(pos.king_sq(us)), rf = file_of(sq);
            if (kf != rf && (kf < rf) == (kf >= FILE_E)) {
                // castling-rights check approximated by king on home squares
                bool can_castle = (us == WHITE) ? (rank_of(pos.king_sq(us)) == RANK_1)
                                                : (rank_of(pos.king_sq(us)) == RANK_8);
                ret += can_castle ? RookTrapped : RookBuried;
            }
        }
        if (distance(sq, pos.king_sq(us)) > 3) ret += FarRook;
        if (b & d.king_zone[us]) {
            d.safety_attackers[us] += 1;
            d.safety_attacks[us] += popcount(b & d.king_zone[us]);
            d.safety_value[us] += RookWeight;
        }
    }
    return ret;
}

static SP eval_queens(const Position& pos, EvalData& d, Bitboard kblockers, Color us) {
    Bitboard occ_xb = pos.pieces() ^ pos.pieces(us, BISHOP);
    Bitboard occ_xr = pos.pieces() ^ pos.pieces(us, ROOK);
    SP ret(0, 0);
    Bitboard bb = pos.pieces(us, QUEEN);
    while (bb) {
        Square sq = pop_lsb(bb);
        Bitboard b = bishop_attacks_bb(sq, occ_xb) | rook_attacks_bb(sq, occ_xr);
        ret += psqt_piece(QueenSQT, QUEEN_MG, QUEEN_EG, relative_square(us, sq));
        if (kblockers & square_bb(sq)) b &= line_through(pos.king_sq(us), sq);
        d.attacked_by[us][QUEEN] |= b;
        d.attacked2[us] |= d.attacked[us] & b;
        d.attacked[us] |= b;
        ret += QueenMobility[popcount(b & d.mobility_zone[us])];
        if (distance(sq, pos.king_sq(us)) > 3) ret += FarQueen;
        if (b & d.king_zone[us]) {
            d.safety_attackers[us] += 1;
            d.safety_attacks[us] += popcount(b & d.king_zone[us]);
            d.safety_value[us] += QueenWeight;
        }
    }
    return ret;
}

static SP eval_passed_eval(const Position& pos, const EvalData& d, const KPEntry& kpe, Color us) {
    Color them = Color(us ^ 1);
    Bitboard occ = pos.pieces();
    Bitboard passed = kpe.passed[us];
    SP ret(0, 0);
    while (passed) {
        Square sq = pop_lsb(passed);
        Rank rank = relative_rank(us, rank_of(sq));
        int qdist = 7 - rank;
        if (qdist > 4) continue;
        if (shift_down_rel(occ, us) & square_bb(sq))
            ret += PassedBlocked[rank - RANK_4];
        if (!(shift_down_rel(d.attacked[them], us) & square_bb(sq)))
            ret += PassedSafeAdvance[rank - RANK_4];
    }
    return ret;
}

static SP eval_threats(const Position& pos, const EvalData& d, Color us) {
    Color them = Color(us ^ 1);
    Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
    SP ret(0, 0);
    auto threats = [&](Bitboard attacked_by, const SP* table) {
        Bitboard t = their_pieces & attacked_by;
        while (t) {
            Square sq = pop_lsb(t);
            PieceType pt = piece_type_of(pos.piece_on(sq));
            if (pt >= PAWN && pt <= QUEEN) ret += table[pt - PAWN];
        }
    };
    threats(d.attacked_by[us][PAWN], PawnThreats);
    threats(d.attacked_by[us][KNIGHT], KnightThreats);
    threats(d.attacked_by[us][BISHOP], BishopThreats);
    threats(d.attacked_by[us][ROOK], RookThreats);
    threats(d.attacked_by[us][QUEEN], QueenThreats);
    Bitboard hanging = pos.pieces(them, PAWN) & ~d.attacked[them] & d.attacked[us];
    if (hanging) ret += HangingPawn * popcount(hanging);
    return ret;
}

static SP safety_file(Bitboard our_pawns, Bitboard their_pawns, File f, Square their_king, Color us) {
    File kf = file_of(their_king);
    int offset = (kf == f) ? 8 : ((kf >= FILE_E) == (kf < f)) ? 0 : 16;
    SP ret(0, 0);
    // Stash: rank distance from the MOST-ADVANCED (us-relative) pawn on the
    // file to their king; 7 if the file has no such pawns.
    auto most_advanced_dist = [&](Bitboard pawns) -> int {
        if (!pawns) return 7;
        int best_rel = -1, best_sq = -1;
        Bitboard b = pawns;
        while (b) {
            Square s = pop_lsb(b);
            int rel = relative_rank(us, rank_of(s));
            if (rel > best_rel) { best_rel = rel; best_sq = s; }
        }
        return std::abs(int(rank_of(Square(best_sq))) - int(rank_of(their_king)));
    };
    ret += KingStorm[offset + most_advanced_dist(our_pawns & file_bb(f))];
    ret += KingShelter[offset + most_advanced_dist(their_pawns & file_bb(f))];
    return ret;
}

static SP eval_safety(const Position& pos, const EvalData& d, Color us) {
    bool queenless = !pos.pieces(us, QUEEN);
    if (d.safety_attackers[us] < 1 + queenless) return SP(0, 0);
    Color them = Color(us ^ 1);
    Square their_king = pos.king_sq(them);

    Bitboard weak = d.attacked[us] & ~d.attacked2[them]
                  & (~d.attacked[them] | d.attacked_by[them][KING]);
    Bitboard safe = ~pos.pieces(us)
                  & (~d.attacked[them] | (weak & d.attacked2[us]));

    Bitboard occ = pos.pieces();
    Bitboard rspan = rook_attacks_bb(their_king, occ);
    Bitboard bspan = bishop_attacks_bb(their_king, occ);
    Bitboard nchecks = d.attacked_by[us][KNIGHT] & knight_attacks_bb(their_king);
    Bitboard bchecks = d.attacked_by[us][BISHOP] & bspan;
    Bitboard rchecks = d.attacked_by[us][ROOK] & rspan;
    Bitboard qchecks = d.attacked_by[us][QUEEN] & (bspan | rspan);
    Bitboard allchecks = nchecks | bchecks | rchecks | qchecks;

    SP bonus = d.safety_value[us] + SafetyOffset;
    bonus += AttackWeight * d.safety_attacks[us];
    bonus += WeakKingZone * popcount(weak & d.king_zone[us]);
    if (queenless) bonus += QueenlessAttack;
    bonus += SafeKnightCheck * popcount(nchecks & safe);
    bonus += SafeBishopCheck * popcount(bchecks & safe);
    bonus += SafeRookCheck * popcount(rchecks & safe);
    bonus += SafeQueenCheck * popcount(qchecks & safe);
    bonus += UnsafeCheck * popcount(allchecks & ~safe);
    {
        Bitboard op = pos.pieces(us, PAWN), tp = pos.pieces(them, PAWN);
        File kf = file_of(their_king);
        if (kf > FILE_A) bonus += safety_file(op, tp, File(kf - 1), their_king, us);
        bonus += safety_file(op, tp, kf, their_king, us);
        if (kf < FILE_H) bonus += safety_file(op, tp, File(kf + 1), their_king, us);
    }
    // MG quadratic, EG linear, floor at 0 (their /256 and /16 scaling)
    int mg = std::max(bonus.mg, 0);
    int eg = std::max(bonus.eg, 0);
    return SP(mg * mg / 256, eg / 16);
}

// ============================================================
// Endgame scaling (simplified: no specialized table probe)
// ============================================================
static bool is_ocb(const Position& pos) {
    if (popcount(pos.pieces(WHITE, BISHOP)) != 1 || popcount(pos.pieces(BLACK, BISHOP)) != 1)
        return false;
    Bitboard b = pos.pieces(BISHOP);
    return ((b & dark_squares()) && (b & ~dark_squares()));
}

static int scale_endgame(const Position& pos, const KPEntry& kpe, int eg) {
    Color strong = eg > 0 ? WHITE : BLACK;
    Color weak = Color(strong ^ 1);
    int sm = material_mg(pos, strong), wm = material_mg(pos, weak);
    Bitboard sp = pos.pieces(strong, PAWN), wp2 = pos.pieces(weak, PAWN);
    int factor;
    if (!sp && sm - wm <= BISHOP_MG) {
        factor = (sm <= BISHOP_MG) ? SCALE_DRAW
                                   : std::max((sm - wm) * 8 / BISHOP_MG, (int)SCALE_DRAW);
    } else if (is_ocb(pos)) {
        factor = (sm + wm > 2 * BISHOP_MG)
                   ? 71 + popcount(pos.pieces(strong) & ~pos.pieces(strong, KING)) * 9
                   : 33 + popcount(kpe.passed[strong]) * 21;
    } else if (sm == ROOK_MG && wm == ROOK_MG
               && popcount(sp) < 2 + popcount(wp2)
               && !!(kingside_bb() & sp) != !!(queenside_bb() & sp)
               && !!(king_attacks_bb(pos.king_sq(weak)) & wp2)) {
        factor = 130;
    } else {
        factor = std::min(177 + 13 * popcount(sp), (int)SCALE_NORMAL);
    }
    return eg * factor / SCALE_NORMAL;
}

// KXK (their eval_kxk): lone king vs mating material
static bool is_kxk(const Position& pos, Color us) {
    Color them = Color(us ^ 1);
    if (popcount(pos.pieces(them) & ~pos.pieces(them, KING)) > 0) return false;
    return material_mg(pos, us) >= ROOK_MG;
}
static int eval_kxk_score(const Position& pos, Color us) {
    Color them = Color(us ^ 1);
    // stalemate guard: if it's their move and not in check, ensure a move exists
    if (pos.side_to_move() == them && !pos.is_check()) {
        ExtMove m[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(const_cast<Position&>(pos), m);
        if (end == m) return 0;
    }
    Square wk = pos.king_sq(us), lk = pos.king_sq(them);
    int score = material_mg(pos, us) + popcount(pos.pieces(us, PAWN)) * PAWN_MG;
    int wf = std::abs(int(file_of(lk)) - 3), wr = std::abs(int(rank_of(lk)) - 3);
    score += (std::max(wf, 0) * 0); // placeholder; corner bonus below
    score += (wf * wf + wr * wr);   // push to corner (their corner_bonus equivalent)
    int kd = std::abs(int(file_of(wk)) - int(file_of(lk))) + std::abs(int(rank_of(wk)) - int(rank_of(lk)));
    score += 70 - kd * 10;          // keep kings close
    Bitboard n = pos.pieces(KNIGHT), b = pos.pieces(BISHOP);
    if (pos.pieces(QUEEN) || pos.pieces(ROOK) || (n && b)
        || popcount(b & dark_squares()) + popcount(b & ~dark_squares()) >= 2
        || popcount(n) >= 3)
        score += VICTORY;
    return pos.side_to_move() == us ? score : -score;
}

// ============================================================
// Main entry
// ============================================================
int evaluate_stash(const Position& pos) {
    if (is_kxk(pos, WHITE)) return eval_kxk_score(pos, WHITE);
    if (is_kxk(pos, BLACK)) return eval_kxk_score(pos, BLACK);

    Bitboard dark = dark_squares();
    EvalData d;
    KPEntry kpe;

    evaldata_init(d, pos, WHITE);
    evaldata_init(d, pos, BLACK);
    evaldata_init_next(d, pos, WHITE);
    evaldata_init_next(d, pos, BLACK);
    evaldata_set_closed(d, pos);

    kpe = king_pawn_probe(pos);

    Bitboard kbw = king_blockers(pos, WHITE);
    Bitboard kbb = king_blockers(pos, BLACK);

    // king PSQT
    SP tapered(0, 0);
    tapered += psqt_piece(KingSQT, 0, 0, relative_square(WHITE, pos.king_sq(WHITE)));
    tapered -= psqt_piece(KingSQT, 0, 0, relative_square(BLACK, pos.king_sq(BLACK)));

    tapered += kpe.value;
    tapered += eval_knights(pos, d, kpe, kbw, WHITE);
    tapered -= eval_knights(pos, d, kpe, kbb, BLACK);
    tapered += eval_bishops(pos, d, kpe, kbw, WHITE, dark);
    tapered -= eval_bishops(pos, d, kpe, kbb, BLACK, dark);
    tapered += eval_rooks(pos, d, kpe, kbw, WHITE);
    tapered -= eval_rooks(pos, d, kpe, kbb, BLACK);
    tapered += eval_queens(pos, d, kbw, WHITE);
    tapered -= eval_queens(pos, d, kbb, BLACK);
    tapered += eval_passed_eval(pos, d, kpe, WHITE);
    tapered -= eval_passed_eval(pos, d, kpe, BLACK);
    tapered += eval_threats(pos, d, WHITE);
    tapered -= eval_threats(pos, d, BLACK);
    tapered += eval_safety(pos, d, WHITE);
    tapered -= eval_safety(pos, d, BLACK);
    tapered += (pos.side_to_move() == WHITE) ? Initiative : -Initiative;

    int mg = tapered.mg;
    int eg = scale_endgame(pos, kpe, tapered.eg);
    // Scale probe: Stash EG scores live at ~2x our search's expected cp
    // scale; halve the EG blend input so margins fire as our search expects.
    eg /= 2;
    int phase = 4 * popcount(pos.pieces(QUEEN)) + 2 * popcount(pos.pieces(ROOK))
              + popcount(pos.pieces(BISHOP)) + popcount(pos.pieces(KNIGHT));
    phase = std::max((int)ENDGAME_COUNT, std::min((int)MIDGAME_COUNT, phase));
    int score = mg * (phase - ENDGAME_COUNT) / (MIDGAME_COUNT - ENDGAME_COUNT)
              + eg * (MIDGAME_COUNT - phase) / (MIDGAME_COUNT - ENDGAME_COUNT);

    return pos.side_to_move() == WHITE ? score : -score;
}

} // namespace stash_eval

// UCI toggle
static bool g_stash_eval = false;
void set_stash_eval(bool on) { g_stash_eval = on; }
bool stash_eval_enabled() { return g_stash_eval; }

} // namespace luminex
