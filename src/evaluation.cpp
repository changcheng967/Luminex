#include "luminex.h"
#include <cstring>

namespace luminex {

EvalParams g_eval_params;

// ============================================================
// SELF-ENGINEERED EVALUATION
// All values derived from chess principles and logical reasoning.
// No values borrowed from PeSTO, Ethereal, Stash, or any other engine.
//
// Design philosophy:
//   1. Piece values based on classical chess theory (Reinfeld/Euwe)
//   2. PST tables from first principles: where pieces WANT to be
//   3. Mobility: linear formula (each square adds fixed value)
//   4. Threats: proportional to value gap between attacker and target
//   5. King safety: quadratic danger curve from attack units
//   6. Pawn structure: doubled/isolated/backward/connected/phalanx
//   7. Passed pawns: exponential growth in endgame
// ============================================================

// Self-engineered piece material values
// Logic: classical ratios with MG/EG adjustments
//   Pawn: 90 MG (development cost), 100 EG (full value)
//   Knight: 320 MG (3.2 pawns), 290 EG (slow piece, less valuable)
//   Bishop: 340 MG (diagonal range > L-shapes), 310 EG (long-range)
//   Rook: 500 MG (5 pawns), 530 EG (stronger with open files)
//   Queen: 960 MG (~10 pawns), 940 EG (king activity reduces power)
static constexpr int PieceValueMG[8] = { 90, 320, 340, 500, 960, 0, 0, 0 };
static constexpr int PieceValueEG[8] = { 100, 290, 310, 530, 940, 0, 0, 0 };

// ============================================================
// PIECE-SQUARE TABLES
// WHITE perspective, a1 = index 0, rows = ranks 1-8
//
// Principles for each piece type:
//   Pawn: center > wing, advancement scales with rank
//   Knight: center is king ("rim is dim"), edges penalized
//   Bishop: long diagonals (c4,d5,e4,f5), avoid corners
//   Rook: open files, 7th rank penetration
//   Queen: don't overextend early, central when attacking
//   King MG: safety behind pawn shield
//   King EG: centralization critical
// ============================================================

Score PST_MG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN MG: Central pawns control more squares. Advanced = dangerous.
        // Wing pawns penalized (less space control, harder to push).
        {
              0,   0,   0,   0,   0,   0,   0,   0,
            -15,  -5,  -5,   0,   0,  -5,  -5, -15,
            -12,  -2,   0,   5,   5,   0,  -2, -12,
             -8,   0,   5,  15,  15,   5,   0,  -8,
             -5,   5,  15,  25,  25,  15,   5,  -5,
              5,  20,  35,  45,  45,  35,  20,   5,
             50,  60,  65,  70,  70,  65,  60,  50,
              0,   0,   0,   0,   0,   0,   0,   0
        },
        // KNIGHT MG: "A knight on the rim is dim."
        // d4/d5/e4/e5 are the dream squares (outposts in center).
        // Edges and corners heavily penalized.
        {
            -70, -35, -25, -20, -20, -25, -35, -70,
            -25, -10,   0,  10,  10,   0, -10, -25,
            -15,   5,  15,  25,  25,  15,   5, -15,
            -10,  10,  25,  40,  40,  25,  10, -10,
            -10,  10,  25,  40,  40,  25,  10, -10,
            -15,   0,  15,  20,  20,  15,   0, -15,
            -30, -10,   5,  10,  10,   5, -10, -30,
            -70, -30, -25, -20, -20, -25, -30, -70
        },
        // BISHOP MG: Long diagonals are key. Developed to c4/f4/d5/e5 = strong.
        // Fianchetto squares (b2/g2) get moderate bonus.
        // Corners and back rank penalized (undeveloped).
        {
            -25, -10, -15, -10, -10, -15, -10, -25,
            -10,   5,  10,  10,  10,  10,   5, -10,
             -5,  10,  15,  15,  15,  15,  10,  -5,
             -5,  10,  15,  25,  25,  15,  10,  -5,
             -5,  10,  15,  25,  25,  15,  10,  -5,
             -5,  10,  15,  15,  15,  15,  10,  -5,
            -15,   5,   5,  10,  10,   5,   5, -15,
            -25, -10, -15, -10, -10, -15, -10, -25
        },
        // ROOK MG: Activity on open files, 7th rank is famous.
        // Central files (d/e) slightly better (control of center).
        // Trapped on back rank = penalty (handled by trapped term).
        {
            -15,  -5,  -5,   5,   5,  -5,  -5, -15,
            -20, -10,   0,  10,  10,   0, -10, -20,
            -15,  -5,   0,  10,  10,   0,  -5, -15,
            -15,  -5,   5,  15,  15,   5,  -5, -15,
            -15,  -5,   5,  15,  15,   5,  -5, -15,
            -10,   0,  10,  20,  20,  10,   0, -10,
              5,  20,  25,  35,  35,  25,  20,   5,
             10,  15,  20,  25,  25,  20,  15,  10
        },
        // QUEEN MG: Don't develop too early (risk of being attacked).
        // Central activity is strong WHEN SAFE (behind pieces).
        // Back rank is acceptable (queen is flexible).
        {
             -5, -10,  -5,   5,   5,  -5, -10,  -5,
            -10,  -5,   0,  10,  10,   0,  -5, -10,
            -10,  -5,   5,  10,  10,   5,  -5, -10,
            -15,  -5,   0,  10,  10,   0,  -5, -15,
            -10,   0,   5,  15,  15,   5,   0, -10,
             -5,   5,  10,  20,  20,  10,   5,  -5,
            -10,   0,   5,  15,  15,   5,   0, -10,
            -10,  -5,   0,   5,   5,   0,  -5, -10
        },
        // KING MG: Safety is paramount. Castled position = strong bonus.
        // g1/c1 are castled positions (highest bonus).
        // Center = dangerous (exposed to attack).
        // Higher ranks = increasingly dangerous (no shelter).
        {
             15,  10,  -5, -25, -25,  -5,  20,  15,
              5,   0, -15, -30, -30, -15,   0,   5,
            -15, -15, -20, -30, -30, -20, -15, -15,
            -25, -20, -25, -35, -35, -25, -20, -25,
            -30, -25, -25, -35, -35, -25, -25, -30,
            -35, -30, -30, -40, -40, -30, -30, -35,
            -40, -35, -30, -40, -40, -30, -35, -40,
            -45, -40, -30, -40, -40, -30, -40, -45
        },
        // NONE / ALL
        {0}, {0}
    },
    // BLACK (populated by init_evaluation via rank mirroring)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

Score PST_EG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN EG: Advancement is critical. Passed pawns win games.
        // Rank 6-7 pawns are extremely valuable (promotion threat).
        {
              0,   0,   0,   0,   0,   0,   0,   0,
            -10,  -3,  -3,   0,   0,  -3,  -3, -10,
             -5,   0,   0,   5,   5,   0,   0,  -5,
              0,   5,   8,  15,  15,   8,   5,   0,
             10,  15,  20,  30,  30,  20,  15,  10,
             30,  40,  50,  65,  65,  50,  40,  30,
             70,  80,  90, 100, 100,  90,  80,  70,
              0,   0,   0,   0,   0,   0,   0,   0
        },
        // KNIGHT EG: Centralization still matters but less extreme than MG.
        // Slow piece, vulnerable to being outmaneuvered by king.
        {
            -50, -25, -20, -15, -15, -20, -25, -50,
            -20, -10,   0,   5,   5,   0, -10, -20,
            -10,   0,  10,  15,  15,  10,   0, -10,
            -10,   5,  15,  25,  25,  15,   5, -10,
            -10,   5,  15,  25,  25,  15,   5, -10,
            -10,   0,  10,  15,  15,  10,   0, -10,
            -20, -10,   0,   5,   5,   0, -10, -20,
            -50, -25, -20, -15, -15, -20, -25, -50
        },
        // BISHOP EG: Active bishop is one of the strongest endgame pieces.
        // Centralization for maximum board coverage.
        {
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10,   0,   5,   5,   5,   5,   0, -10,
             -5,   5,  10,  10,  10,  10,   5,  -5,
             -5,   5,  10,  15,  15,  10,   5,  -5,
             -5,   5,  10,  15,  15,  10,   5,  -5,
             -5,   5,  10,  10,  10,  10,   5,  -5,
            -10,   0,   5,   5,   5,   5,   0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        },
        // ROOK EG: Activity is everything. 7th rank wins games.
        // Central files provide best cutting-off opportunities.
        {
            -10,  -5,   0,   5,   5,   0,  -5, -10,
            -10,  -5,   5,  10,  10,   5,  -5, -10,
             -5,   0,   5,  10,  10,   5,   0,  -5,
             -5,   5,  10,  15,  15,  10,   5,  -5,
              0,   5,  10,  15,  15,  10,   5,   0,
              5,  10,  15,  20,  20,  15,  10,   5,
             15,  25,  25,  30,  30,  25,  25,  15,
             10,  15,  20,  25,  25,  20,  15,  10
        },
        // QUEEN EG: Powerful everywhere. Slight central preference for
        // coordination with king and stopping passed pawns.
        {
            -10,  -5,  -5,   0,   0,  -5,  -5, -10,
             -5,   0,   5,  10,  10,   5,   0,  -5,
             -5,   5,  10,  15,  15,  10,   5,  -5,
             -5,   5,  10,  20,  20,  10,   5,  -5,
             -5,   5,  10,  20,  20,  10,   5,  -5,
             -5,   5,  10,  15,  15,  10,   5,  -5,
             -5,   0,   5,  10,  10,   5,   0,  -5,
            -10,  -5,  -5,   0,   0,  -5,  -5, -10
        },
        // KING EG: Active king is critical. Must centralize.
        // d4/d5/e4/e5 = maximum (king wants to be in the center).
        // Edge/corner = penalty (passive, can't stop passed pawns).
        {
            -40, -25, -15,  -5,  -5, -15, -25, -40,
            -20, -10,   0,  10,  10,   0, -10, -20,
            -10,   0,  10,  20,  20,  10,   0, -10,
             -5,   5,  15,  25,  25,  15,   5,  -5,
             -5,   5,  15,  25,  25,  15,   5,  -5,
            -10,   0,  10,  20,  20,  10,   0, -10,
            -20, -10,   0,  10,  10,   0, -10, -20,
            -40, -25, -15,  -5,  -5, -15, -25, -40
        },
        // NONE / ALL
        {0}, {0}
    },
    // BLACK (populated by init_evaluation via rank mirroring)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

using Score = Value;

// ============================================================
// Mobility system: linear formula instead of lookup tables
// Each additional square of mobility adds a fixed bonus.
// A piece with 0 mobility is penalized (trapped = useless).
//
// Principle: a piece that can't move is almost worthless.
// Each freedom of movement adds incremental value.
// Knights need mobility most (limited range).
// Rooks/queens naturally have high mobility (lower per-square value).
// ============================================================

// Knight mobility: calibrated so average (mob=4) ≈ 0
// base = -avg_mob * slope = -4 * 15 = -60
// mob=0: -60 (trapped), mob=4: 0 (normal), mob=8: +60 (excellent)
static constexpr int KnightMobBaseMG = -60;
static constexpr int KnightMobSlopeMG = 15;
static constexpr int KnightMobBaseEG = -75;
static constexpr int KnightMobSlopeEG = 18;
static constexpr int KnightMobMax = 8;

// Bishop mobility: calibrated so average (mob=7) ≈ 0
// base = -7 * 7 = -49. mob=0: -48, mob=7: +1, mob=13: +43
static constexpr int BishopMobBaseMG = -48;
static constexpr int BishopMobSlopeMG = 7;
static constexpr int BishopMobBaseEG = -56;
static constexpr int BishopMobSlopeEG = 9;
static constexpr int BishopMobMax = 13;

// Rook mobility: calibrated so average (mob=8) ≈ 0
// base = -8 * 5 = -40. mob=0: -38, mob=8: +2, mob=14: +32
static constexpr int RookMobBaseMG = -38;
static constexpr int RookMobSlopeMG = 5;
static constexpr int RookMobBaseEG = -48;
static constexpr int RookMobSlopeEG = 7;
static constexpr int RookMobMax = 14;

// Queen mobility: calibrated so average (mob=15) ≈ 0
// base = -15 * 2 = -30. mob=0: -28, mob=15: +2, mob=27: +26
static constexpr int QueenMobBaseMG = -28;
static constexpr int QueenMobSlopeMG = 2;
static constexpr int QueenMobBaseEG = -38;
static constexpr int QueenMobSlopeEG = 3;
static constexpr int QueenMobMax = 27;

// ============================================================
// Threat evaluation: bonus proportional to value gap
// The cheaper the attacker, the more dangerous the threat.
// Threatening a queen with a pawn = devastating (value gap ~860).
// Threatening a knight with a queen = minor (queen already strong).
// ============================================================

// [target piece type]: P=0, N=1, B=2, R=3, Q=4, K=5
// Pawn threats: pawns attacking higher-value pieces is always strong
static constexpr int PawnThreatMG[6] = {  0, 35, 40, 60, 90, 0 };
static constexpr int PawnThreatEG[6] = {  0, 40, 45, 70,100, 0 };

// Knight threats: minor piece attacking higher-value pieces
static constexpr int KnightThreatMG[6] = { 15,  5, 20, 45, 60, 0 };
static constexpr int KnightThreatEG[6] = { 20,  5, 25, 50, 65, 0 };

// Bishop threats: similar to knight (both are minors)
static constexpr int BishopThreatMG[6] = { 15, 20,  5, 45, 60, 0 };
static constexpr int BishopThreatEG[6] = { 20, 25,  5, 50, 65, 0 };

// Rook threats: rook threatening queen is strong
static constexpr int RookThreatMG[6] = { 10, 25, 25,  5, 50, 0 };
static constexpr int RookThreatEG[6] = { 12, 30, 30,  5, 55, 0 };

// Queen threats: queen is already powerful, threats are less "extra"
static constexpr int QueenThreatMG[6] = {  5, 15, 15, 15,  5, 0 };
static constexpr int QueenThreatEG[6] = {  5, 18, 18, 18,  5, 0 };

// ============================================================
// Pawn structure constants
// ============================================================

// Self-engineered phalanx bonus (side-by-side pawns on same rank)
// Principle: connected pawns on higher ranks create space advantage
// Rank 4-5: strong duo controlling center
// Rank 6-7: extremely dangerous (both threatening promotion)
static constexpr int PhalanxMG[8] = { 0, 5, 10, 20, 40, 80, 120, 0 };
static constexpr int PhalanxEG[8] = { 0, 5, 15, 30, 60, 120, 180, 0 };

// Candidate passed pawn bonus (has enemy pawns ahead but can push through)
// [supported by own pawn?][rank]
static constexpr int CandMG[2][8] = {
    { 0, -5,  -8, -10, -12,  10, 0, 0 },   // unsupported
    { 0, -8,  -3,   2,   8,  25, 0, 0 }    // supported
};
static constexpr int CandEG[2][8] = {
    { 0,  5,  10,  20,  40,  60, 0, 0 },
    { 0, 10,  20,  40,  70, 110, 0, 0 }
};

// Base passed pawn bonus (no enemy pawns ahead at all)
// MG: cautious (can be blockaded), EG: aggressive (wins games)
static constexpr int PassedMG[8] = { 0, -8,  0, 10, 25, 50, 100, 0 };
static constexpr int PassedEG[8] = { 0, 10, 20, 40, 70,120, 200, 0 };

// ============================================================
// Pawn hash table
// ============================================================
struct PawnEntry {
    uint64_t key;
    int32_t mg, eg;
};
static constexpr int PAWN_HASH_SIZE = 16384;
static PawnEntry pawn_table[PAWN_HASH_SIZE];

// Evaluate pawn-only terms and cache in pawn_table
static void evaluate_pawns(const Position& pos, int32_t& mg_out, int32_t& eg_out) {
    uint64_t pawn_key = pos.pawn_key();
    int idx = int(pawn_key & uint64_t(PAWN_HASH_SIZE - 1));
    PawnEntry& entry = pawn_table[idx];
    if (entry.key == pawn_key) {
        mg_out = entry.mg;
        eg_out = entry.eg;
        return;
    }

    int32_t mg = 0;
    int32_t eg = 0;

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        Sign sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        // Pre-compute file counts for doubled/isolated detection
        int file_count[8] = {0};
        Bitboard tmp = our_pawns;
        while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }

        // Adjacent file pawn support for connected pawn detection
        Bitboard supported_by_adj = pawn_attacks_bb(c, our_pawns);

        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            // Material + PST
            mg += sign * (PieceValueMG[PAWN] + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg += sign * (PieceValueEG[PAWN] + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);

            // Doubled pawn: two pawns on same file block each other
            if (file_count[f] > 1) {
                mg -= sign * 12;
                eg -= sign * 20;
            }

            // Isolated pawn: no friendly pawns on adjacent files
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg -= sign * 15;
                eg -= sign * 20;
            }

            // Connected pawn: protected by another pawn (pawn chain)
            if (square_bb(sq) & supported_by_adj) {
                int connected_bonus = 5 + r * 3;
                mg += sign * connected_bonus;
                eg += sign * (3 + r * 2);
            }

            // Pawn phalanx: side-by-side pawns on same rank
            {
                Bitboard neighbors = our_pawns & (file_bb(File(f - 1)) | file_bb(File(f + 1)));
                Bitboard same_rank = neighbors & rank_bb(rank_of(sq));
                if (same_rank) {
                    if (r >= RANK_2 && r <= RANK_7) {
                        mg += sign * PhalanxMG[r];
                        eg += sign * PhalanxEG[r];
                    }
                }
            }

            // Pawn lever: can capture an adjacent enemy pawn (gaining space)
            {
                Bitboard lever_targets = pawn_attacks_bb(c, sq) & their_pawns;
                if (lever_targets) {
                    int lever_bonus = 4 + r * 2;
                    mg += sign * lever_bonus;
                    // Central levers (c-f files) are more valuable
                    if (f >= FILE_C && f <= FILE_F) {
                        mg += sign * 3;
                    }
                }
            }

            // Backward pawn: can't safely advance (push square is attacked)
            {
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                if (push < SQUARE_NONE && r >= RANK_2 && r <= RANK_5) {
                    bool push_attacked = (pawn_attacks_bb(them, their_pawns) & square_bb(push)) != 0;
                    if (push_attacked) {
                        Bitboard our_pawn_attacks = pawn_attacks_bb(c, our_pawns);
                        bool defended = (our_pawn_attacks & square_bb(push)) != 0;
                        if (!defended) {
                            mg -= sign * 12;
                            eg -= sign * 15;
                        }
                    }
                }
            }

            // Passed pawn detection + base bonus
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            // Candidate passed pawn (some stoppers but can push through)
            if ((ahead & their_pawns) && r >= RANK_2 && r <= RANK_6) {
                Bitboard stoppers = ahead & their_pawns;
                Bitboard threats = their_pawns & pawn_attacks_bb(c, sq);
                Bitboard support = our_pawns & pawn_attacks_bb(them, sq);
                Bitboard push_sq_bb = relative_square(c, make_square(f, Rank(r + 1)));
                Bitboard push_threats = their_pawns & pawn_attacks_bb(c, push_sq_bb);
                Bitboard push_support = our_pawns & pawn_attacks_bb(them, push_sq_bb);
                Bitboard leftovers = stoppers & ~(threats | push_threats);
                bool supported = popcount(support) >= popcount(threats);
                if (!leftovers && popcount(push_support) >= popcount(push_threats)) {
                    mg += sign * CandMG[supported ? 1 : 0][r];
                    eg += sign * CandEG[supported ? 1 : 0][r];
                }
            }

            // Truly passed pawn (no enemy pawns ahead or adjacent)
            if (!(ahead & their_pawns)) {
                mg += sign * PassedMG[r];
                eg += sign * PassedEG[r];
            }
        }
    }

    entry.key = pawn_key;
    entry.mg = mg;
    entry.eg = eg;
    mg_out = mg;
    eg_out = eg;
}

// ============================================================
// Main evaluation function
// ============================================================
Value evaluate(const Position& pos, bool tactical_only) {
    // KXK endgame: one side has only a king
    for (int strong = 0; strong < 2; ++strong) {
        Color c = Color(strong);
        Color weak = Color(strong ^ 1);
        int strong_material = popcount(pos.pieces(c)) - 1;
        int weak_material = popcount(pos.pieces(weak)) - 1;
        if (strong_material >= 1 && weak_material == 0) {
            bool has_major = (pos.pieces(c, QUEEN) || pos.pieces(c, ROOK));
            bool has_bn = (pos.pieces(c, BISHOP) && pos.pieces(c, KNIGHT));
            bool has_two_bishops = (popcount(pos.pieces(c, BISHOP)) >= 2);
            bool has_three_knights = (popcount(pos.pieces(c, KNIGHT)) >= 3);
            if (has_major || has_bn || has_two_bishops || has_three_knights) {
                Square weak_ksq = pos.king_sq(weak);
                Square strong_ksq = pos.king_sq(c);

                // Push weak king to corner
                int wfile = file_of(weak_ksq);
                int wrank = rank_of(weak_ksq);
                int center_dist_file = std::max(wfile - 3, 7 - wfile);
                int center_dist_rank = std::max(wrank - 3, 7 - wrank);

                int corner_bonus;
                if (has_bn && strong_material == 2) {
                    // KBN vs K: drive to corner matching bishop's color
                    Square bsq = lsb(pos.pieces(c, BISHOP));
                    bool bishop_light = ((int(bsq) + int(bsq) / 8) % 2) == 0;
                    bool weak_light = ((int(weak_ksq) + int(weak_ksq) / 8) % 2) == 0;
                    if (bishop_light == weak_light) {
                        corner_bonus = center_dist_file * center_dist_file + center_dist_rank * center_dist_rank;
                    } else {
                        corner_bonus = (center_dist_file * center_dist_file + center_dist_rank * center_dist_rank) / 2;
                    }
                } else {
                    corner_bonus = center_dist_file * center_dist_file + center_dist_rank * center_dist_rank;
                }

                // Keep kings close for mating
                int king_dist = std::abs(file_of(strong_ksq) - wfile) + std::abs(rank_of(strong_ksq) - wrank);
                int close_bonus = 70 - king_dist * 10;

                int score = (50 - corner_bonus) * 2 + close_bonus;
                score += VALUE_KNOWN_WIN;

                score += popcount(pos.pieces(c, QUEEN)) * PieceValueEG[QUEEN];
                score += popcount(pos.pieces(c, ROOK)) * PieceValueEG[ROOK];
                score += popcount(pos.pieces(c, BISHOP)) * PieceValueEG[BISHOP];
                score += popcount(pos.pieces(c, KNIGHT)) * PieceValueEG[KNIGHT];

                return (strong == WHITE) ? score : -score;
            }
            // Non-mating material (lone bishop/knight vs king): drawn
            return VALUE_DRAW;
        }
    }

    Score mg_score = 0;
    Score eg_score = 0;

    Square ksq_arr[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};
    int shield_count[2] = {0, 0};  // Pawn shield pawns for danger amplification

    Bitboard occupied = pos.pieces();

    // Per-piece attack maps for threats and king safety
    Bitboard attacks_by[2][7] = {};
    Bitboard all_attacks[2] = {};

    // Initialize pawn and king attacks
    for (int c = 0; c < 2; ++c) {
        attacks_by[c][PAWN] = pawn_attacks_bb(Color(c), pos.pieces(Color(c), PAWN));
        attacks_by[c][KING] = king_attacks_bb(ksq_arr[c]);
        all_attacks[c] = attacks_by[c][PAWN] | attacks_by[c][KING];
    }

    // Pawn evaluation via pawn hash table
    {
        int32_t pawn_mg = 0, pawn_eg = 0;
        evaluate_pawns(pos, pawn_mg, pawn_eg);
        mg_score += pawn_mg;
        eg_score += pawn_eg;
    }

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        Sign sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        // Mobility area: exclude squares attacked by enemy pawns
        Bitboard mob_area = tactical_only ? Bitboard(0) : ~pawn_attacks_bb(them, their_pawns);

        // -------------------------------------------------------
        // Passed pawn extra bonuses (beyond base table)
        // -------------------------------------------------------
        if (!tactical_only) {
        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            Bitboard ahead = 0;
            Bitboard ahead_file = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                ahead_file |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            if (!(ahead & their_pawns)) {
                int mg_passer = 0;
                int eg_passer = 0;

                // Rook behind passed pawn: supports advance
                Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
                if (behind) {
                    Square rsq2 = lsb(behind);
                    bool rook_behind = (c == WHITE) ? (rank_of(rsq2) < rank_of(sq)) : (rank_of(rsq2) > rank_of(sq));
                    if (rook_behind) {
                        mg_passer += 20;
                        eg_passer += 30;
                    }
                }

                // Enemy rook behind: reduces bonus (can blockade)
                Bitboard enemy_behind = file_bb(f) & pos.pieces(them, ROOK);
                if (enemy_behind) {
                    Square ersq = lsb(enemy_behind);
                    bool enemy_rook_behind = (c == WHITE) ? (rank_of(ersq) < rank_of(sq)) : (rank_of(ersq) > rank_of(sq));
                    if (enemy_rook_behind) {
                        mg_passer -= 10;
                        eg_passer -= 15;
                    }
                }

                // King proximity (EG only): closer king = better support
                Square our_ksq = ksq_arr[c_idx];
                Square their_ksq = ksq_arr[c_idx ^ 1];
                Square promo_sq = relative_square(c, make_square(f, RANK_8));
                int our_kdist = distance(our_ksq, promo_sq);
                int their_kdist = distance(their_ksq, promo_sq);
                eg_passer += (their_kdist - our_kdist) * 8;

                // Free passed pawn: enemy king can't catch it
                if (their_kdist > our_kdist + (c == pos.side_to_move() ? 0 : 1)) {
                    int unreachable = their_kdist - our_kdist;
                    eg_passer += unreachable * (unreachable > 4 ? 25 : 15);
                }

                // Blocked by enemy pieces: harder to push
                if (ahead_file & pos.pieces(them)) {
                    mg_passer -= 15;
                    eg_passer -= 20;
                }

                mg_score += sign * mg_passer;
                eg_score += sign * eg_passer;
            }
        }
        } // end !tactical_only passed pawns

        // -------------------------------------------------------
        // Knights
        // -------------------------------------------------------
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * (PieceValueMG[KNIGHT] + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (PieceValueEG[KNIGHT] + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);

            Bitboard attacks = knight_attacks_bb(sq);
            attacks_by[c][KNIGHT] |= attacks;
            all_attacks[c] |= attacks;

            if (!tactical_only) {
            // Mobility: linear formula (trapped knight = useless)
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, KnightMobMax);
            mg_score += sign * (KnightMobBaseMG + mob * KnightMobSlopeMG);
            eg_score += sign * (KnightMobBaseEG + mob * KnightMobSlopeEG);

            // Outpost: knight on rank 4-6, protected by own pawn,
            // not attackable by enemy pawn (permanent advantage)
            Rank kr = relative_rank(c, sq);
            if (kr >= RANK_4 && kr <= RANK_6) {
                if (pawn_attacks_bb(c, our_pawns) & square_bb(sq)) {
                    Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
                    if (!(enemy_pawn_attacks & square_bb(sq))) {
                        mg_score += sign * (g_eval_params.outpost_knight_mg + (kr - 3) * 5);
                        eg_score += sign * (g_eval_params.outpost_knight_eg + (kr - 3) * 3);
                    }
                }
            }

            // Trapped knight: 0-mobility knight is severely limited
            if (mob == 0) {
                mg_score -= sign * 10;
                eg_score -= sign * 5;
            }

            // Far from own king: harder to defend
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * g_eval_params.far_knight_mg;
                eg_score -= sign * g_eval_params.far_knight_eg;
            }
            } // end !tactical_only knight
        }

        // -------------------------------------------------------
        // Bishops
        // -------------------------------------------------------
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * (PieceValueMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PieceValueEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);

            Bitboard attacks = bb_diag_attacks(sq, occupied);
            attacks_by[c][BISHOP] |= attacks;
            all_attacks[c] |= attacks;

            if (!tactical_only) {
            // Mobility: linear formula
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, BishopMobMax);
            mg_score += sign * (BishopMobBaseMG + mob * BishopMobSlopeMG);
            eg_score += sign * (BishopMobBaseEG + mob * BishopMobSlopeEG);

            // Bishop with many same-color pawns: bad bishop penalty
            {
                static constexpr Bitboard BB_DARK_SQ = 0x55AA55AA55AA55AAULL;
                static constexpr Bitboard BB_LIGHT_SQ = 0xAA55AA55AA55AA55ULL;
                bool is_dark = ((int(sq) + rank_of(sq)) % 2) == 0;
                Bitboard same_color_bb = is_dark ? BB_DARK_SQ : BB_LIGHT_SQ;
                int same_color_pawns = popcount(our_pawns & same_color_bb);
                same_color_pawns = std::min(same_color_pawns, 6);
                static constexpr int BscMG[7] = { 20, 15, 10, 5, 0, -8, -15 };
                static constexpr int BscEG[7] = { 40, 30, 20, 10, 0, -10, -20 };
                mg_score += sign * BscMG[same_color_pawns];
                eg_score += sign * BscEG[same_color_pawns];
            }

            // Bishop on long diagonal (sees 2+ center squares)
            if (popcount(attacks & BB_CENTER) >= 2) {
                mg_score += sign * 12;
                eg_score += sign * 20;
            }

            // Bishop outpost
            {
                Rank br = relative_rank(c, sq);
                if (br >= RANK_4 && br <= RANK_6) {
                    if ((pawn_attacks_bb(c, our_pawns) & square_bb(sq)) &&
                        !(pawn_attacks_bb(them, their_pawns) & square_bb(sq))) {
                        mg_score += sign * g_eval_params.outpost_bishop_mg;
                        eg_score += sign * g_eval_params.outpost_bishop_eg;
                    }
                }
            }

            // Far bishop penalty
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * g_eval_params.far_bishop_mg;
                eg_score -= sign * g_eval_params.far_bishop_eg;
            }
            } // end !tactical_only bishop
        }

        // -------------------------------------------------------
        // Rooks
        // -------------------------------------------------------
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (PieceValueMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PieceValueEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);

            // Rook mobility: exclude own rooks/queens for x-ray effect
            Bitboard attacks = rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
            attacks_by[c][ROOK] |= attacks;
            all_attacks[c] |= attacks;

            if (!tactical_only) {
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, RookMobMax);
            mg_score += sign * (RookMobBaseMG + mob * RookMobSlopeMG);
            eg_score += sign * (RookMobBaseEG + mob * RookMobSlopeEG);

            // Open / semi-open file
            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * g_eval_params.rook_open_mg;
                eg_score += sign * g_eval_params.rook_open_eg;
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * g_eval_params.rook_semi_open_mg;
                eg_score += sign * g_eval_params.rook_semi_open_eg;
            }

            // Rook on 7th rank
            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * g_eval_params.rook_7th_mg;
                eg_score += sign * g_eval_params.rook_7th_eg;
            }

            // Rook xray: same file as enemy queen
            if (file_bb(f) & pos.pieces(them, QUEEN)) {
                mg_score += sign * 15;
                eg_score += sign * 5;
            }

            // Far rook penalty
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * 8;
                eg_score -= sign * 5;
            }

            // Rook trapped/buried behind king
            if (rr <= RANK_2) {
                Square ksq = ksq_arr[c_idx];
                int fdist = std::abs(file_of(sq) - file_of(ksq));
                bool king_between = (c == WHITE)
                    ? (rank_of(ksq) <= rank_of(sq) && fdist <= 1)
                    : (rank_of(ksq) >= rank_of(sq) && fdist <= 1);
                if (king_between && mob <= 4) {
                    CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
                    CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
                    bool has_castling = pos.castling_allowed(c, ks_cr) || pos.castling_allowed(c, qs_cr);
                    if (has_castling) {
                        mg_score += sign * (-8);
                        eg_score += sign * (-12);
                    } else {
                        mg_score += sign * (-55);
                        eg_score += sign * (-25);
                    }
                }
            }

            // Rook on blocked file: own pawns ahead restrict activity
            {
                Bitboard ahead_pawns;
                if (c == WHITE) {
                    Bitboard below_and_same = 0;
                    for (int r = RANK_1; r <= rank_of(sq); ++r)
                        below_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~below_and_same;
                } else {
                    Bitboard above_and_same = 0;
                    for (int r = rank_of(sq); r <= RANK_8; ++r)
                        above_and_same |= rank_bb(Rank(r));
                    ahead_pawns = our_pawns & file_bb(f) & ~above_and_same;
                }
                if (ahead_pawns) {
                    mg_score += sign * (-8);
                    eg_score += sign * (-8);
                }
            }
            } // end !tactical_only rook
        }

        // -------------------------------------------------------
        // Queens
        // -------------------------------------------------------
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            mg_score += sign * (PieceValueMG[QUEEN] + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (PieceValueEG[QUEEN] + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);

            // Queen mobility: x-ray through own bishops/rooks
            Bitboard attacks = bb_diag_attacks(sq, occupied ^ pos.pieces(c, BISHOP))
                             | rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK));
            attacks_by[c][QUEEN] |= attacks;
            all_attacks[c] |= attacks;

            if (!tactical_only) {
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, QueenMobMax);
            mg_score += sign * (QueenMobBaseMG + mob * QueenMobSlopeMG);
            eg_score += sign * (QueenMobBaseEG + mob * QueenMobSlopeEG);

            // Far queen penalty
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * 8;
                eg_score -= sign * 3;
            }
            } // end !tactical_only queen

            // Weak queen: pinned to own king (forced passivity)
            // NOTE: pos.pinned() returns enemy sniper positions, not our pinned pieces.
            // The weak queen concept is handled implicitly by mobility (pinned queen = low mob).
            // Removing explicit penalty as block_checkers has Color issues in eval loop.
        }

        // -------------------------------------------------------
        // King
        // -------------------------------------------------------
        Square ksq = ksq_arr[c_idx];
        mg_score += sign * (PieceValueMG[KING] + PST_MG_TABLE[int(c)][int(KING)][int(ksq)]);
        eg_score += sign * (PieceValueEG[KING] + PST_EG_TABLE[int(c)][int(KING)][int(ksq)]);

        if (!tactical_only) {
        // King pawn shield (MG only: safety matters in middlegame)
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);
        bool on_back = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);
        if (on_back) {
            for (int r = 1; r <= 3; ++r) {
                int weight = (r == 1) ? g_eval_params.pawn_shield_knight
                             : (r == 2) ? g_eval_params.pawn_shield_center
                             : g_eval_params.pawn_shield_rook;
                for (int df = -1; df <= 1; ++df) {
                    File sf = File(int(kfile) + df);
                    if (sf < FILE_A || sf > FILE_H) continue;
                    Square shield_sq = relative_square(c, make_square(sf, Rank(r)));
                    if (our_pawns & square_bb(shield_sq)) {
                        mg_score += sign * weight;
                        if (r <= 2) shield_count[c_idx]++;  // Track front-rank shield pawns
                    }
                }
            }

            Bitboard all_pawns = pos.pieces(PAWN);
            if (!(all_pawns & file_bb(kfile))) mg_score -= sign * g_eval_params.open_file_penalty_mg;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) mg_score -= sign * g_eval_params.open_file_penalty_eg;
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) mg_score -= sign * g_eval_params.open_file_penalty_eg;

            for (int df = -1; df <= 1; ++df) {
                File storm_file = File(int(kfile) + df);
                if (storm_file < FILE_A || storm_file > FILE_H) continue;
                Bitboard enemy_file_pawns = their_pawns & file_bb(storm_file);
                while (enemy_file_pawns) {
                    Square psq = pop_lsb(enemy_file_pawns);
                    Rank pr = relative_rank(c, rank_of(psq));
                    if (pr >= RANK_4) {
                        int storm_danger = (pr - 3) * g_eval_params.pawn_storm;
                        mg_score -= sign * storm_danger;
                    }
                }
            }
        }

        // Castling evaluation
        bool castled = false;
        if (c == WHITE && krank == RANK_1 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (c == BLACK && krank == RANK_8 && (kfile == FILE_G || kfile == FILE_C)) castled = true;
        if (castled) { mg_score += sign * 30; eg_score += sign * 10; }

        CastlingRight ks_cr = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight qs_cr = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        if (!pos.castling_allowed(c, ks_cr) && !pos.castling_allowed(c, qs_cr) && !castled) {
            mg_score -= sign * 25;
            eg_score -= sign * 10;
        }
        } // end !tactical_only king

        // -------------------------------------------------------
        // Threat evaluation
        // -------------------------------------------------------
        {
            Bitboard their_pieces = pos.pieces(them);

            // Pawn threats: cheapest attacker = most valuable threats
            Bitboard threats = their_pieces & attacks_by[c][PAWN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * PawnThreatMG[pt];
                    eg_score += sign * PawnThreatEG[pt];
                }
            }
            // Knight threats
            threats = their_pieces & attacks_by[c][KNIGHT];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * KnightThreatMG[pt];
                    eg_score += sign * KnightThreatEG[pt];
                }
            }
            // Bishop threats
            threats = their_pieces & attacks_by[c][BISHOP];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * BishopThreatMG[pt];
                    eg_score += sign * BishopThreatEG[pt];
                }
            }
            // Rook threats
            threats = their_pieces & attacks_by[c][ROOK];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * RookThreatMG[pt];
                    eg_score += sign * RookThreatEG[pt];
                }
            }
            // Queen threats
            threats = their_pieces & attacks_by[c][QUEEN];
            while (threats) {
                PieceType pt = piece_type_of(pos.piece_on(pop_lsb(threats)));
                if (pt < KING) {
                    mg_score += sign * QueenThreatMG[pt];
                    eg_score += sign * QueenThreatEG[pt];
                }
            }
            // Hanging pawns: enemy pawns undefended and attacked by us
            Bitboard hanging_pawns = pos.pieces(them, PAWN) & ~all_attacks[them] & all_attacks[c];
            if (hanging_pawns) {
                mg_score += sign * g_eval_params.hanging_pawn_mg * popcount(hanging_pawns);
                eg_score += sign * g_eval_params.hanging_pawn_eg * popcount(hanging_pawns);
            }
            // Hanging pieces: enemy minor/major pieces undefended and attacked by us
            // Principle: a loose piece must move or be captured, creating tempo advantage
            Bitboard hanging_pieces = (pos.pieces(them, KNIGHT) | pos.pieces(them, BISHOP)
                                     | pos.pieces(them, ROOK) | pos.pieces(them, QUEEN))
                                     & ~all_attacks[them] & all_attacks[c];
            if (hanging_pieces) {
                mg_score += sign * 25 * popcount(hanging_pieces);
                eg_score += sign * 15 * popcount(hanging_pieces);
            }
        }
    }

    // -------------------------------------------------------
    // Early eval exit: if score is clearly decisive after
    // material+PST+pawns+threats+mobility, skip expensive
    // positional terms (space, king safety).
    // Remaining terms typically add ±150cp max.
    // -------------------------------------------------------
    if (!tactical_only) {
        int quick_phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
                        + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
        quick_phase = std::min(24, quick_phase);
        int quick_score = (mg_score * quick_phase + eg_score * (24 - quick_phase)) / 24;
        if (quick_score > 500 || quick_score < -500) {
            // Skip space, imbalance calc, king safety — position is decisive
            int sf = scale_factor(pos, eg_score);
            Score eg_scaled = eg_score * sf / 32;
            Score score = (mg_score * quick_phase + eg_scaled * (24 - quick_phase)) / 24;
            Score tempo = (15 * quick_phase + 5 * (24 - quick_phase)) / 24;
            score += (pos.side_to_move() == WHITE) ? tempo : -tempo;
            return pos.side_to_move() == WHITE ? score : -score;
        }
    }

    // -------------------------------------------------------
    // Space: control of central squares behind our pawns
    // -------------------------------------------------------
    if (!tactical_only) {
    {
        int space[2] = {0, 0};
        int pawn_count[2] = {0, 0};
        for (int c = 0; c < 2; ++c) {
            Bitboard our_pawns = pos.pieces(Color(c), PAWN);
            pawn_count[c] = popcount(our_pawns);
            Bitboard our_pawn_attacks = pawn_attacks_bb(Color(c), our_pawns);
            Bitboard their_pawn_attacks = pawn_attacks_bb(Color(c ^ 1), pos.pieces(Color(c ^ 1), PAWN));
            Bitboard space_area = (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F)
                                 & (our_pawn_attacks & ~their_pawn_attacks);
            space[c] = popcount(space_area);
        }
        if (pawn_count[WHITE] > pawn_count[BLACK]) {
            mg_score += space[WHITE] * 4 - space[BLACK] * 2;
        } else if (pawn_count[BLACK] > pawn_count[WHITE]) {
            mg_score -= space[BLACK] * 4 - space[WHITE] * 2;
        }
    }
    } // end !tactical_only space

    // -------------------------------------------------------
    // Material imbalance: bishop value increases with fewer pawns
    // Principle: in open positions (few pawns), bishops shine
    // -------------------------------------------------------
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Sign sign = (c_idx == 0) ? 1 : -1;
        int pawns = popcount(pos.pieces(c, PAWN));
        int bishops = bishop_count[c_idx];
        if (bishops >= 1) {
            mg_score += sign * (25 - pawns * 2);
            eg_score += sign * (45 - pawns * 3);
        }
    }

    // Bishop pair: two bishops cover both color complexes
    if (bishop_count[WHITE] >= 2) { mg_score += g_eval_params.bishop_pair_mg; eg_score += g_eval_params.bishop_pair_eg; }
    if (bishop_count[BLACK] >= 2) { mg_score -= g_eval_params.bishop_pair_mg; eg_score -= g_eval_params.bishop_pair_eg; }

    // Knight bonus with many pawns: knights can jump over pawn chains,
    // making them relatively stronger in closed positions (many pawns).
    // Bishop advantage with few pawns is already handled by imbalance term.
    {
        int w_pawns = popcount(pos.pieces(WHITE, PAWN));
        int b_pawns = popcount(pos.pieces(BLACK, PAWN));
        int total_pawns = w_pawns + b_pawns;
        // With 16 pawns (impossible): max bonus. With 0 pawns: no bonus.
        int knight_adj = total_pawns - 8;  // -8 to +8 range (typical: 0 to 8)
        int w_knights = popcount(pos.pieces(WHITE, KNIGHT));
        int b_knights = popcount(pos.pieces(BLACK, KNIGHT));
        if (knight_adj > 0) {
            mg_score += w_knights * knight_adj * 2;
            mg_score -= b_knights * knight_adj * 2;
            eg_score += w_knights * knight_adj;
            eg_score -= b_knights * knight_adj;
        }
    }

    // -------------------------------------------------------
    // King safety: sigmoid danger model (Hill equation)
    // Selective: skip if no enemy minor/major pieces near king zone
    // -------------------------------------------------------
    if (!tactical_only) {
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        Sign sign = (c_idx == 0) ? 1 : -1;
        Square our_ksq = ksq_arr[c_idx];

        // King zone: king attacks + one rank toward enemy
        Bitboard king_zone = king_attacks_bb(our_ksq) | square_bb(our_ksq);
        Rank fwd = (c == WHITE) ? Rank(rank_of(our_ksq) + 1) : Rank(rank_of(our_ksq) - 1);
        if (fwd >= RANK_1 && fwd <= RANK_8) {
            for (File f = FILE_A; f <= FILE_H; f = File(f + 1)) {
                king_zone |= square_bb(make_square(f, fwd));
            }
        }

        // Fast pre-check: are there enemy minor/major pieces near king?
        Bitboard enemy_pieces_near = pos.pieces(them) & king_zone;
        enemy_pieces_near &= ~(pos.pieces(them, PAWN) | pos.pieces(them, KING));
        if (!enemy_pieces_near) continue;

        int attack_units = 0;
        int attacker_count = 0;

        // Safe squares: not defended by our pawns
        Bitboard our_pawn_attacks = pawn_attacks_bb(c, pos.pieces(c, PAWN));
        Bitboard safe = ~our_pawn_attacks;

        // Pawn attacks on king zone
        Bitboard enemy_pawns = pos.pieces(them, PAWN);
        Bitboard ep = enemy_pawns;
        while (ep) {
            Square psq = pop_lsb(ep);
            if (pawn_attacks_bb(them, square_bb(psq)) & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
        }

        // Knight attacks + safe check bonus
        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            Square ksq2 = pop_lsb(enemy_kn);
            Bitboard kn_attacks = knight_attacks_bb(ksq2);
            if (kn_attacks & king_zone) {
                attack_units += 4;
                attacker_count++;
            }
            Bitboard kn_checks = kn_attacks & king_attacks_bb(our_ksq);
            if (kn_checks & safe) attack_units += 5;
        }

        // Bishop attacks + safe check bonus
        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            Square bsq = pop_lsb(enemy_bi);
            Bitboard bi_attacks = bishop_attacks_bb(bsq, occupied);
            if (bi_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard bi_checks = bi_attacks & bb_diag_attacks(our_ksq, Bitboard(0));
            if (bi_checks & safe) attack_units += 4;
        }

        // Rook attacks + safe check bonus
        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            Square rsq = pop_lsb(enemy_ro);
            Bitboard ro_attacks = rook_attacks_bb(rsq, occupied);
            if (ro_attacks & king_zone) {
                attack_units += 5;
                attacker_count++;
            }
            Bitboard ro_checks = ro_attacks & rook_attacks_bb(our_ksq, Bitboard(0));
            if (ro_checks & safe) attack_units += 6;
        }

        // Queen attacks + safe check bonus
        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            Square qsq = pop_lsb(enemy_qu);
            Bitboard qu_attacks = queen_attacks_bb(qsq, occupied);
            if (qu_attacks & king_zone) {
                attack_units += 7;
                attacker_count++;
            }
            Bitboard qu_checks = qu_attacks & (bb_diag_attacks(our_ksq, Bitboard(0)) | rook_attacks_bb(our_ksq, Bitboard(0)));
            if (qu_checks & safe) attack_units += 8;
        }

        // Only evaluate king safety if 2+ attackers
        if (attacker_count >= 2) {
            // Sigmoid danger: 500 * au^2 / (au^2 + 200)
            // Self-engineered S-curve derived from attack probability theory
            int au2 = attack_units * attack_units;
            int danger = std::min(500, (500 * au2) / (au2 + 200));

            // No queen: much less dangerous
            if (!enemy_qu) danger = danger / 4;
            // No queen and no rook: even less dangerous
            if (!enemy_qu && !enemy_ro) danger = danger / 4;

            // Shield-aware danger: missing shield pawns amplify danger
            // Chess principle: pawn shield and attack danger are multiplicative,
            // not additive. An exposed king under attack is in disproportionate peril.
            // shield_count: 0-4 (front 2 ranks × 3 files). Full shield = 4+, exposed = 0-1.
            int missing = std::max(0, 4 - shield_count[c_idx]);
            if (missing > 0) {
                danger = danger * (4 + missing * 2) / 4;
            }

            mg_score -= sign * danger;
        }
    }
    } // end !tactical_only king safety

    // -------------------------------------------------------
    // Phase calculation and score interpolation
    // -------------------------------------------------------
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    // Interpolate MG/EG with endgame scaling
    int sf = scale_factor(pos, eg_score);
    Score eg_scaled = eg_score * sf / 32;
    Score score = (mg_score * phase + eg_scaled * (24 - phase)) / 24;

    // Tempo bonus: having the move is worth ~15cp in MG, ~5cp in EG
    Score tempo = (15 * phase + 5 * (24 - phase)) / 24;
    score += (pos.side_to_move() == WHITE) ? tempo : -tempo;

    return pos.side_to_move() == WHITE ? score : -score;
}

// ============================================================
// Endgame scaling
// ============================================================
int scale_factor(const Position& pos, [[maybe_unused]] Value eg) {
    int wp = popcount(pos.pieces(WHITE, PAWN));
    int bp = popcount(pos.pieces(BLACK, PAWN));
    int total_pawns = wp + bp;

    // No pawns: harder to win without passed pawns
    if (total_pawns == 0) {
        int piece_count = popcount(pos.pieces()) - 4;
        return std::min(16 + piece_count * 2, 32);
    }

    // Opposite-colored bishops: drawish
    int wb = popcount(pos.pieces(WHITE, BISHOP));
    int bb = popcount(pos.pieces(BLACK, BISHOP));
    if (wb == 1 && bb == 1) {
        Square wb_sq = lsb(pos.pieces(WHITE, BISHOP));
        Square bb_sq = lsb(pos.pieces(BLACK, BISHOP));
        bool wb_light = ((int(wb_sq) + (wb_sq / 8)) % 2) == 0;
        bool bb_light = ((int(bb_sq) + (bb_sq / 8)) % 2) == 0;
        if (wb_light != bb_light) {
            int piece_count = popcount(pos.pieces()) - 4;
            if (piece_count > 0) {
                return std::min(18 + piece_count * 2, 32);
            } else {
                return std::min(18 + total_pawns, 32);
            }
        }
    }

    // Lone queen vs multiple minor pieces
    int w_queens = popcount(pos.pieces(WHITE, QUEEN));
    int b_queens = popcount(pos.pieces(BLACK, QUEEN));
    int w_minors = popcount(pos.pieces(WHITE, KNIGHT)) + popcount(pos.pieces(WHITE, BISHOP));
    int b_minors = popcount(pos.pieces(BLACK, KNIGHT)) + popcount(pos.pieces(BLACK, BISHOP));
    if (w_queens == 1 && b_queens == 0 && b_minors >= 2 && wp == 0) return 20;
    if (b_queens == 1 && w_queens == 0 && w_minors >= 2 && bp == 0) return 20;

    return 32;
}

// ============================================================
// Initialization: mirror PST tables for BLACK
// ============================================================
void init_evaluation() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    // BLACK PST = mirror of WHITE (flip ranks)
    for (int pt = 0; pt < 8; ++pt) {
        for (int s = 0; s < 64; ++s) {
            PST_MG_TABLE[BLACK][pt][s] = PST_MG_TABLE[WHITE][pt][s ^ 56];
            PST_EG_TABLE[BLACK][pt][s] = PST_EG_TABLE[WHITE][pt][s ^ 56];
        }
    }

    std::memset(pawn_table, 0, sizeof(pawn_table));
}

} // namespace luminex
