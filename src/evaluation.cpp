#include "luminex.h"
#include <cstring>

namespace luminex {

EvalParams g_eval_params;

// PeSTO piece values (MG and EG)
static constexpr int PieceValueMG[8] = { 82, 337, 365, 477, 1025, 0, 0, 0 };
static constexpr int PieceValueEG[8] = { 94, 281, 297, 512, 936, 0, 0, 0 };

// Ethereal-style mobility tables (MG, EG) - better tuned non-linear values
static constexpr int KnightMobilityMG[9] = { -104, -45, -22, -8, 6, 11, 19, 30, 43 };
static constexpr int KnightMobilityEG[9] = { -139, -114, -37, 3, 15, 34, 38, 37, 17 };
static constexpr int BishopMobilityMG[14] = { -99, -46, -16, -4, 6, 14, 17, 19, 19, 27, 26, 52, 55, 83 };
static constexpr int BishopMobilityEG[14] = { -186, -124, -54, -14, 1, 20, 35, 39, 49, 48, 48, 32, 47, 2 };
static constexpr int RookMobilityMG[15] = { -127, -56, -25, -12, -10, -12, -11, -4, 4, 9, 11, 19, 19, 37, 97 };
static constexpr int RookMobilityEG[15] = { -148, -127, -85, -28, 2, 27, 42, 46, 52, 55, 64, 68, 73, 60, 15 };
static constexpr int QueenMobilityMG[28] = { -111, -253, -127, -46, -18, 10, 9, 13, 7, 7, 7, 16, 21, 27, 24, 31, 32, 35, 38, 41, 35, 35, 35, 42, 44, 47, 50, 52 };
static constexpr int QueenMobilityEG[28] = { -273, -401, -228, -236, -144, -81, -41, 13, 32, 62, 79, 96, 110, 114, 115, 120, 122, 113, 101, 96, 83, 74, 62, 50, 36, 25, 18, 7 };

// SF11-style threat tables indexed by attacked piece type
// ThreatByMinor: bonus when our knight/bishop attacks enemy piece
static constexpr int ThreatByMinorMG[6] = { 0, 6, 59, 79, 90, 79 };
static constexpr int ThreatByMinorEG[6] = { 0, 32, 41, 56, 119, 161 };
// ThreatByRook: bonus when our rook attacks weak enemy piece
static constexpr int ThreatByRookMG[6] = { 0, 3, 38, 38, 0, 51 };
static constexpr int ThreatByRookEG[6] = { 0, 44, 71, 61, 38, 38 };

// Far piece penalty: pieces more than 3 squares from own king (only rook/queen remain hardcoded)
static constexpr int FarRookMG = -10, FarRookEG = 5;
static constexpr int FarQueenMG = -8, FarQueenEG = 15;

// PeSTO piece-square tables (WHITE, a1=0 layout)
// Transposed from original a8=0 layout by reversing row order.
// Verified against chessprogramming.org/PeSTO's_Evaluation_Function
Score PST_MG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN
        {
              0,   0,   0,   0,   0,   0,   0,   0,
            -35,  -1, -20, -23, -15,  24,  38, -22,
            -26,  -4,  -4, -10,   3,   3,  33, -12,
            -27,  -2,  -5,  12,  17,   6,  10, -25,
            -14,  13,   6,  21,  23,  12,  17, -23,
             -6,   7,  26,  31,  65,  56,  25, -20,
             98, 134,  61,  95,  68, 126,  34, -11,
              0,   0,   0,   0,   0,   0,   0,   0
        },
        // KNIGHT
        {
            -105, -21, -58, -33, -17, -28, -19, -23,
             -29, -53, -12,  -3,  -1,  18, -14, -19,
             -23,  -9,  12,  10,  19,  17,  25, -16,
             -13,   4,  16,  13,  28,  19,  21,  -8,
              -9,  17,  19,  53,  37,  69,  18,  22,
             -47,  60,  37,  65,  84, 129,  73,  44,
             -73, -41,  72,  36,  23,  62,   7, -17,
            -167, -89, -34, -49,  61, -97, -15,-107
        },
        // BISHOP
        {
            -33,  -3, -14, -21, -13, -12, -39, -21,
              4,  15,  16,   0,   7,  21,  33,   1,
              0,  15,  15,  15,  14,  27,  18,  10,
             -6,  13,  13,  26,  34,  12,  10,   4,
             -4,   5,  19,  50,  37,  37,   7,  -2,
            -16,  37,  43,  40,  35,  50,  37,  -2,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -29,   4, -82, -37, -25, -42,   7,  -8
        },
        // ROOK
        {
            -26, -71, -19, -13,   1,  17,  16,   7,
            -44, -16, -20,  -9,  -1,  11,  -6, -37,
            -45, -25, -16, -17,   3,   0,  -5, -33,
            -36, -26, -12,  -1,   9,  -7,   6, -23,
            -24, -11,   7,  26,  24,  35,  -8, -20,
             -5,  19,  26,  36,  17,  45,  61,  16,
             27,  32,  58,  62,  80,  67,  26,  44,
             32,  42,  32,  51,  63,   9,  31,  43
        },
        // QUEEN
        {
             -1, -18,  -9,  10, -15, -25, -31, -50,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
             -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -28,   0,  29,  12,  59,  44,  43,  45
        },
        // KING (MG - safety-oriented)
        {
            -15,  36,  12, -54,   8, -28,  24,  14,
              1,   7,  -8, -64, -43, -16,   9,   8,
            -14, -14, -22, -46, -44, -30, -15, -27,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -17, -20, -12, -27, -30, -25, -14, -36,
             -9,  24,   2, -16, -20,   6,  22, -22,
             29,  -1, -20,  -7,  -8,  -4, -38, -29,
            -65,  23,  16, -15, -56, -34,   2,  13
        },
        // NONE
        {0},
        // ALL
        {0}
    },
    // BLACK (will be populated by init_evaluation via mirroring)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

Score PST_EG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN
        {
              0,   0,   0,   0,   0,   0,   0,   0,
             13,   8,   8,  10,  13,   0,   2,  -7,
              4,   7,  -6,   1,   0,  -5,  -1,  -8,
             13,   9,  -3,  -7,  -7,  -8,   3,  -1,
             32,  24,  13,   5,  -2,   4,  17,  17,
             94, 100,  85,  67,  56,  53,  82,  84,
            178, 173, 158, 134, 147, 132, 165, 187,
              0,   0,   0,   0,   0,   0,   0,   0
        },
        // KNIGHT
        {
            -29, -51, -23, -15, -22, -18, -50, -64,
            -42, -20, -10,  -5,  -2, -20, -23, -44,
            -23,  -3,  -1,  15,  10,  -3, -20, -22,
            -18,  -6,  16,  25,  16,  17,   4, -18,
            -17,   3,  22,  22,  22,  11,   8, -18,
            -24, -20,  10,   9,  -1,  -9, -19, -41,
            -25,  -8, -25,  -2,  -9, -25, -24, -52,
            -58, -38, -13, -28, -31, -27, -63, -99
        },
        // BISHOP
        {
            -27, -23,  -9, -23,  -5,  -9, -16,  -5,
            -14, -18,  -7,  -1,   4,  -9, -15, -27,
            -12,  -3,   8,  10,  13,   3,  -7, -15,
             -6,   3,  13,  19,   7,  10,  -3,  -9,
             -3,   9,  12,   9,  14,  10,   3,   2,
              2,  -8,   0,  -1,  -2,   6,   0,   4,
             -8,  -4,   7, -12,  -3, -13,  -4, -14,
            -14, -21, -11,  -8,  -7,  -9, -17, -24
        },
        // ROOK
        {
             -9,   2,   3,  -1,  -5, -13,   4, -20,
             -6,  -6,   0,   2,  -9,  -9, -11,  -3,
             -4,   0,  -5,  -1,  -7, -12,  -8, -16,
              3,   5,   8,   4,  -5,  -6,  -8, -11,
              4,   3,  13,   1,   2,   1,  -1,   2,
              7,   7,   7,   5,   4,  -3,  -5,  -3,
             11,  13,  13,  11,  -3,   3,   8,   3,
             13,  10,  18,  15,  12,  12,   8,   5
        },
        // QUEEN
        {
            -33, -28, -22, -43,  -5, -32, -20, -41,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -16, -27,  15,   6,   9,  17,  10,   5,
            -18,  28,  19,  47,  31,  34,  39,  23,
              3,  22,  24,  45,  57,  40,  57,  36,
            -20,   6,   9,  49,  47,  35,  19,   9,
            -17,  20,  32,  41,  58,  25,  30,   0,
             -9,  22,  22,  27,  27,  19,  10,  20
        },
        // KING (EG - center-oriented)
        {
            -53, -34, -21, -11, -28, -14, -24, -43,
            -27, -11,   4,  13,  14,   4,  -5, -17,
            -19,  -3,  11,  21,  23,  16,   7,  -9,
            -18,  -4,  21,  24,  27,  23,   9, -11,
             -8,  22,  24,  27,  26,  33,  26,   3,
             10,  17,  23,  15,  20,  45,  44,  13,
            -12,  17,  14,  17,  17,  38,  23,  11,
            -74, -35, -18, -18, -11,  15,   4, -17
        },
        // NONE
        {0},
        // ALL
        {0}
    },
    // BLACK (will be populated by init_evaluation via mirroring)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

using Score = Value;

// ============================================================
// Pawn hash table
// ============================================================
struct PawnEntry {
    uint64_t key;
    int32_t mg, eg;
};
static constexpr int PAWN_HASH_SIZE = 16384;
static PawnEntry pawn_table[PAWN_HASH_SIZE];

// Evaluate pawn-only terms and cache the result in pawn_table.
// Returns true on cache hit, false on cache miss (but fills mg_out/eg_out either way).
static void evaluate_pawns(const Position& pos, int32_t& mg_out, int32_t& eg_out) {
    // Use incrementally maintained pawn key
    uint64_t pawn_key = pos.pawn_key();

    int idx = int(pawn_key & uint64_t(PAWN_HASH_SIZE - 1));
    PawnEntry& entry = pawn_table[idx];
    if (entry.key == pawn_key) {
        mg_out = entry.mg;
        eg_out = entry.eg;
        return;
    }

    // Cache miss: compute pawn-only eval from scratch
    int32_t mg = 0;
    int32_t eg = 0;

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        Sign sign = (c == WHITE) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        // Pre-compute file counts for pawn structure
        int file_count[8] = {0};
        Bitboard tmp = our_pawns;
        while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }

        // Adjacent file pawn support lookup (for connected pawns)
        Bitboard supported_by_adj = pawn_attacks_bb(c, our_pawns);

        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            mg += sign * (PieceValueMG[PAWN] + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg += sign * (PieceValueEG[PAWN] + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);

            // Doubled pawn penalty
            if (file_count[f] > 1) {
                mg -= sign * 11;
                eg -= sign * 22;
            }

            // Isolated pawn penalty
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg -= sign * 14;
                eg -= sign * 18;
            }

            // Connected pawn bonus: pawn protected by another pawn
            if (square_bb(sq) & supported_by_adj) {
                int connected_bonus = 4 + r * 3;
                mg += sign * connected_bonus;
                eg += sign * (2 + r * 2);
            }

            // Pawn phalanx: side-by-side pawns on same rank (Weiss-style)
            {
                Bitboard neighbors = our_pawns & (file_bb(File(f - 1)) | file_bb(File(f + 1)));
                Bitboard same_rank = neighbors & rank_bb(rank_of(sq));
                if (same_rank) {
                    // Phalanx bonuses by rank (from Weiss)
                    static constexpr int PhalanxMG[8] = { 0, 10, 20, 35, 72, 231, 168, 0 };
                    static constexpr int PhalanxEG[8] = { 0, -2, 14, 37, 121, 367, 394, 0 };
                    if (r >= RANK_2 && r <= RANK_7) {
                        mg += sign * PhalanxMG[r];
                        eg += sign * PhalanxEG[r];
                    }
                }
            }

            // Pawn lever bonus: can capture an adjacent enemy pawn
            {
                Bitboard lever_targets = pawn_attacks_bb(c, sq) & their_pawns;
                if (lever_targets) {
                    int lever_bonus = 3 + r * 2;
                    mg += sign * lever_bonus;
                    // Extra for central levers
                    if (f >= FILE_C && f <= FILE_F) {
                        mg += sign * 3;
                    }
                }
            }

            // Backward pawn detection
            {
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                if (push < SQUARE_NONE && r >= RANK_2 && r <= RANK_5) {
                    bool push_attacked = (pawn_attacks_bb(them, their_pawns) & square_bb(push)) != 0;
                    if (push_attacked) {
                        Bitboard our_pawn_attacks = pawn_attacks_bb(c, our_pawns);
                        bool defended = (our_pawn_attacks & square_bb(push)) != 0;
                        if (!defended) {
                            mg -= sign * 10;
                            eg -= sign * 12;
                        }
                    }
                }
            }

            // Passed pawn detection + BASE bonus (table values only, no rook/king terms)
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            // Candidate passed pawn
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
                    static constexpr int CandMG[2][8] = {
                        { 0, -8, -10, -12, -15, 15, 0, 0 },
                        { 0, -10, -5,   2,   8, 30, 0, 0 }
                    };
                    static constexpr int CandEG[2][8] = {
                        { 0,   5,  12,  25,  50, 70, 0, 0 },
                        { 0,  12,  25,  50,  85,120, 0, 0 }
                    };
                    mg += sign * CandMG[supported ? 1 : 0][r];
                    eg += sign * CandEG[supported ? 1 : 0][r];
                }
            }

            // Base passed pawn bonus (PassedMG/PassedEG tables only)
            if (!(ahead & their_pawns)) {
                static constexpr int PassedMG[8] = { 0, -10, -14, -27, 16, 67, 107, 0 };
                static constexpr int PassedEG[8] = { 0,   7,  15,  41, 99,189, 350, 0 };
                mg += sign * PassedMG[r];
                eg += sign * PassedEG[r];
            }
        }
    }

    // Store in pawn hash table
    entry.key = pawn_key;
    entry.mg = mg;
    entry.eg = eg;

    mg_out = mg;
    eg_out = eg;
}

Value evaluate(const Position& pos) {
    // KXK endgame: one side has only a king
    for (int strong = 0; strong < 2; ++strong) {
        Color c = Color(strong);
        Color weak = Color(strong ^ 1);
        int strong_material = popcount(pos.pieces(c)) - 1; // exclude king
        int weak_material = popcount(pos.pieces(weak)) - 1; // exclude king
        if (strong_material >= 1 && weak_material == 0) {
            // Check if strong side has mating material
            bool has_major = (pos.pieces(c, QUEEN) || pos.pieces(c, ROOK));
            bool has_bn = (pos.pieces(c, BISHOP) && pos.pieces(c, KNIGHT));
            bool has_two_bishops = (popcount(pos.pieces(c, BISHOP)) >= 2);
            bool has_three_knights = (popcount(pos.pieces(c, KNIGHT)) >= 3);
            if (has_major || has_bn || has_two_bishops || has_three_knights) {
                Square weak_ksq = pos.king_sq(weak);
                Square strong_ksq = pos.king_sq(c);

                // Push weak king to corner: bonus based on distance from center
                int wfile = file_of(weak_ksq);
                int wrank = rank_of(weak_ksq);
                int center_dist_file = std::max(wfile - 3, 7 - wfile);
                int center_dist_rank = std::max(wrank - 3, 7 - wrank);

                // For KBN vs K: drive to corner matching bishop's color
                int corner_bonus;
                if (has_bn && strong_material == 2) {
                    // KBN vs K: find bishop, check its square color
                    Square bsq = lsb(pos.pieces(c, BISHOP));
                    bool bishop_light = ((int(bsq) + int(bsq) / 8) % 2) == 0;
                    bool weak_light = ((int(weak_ksq) + int(weak_ksq) / 8) % 2) == 0;

                    // Push to corner of same color as bishop
                    // Corners: a1(light), h1(dark), a8(dark), h8(light)
                    // If bishop on light squares, target a1/h8; if dark, target a8/h1
                    if (bishop_light == weak_light) {
                        // Weak king already on right color - push to corner
                        corner_bonus = center_dist_file * center_dist_file + center_dist_rank * center_dist_rank;
                    } else {
                        // Wrong color - bonus for moving toward correct color first
                        corner_bonus = (center_dist_file * center_dist_file + center_dist_rank * center_dist_rank) / 2;
                    }
                } else {
                    corner_bonus = center_dist_file * center_dist_file + center_dist_rank * center_dist_rank;
                }

                // Keep kings close: bonus for strong king being near weak king
                int king_dist = std::abs(file_of(strong_ksq) - wfile) + std::abs(rank_of(strong_ksq) - wrank);
                int close_bonus = 70 - king_dist * 10;

                int score = (50 - corner_bonus) * 2 + close_bonus;
                score += VALUE_KNOWN_WIN; // Add known win bonus to ensure engine doesn't draw

                // Add value of winning material
                score += popcount(pos.pieces(c, QUEEN)) * PieceValueEG[QUEEN];
                score += popcount(pos.pieces(c, ROOK)) * PieceValueEG[ROOK];
                score += popcount(pos.pieces(c, BISHOP)) * PieceValueEG[BISHOP];
                score += popcount(pos.pieces(c, KNIGHT)) * PieceValueEG[KNIGHT];

                return (strong == WHITE) ? score : -score;
            }
            // Non-mating material (e.g., lone K+B vs K) - return 0 (draw)
            // Add a tiny score based on material difference so engine doesn't trade down to nothing
            int mat_diff = strong_material * 50;
            return (strong == WHITE) ? mat_diff : -mat_diff;
        }
    }

    Score mg_score = 0;
    Score eg_score = 0;

    Square ksq_arr[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};

    Bitboard occupied = pos.pieces();

    // Per-piece-type attack maps for threat evaluation and king safety
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
        Bitboard mob_area = ~pawn_attacks_bb(them, their_pawns);

        // Non-pawn passed pawn bonuses (rook behind, enemy rook, king proximity, blocked by pieces)
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

                // Rook behind passed pawn bonus
                Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
                if (behind) {
                    Square rsq2 = lsb(behind);
                    bool rook_behind = (c == WHITE) ? (rank_of(rsq2) < rank_of(sq)) : (rank_of(rsq2) > rank_of(sq));
                    if (rook_behind) {
                        mg_passer += 18;
                        eg_passer += 25;
                    }
                }

                // Enemy rook behind our passed pawn — reduces bonus
                Bitboard enemy_behind = file_bb(f) & pos.pieces(them, ROOK);
                if (enemy_behind) {
                    Square ersq = lsb(enemy_behind);
                    bool enemy_rook_behind = (c == WHITE) ? (rank_of(ersq) < rank_of(sq)) : (rank_of(ersq) > rank_of(sq));
                    if (enemy_rook_behind) {
                        mg_passer -= 8;
                        eg_passer -= 12;
                    }
                }

                // King proximity (EG only)
                Square our_ksq = ksq_arr[c_idx];
                Square their_ksq = ksq_arr[c_idx ^ 1];
                Square promo_sq = relative_square(c, make_square(f, RANK_8));
                int our_kdist = distance(our_ksq, promo_sq);
                int their_kdist = distance(their_ksq, promo_sq);
                eg_passer += (their_kdist - our_kdist) * 8;

                // Free passed pawn bonus: enemy king can't catch
                // Bonus scales with how far the enemy king is
                if (their_kdist > our_kdist + (c == pos.side_to_move() ? 0 : 1)) {
                    int unreachable = their_kdist - our_kdist;
                    eg_passer += unreachable * (unreachable > 4 ? 25 : 15);
                }

                // Blocked by enemy pieces penalty
                if (ahead_file & pos.pieces(them)) {
                    mg_passer -= 12;
                    eg_passer -= 18;
                }

                mg_score += sign * mg_passer;
                eg_score += sign * eg_passer;
            }
        }

        // Knights
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * (PieceValueMG[KNIGHT] + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (PieceValueEG[KNIGHT] + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);

            Bitboard attacks = knight_attacks_bb(sq);
            attacks_by[c][KNIGHT] |= attacks;
            all_attacks[c] |= attacks;
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 8);
            mg_score += sign * KnightMobilityMG[mob];
            eg_score += sign * KnightMobilityEG[mob];

            // Outpost knight: on rank 4-6, protected by own pawn, cannot be attacked by enemy pawn
            Rank kr = relative_rank(c, sq);
            if (kr >= RANK_4 && kr <= RANK_6) {
                if (pawn_attacks_bb(c, our_pawns) & square_bb(sq)) {
                    Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
                    bool can_be_attacked = (enemy_pawn_attacks & square_bb(sq)) != 0;
                    if (!can_be_attacked) {
                        mg_score += sign * (g_eval_params.outpost_knight_mg + (kr - 3) * 5);
                        eg_score += sign * (g_eval_params.outpost_knight_eg + (kr - 3) * 3);
                    }
                }
            }

            // Far knight penalty: knight far from own king
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score -= sign * g_eval_params.far_knight_mg;
                eg_score -= sign * g_eval_params.far_knight_eg;
            }

            // Knight shielded: friendly pawn above knight (Stash: 4/23)
            Square shield_sq = (c == WHITE)
                ? make_square(file_of(sq), Rank(rank_of(sq) + 1))
                : make_square(file_of(sq), Rank(rank_of(sq) - 1));
            if (shield_sq < SQUARE_NONE && (our_pawns & square_bb(shield_sq))) {
                mg_score += sign * 4;
                eg_score += sign * 23;
            }
        }

        // Bishops
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * (PieceValueMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PieceValueEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            Bitboard attacks = bb_diag_attacks(sq, occupied);
            attacks_by[c][BISHOP] |= attacks;
            all_attacks[c] |= attacks;
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 13);
            mg_score += sign * BishopMobilityMG[mob];
            eg_score += sign * BishopMobilityEG[mob];

            // Bishop shielded: friendly pawn above bishop (Stash: 1/3)
            {
                Square shield_sq = (c == WHITE)
                    ? make_square(file_of(sq), Rank(rank_of(sq) + 1))
                    : make_square(file_of(sq), Rank(rank_of(sq) - 1));
                if (shield_sq < SQUARE_NONE && (our_pawns & square_bb(shield_sq))) {
                    mg_score += sign * 1;
                    eg_score += sign * 3;
                }
            }

            // Bishop pawns same color penalty (Stash: BishopPawnsSameColor)
            {
                static constexpr Bitboard BB_DARK_SQ = 0x55AA55AA55AA55AAULL;
                static constexpr Bitboard BB_LIGHT_SQ = 0xAA55AA55AA55AA55ULL;
                bool is_dark = ((int(sq) + rank_of(sq)) % 2) == 0;
                Bitboard same_color_bb = is_dark ? BB_DARK_SQ : BB_LIGHT_SQ;
                int same_color_pawns = popcount(our_pawns & same_color_bb);
                same_color_pawns = std::min(same_color_pawns, 6);
                // Stash values: 15/36, 15/26, 13/17, 9/11, 6/3, 3/-2, -3/-10
                static constexpr int BscMG[7] = { 15, 15, 13, 9, 6, 3, -3 };
                static constexpr int BscEG[7] = { 36, 26, 17, 11, 3, -2, -10 };
                mg_score += sign * BscMG[same_color_pawns];
                eg_score += sign * BscEG[same_color_pawns];
            }

            // Bishop long diagonal bonus (Stash: BishopLongDiagonal 13/22)
            // Check if bishop sees 2+ center squares (d4,d5,e4,e5)
            if (popcount(attacks & BB_CENTER) >= 2) {
                mg_score += sign * 13;
                eg_score += sign * 22;
            }

            // Bishop outpost (Stash: BishopOutpost 47/24)
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
        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (PieceValueMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PieceValueEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);

            // Rook mobility: exclude own rooks and queens from occupancy (Stash)
            Bitboard attacks = rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN));
            attacks_by[c][ROOK] |= attacks;
            all_attacks[c] |= attacks;
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 14);
            mg_score += sign * RookMobilityMG[mob];
            eg_score += sign * RookMobilityEG[mob];

            // Open/semi-open file
            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * g_eval_params.rook_open_mg;
                eg_score += sign * g_eval_params.rook_open_eg;
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * g_eval_params.rook_semi_open_mg;
                eg_score += sign * g_eval_params.rook_semi_open_eg;
            }

            // Rook on 7th rank bonus
            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * g_eval_params.rook_7th_mg;
                eg_score += sign * g_eval_params.rook_7th_eg;
            }

            // Rook xray queen: bonus for rook on same file as enemy queen (Stash: 15/4)
            if ((file_bb(f) & pos.pieces(them, QUEEN))) {
                mg_score += sign * 15;
                eg_score += sign * 4;
            }

            // Far rook penalty
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FarRookMG;
                eg_score += sign * FarRookEG;
            }

            // Rook trapped/buried (Stash: -8/-16 trapped, -69/-33 buried)
            if (rr <= RANK_2) {
                Square ksq = ksq_arr[c_idx];
                int fdist = std::abs(file_of(sq) - file_of(ksq));
                bool king_between = (c == WHITE)
                    ? (rank_of(ksq) <= rank_of(sq) && fdist <= 1)
                    : (rank_of(ksq) >= rank_of(sq) && fdist <= 1);
                if (king_between && mob <= 4) {
                    bool has_castling = pos.castling_allowed(c, WHITE_KINGSIDE) || pos.castling_allowed(c, WHITE_QUEENSIDE);
                    if (has_castling) {
                        mg_score += sign * (-8);
                        eg_score += sign * (-16);
                    } else {
                        mg_score += sign * (-69);
                        eg_score += sign * (-33);
                    }
                }
            }

            // Rook on blocked file: friendly pawn ahead on same file (Stash: -8/-8)
            {
                Bitboard ahead_pawns;
                if (c == WHITE) {
                    // White: pawns on ranks above the rook
                    ahead_pawns = our_pawns & file_bb(f) & ~rank_bb(Rank(rank_of(sq)));
                } else {
                    // Black: pawns on ranks below the rook
                    Bitboard rook_rank = rank_bb(Rank(rank_of(sq)));
                    ahead_pawns = our_pawns & file_bb(f) & (rook_rank - 1);
                }
                if (ahead_pawns) {
                    mg_score += sign * (-8);
                    eg_score += sign * (-8);
                }
            }
        }

        // Queens
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            mg_score += sign * (PieceValueMG[QUEEN] + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (PieceValueEG[QUEEN] + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);

            // Queen mobility: exclude own bishop from diag occupancy, own rook from rook occupancy (Stash)
            Bitboard attacks = bb_diag_attacks(sq, occupied ^ pos.pieces(c, BISHOP))
                             | rook_attacks_bb(sq, occupied ^ pos.pieces(c, ROOK));
            attacks_by[c][QUEEN] |= attacks;
            all_attacks[c] |= attacks;
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 27);
            mg_score += sign * QueenMobilityMG[mob];
            eg_score += sign * QueenMobilityEG[mob];

            // Far queen penalty
            if (distance(sq, ksq_arr[c_idx]) > 3) {
                mg_score += sign * FarQueenMG;
                eg_score += sign * FarQueenEG;
            }
        }

        // King PST
        Square ksq = ksq_arr[c_idx];
        mg_score += sign * (PieceValueMG[KING] + PST_MG_TABLE[int(c)][int(KING)][int(ksq)]);
        eg_score += sign * (PieceValueEG[KING] + PST_EG_TABLE[int(c)][int(KING)][int(ksq)]);

        // King pawn shield (MG only)
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);
        bool on_back = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);
        if (on_back) {
            // Per-rank shield evaluation: check 3 ranks in front of king
            for (int r = 1; r <= 3; ++r) {
                int weight = (r == 1) ? g_eval_params.pawn_shield_knight : (r == 2) ? g_eval_params.pawn_shield_center : g_eval_params.pawn_shield_rook;
                for (int df = -1; df <= 1; ++df) {
                    File sf = File(int(kfile) + df);
                    if (sf < FILE_A || sf > FILE_H) continue;
                    Square shield_sq = relative_square(c, make_square(sf, Rank(r)));
                    if (our_pawns & square_bb(shield_sq)) {
                        mg_score += sign * weight;
                    }
                }
            }

            Bitboard all_pawns = pos.pieces(PAWN);
            if (!(all_pawns & file_bb(kfile))) mg_score -= sign * g_eval_params.open_file_penalty_mg;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) mg_score -= sign * g_eval_params.open_file_penalty_eg;
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) mg_score -= sign * g_eval_params.open_file_penalty_eg;

            // Pawn storm: enemy pawns advancing toward our castled king
            {
                // Check files near the king for enemy storm pawns
                for (int df = -1; df <= 1; ++df) {
                    File storm_file = File(int(kfile) + df);
                    if (storm_file < FILE_A || storm_file > FILE_H) continue;
                    Bitboard enemy_file_pawns = their_pawns & file_bb(storm_file);
                    while (enemy_file_pawns) {
                        Square psq = pop_lsb(enemy_file_pawns);
                        Rank pr = relative_rank(c, rank_of(psq));
                        // Enemy pawn advancing toward our king (higher relative rank = closer)
                        if (pr >= RANK_4) {
                            int storm_danger = (pr - 3) * g_eval_params.pawn_storm;
                            mg_score -= sign * storm_danger;
                        }
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

        // SF11-style threat evaluation
        {
            Bitboard nonPawnEnemies = pos.pieces(them) & ~pos.pieces(them, PAWN);

            // Strongly protected: defended by enemy pawn, or defended twice where we don't
            Bitboard stronglyProtected = attacks_by[them][PAWN]
                                      | (all_attacks[them] & ~all_attacks[c]);
            // Weak enemy pieces: attacked by us but not strongly protected
            Bitboard weak = nonPawnEnemies & ~stronglyProtected & all_attacks[c];

            // ThreatByMinor: knight/bishop threats on defended+weak pieces (SF11 values)
            if (weak | nonPawnEnemies) {
                Bitboard b = (weak | nonPawnEnemies) & (attacks_by[c][KNIGHT] | attacks_by[c][BISHOP]);
                b &= nonPawnEnemies & all_attacks[c];
                while (b) {
                    PieceType pt = piece_type_of(pos.piece_on(pop_lsb(b)));
                    if (pt <= QUEEN) {
                        mg_score += sign * ThreatByMinorMG[pt];
                        eg_score += sign * ThreatByMinorEG[pt];
                    }
                }
            }

            // ThreatByRook: rook threats on weak pieces only (SF11 values)
            {
                Bitboard b = weak & attacks_by[c][ROOK];
                while (b) {
                    PieceType pt = piece_type_of(pos.piece_on(pop_lsb(b)));
                    if (pt <= QUEEN) {
                        mg_score += sign * ThreatByRookMG[pt];
                        eg_score += sign * ThreatByRookEG[pt];
                    }
                }
            }

            // ThreatByKing: king attacking weak pieces
            if (weak & attacks_by[c][KING]) {
                mg_score += sign * 24;
                eg_score += sign * 89;
            }

            // Hanging pieces: weak pieces not defended at all, or doubly attacked by us
            {
                Bitboard b = weak & (~all_attacks[them] | all_attacks[c]);
                mg_score += sign * 69 * popcount(b);
                eg_score += sign * 36 * popcount(b);
            }

            // RestrictedPiece: enemy pieces attacked by us, not strongly protected
            {
                Bitboard b = pos.pieces(them) & ~stronglyProtected & all_attacks[c];
                mg_score += sign * 7 * popcount(b);
                eg_score += sign * 7 * popcount(b);
            }

            // ThreatBySafePawn: our pawns on safe squares attacking non-pawn enemies
            {
                Bitboard safe = ~all_attacks[them] | all_attacks[c];
                Bitboard safe_pawns = pos.pieces(c, PAWN) & safe;
                Bitboard b = pawn_attacks_bb(c, safe_pawns) & nonPawnEnemies;
                mg_score += sign * 173 * popcount(b);
                eg_score += sign * 94 * popcount(b);
            }

            // ThreatByPawnPush: pawns that can push to threaten non-pawn enemies
            {
                Bitboard TRank3BB = (c == WHITE) ? BB_RANK_3 : BB_RANK_6;
                Bitboard push_one = (c == WHITE) ? shift_n(pos.pieces(c, PAWN)) : shift_s(pos.pieces(c, PAWN));
                Bitboard b = push_one & ~pos.pieces();
                Bitboard dbl = (b & TRank3BB);
                dbl = (c == WHITE) ? shift_n(dbl) : shift_s(dbl);
                b |= dbl & ~pos.pieces();
                b &= ~attacks_by[them][PAWN];
                Bitboard safe = ~all_attacks[them] | all_attacks[c];
                b &= safe;
                b = pawn_attacks_bb(c, b) & nonPawnEnemies;
                mg_score += sign * 48 * popcount(b);
                eg_score += sign * 39 * popcount(b);
            }

            // Threats on enemy queen: KnightOnQueen + SliderOnQueen
            if (pos.pieces(them, QUEEN)) {
                Square qsq = lsb(pos.pieces(them, QUEEN));
                Bitboard safe = ~stronglyProtected & ~pos.pieces(c);
                Bitboard kn_on_q = attacks_by[c][KNIGHT] & knight_attacks_bb(qsq) & safe;
                mg_score += sign * 16 * popcount(kn_on_q);
                eg_score += sign * 12 * popcount(kn_on_q);
                Bitboard slider_on_q = (attacks_by[c][BISHOP] & bishop_attacks_bb(qsq, occupied))
                                     | (attacks_by[c][ROOK] & rook_attacks_bb(qsq, occupied));
                slider_on_q &= all_attacks[c] & safe;
                mg_score += sign * 59 * popcount(slider_on_q);
                eg_score += sign * 18 * popcount(slider_on_q);
            }

            // Hanging pawns: enemy pawns not defended by any enemy piece
            Bitboard hanging_pawns = pos.pieces(them, PAWN) & ~all_attacks[them] & all_attacks[c];
            if (hanging_pawns) {
                mg_score += sign * g_eval_params.hanging_pawn_mg * popcount(hanging_pawns);
                eg_score += sign * g_eval_params.hanging_pawn_eg * popcount(hanging_pawns);
            }
        }

    }

    // Space evaluation: count squares behind our pawns that are safe (not attacked by enemy pawns)
    // Safe squares = squares attacked by our pawns but NOT by enemy pawns
    {
        int space[2] = {0, 0};
        int pawn_count[2] = {0, 0};
        for (int c = 0; c < 2; ++c) {
            Bitboard our_pawns = pos.pieces(Color(c), PAWN);
            pawn_count[c] = popcount(our_pawns);
            // Our pawn attacks minus enemy pawn attacks = our territory
            Bitboard our_pawn_attacks = pawn_attacks_bb(Color(c), our_pawns);
            Bitboard their_pawn_attacks = pawn_attacks_bb(Color(c ^ 1), pos.pieces(Color(c ^ 1), PAWN));
            // Only count space in the center four files (C through F) and between ranks 2-7
            Bitboard space_area = (BB_FILE_C | BB_FILE_D | BB_FILE_E | BB_FILE_F)
                                 & (our_pawn_attacks & ~their_pawn_attacks);
            space[c] = popcount(space_area);
        }
        // Only apply space bonus if we have more pawns (otherwise pushing creates weaknesses)
        if (pawn_count[WHITE] > pawn_count[BLACK]) {
            mg_score += space[WHITE] * 4 - space[BLACK] * 2;
        } else if (pawn_count[BLACK] > pawn_count[WHITE]) {
            mg_score -= space[BLACK] * 4 - space[WHITE] * 2;
        }
    }

    // Material imbalance: bishop value increases with fewer pawns (endgame piece)
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Sign sign = (c_idx == 0) ? 1 : -1;
        int pawns = popcount(pos.pieces(c, PAWN));
        int bishops = bishop_count[c_idx];
        if (bishops >= 1) {
            mg_score += sign * (30 - pawns * 2);
            eg_score += sign * (50 - pawns * 3);
        }
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) { mg_score += g_eval_params.bishop_pair_mg; eg_score += g_eval_params.bishop_pair_eg; }
    if (bishop_count[BLACK] >= 2) { mg_score -= g_eval_params.bishop_pair_mg; eg_score -= g_eval_params.bishop_pair_eg; }

    // King safety using SafetyTable (Stash-style linear formula, capped at 200)
    static const int SafetyTable[100] = {
        0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 39,
        44, 50, 56, 62, 68, 75, 82, 85, 89, 97, 105, 113, 122, 131, 140,
        150, 160, 168, 174, 178, 182, 185, 188, 190, 192, 194, 195, 196,
        197, 198, 198, 199, 199, 199, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
        200, 200
    };

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
                attack_units += 3;
                attacker_count++;
            }
        }

        // Knight attacks on king zone + safe check bonus
        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            Square ksq2 = pop_lsb(enemy_kn);
            Bitboard kn_attacks = knight_attacks_bb(ksq2);
            if (kn_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard kn_checks = kn_attacks & king_attacks_bb(our_ksq);
            if (kn_checks & safe) attack_units += 4;
        }

        // Bishop attacks on king zone + safe check bonus
        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            Square bsq = pop_lsb(enemy_bi);
            Bitboard bi_attacks = bishop_attacks_bb(bsq, occupied);
            if (bi_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard bi_checks = bi_attacks & bb_diag_attacks(our_ksq, Bitboard(0));
            if (bi_checks & safe) attack_units += 3;
        }

        // Rook attacks on king zone + safe check bonus
        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            Square rsq = pop_lsb(enemy_ro);
            Bitboard ro_attacks = rook_attacks_bb(rsq, occupied);
            if (ro_attacks & king_zone) {
                attack_units += 4;
                attacker_count++;
            }
            Bitboard ro_checks = ro_attacks & rook_attacks_bb(our_ksq, Bitboard(0));
            if (ro_checks & safe) attack_units += 5;
        }

        // Queen attacks on king zone + safe check bonus
        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            Square qsq = pop_lsb(enemy_qu);
            Bitboard qu_attacks = queen_attacks_bb(qsq, occupied);
            if (qu_attacks & king_zone) {
                attack_units += 6;
                attacker_count++;
            }
            Bitboard qu_checks = qu_attacks & (bb_diag_attacks(our_ksq, Bitboard(0)) | rook_attacks_bb(our_ksq, Bitboard(0)));
            if (qu_checks & safe) attack_units += 6;
        }

        // Only evaluate king safety if we have enough attackers
        if (attacker_count >= 2) {
            int idx = std::min(attack_units, 99);
            int danger = SafetyTable[idx];

            if (!enemy_qu) danger = danger / 4;
            if (!enemy_qu && !enemy_ro) danger = danger / 4;

            mg_score -= sign * danger;
        }
    }

    // Phase calculation
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    // Interpolate MG/EG with endgame scaling
    int sf = scale_factor(pos, eg_score);
    Score eg_scaled = eg_score * sf / 32;
    Score score = (mg_score * phase + eg_scaled * (24 - phase)) / 24;

    // Tempo bonus
    Score tempo = (15 * phase + 5 * (24 - phase)) / 24;
    score += (pos.side_to_move() == WHITE) ? tempo : -tempo;

    return pos.side_to_move() == WHITE ? score : -score;
}

int scale_factor(const Position& pos, [[maybe_unused]] Value eg) {
    // Scale endgame scores based on material configuration
    int wp = popcount(pos.pieces(WHITE, PAWN));
    int bp = popcount(pos.pieces(BLACK, PAWN));
    int total_pawns = wp + bp;

    // No pawns endgame: harder to win
    if (total_pawns == 0) {
        int piece_count = popcount(pos.pieces()) - 4; // exclude kings
        return std::min(16 + piece_count * 2, 32);
    }

    // Opposite-colored bishops: drawish endgames
    int wb = popcount(pos.pieces(WHITE, BISHOP));
    int bb = popcount(pos.pieces(BLACK, BISHOP));
    if (wb == 1 && bb == 1) {
        // Check if bishops are on opposite colors
        Square wb_sq = lsb(pos.pieces(WHITE, BISHOP));
        Square bb_sq = lsb(pos.pieces(BLACK, BISHOP));
        bool wb_light = ((int(wb_sq) + (wb_sq / 8)) % 2) == 0;
        bool bb_light = ((int(bb_sq) + (bb_sq / 8)) % 2) == 0;
        if (wb_light != bb_light) {
            // OCB scaling: more drawish with fewer pieces
            int piece_count = popcount(pos.pieces()) - 4; // exclude kings and bishops
            if (piece_count > 0) {
                return std::min(18 + piece_count * 2, 32);
            } else {
                // Passed pawns only: very drawish
                return std::min(18 + total_pawns, 32);
            }
        }
    }

    // Lone queen vs multiple minor pieces - harder to convert
    int w_queens = popcount(pos.pieces(WHITE, QUEEN));
    int b_queens = popcount(pos.pieces(BLACK, QUEEN));
    int w_minors = popcount(pos.pieces(WHITE, KNIGHT)) + popcount(pos.pieces(WHITE, BISHOP));
    int b_minors = popcount(pos.pieces(BLACK, KNIGHT)) + popcount(pos.pieces(BLACK, BISHOP));
    if (w_queens == 1 && b_queens == 0 && b_minors >= 2 && wp == 0) return 20;
    if (b_queens == 1 && w_queens == 0 && w_minors >= 2 && bp == 0) return 20;

    return 32;  // Normal scale (32 = multiply by 1)
}

void init_evaluation() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    for (int pt = 0; pt < 8; ++pt) {
        for (int s = 0; s < 64; ++s) {
            PST_MG_TABLE[BLACK][pt][s] = PST_MG_TABLE[WHITE][pt][s ^ 56];
            PST_EG_TABLE[BLACK][pt][s] = PST_EG_TABLE[WHITE][pt][s ^ 56];
        }
    }

    // Zero-initialize pawn hash table
    std::memset(pawn_table, 0, sizeof(pawn_table));
}

} // namespace luminex
