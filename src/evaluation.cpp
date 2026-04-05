#include "luminex.h"

namespace luminex {

// PeSTO piece values (MG and EG)
static constexpr int PieceValueMG[8] = { 82, 337, 365, 477, 1025, 0, 0, 0 };
static constexpr int PieceValueEG[8] = { 94, 281, 297, 512, 936, 0, 0, 0 };

// Stash-style mobility tables (MG, EG) - tuned non-linear values
static constexpr int KnightMobilityMG[9] = { -56, -45, -35, -26, -19, -13, -8, 0, 4 };
static constexpr int KnightMobilityEG[9] = {  20, -28,  36,  58,  76,  95,104,109,103 };
static constexpr int BishopMobilityMG[14] = { -56, -44, -27, -25, -17, -13, -10, -8, -7, -5, -2, 7, 9, 35 };
static constexpr int BishopMobilityEG[14] = { -44, -39, -15,  15,  32,  46,  56, 61, 64, 67, 61, 56, 57, 48 };
static constexpr int RookMobilityMG[15] = { -88, -39, -27, -31, -29, -33, -34, -29, -26, -18, -17, -13, -6, 4, 23 };
static constexpr int RookMobilityEG[15] = { -47,  40,  76,  92, 103, 117, 125, 131, 140, 149, 156, 163, 167, 169, 161 };
static constexpr int QueenMobilityMG[28] = { -14, 19, 0, -6, 1, -2, -4, -3, -2, -2, -1, 2, 2, 5, 5, 5, 6, 9, 14, 12, 36, 39, 48, 34, 59, 11, 11, 36 };
static constexpr int QueenMobilityEG[28] = { -93, 216, 147, 95, 65, 118, 151, 175, 185, 205, 214, 219, 228, 232, 237, 243, 244, 241, 232, 234, 204, 198, 183, 163, 156, 164, 170, 141 };

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

Value evaluate(const Position& pos) {
    Score mg_score = 0;
    Score eg_score = 0;

    Square ksq_arr[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};

    Bitboard occupied = pos.pieces();

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

        // Mobility area: exclude squares attacked by enemy pawns
        Bitboard mob_area = ~pawn_attacks_bb(them, their_pawns);

        // Pawns: material + PST + structure
        Bitboard pawns = our_pawns;
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            Rank r = relative_rank(c, sq);

            mg_score += sign * (PieceValueMG[PAWN] + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg_score += sign * (PieceValueEG[PAWN] + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);

            // Doubled pawn penalty
            if (file_count[f] > 1) {
                mg_score -= sign * 11;
                eg_score -= sign * 22;
            }

            // Isolated pawn penalty
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg_score -= sign * 14;
                eg_score -= sign * 18;
            }

            // Connected pawn bonus: pawn protected by another pawn
            if (square_bb(sq) & supported_by_adj) {
                int connected_bonus = 4 + r * 3;
                mg_score += sign * connected_bonus;
                eg_score += sign * (2 + r * 2);
            }

            // Pawn lever bonus: can capture an adjacent enemy pawn
            {
                Bitboard lever_targets = pawn_attacks_bb(c, sq) & their_pawns;
                if (lever_targets) {
                    int lever_bonus = 3 + r * 2;
                    mg_score += sign * lever_bonus;
                    // Extra for central levers
                    if (f >= FILE_C && f <= FILE_F) {
                        mg_score += sign * 3;
                    }
                }
            }

            // Backward pawn detection: a pawn that cannot advance safely
            // because the push square is attacked by enemy pawns, and no
            // adjacent-file friendly pawn can defend the push square
            {
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                if (push < SQUARE_NONE && r >= RANK_2 && r <= RANK_5) {
                    // Check if push square is attacked by enemy pawns
                    bool push_attacked = (pawn_attacks_bb(them, their_pawns) & square_bb(push)) != 0;
                    if (push_attacked) {
                        // Check if our pawns attack the push square (can defend it)
                        Bitboard our_pawn_attacks = pawn_attacks_bb(c, our_pawns);
                        bool defended = (our_pawn_attacks & square_bb(push)) != 0;
                        if (!defended) {
                            mg_score -= sign * 10;
                            eg_score -= sign * 12;
                        }
                    }
                }
            }

            // Passed pawn bonus with table-based evaluation (Stash-inspired)
            Bitboard ahead = 0;
            Bitboard ahead_file = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                ahead_file |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }

            // Candidate passed pawn: would be passed if stoppers were exchanged
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
                    mg_score += sign * CandMG[supported ? 1 : 0][r];
                    eg_score += sign * CandEG[supported ? 1 : 0][r];
                }
            }

            if (!(ahead & their_pawns)) {
                // Stash-style table bonuses by rank
                static constexpr int PassedMG[8] = { 0, -10, -14, -27, 16, 67, 107, 0 };
                static constexpr int PassedEG[8] = { 0,   7,  15,  41, 99,189, 350, 0 };
                int mg_passer = PassedMG[r];
                int eg_passer = PassedEG[r];

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
                eg_passer += (their_kdist - our_kdist) * 6;

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
            int mob = popcount(attacks & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 8);
            mg_score += sign * KnightMobilityMG[mob];
            eg_score += sign * KnightMobilityEG[mob];

            // Outpost knight: on rank 4-6, protected by own pawn, cannot be attacked by enemy pawn
            Rank kr = relative_rank(c, sq);
            if (kr >= RANK_4 && kr <= RANK_6) {
                if (pawn_attacks_bb(c, our_pawns) & square_bb(sq)) {
                    // Check if any enemy pawn could attack this square
                    Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
                    bool can_be_attacked = (enemy_pawn_attacks & square_bb(sq)) != 0;
                    if (!can_be_attacked) {
                        // True outpost: protected, cannot be challenged by enemy pawn
                        mg_score += sign * (20 + (kr - 3) * 5);
                        eg_score += sign * (10 + (kr - 3) * 3);
                    }
                }
            }
        }

        // Bishops
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * (PieceValueMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PieceValueEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            int mob = popcount(bb_diag_attacks(sq, occupied) & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 13);
            mg_score += sign * BishopMobilityMG[mob];
            eg_score += sign * BishopMobilityEG[mob];

        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (PieceValueMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PieceValueEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);

            int mob = popcount(rook_attacks_bb(sq, occupied) & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 14);
            mg_score += sign * RookMobilityMG[mob];
            eg_score += sign * RookMobilityEG[mob];

            // Open/semi-open file
            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * 38;
                eg_score += sign * 38;
            } else if (!(our_pawns & file_bb(f))) {
                mg_score += sign * 15;
                eg_score += sign * 18;
            }

            // Rook on 7th rank bonus
            Rank rr = relative_rank(c, rank_of(sq));
            if (rr == RANK_7) {
                mg_score += sign * 25;
                eg_score += sign * 30;
            }
        }

        // Queens
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            mg_score += sign * (PieceValueMG[QUEEN] + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (PieceValueEG[QUEEN] + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);

            int mob = popcount(queen_attacks_bb(sq, occupied) & mob_area & ~pos.pieces(c));
            mob = std::min(mob, 27);
            mg_score += sign * QueenMobilityMG[mob];
            eg_score += sign * QueenMobilityEG[mob];
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
                int weight = (r == 1) ? 15 : (r == 2) ? 8 : 3;
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
            if (!(all_pawns & file_bb(kfile))) mg_score -= sign * 20;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) mg_score -= sign * 15;
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) mg_score -= sign * 15;

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
                            int storm_danger = (pr - 3) * 8;  // 8 per rank above 3
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

        // Threat evaluation: pieces attacked by enemy pawns
        {
            Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
            Bitboard our_non_pawns = (pos.pieces(c) ^ pos.pieces(c, PAWN)) ^ pos.pieces(c, KING);
            Bitboard threatened = our_non_pawns & enemy_pawn_attacks;
            while (threatened) {
                Square sq = pop_lsb(threatened);
                PieceType pt = piece_type_of(pos.piece_on(sq));
                int penalty = 0;
                switch (pt) {
                    case KNIGHT: penalty = 40; break;
                    case BISHOP: penalty = 40; break;
                    case ROOK:   penalty = 30; break;
                    case QUEEN:  penalty = 55; break;
                    default: break;
                }
                mg_score -= sign * penalty;
                eg_score -= sign * (penalty * 2 / 3);
            }
        }

    }

    // Space evaluation: simplified - just count center pawn presence
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Sign sign = (c_idx == 0) ? 1 : -1;
        Bitboard our_pawns = pos.pieces(c, PAWN);
        int center_count = popcount(our_pawns & (BB_FILE_D | BB_FILE_E));
        if (center_count > 0 && popcount(our_pawns) >= 5) {
            mg_score += sign * center_count * 6;
        }
    }

    // Bishop pair bonus (Stash values: 21/93)
    if (bishop_count[WHITE] >= 2) { mg_score += 21; eg_score += 93; }
    if (bishop_count[BLACK] >= 2) { mg_score -= 21; eg_score -= 93; }

    // King safety using Stockfish-style SafetyTable (from chessprogramming.org)
    // Smooth S-curve scaling instead of crude quadratic
    static const int SafetyTable[100] = {
        0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 39,
        44, 50, 56, 62, 68, 75, 82, 85, 89, 97, 105, 113, 122, 131, 140,
        150, 169, 180, 191, 202, 213, 225, 237, 248, 260, 272, 283, 295,
        307, 319, 330, 342, 354, 366, 377, 389, 401, 412, 424, 436, 448,
        459, 471, 483, 494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500
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

        // Pawn attacks on king zone (2 units per pawn)
        Bitboard enemy_pawns = pos.pieces(them, PAWN);
        Bitboard ep = enemy_pawns;
        while (ep) {
            Square psq = pop_lsb(ep);
            if (pawn_attacks_bb(them, square_bb(psq)) & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
        }

        // Knight attacks on king zone + safe check bonus
        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            Square ksq2 = pop_lsb(enemy_kn);
            Bitboard kn_attacks = knight_attacks_bb(ksq2);
            if (kn_attacks & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
            // Safe check: knight can give check and the check square is not defended
            Bitboard kn_checks = kn_attacks & king_attacks_bb(our_ksq);
            if (kn_checks & safe) attack_units += 3;
        }

        // Bishop attacks on king zone + safe check bonus
        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            Square bsq = pop_lsb(enemy_bi);
            Bitboard bi_attacks = bishop_attacks_bb(bsq, occupied);
            if (bi_attacks & king_zone) {
                attack_units += 2;
                attacker_count++;
            }
            Bitboard bi_checks = bi_attacks & bb_diag_attacks(our_ksq, Bitboard(0));
            if (bi_checks & safe) attack_units += 2;
        }

        // Rook attacks on king zone + safe check bonus
        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            Square rsq = pop_lsb(enemy_ro);
            Bitboard ro_attacks = rook_attacks_bb(rsq, occupied);
            if (ro_attacks & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
            Bitboard ro_checks = ro_attacks & rook_attacks_bb(our_ksq, Bitboard(0));
            if (ro_checks & safe) attack_units += 4;
        }

        // Queen attacks on king zone + safe check bonus
        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            Square qsq = pop_lsb(enemy_qu);
            Bitboard qu_attacks = queen_attacks_bb(qsq, occupied);
            if (qu_attacks & king_zone) {
                attack_units += 5;
                attacker_count++;
            }
            // Queen checks are devastating
            Bitboard qu_checks = qu_attacks & (bb_diag_attacks(our_ksq, Bitboard(0)) | rook_attacks_bb(our_ksq, Bitboard(0)));
            if (qu_checks & safe) attack_units += 5;
        }

        // Only evaluate king safety if we have enough attackers
        if (attacker_count >= 1) {
            int idx = std::min(attack_units, 99);
            int danger = SafetyTable[idx];

            // No queen = much less danger
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
    if (total_pawns == 0) return 16;

    // Opposite-colored bishops: drawish endgames
    int wb = popcount(pos.pieces(WHITE, BISHOP));
    int bb = popcount(pos.pieces(BLACK, BISHOP));
    if (wb == 1 && bb == 1) {
        // Check if bishops are on opposite colors
        Square wb_sq = lsb(pos.pieces(WHITE, BISHOP));
        Square bb_sq = lsb(pos.pieces(BLACK, BISHOP));
        bool wb_light = ((int(wb_sq) + (wb_sq / 8)) % 2) == 0;
        bool bb_light = ((int(bb_sq) + (bb_sq / 8)) % 2) == 0;
        if (wb_light != bb_light && total_pawns <= 4) {
            return 20;  // Drawish with OCB and few pawns
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
}

} // namespace luminex
