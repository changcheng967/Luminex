#include "luminex.h"

namespace luminex {

// PeSTO piece values (MG and EG)
static constexpr int PieceValueMG[8] = { 82, 337, 365, 477, 1025, 0, 0, 0 };
static constexpr int PieceValueEG[8] = { 94, 281, 297, 512, 936, 0, 0, 0 };

// PeSTO piece-square tables (WHITE, standard layout a1=0)
// These are positional-only bonuses; material is added separately
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
            -105, -21, -58, -33, -17, -28, -19,-105,
             -29, -53, -12,  -3,  -1,  18, -14, -19,
             -23,  -9,  12,  10,  19,  17,  25, -16,
             -13,   4,  16,  13,  28,  19,  21,  -8,
              -9,  17,  19,  53,  37,  69,  18,  22,
             -47,  60,  37,  65,  84,  71,  62,   7,
             -73,  41,  72,  36,  23,  62,   7, -17,
            -167, -89, -34, -49,  61, -97, -40,-162
        },
        // BISHOP
        {
            -24,  -1, -20, -23,  -9, -24, -14, -30,
              -3,  14,  21,  28,  32,  43,  27, -13,
               0,   8,  28,  29,  37,  32,  26,  -3,
              -6,  13,  13,  26,  34,  23,  12,  -6,
              -4,   5,  19,  50,  37,  67,  26,  -3,
            -16,  37,  43,  40,  35,  50,  37,  -2,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -29,   4, -82, -37, -25, -42,   7,  -8
        },
        // ROOK
        {
            -21,  -3,  -7,  12,  22,  28,  46, -14,
            -31,  -4,   9,  17,  21,  43,  52, -25,
            -25,  -6,  10,  14,  23,  49,  52, -17,
            -20,   1,  12,  17,  15,  55,  52, -14,
            -24,  -7,  26,  25,  24,  49,  55, -24,
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
             13,   8,   8, -10,  -6,  -1,  -2,   6,
              4,   7,  -6,   1,   0,  -5,  -1,   8,
             13,   9,  -3,  -7,  -7,  -8,   3,  -1,
             32,  24,  13,   5,  -2,   4,  17,  17,
             94, 100,  85,  67,  56,  53,  82,  84,
            178, 173, 158, 134, 147, 132, 165, 187,
              0,   0,   0,   0,   0,   0,   0,   0
        },
        // KNIGHT
        {
            -47, -14, -25, -17, -14, -21, -28, -52,
            -23,  -9,   0,  -3,  -3,  -9, -10,  -7,
            -25,  -2,  -6,   0,   1,  -4,  -6,  -5,
            -27,   0,  -3,   0,  -3,   1,  -3,  -2,
            -23,   2,  -5,  -1,  -5,  -2,  -1,   1,
            -21,  -2,  -7, -10,  -8,  -6,  -3,  -4,
            -30,  -7, -14, -22, -14, -11, -18,  -9,
            -58, -38, -13, -28,  31, -27, -30, -68
        },
        // BISHOP
        {
             -8,   0,  -5,  -1,  -3,  -1,  -5, -10,
              0,   2,  -1,   0,   0,   1,   1,   2,
              2,   4,   2,   3,   2,   4,   3,   3,
              2,   4,   2,   1,   0,   3,   2,   1,
              3,  -1,  -1,   0,  -1,   4,   0,   0,
              2,  -8,   0,  -1,  -2,   0,  -4,   1,
             -8,  -4,   7,  -5,  -4,  -8, -16,  -9,
            -14, -21, -11,  -8,  -7,  -9, -17, -19
        },
        // ROOK
        {
              0,   0,   0,   0,   0,   0,   0,   1,
              0,   0,  -1,   0,   0,   0,   0,   1,
              1,   0,  -1,  -1,   0,   0,  -1,   2,
              1,   0,   0,  -1,   0,   0,  -1,   3,
              2,   0,   0,   0,   0,  -1,   0,   3,
              3,   1,   0,   1,   0,   0,   0,   2,
              6,   1,   0,   1,   1,   0,  -1,   2,
              7,  -1,  -1,   1,   1,  -1,   0,   0
        },
        // QUEEN
        {
             -6,   1,  -3,   0,  -1,  -5,  -3,  -6,
              3,  -4,  -2,  -6,  -2,  -6,  -2,  -4,
              0,  -3,  -6,  -8,  -6,  -9,  -5,  -3,
            -10,  -5, -12,  -9,  -4,  -7,  -8,  -6,
            -16,  -3, -11,  -6,   3,   0,  -3,  -3,
            -17,  -3,  12,  -1,   5,  -1,  -7,   6,
            -17,  20,  22,  18,  22,  13, -10,   1,
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

    Square king_sq[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};
    int bishop_count[2] = {0, 0};
    Bitboard occupied = pos.pieces();

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Color them = Color(c_idx ^ 1);
        Sign sign = (c == WHITE) ? 1 : -1;
        Bitboard their_pawns = pos.pieces(them, PAWN);

        // Pre-compute file counts for pawn structure
        int file_count[8] = {0};
        Bitboard tmp = pos.pieces(c, PAWN);
        while (tmp) { file_count[file_of(pop_lsb(tmp))]++; }

        // Pawns: material + PST + structure
        Bitboard pawns = pos.pieces(c, PAWN);
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

            // Passed pawn bonus
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }
            if (!(ahead & their_pawns)) {
                mg_score += sign * (15 + r * 10);
                eg_score += sign * (30 + r * 25);
            }
        }

        // Knights
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * (PieceValueMG[KNIGHT] + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (PieceValueEG[KNIGHT] + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            int mob = popcount(knight_attacks_bb(sq) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 4;
        }

        // Bishops
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * (PieceValueMG[BISHOP] + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (PieceValueEG[BISHOP] + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            int mob = popcount(bb_diag_attacks(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 3;
            eg_score += sign * mob * 5;
        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (PieceValueMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PieceValueEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);
            int mob = popcount((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 6;

            // Open/semi-open file
            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * 38;
                eg_score += sign * 38;
            } else if (!(pos.pieces(c, PAWN) & file_bb(f))) {
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
            int mob = popcount(queen_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 4;
        }

        // King PST
        Square ksq = king_sq[c_idx];
        mg_score += sign * (PieceValueMG[KING] + PST_MG_TABLE[int(c)][int(KING)][int(ksq)]);
        eg_score += sign * (PieceValueEG[KING] + PST_EG_TABLE[int(c)][int(KING)][int(ksq)]);

        // King pawn shield (MG only)
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);
        bool on_back = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);
        if (on_back) {
            Square front = relative_square(c, make_square(kfile, RANK_2));
            Bitboard shield = square_bb(front);
            if (kfile > FILE_A) shield |= square_bb(relative_square(c, make_square(File(kfile - 1), RANK_2)));
            if (kfile < FILE_H) shield |= square_bb(relative_square(c, make_square(File(kfile + 1), RANK_2)));
            int count = popcount(pos.pieces(c, PAWN) & shield);
            mg_score += sign * count * 15;

            Bitboard all_pawns = pos.pieces(PAWN);
            if (!(all_pawns & file_bb(kfile))) mg_score -= sign * 20;
            if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) mg_score -= sign * 15;
            if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) mg_score -= sign * 15;
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
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) { mg_score += 60; eg_score += 80; }
    if (bishop_count[BLACK] >= 2) { mg_score -= 60; eg_score -= 80; }

    // Simplified king safety: count pieces attacking near enemy king
    // Use a simple 3x3 zone around the king (much cheaper than full danger zone)
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Square opp_ksq = king_sq[c_idx ^ 1];
        // Simple king neighborhood: king attacks + king square
        Bitboard zone = king_attacks_bb(opp_ksq) | square_bb(opp_ksq);

        int attack_units = 0;

        Bitboard kn = pos.pieces(us, KNIGHT);
        while (kn) { if (knight_attacks_bb(pop_lsb(kn)) & zone) attack_units += 4; }

        Bitboard bi = pos.pieces(us, BISHOP);
        while (bi) { if (bb_diag_attacks(pop_lsb(bi), occupied) & zone) attack_units += 5; }

        Bitboard ro = pos.pieces(us, ROOK);
        while (ro) { Square sq = pop_lsb(ro); if ((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & zone) attack_units += 7; }

        Bitboard qu = pos.pieces(us, QUEEN);
        while (qu) { if (queen_attacks_bb(pop_lsb(qu), occupied) & zone) attack_units += 12; }

        // Quadratic bonus for multiple attackers
        if (attack_units >= 8) attack_units += attack_units * attack_units / 4;

        Sign sign = (c_idx == 0) ? 1 : -1;
        mg_score += sign * attack_units * 4;
    }

    // Phase calculation
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    // Interpolate MG/EG
    Score score = (mg_score * phase + eg_score * (24 - phase)) / 24;

    // Tempo bonus
    Score tempo = (15 * phase + 5 * (24 - phase)) / 24;
    score += (pos.side_to_move() == WHITE) ? tempo : -tempo;

    return pos.side_to_move() == WHITE ? score : -score;
}

int scale_factor(const Position&, Value) {
    return 1;
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
