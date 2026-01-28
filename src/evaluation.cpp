#include "luminex.h"

namespace luminex {

// Piece-square tables for middle game
Score PST_MG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN
        {0, 0, 0, 0, 0, 0, 0, 0,
         50, 50, 50, 50, 50, 50, 50, 50,
         10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
        },
        // KNIGHT
        {
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        },
        // BISHOP
        {
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        },
        // ROOK
        {
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        },
        // QUEEN
        {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        },
        // KING (middle game)
        {
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        },
        // NONE
        {0},
        // ALL
        {0}
    },
    // BLACK (mirrored at runtime)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

// Piece-square tables for endgame
Score PST_EG_TABLE[2][8][64] = {
    // WHITE
    {
        // PAWN
        {
            0,  0,  0,  0,  0,  0,  0,  0,
            100, 100, 100, 100, 100, 100, 100, 100,
            90, 90, 90, 90, 90, 90, 90, 90,
            70, 70, 70, 70, 70, 70, 70, 70,
            50, 50, 50, 50, 50, 50, 50, 50,
            30, 30, 30, 30, 30, 30, 30, 30,
            10, 10, 10, 10, 10, 10, 10, 10,
            0,  0,  0,  0,  0,  0,  0,  0
        },
        // KNIGHT
        {
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  5, 20, 25, 25, 20,  5,-30,
            -30,  5, 20, 25, 25, 20,  5,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        },
        // BISHOP
        {
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10, 10, 15, 15, 15, 15, 10,-10,
            -10, 10, 15, 20, 20, 15, 10,-10,
            -10, 10, 15, 20, 20, 15, 10,-10,
            -10, 10, 15, 15, 15, 15, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        },
        // ROOK
        {
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        },
        // QUEEN
        {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        },
        // KING (endgame - wants to be active)
        {
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        },
        // NONE
        {0},
        // ALL
        {0}
    },
    // BLACK (mirrored)
    {
        {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
    }
};

using Score = Value;

Value evaluate(const Position& pos) {
    Score mg_score = 0;
    Score eg_score = 0;

    // Track bishop pairs
    int bishop_count[2] = {0, 0};

    // Material and position evaluation
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Sign sign = (c == WHITE) ? 1 : -1;

        // Pawns
        Bitboard pawns = pos.pieces(c, PAWN);
        // Check for doubled pawns
        while (pawns) {
            Square sq = pop_lsb(pawns);
            mg_score += sign * PAWN_VALUE;
            eg_score += sign * PAWN_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(PAWN)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(PAWN)][int(sq)];

            // Doubled pawn penalty
            File f = file_of(sq);
            Bitboard file_pawns = pos.pieces(c, PAWN) & file_bb(f);
            if (popcount(file_pawns) > 1) {
                mg_score -= sign * 10;
                eg_score -= sign * 20;
            }

            // Isolated pawn penalty
            Bitboard adjacent_files = 0;
            if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
            if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
            Bitboard friendly_pawns_adjacent = pos.pieces(c, PAWN) & adjacent_files;
            if (!friendly_pawns_adjacent) {
                mg_score -= sign * 20;
                eg_score -= sign * 20;
            }

            // Passed pawn bonus
            Bitboard ahead = 0;
            Rank r = relative_rank(c, sq);
            if (r < RANK_7) {
                // Squares ahead of this pawn
                for (int rr = int(r) + 1; rr <= int(RANK_7); ++rr) {
                    Square rank_sq = relative_square(c, make_square(file_of(sq), Rank(rr)));
                    ahead |= square_bb(rank_sq);
                }
            }
            // No enemy pawns ahead
            if (!(ahead & pos.pieces(Color(c ^ 1), PAWN))) {
                mg_score += sign * (20 + r * 10);
                eg_score += sign * (50 + r * 20);
            }
        }

        // Knights
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * KNIGHT_VALUE;
            eg_score += sign * KNIGHT_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)];

            // Knight mobility
            int mobility = popcount(knight_attacks_bb(sq) & ~pos.pieces(c));
            mg_score += sign * mobility * 4;
            eg_score += sign * mobility * 8;
        }

        // Bishops
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[int(c)] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * BISHOP_VALUE;
            eg_score += sign * BISHOP_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)];

            // Bishop mobility
            int mobility = popcount(bb_diag_attacks(sq, pos.pieces()) & ~pos.pieces(c));
            mg_score += sign * mobility * 5;
            eg_score += sign * mobility * 10;
        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * ROOK_VALUE;
            eg_score += sign * ROOK_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(ROOK)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(ROOK)][int(sq)];

            // Rook mobility
            Bitboard occupied = pos.pieces();
            int mobility = popcount((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & ~pos.pieces(c));
            mg_score += sign * mobility * 2;
            eg_score += sign * mobility * 8;
        }

        // Queens
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            mg_score += sign * QUEEN_VALUE;
            eg_score += sign * QUEEN_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)];

            // Queen mobility
            Bitboard occupied = pos.pieces();
            int mobility = popcount(queen_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mobility * 1;
            eg_score += sign * mobility * 2;
        }

        // King (position only, no material value)
        Square ksq = pos.king_sq(c);
        mg_score += sign * PST_MG_TABLE[int(c)][int(KING)][int(ksq)];
        eg_score += sign * PST_EG_TABLE[int(c)][int(KING)][int(ksq)];
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) {
        mg_score += 50;
        eg_score += 70;
    }
    if (bishop_count[BLACK] >= 2) {
        mg_score -= 50;
        eg_score -= 70;
    }

    // Game phase detection (simplified)
    int material = 0;
    Bitboard all = pos.pieces();
    while (all) {
        Square s = pop_lsb(all);
        if (s >= SQUARE_NONE) continue;
        PieceType pt = pos.piece_type_on(s);
        if (pt != PAWN && pt != KING) {
            material += PAWN_VALUE;
        }
    }

    int phase = std::min(24, material / 200);

    // Interpolate between middle game and endgame
    Score score = (mg_score * phase + eg_score * (24 - phase)) / 24;

    return pos.side_to_move() == WHITE ? score : -score;
}

int scale_factor(const Position&, Value) {
    return 1;
}

void init_evaluation() {
    // Initialize black PST as mirrored white PST
    for (int pt = 0; pt < 8; ++pt) {
        for (int s = 0; s < 64; ++s) {
            PST_MG_TABLE[BLACK][pt][s] = PST_MG_TABLE[WHITE][pt][s ^ 56];
            PST_EG_TABLE[BLACK][pt][s] = PST_EG_TABLE[WHITE][pt][s ^ 56];
        }
    }
}

} // namespace luminex
