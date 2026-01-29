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

// Helper: get king danger zone (3x3 area around king, plus pawn attack squares)
inline Bitboard king_danger_zone(Square ksq) {
    Bitboard zone = 0;
    // 3x3 area around king
    Bitboard kbb = square_bb(ksq);
    zone |= kbb;
    if (file_of(ksq) > FILE_A) zone |= shift_w(kbb) | shift_nw(kbb) | shift_sw(kbb);
    if (file_of(ksq) < FILE_H) zone |= shift_e(kbb) | shift_ne(kbb) | shift_se(kbb);
    zone |= shift_n(kbb) | shift_s(kbb);
    // Also include squares that pawns could attack the king from
    if (rank_of(ksq) > RANK_1) zone |= shift_s(zone);  // One rank below
    if (rank_of(ksq) < RANK_8) zone |= shift_n(zone);  // One rank above
    return zone;
}

Value evaluate(const Position& pos) {
    Score mg_score = 0;
    Score eg_score = 0;

    // Track bishop pairs and king danger
    int bishop_count[2] = {0, 0};
    int king_attackers[2] = {0, 0};
    Square king_sq[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};

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

            // Rook on open file bonus
            File f = file_of(sq);
            Bitboard file_pawns = pos.pieces(PAWN) & file_bb(f);
            if (!file_pawns) {
                // Open file - very valuable
                mg_score += sign * 30;
                eg_score += sign * 50;
            } else {
                // Check if only enemy pawns on this file (semi-open)
                Bitboard our_pawns = pos.pieces(c, PAWN) & file_bb(f);
                if (!our_pawns) {
                    // Semi-open file for us
                    mg_score += sign * 15;
                    eg_score += sign * 25;
                }
            }

            // Rook on 7th rank bonus (for white) or 2nd rank (for black)
            Rank r = rank_of(sq);
            if ((c == WHITE && r == RANK_7) || (c == BLACK && r == RANK_2)) {
                mg_score += sign * 30;
                eg_score += sign * 20;
            }
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

        // King safety - pawn shield (middle game only)
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);

        // Only evaluate pawn shield when king is on ranks 1-2 (white) or 6-7 (black)
        bool king_on_back_rank = (c == WHITE && krank <= RANK_2) || (c == BLACK && krank >= RANK_7);

        if (king_on_back_rank) {
            // Check pawn shield squares
            // In front of king
            Square front_sq = relative_square(c, make_square(kfile, RANK_2));
            Bitboard shield_squares = square_bb(front_sq);

            // Diagonally in front
            if (kfile > FILE_A) {
                shield_squares |= square_bb(relative_square(c, make_square(File(kfile - 1), RANK_2)));
            }
            if (kfile < FILE_H) {
                shield_squares |= square_bb(relative_square(c, make_square(File(kfile + 1), RANK_2)));
            }

            // Bonus for pawn shield
            int shield_count = popcount(pos.pieces(c, PAWN) & shield_squares);
            mg_score += sign * shield_count * 15;

            // Penalty for missing shield (especially in center files)
            if (kfile >= FILE_C && kfile <= FILE_F) {
                mg_score -= sign * (3 - shield_count) * 20;
            }

            // Second rank shield bonus
            Square front_sq2 = relative_square(c, make_square(kfile, RANK_3));
            Bitboard shield_squares2 = square_bb(front_sq2);
            if (kfile > FILE_A) {
                shield_squares2 |= square_bb(relative_square(c, make_square(File(kfile - 1), RANK_3)));
            }
            if (kfile < FILE_H) {
                shield_squares2 |= square_bb(relative_square(c, make_square(File(kfile + 1), RANK_3)));
            }

            int shield_count2 = popcount(pos.pieces(c, PAWN) & shield_squares2);
            mg_score += sign * shield_count2 * 5;
        }

        // Penalty for open files near king (files with no pawns)
        Bitboard all_pawns = pos.pieces(PAWN);
        int open_file_penalty = 0;

        // Check king's file
        if (!(all_pawns & file_bb(kfile))) {
            open_file_penalty += 20;
        }

        // Check adjacent files
        if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) {
            open_file_penalty += 15;
        }
        if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) {
            open_file_penalty += 15;
        }

        mg_score -= sign * open_file_penalty;
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

    // King danger and queen tropism evaluation
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Square their_king = king_sq[int(them)];
        Square our_king = king_sq[int(us)];

        // Queen tropism: penalty when enemy queen is close to our king
        Bitboard their_queens = pos.pieces(them, QUEEN);
        while (their_queens) {
            Square qsq = pop_lsb(their_queens);
            int dist = distance(our_king, qsq);
            if (dist <= 3) {
                mg_score -= (us == WHITE ? 1 : -1) * (50 * (4 - dist));
                eg_score -= (us == WHITE ? 1 : -1) * (30 * (4 - dist));
            } else if (dist <= 5) {
                mg_score -= (us == WHITE ? 1 : -1) * (20 * (6 - dist));
            }
        }

        // King danger: count attackers and their weight
        Bitboard danger_zone = king_danger_zone(their_king);
        int danger = 0;

        // Count pieces attacking king zone
        Bitboard our_knights = pos.pieces(us, KNIGHT);
        while (our_knights) {
            Square sq = pop_lsb(our_knights);
            if (knight_attacks_bb(sq) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 40;  // Knight attacks
            }
        }

        Bitboard our_bishops = pos.pieces(us, BISHOP);
        while (our_bishops) {
            Square sq = pop_lsb(our_bishops);
            if (bb_diag_attacks(sq, pos.pieces()) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 50;  // Bishop attacks
            }
        }

        Bitboard our_rooks = pos.pieces(us, ROOK);
        while (our_rooks) {
            Square sq = pop_lsb(our_rooks);
            Bitboard attacks = (bb_rank_attacks(sq, pos.pieces()) | bb_file_attacks(sq, pos.pieces()));
            if (attacks & danger_zone) {
                king_attackers[int(us)]++;
                danger += 70;  // Rook attacks are dangerous
            }
        }

        Bitboard our_queens = pos.pieces(us, QUEEN);
        while (our_queens) {
            Square sq = pop_lsb(our_queens);
            if (queen_attacks_bb(sq, pos.pieces()) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 120;  // Queen attacks are very dangerous
            }
        }

        // Bonus for multiple attackers (coordination)
        if (king_attackers[int(us)] >= 2) {
            danger += (king_attackers[int(us)] - 1) * 30;
        }

        // Penalty when we have no safe king position (king in center)
        Rank krank = rank_of(our_king);
        if ((us == WHITE && krank >= RANK_3 && krank <= RANK_5) ||
            (us == BLACK && krank >= RANK_4 && krank <= RANK_6)) {
            // King in center - very dangerous in middle game
            mg_score -= (us == WHITE ? 1 : -1) * 40;
            // Even worse when under attack
            if (king_attackers[int(them)] > 0) {
                mg_score -= (us == WHITE ? 1 : -1) * 60 * king_attackers[int(them)];
            }
        }

        // Apply king danger score (scaled by number of attackers)
        if (king_attackers[int(us)] >= 2) {
            mg_score += (us == WHITE ? 1 : -1) * danger;
        } else if (king_attackers[int(us)] == 1 && danger > 50) {
            mg_score += (us == WHITE ? 1 : -1) * (danger / 2);
        }
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

    // Trade logic: simplify when ahead, complicate when behind
    // Calculate material difference (without pawns)
    int white_pieces = popcount(pos.pieces(WHITE, KNIGHT)) + popcount(pos.pieces(WHITE, BISHOP)) * 2
                       + popcount(pos.pieces(WHITE, ROOK)) * 2 + popcount(pos.pieces(WHITE, QUEEN)) * 4;
    int black_pieces = popcount(pos.pieces(BLACK, KNIGHT)) + popcount(pos.pieces(BLACK, BISHOP)) * 2
                       + popcount(pos.pieces(BLACK, ROOK)) * 2 + popcount(pos.pieces(BLACK, QUEEN)) * 4;
    int piece_diff = white_pieces - black_pieces;

    // When ahead (positive piece difference), favor trades
    // When behind (negative piece difference), avoid trades
    if (piece_diff > 0) {
        // White is ahead - encourage simplification
        mg_score += piece_diff * 5;
        eg_score += piece_diff * 3;
    } else if (piece_diff < 0) {
        // White is behind - discourage simplification (add negative becomes positive for black)
        mg_score += piece_diff * 5;
        eg_score += piece_diff * 3;
    }

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
