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

            // Knight outpost bonus: knight in enemy territory, supported by pawn, not attackable by enemy pawns
            Rank r = rank_of(sq);
            File f = file_of(sq);
            bool in_enemy_territory = (c == WHITE && r >= RANK_4 && r <= RANK_6) || (c == BLACK && r >= RANK_3 && r <= RANK_5);

            if (in_enemy_territory) {
                // Check if supported by our pawn
                bool pawn_support = false;
                Bitboard supporting_pawns = pos.pieces(c, PAWN);
                if (c == WHITE) {
                    // White pawns on rank 3 can support knights on rank 4
                    Bitboard rank3 = rank_bb(RANK_3);
                    Bitboard support_squares = 0;
                    if (f > FILE_A) support_squares |= shift_sw(rank3 & file_bb(File(f - 1)));
                    if (f < FILE_H) support_squares |= shift_se(rank3 & file_bb(File(f + 1)));
                    if (supporting_pawns & support_squares) pawn_support = true;
                } else {
                    // Black pawns on rank 6 can support knights on rank 5
                    Bitboard rank6 = rank_bb(RANK_6);
                    Bitboard support_squares = 0;
                    if (f > FILE_A) support_squares |= shift_nw(rank6 & file_bb(File(f - 1)));
                    if (f < FILE_H) support_squares |= shift_ne(rank6 & file_bb(File(f + 1)));
                    if (supporting_pawns & support_squares) pawn_support = true;
                }

                // Check if enemy pawns can attack this square
                Bitboard enemy_pawns = pos.pieces(Color(c ^ 1), PAWN);
                Bitboard pawn_attacks = 0;
                if (c == WHITE) {
                    // Black pawns attack from ranks above
                    if (f > FILE_A) pawn_attacks |= shift_ne(square_bb(sq));
                    if (f < FILE_H) pawn_attacks |= shift_nw(square_bb(sq));
                } else {
                    // White pawns attack from ranks below
                    if (f > FILE_A) pawn_attacks |= shift_se(square_bb(sq));
                    if (f < FILE_H) pawn_attacks |= shift_sw(square_bb(sq));
                }
                bool safe_from_pawns = !(enemy_pawns & pawn_attacks);

                if (pawn_support && safe_from_pawns) {
                    // Outpost knight - very valuable
                    mg_score += sign * 50;
                    eg_score += sign * 30;
                } else if (safe_from_pawns) {
                    // Still a good square if safe from pawns
                    mg_score += sign * 20;
                    eg_score += sign * 15;
                }
            }
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

        // Bad bishop penalty: bishop blocked by own pawns on same color
        if (bishop_count[int(c)] == 1) {
            // We have exactly one bishop
            Square bishop_sq = lsb(pos.pieces(c, BISHOP));
            bool bishop_on_light = ((int(bishop_sq) + int(rank_of(bishop_sq))) % 2) == 0;

            // Count our pawns on the same color as our bishop
            Bitboard our_pawns = pos.pieces(c, PAWN);
            int blocking_pawns = 0;
            while (our_pawns) {
                Square pawn_sq = pop_lsb(our_pawns);
                bool pawn_on_light = ((int(pawn_sq) + int(rank_of(pawn_sq))) % 2) == 0;
                if (pawn_on_light == bishop_on_light) {
                    blocking_pawns++;
                }
            }

            if (blocking_pawns >= 4) {
                // Bad bishop - many pawns on same color
                mg_score -= sign * 50;
                eg_score -= sign * 30;
            } else if (blocking_pawns >= 3) {
                // Slightly bad
                mg_score -= sign * 25;
                eg_score -= sign * 15;
            }
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

    // Weak square evaluation: find squares that can't be defended by pawns
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Sign sign = (us == WHITE) ? 1 : -1;

        // For each file, check if we have pawn weak points
        for (int fi = int(FILE_A); fi <= int(FILE_H); ++fi) {
            File f = File(fi);
            Bitboard file_bb_f = file_bb(f);

            // Find our pawns on this file
            Bitboard our_pawns_on_file = pos.pieces(us, PAWN) & file_bb_f;

            // Check if we have pawn weak points in the middle ranks (3-6 for white, 3-6 for black)
            Bitboard weak_squares = 0;

            if (!our_pawns_on_file) {
                // No pawns on this file - all squares are potentially weak
                for (int ri = int(RANK_3); ri <= int(RANK_6); ++ri) {
                    Square sq = make_square(f, Rank(ri));
                    weak_squares |= square_bb(sq);
                }
            } else {
                // Check for holes in our pawn structure
                Square frontmost_pawn = us == WHITE ?
                    msb(our_pawns_on_file) :  // White: highest rank pawn
                    lsb(our_pawns_on_file);   // Black: lowest rank pawn

                Rank front_rank = relative_rank(us, frontmost_pawn);

                // Squares ahead of our frontmost pawn are weak
                if (us == WHITE) {
                    for (int ri = int(front_rank) + 1; ri <= int(RANK_6); ++ri) {
                        Square sq = make_square(f, Rank(ri));
                        weak_squares |= square_bb(sq);
                    }
                } else {
                    for (int ri = int(RANK_6); ri >= int(front_rank) + 1; --ri) {
                        Square sq = relative_square(us, make_square(f, Rank(ri)));
                        weak_squares |= square_bb(sq);
                    }
                }
            }

            // Check if enemy pieces can control these weak squares
            if (weak_squares) {
                Bitboard enemy_knights = pos.pieces(them, KNIGHT);
                Bitboard enemy_bishops = pos.pieces(them, BISHOP);

                int weak_square_count = 0;
                while (weak_squares) {
                    Square sq = pop_lsb(weak_squares);

                    // Check if enemy minor pieces can attack this square
                    bool attacked = false;

                    Bitboard temp_knights = enemy_knights;
                    while (temp_knights) {
                        Square ksq = pop_lsb(temp_knights);
                        if (knight_attacks_bb(ksq) & square_bb(sq)) {
                            attacked = true;
                            break;
                        }
                    }

                    if (!attacked) {
                        Bitboard temp_bishops = enemy_bishops;
                        while (temp_bishops) {
                            Square bsq = pop_lsb(temp_bishops);
                            if (bb_diag_attacks(bsq, pos.pieces()) & square_bb(sq)) {
                                attacked = true;
                                break;
                            }
                        }
                    }

                    if (attacked) {
                        weak_square_count++;
                    }
                }

                if (weak_square_count >= 2) {
                    // Multiple weak squares on this file
                    mg_score -= sign * 20 * weak_square_count;
                    eg_score -= sign * 10 * weak_square_count;
                }
            }
        }

        // Backward pawn penalty: pawn that can't be defended by other pawns
        Bitboard our_pawns = pos.pieces(us, PAWN);
        while (our_pawns) {
            Square pawn_sq = pop_lsb(our_pawns);
            File f = file_of(pawn_sq);
            Rank r = rank_of(pawn_sq);

            // Check if this pawn has friendly pawn support behind it
            bool has_support = false;
            if (us == WHITE && r > RANK_2) {
                // Check for pawns on adjacent files on rank below
                Bitboard support_rank = rank_bb(Rank(int(r) - 1));
                Bitboard adjacent_files = 0;
                if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
                if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
                if (pos.pieces(us, PAWN) & support_rank & adjacent_files) {
                    has_support = true;
                }
            } else if (us == BLACK && r < RANK_7) {
                // Check for pawns on adjacent files on rank above
                Bitboard support_rank = rank_bb(Rank(int(r) + 1));
                Bitboard adjacent_files = 0;
                if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
                if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
                if (pos.pieces(us, PAWN) & support_rank & adjacent_files) {
                    has_support = true;
                }
            }

            // Check if there are enemy pawns on adjacent files ahead (stopping it from advancing)
            bool stopped_by_enemy = false;
            if (us == WHITE && r < RANK_7) {
                Bitboard ahead_ranks = 0;
                for (int ar = int(r) + 1; ar <= int(RANK_7); ++ar) {
                    ahead_ranks |= rank_bb(Rank(ar));
                }
                Bitboard adjacent_files = 0;
                if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
                if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
                if (pos.pieces(them, PAWN) & ahead_ranks & adjacent_files) {
                    stopped_by_enemy = true;
                }
            } else if (us == BLACK && r > RANK_2) {
                Bitboard ahead_ranks = 0;
                for (int ar = int(RANK_2); ar < int(r); ++ar) {
                    ahead_ranks |= rank_bb(Rank(ar));
                }
                Bitboard adjacent_files = 0;
                if (f > FILE_A) adjacent_files |= file_bb(File(f - 1));
                if (f < FILE_H) adjacent_files |= file_bb(File(f + 1));
                if (pos.pieces(them, PAWN) & ahead_ranks & adjacent_files) {
                    stopped_by_enemy = true;
                }
            }

            if (!has_support && stopped_by_enemy) {
                // Backward pawn
                mg_score -= sign * 15;
                eg_score -= sign * 25;
            }
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

    // Threat detection: find pieces under attack and hanging pieces
    // For each side, check if their pieces are attacked by opponent
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Sign sign = (us == WHITE) ? 1 : -1;

        // Find all our pieces that might be under attack
        Bitboard our_pieces = pos.pieces(us);
        Bitboard their_attacks = 0;

        // Calculate their attack squares
        Bitboard their_pawns = pos.pieces(them, PAWN);
        while (their_pawns) {
            Square sq = pop_lsb(their_pawns);
            File f = file_of(sq);
            if (us == WHITE) {
                // White pawns attack diagonally downward (from black's perspective)
                Bitboard pb = square_bb(sq);
                if (f > FILE_A) their_attacks |= shift_sw(pb);
                if (f < FILE_H) their_attacks |= shift_se(pb);
            } else {
                // Black pawns attack diagonally upward (from black's perspective)
                Bitboard pb = square_bb(sq);
                if (f > FILE_A) their_attacks |= shift_nw(pb);
                if (f < FILE_H) their_attacks |= shift_ne(pb);
            }
        }

        Bitboard their_knights = pos.pieces(them, KNIGHT);
        while (their_knights) {
            Square sq = pop_lsb(their_knights);
            their_attacks |= knight_attacks_bb(sq);
        }

        Bitboard their_bishops = pos.pieces(them, BISHOP);
        while (their_bishops) {
            Square sq = pop_lsb(their_bishops);
            their_attacks |= bb_diag_attacks(sq, pos.pieces());
        }

        Bitboard their_rooks = pos.pieces(them, ROOK);
        while (their_rooks) {
            Square sq = pop_lsb(their_rooks);
            their_attacks |= bb_rank_attacks(sq, pos.pieces()) | bb_file_attacks(sq, pos.pieces());
        }

        Bitboard their_queens = pos.pieces(them, QUEEN);
        while (their_queens) {
            Square sq = pop_lsb(their_queens);
            their_attacks |= queen_attacks_bb(sq, pos.pieces());
        }

        Bitboard their_king = pos.pieces(them, KING);
        while (their_king) {
            Square sq = pop_lsb(their_king);
            Bitboard kbb = square_bb(sq);
            their_attacks |= kbb;
            if (file_of(sq) > FILE_A) their_attacks |= shift_w(kbb) | shift_nw(kbb) | shift_sw(kbb);
            if (file_of(sq) < FILE_H) their_attacks |= shift_e(kbb) | shift_ne(kbb) | shift_se(kbb);
            their_attacks |= shift_n(kbb) | shift_s(kbb);
        }

        // Now check which of our pieces are under attack
        Bitboard attacked_pieces = our_pieces & their_attacks;
        while (attacked_pieces) {
            Square sq = pop_lsb(attacked_pieces);
            PieceType pt = pos.piece_type_on(sq);
            if (pt == KING) continue;  // Skip king

            // Base threat: penalty when our piece is attacked
            Value piece_value = VALUE_ZERO;
            if (pt == PAWN) piece_value = PAWN_VALUE;
            else if (pt == KNIGHT) piece_value = KNIGHT_VALUE;
            else if (pt == BISHOP) piece_value = BISHOP_VALUE;
            else if (pt == ROOK) piece_value = ROOK_VALUE;
            else if (pt == QUEEN) piece_value = QUEEN_VALUE;

            // Check if this piece is defended (hanging piece detection)
            Bitboard our_attacks = 0;

            // Calculate our defense for this square
            Bitboard our_pieces_check = pos.pieces(us) ^ square_bb(sq);  // All our pieces except this one

            Bitboard our_pawns = pos.pieces(us, PAWN) & our_pieces_check;
            while (our_pawns) {
                Square psq = pop_lsb(our_pawns);
                File pf = file_of(psq);
                // White pawns attack diagonally upward, black pawns attack diagonally downward
                Bitboard pb = square_bb(psq);
                if (pf > FILE_A) our_attacks |= us == WHITE ? shift_ne(pb) : shift_se(pb);
                if (pf < FILE_H) our_attacks |= us == WHITE ? shift_nw(pb) : shift_sw(pb);
            }

            Bitboard our_knights = pos.pieces(us, KNIGHT);
            while (our_knights) {
                Square psq = pop_lsb(our_knights);
                our_attacks |= knight_attacks_bb(psq);
            }

            Bitboard our_bishops = pos.pieces(us, BISHOP);
            while (our_bishops) {
                Square psq = pop_lsb(our_bishops);
                our_attacks |= bb_diag_attacks(psq, pos.pieces());
            }

            Bitboard our_rooks = pos.pieces(us, ROOK);
            while (our_rooks) {
                Square psq = pop_lsb(our_rooks);
                our_attacks |= bb_rank_attacks(psq, pos.pieces()) | bb_file_attacks(psq, pos.pieces());
            }

            Bitboard our_queens = pos.pieces(us, QUEEN);
            while (our_queens) {
                Square psq = pop_lsb(our_queens);
                our_attacks |= queen_attacks_bb(psq, pos.pieces());
            }

            Bitboard our_king_bb = pos.pieces(us, KING);
            while (our_king_bb) {
                Square psq = pop_lsb(our_king_bb);
                Bitboard kbb = square_bb(psq);
                our_attacks |= kbb;
                if (file_of(psq) > FILE_A) our_attacks |= shift_w(kbb) | shift_nw(kbb) | shift_sw(kbb);
                if (file_of(psq) < FILE_H) our_attacks |= shift_e(kbb) | shift_ne(kbb) | shift_se(kbb);
                our_attacks |= shift_n(kbb) | shift_s(kbb);
            }

            bool is_defended = (our_attacks & square_bb(sq)) != 0;
            bool is_hanging = !is_defended;

            // Heavy penalty for hanging pieces
            if (is_hanging) {
                mg_score -= sign * (piece_value * 3 / 2);
                eg_score -= sign * (piece_value * 3 / 2);
            } else {
                // Smaller penalty for attacked but defended pieces
                mg_score -= sign * (piece_value / 4);
                eg_score -= sign * (piece_value / 6);
            }
        }
    }

    // Tempo bonus: small advantage for having the move
    // In middle game, tempo is more valuable; in endgame, less so
    mg_score += 15;
    eg_score += 5;

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
