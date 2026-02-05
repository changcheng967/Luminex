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
        // KING (middle game - wants safety in corners/castled)
        {
            50, 60, 40, 30, 30, 40, 60, 50,
            40, 50, 30, 20, 20, 30, 50, 40,
            10, 20, 10,  0,  0, 10, 20, 10,
            -20,-10,-10,-20,-20,-10,-10,-20,
            -30,-30,-30,-40,-40,-30,-30,-30,
            -40,-40,-40,-50,-50,-40,-40,-40,
            -50,-50,-50,-50,-50,-50,-50,-50,
            -50,-50,-50,-50,-50,-50,-50,-50
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
        // KING (endgame - wants activity and center)
        {
            -60,-40,-20,-10,-10,-20,-40,-60,
            -40,-20, 10, 20, 20, 10,-20,-40,
            -20, 10, 30, 40, 40, 30, 10,-20,
            -10, 20, 40, 50, 50, 40, 20,-10,
            -10, 20, 40, 50, 50, 40, 20,-10,
            -20, 10, 30, 40, 40, 30, 10,-20,
            -40,-20, 10, 20, 20, 10,-20,-40,
            -60,-40,-20,-10,-10,-20,-40,-60
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

// Helper: evaluate pawn shield for king safety
inline Score evaluate_pawn_shield(const Position& pos, Color c, Square ksq) {
    Score shield_mg = 0;
    Score shield_eg = 0;
    Sign sign = (c == WHITE) ? 1 : -1;

    // Only evaluate pawn shield when king is on ranks 1-3 (white) or 6-8 (black)
    // i.e., when king hasn't castled or castled short/long
    Rank kr = rank_of(ksq);
    bool king_on_back_rank = (c == WHITE && kr <= RANK_3) || (c == BLACK && kr >= RANK_6);

    if (!king_on_back_rank) return 0;

    File kf = file_of(ksq);

    // Get squares in front of king (pawn shield squares)
    Bitboard shield_squares = 0;

    if (c == WHITE) {
        // For white, squares ahead are ranks 2 and 3 in front of king
        if (kr < RANK_8) {
            Square ahead1 = make_square(kf, Rank(kr + 1));
            shield_squares |= square_bb(ahead1);
            if (kf > FILE_A) shield_squares |= square_bb(make_square(File(kf - 1), Rank(kr + 1)));
            if (kf < FILE_H) shield_squares |= square_bb(make_square(File(kf + 1), Rank(kr + 1)));

            if (kr < RANK_7) {
                Square ahead2 = make_square(kf, Rank(kr + 2));
                shield_squares |= square_bb(ahead2);
                if (kf > FILE_A) shield_squares |= square_bb(make_square(File(kf - 1), Rank(kr + 2)));
                if (kf < FILE_H) shield_squares |= square_bb(make_square(File(kf + 1), Rank(kr + 2)));
            }
        }
    } else {
        // For black, squares ahead are ranks 1 and 2 in front (from black's perspective)
        if (kr > RANK_1) {
            Square ahead1 = make_square(kf, Rank(kr - 1));
            shield_squares |= square_bb(ahead1);
            if (kf > FILE_A) shield_squares |= square_bb(make_square(File(kf - 1), Rank(kr - 1)));
            if (kf < FILE_H) shield_squares |= square_bb(make_square(File(kf + 1), Rank(kr - 1)));

            if (kr > RANK_2) {
                Square ahead2 = make_square(kf, Rank(kr - 2));
                shield_squares |= square_bb(ahead2);
                if (kf > FILE_A) shield_squares |= square_bb(make_square(File(kf - 1), Rank(kr - 2)));
                if (kf < FILE_H) shield_squares |= square_bb(make_square(File(kf + 1), Rank(kr - 2)));
            }
        }
    }

    // Count how many shield squares are occupied by our pawns
    Bitboard our_pawns = pos.pieces(c, PAWN);
    Bitboard shield_pawns = shield_squares & our_pawns;
    int pawn_count = popcount(shield_pawns);

    // Bonus for pawn shield
    shield_mg += sign * pawn_count * 15;
    shield_eg += sign * pawn_count * 10;

    // Penalty for missing pawns in front of king
    int missing_shield = 6 - pawn_count;  // At most 6 shield squares
    if (missing_shield > 0) {
        shield_mg -= sign * missing_shield * 10;
        shield_eg -= sign * missing_shield * 5;
    }

    // Penalty for open files near king (no friendly pawn, no enemy pawn)
    Bitboard file_mask = 0;
    if (kf > FILE_A) file_mask |= file_bb(File(kf - 1));
    file_mask |= file_bb(kf);
    if (kf < FILE_H) file_mask |= file_bb(File(kf + 1));

    Bitboard pawns_on_files = our_pawns & file_mask;
    if (pawns_on_files == 0) {
        // No pawns on these files at all - very dangerous
        shield_mg -= sign * 30;
        shield_eg -= sign * 20;
    }

    return shield_mg + shield_eg;
}

Value evaluate(const Position& pos) {
    Score mg_score = 0;
    Score eg_score = 0;

    // Track bishop pairs and king danger
    int bishop_count[2] = {0, 0};
    int king_attackers[2] = {0, 0};
    Square king_sq[2] = {pos.king_sq(WHITE), pos.king_sq(BLACK)};

    // Pre-compute pawn file counts for O(n) instead of O(n²) evaluation
    int pawn_count_by_file[2][8] = {{0}};
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color c = Color(c_idx);
        Bitboard pawns = pos.pieces(c, PAWN);
        while (pawns) {
            Square sq = pop_lsb(pawns);
            File f = file_of(sq);
            pawn_count_by_file[c_idx][f]++;
        }
    }

    // Pre-compute which files have pawns for isolated pawn detection
    bool file_has_pawn[2][8] = {{0}};
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        for (int f = FILE_A; f <= FILE_H; ++f) {
            file_has_pawn[c_idx][f] = (pawn_count_by_file[c_idx][f] > 0);
        }
    }

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

            // Doubled pawn penalty - O(1) lookup
            File f = file_of(sq);
            if (pawn_count_by_file[c_idx][f] > 1) {
                mg_score -= sign * 10;
                eg_score -= sign * 20;
            }

            // Isolated pawn penalty - O(1) lookup
            bool has_left_support = (f > FILE_A && file_has_pawn[c_idx][f - 1]);
            bool has_right_support = (f < FILE_H && file_has_pawn[c_idx][f + 1]);
            if (!has_left_support && !has_right_support) {
                mg_score -= sign * 20;
                eg_score -= sign * 20;
            }

            // Center pawn bonus: pawns on e4/d4 (white) or e5/d5 (black)
            bool is_center_pawn = false;
            if (c == WHITE) {
                if ((sq == E4) || (sq == D4)) is_center_pawn = true;
            } else {
                if ((sq == E5) || (sq == D5)) is_center_pawn = true;
            }

            if (is_center_pawn) {
                mg_score += sign * 50;
                eg_score += sign * 30;

                // Extra bonus if the pawn is protected (not isolated)
                if (has_left_support || has_right_support) {
                    mg_score += sign * 15;
                    eg_score += sign * 10;
                }
            }
        }

        // Advanced passed pawn evaluation - done after all pawns are processed
        Bitboard our_pawns = pos.pieces(c, PAWN);
        Bitboard their_pawns = pos.pieces(Color(c ^ 1), PAWN);
        Bitboard passed_pawns = 0;

        while (our_pawns) {
            Square sq = pop_lsb(our_pawns);
            Rank r = relative_rank(c, sq);
            File f = file_of(sq);

            // Check if this pawn is passed (no enemy pawns ahead on same or adjacent files)
            Bitboard ahead = 0;
            if (r < RANK_7) {
                for (int rr = int(r) + 1; rr <= int(RANK_7); ++rr) {
                    Square rank_sq = relative_square(c, make_square(f, Rank(rr)));
                    ahead |= square_bb(rank_sq);
                    if (f > FILE_A) {
                        Square left_sq = relative_square(c, make_square(File(f - 1), Rank(rr)));
                        ahead |= square_bb(left_sq);
                    }
                    if (f < FILE_H) {
                        Square right_sq = relative_square(c, make_square(File(f + 1), Rank(rr)));
                        ahead |= square_bb(right_sq);
                    }
                }
            }

            // No enemy pawns ahead means this is a passed pawn
            if (!(ahead & their_pawns)) {
                passed_pawns |= square_bb(sq);

                // Base passed pawn bonus increases with rank
                mg_score += sign * (20 + r * 10);
                eg_score += sign * (50 + r * 20);

                // Bonus for protected passed pawn (supported by another pawn)
                Bitboard pawn_attacks = 0;
                Bitboard pb = square_bb(sq);
                if (c == WHITE) {
                    if (f > FILE_A) pawn_attacks |= shift_sw(pb);
                    if (f < FILE_H) pawn_attacks |= shift_se(pb);
                } else {
                    if (f > FILE_A) pawn_attacks |= shift_nw(pb);
                    if (f < FILE_H) pawn_attacks |= shift_ne(pb);
                }
                if (pawn_attacks & pos.pieces(c, PAWN)) {
                    mg_score += sign * 20;
                    eg_score += sign * 30;
                }

                // Outside passed pawn bonus: passed pawns on queenside when opponent has no passed pawns there
                bool is_outside = (f <= FILE_C);
                if (is_outside) {
                    // Check if opponent has any passed pawns on the same side
                    Bitboard their_passed_on_queenside = 0;

                    // Find their passed pawns
                    Bitboard tp = their_pawns;
                    while (tp) {
                        Square tp_sq = pop_lsb(tp);
                        Rank tr = relative_rank(Color(c ^ 1), tp_sq);
                        File tf = file_of(tp_sq);

                        Bitboard t_ahead = 0;
                        if (tr < RANK_7) {
                            for (int trr = int(tr) + 1; trr <= int(RANK_7); ++trr) {
                                Square t_rank_sq = relative_square(Color(c ^ 1), make_square(tf, Rank(trr)));
                                t_ahead |= square_bb(t_rank_sq);
                                if (tf > FILE_A) {
                                    Square t_left_sq = relative_square(Color(c ^ 1), make_square(File(tf - 1), Rank(trr)));
                                    t_ahead |= square_bb(t_left_sq);
                                }
                                if (tf < FILE_H) {
                                    Square t_right_sq = relative_square(Color(c ^ 1), make_square(File(tf + 1), Rank(trr)));
                                    t_ahead |= square_bb(t_right_sq);
                                }
                            }
                        }

                        if (!(t_ahead & pos.pieces(c, PAWN))) {
                            if (tf <= FILE_C) their_passed_on_queenside |= square_bb(tp_sq);
                        }
                    }

                    // If we have outside passed pawn and they don't, big bonus
                    if (is_outside && !their_passed_on_queenside) {
                        eg_score += sign * 50;
                    }
                }

                // Connected passed pawns bonus (two or more passed pawns supporting each other)
                Bitboard adjacent_passed = 0;
                if (f > FILE_A) adjacent_passed |= passed_pawns & file_bb(File(f - 1));
                if (f < FILE_H) adjacent_passed |= passed_pawns & file_bb(File(f + 1));
                if (adjacent_passed) {
                    mg_score += sign * 50;
                    eg_score += sign * 40;
                }

                // Key squares evaluation: bonus for king near passed pawn's promotion path
                Square our_king = king_sq[int(c)];
                Square their_king = king_sq[int(c ^ 1)];

                // Calculate king distance to the passed pawn's file and promotion rank
                int our_king_dist = std::abs(int(file_of(our_king)) - int(f)) + std::abs(int(rank_of(our_king)) - int(r));
                int their_king_dist = std::abs(int(file_of(their_king)) - int(f)) + std::abs(int(rank_of(their_king)) - int(r));

                // Bonus if our king is closer to supporting the passed pawn
                if (our_king_dist < their_king_dist) {
                    eg_score += sign * (their_king_dist - our_king_dist) * 10;
                }
            }
        }

        // Pawn islands: count separate groups of pawns
        // More islands = weaker pawn structure
        int pawn_islands = 0;
        bool in_island = false;
        for (int f = FILE_A; f <= FILE_H; ++f) {
            Bitboard file_pawns = pos.pieces(c, PAWN) & file_bb(File(f));
            if (file_pawns) {
                if (!in_island) {
                    pawn_islands++;
                    in_island = true;
                }
            } else {
                in_island = false;
            }
        }
        // Penalty for having too many pawn islands (more than 3 is bad)
        if (pawn_islands > 3) {
            mg_score -= sign * (pawn_islands - 3) * 15;
            eg_score -= sign * (pawn_islands - 3) * 20;
        }

        // Knights
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * KNIGHT_VALUE;
            eg_score += sign * KNIGHT_VALUE;
            mg_score += sign * PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)];
            eg_score += sign * PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)];

            // Knight mobility - INCREASED for better positional play
            int mobility = popcount(knight_attacks_bb(sq) & ~pos.pieces(c));
            mg_score += sign * mobility * 10;  // Increased from 4
            eg_score += sign * mobility * 20;  // Increased from 8

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

            // Bishop mobility - INCREASED for better positional play
            int mobility = popcount(bb_diag_attacks(sq, pos.pieces()) & ~pos.pieces(c));
            mg_score += sign * mobility * 12;  // Increased from 5
            eg_score += sign * mobility * 25;  // Increased from 10

            // Trapped bishop detection: bishop on starting square blocked by own pawns
            // White: c1 blocked by b2+d2 pawns, f1 blocked by e2+g2 pawns
            // Black: c8 blocked by b7+d7 pawns, f8 blocked by e7+g7 pawns
            Bitboard our_pawns = pos.pieces(c, PAWN);
            if (c == WHITE) {
                if (sq == C1) {
                    // White bishop on c1 trapped if d2 and b2 occupied by white pawns
                    if ((our_pawns & square_bb(D2)) && (our_pawns & square_bb(B2))) {
                        mg_score -= sign * 50;
                        eg_score -= sign * 50;
                    }
                } else if (sq == F1) {
                    // White bishop on f1 trapped if e2 and g2 occupied by white pawns
                    if ((our_pawns & square_bb(E2)) && (our_pawns & square_bb(G2))) {
                        mg_score -= sign * 50;
                        eg_score -= sign * 50;
                    }
                }
            } else {  // BLACK
                if (sq == C8) {
                    // Black bishop on c8 trapped if d7 and b7 occupied by black pawns
                    if ((our_pawns & square_bb(D7)) && (our_pawns & square_bb(B7))) {
                        mg_score -= sign * 50;
                        eg_score -= sign * 50;
                    }
                } else if (sq == F8) {
                    // Black bishop on f8 trapped if e7 and g7 occupied by black pawns
                    if ((our_pawns & square_bb(E7)) && (our_pawns & square_bb(G7))) {
                        mg_score -= sign * 50;
                        eg_score -= sign * 50;
                    }
                }
            }
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

            // Rook mobility - INCREASED for better positional play
            Bitboard occupied = pos.pieces();
            int mobility = popcount((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & ~pos.pieces(c));
            mg_score += sign * mobility * 6;   // Increased from 2
            eg_score += sign * mobility * 20;  // Increased from 8

            // Rook on open file bonus
            File f = file_of(sq);
            Bitboard file_pawns = pos.pieces(PAWN) & file_bb(f);
            if (!file_pawns) {
                // Open file - very valuable
                mg_score += sign * 50;
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

            // Rook on enemy king rank bonus - very valuable for cutting off king
            Color them = Color(c ^ 1);
            Square enemy_king = pos.king_sq(them);
            Rank rook_rank = rank_of(sq);
            Rank enemy_king_rank = rank_of(enemy_king);
            if (rook_rank == enemy_king_rank) {
                mg_score += sign * 20;
                eg_score += sign * 40;  // Even more valuable in endgame
            }

            // Rook on 7th rank bonus - reduced MG, increased EG (endgame monsters)
            Rank r = rank_of(sq);
            if ((c == WHITE && r == RANK_7) || (c == BLACK && r == RANK_2)) {
                mg_score += sign * 20;  // Reduced from 50 (less critical in middlegame)
                eg_score += sign * 60;  // Increased from 30 (devastating in endgame)
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

            // Queen mobility - INCREASED for better positional play
            Bitboard occupied = pos.pieces();
            int mobility = popcount(queen_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mobility * 20;  // Increased from 8
            eg_score += sign * mobility * 40;  // Increased from 15
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

        // Penalty for advanced pawns near king (weakening king safety)
        // Especially bad in opening: pawns like g4 for white or g5 for black
        if (pos.game_ply() < 40) {  // Opening phase
            Bitboard king_side_pawns = pos.pieces(c, PAWN);
            int advanced_pawn_penalty = 0;

            while (king_side_pawns) {
                Square psq = pop_lsb(king_side_pawns);
                Rank pr = rank_of(psq);
                File pf = file_of(psq);

                // Check if pawn is advanced near the king
                bool near_king = (pf >= kfile - 2 && pf <= kfile + 2);

                if (near_king) {
                    // White penalty: pawns on rank 3+ in front of king
                    // Black penalty: pawns on rank 6- in front of king
                    bool advanced = false;
                    if (c == WHITE && pr >= RANK_3) advanced = true;
                    if (c == BLACK && pr <= RANK_6) advanced = true;

                    if (advanced) {
                        // Extra penalty for f/g/h pawns (kingside)
                        if (pf >= FILE_F) {
                            advanced_pawn_penalty += 40;  // Large penalty for kingside pawn advances
                        } else {
                            advanced_pawn_penalty += 20;  // Smaller penalty for central pawn advances
                        }
                    }
                }
            }

            mg_score -= sign * advanced_pawn_penalty;
        }
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) {
        mg_score += 75;
        eg_score += 100;
    }
    if (bishop_count[BLACK] >= 2) {
        mg_score -= 75;
        eg_score -= 100;
    }

    // Bishop vs Knight imbalance: bishops are better in open positions
    // Count the total number of pawns to determine how open the position is
    int total_pawns = popcount(pos.pieces(PAWN));
    int white_bishops = bishop_count[WHITE];
    int black_bishops = bishop_count[BLACK];
    int white_knights = popcount(pos.pieces(WHITE, KNIGHT));
    int black_knights = popcount(pos.pieces(BLACK, KNIGHT));

    // In open positions (few pawns), bishops are generally better than knights
    // In closed positions (many pawns), knights are generally better than bishops
    int open_position_bonus = 0;
    if (total_pawns <= 8) {
        // Very open position - bishops are much better
        open_position_bonus = 30;
    } else if (total_pawns <= 12) {
        // Moderately open - bishops are slightly better
        open_position_bonus = 15;
    } else if (total_pawns >= 20) {
        // Very closed position - knights are better
        open_position_bonus = -20;
    }

    // Apply the bonus/penalty based on piece imbalance
    // White bishop advantage
    if (white_bishops > black_bishops && white_knights < black_knights) {
        mg_score += open_position_bonus;
        eg_score += open_position_bonus * 2;  // Endgame advantage is larger
    }
    // Black bishop advantage
    if (black_bishops > white_bishops && black_knights < white_knights) {
        mg_score -= open_position_bonus;
        eg_score -= open_position_bonus * 2;
    }

    // Space evaluation: count squares we control in enemy territory
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Sign sign = (us == WHITE) ? 1 : -1;

        // Define the space area: ranks 2-5 for white, 3-6 for black
        Bitboard space_area = 0;
        for (int r = int(RANK_2); r <= int(RANK_5); ++r) {
            for (int f = int(FILE_C); f <= int(FILE_F); ++f) {
                Square sq = make_square(File(f), Rank(r));
                if (us == BLACK) {
                    // Mirror for black
                    sq = Square(sq ^ 56);  // Mirror the square
                }
                space_area |= square_bb(sq);
            }
        }

        // Count controlled squares in space area
        int controlled_squares = 0;

        // Pawn attacks
        Bitboard our_pawns = pos.pieces(us, PAWN);
        Bitboard pawn_attacks = 0;
        while (our_pawns) {
            Square psq = pop_lsb(our_pawns);
            Bitboard pb = square_bb(psq);
            if (us == WHITE) {
                if (file_of(psq) > FILE_A) pawn_attacks |= shift_nw(pb);
                if (file_of(psq) < FILE_H) pawn_attacks |= shift_ne(pb);
            } else {
                if (file_of(psq) > FILE_A) pawn_attacks |= shift_sw(pb);
                if (file_of(psq) < FILE_H) pawn_attacks |= shift_se(pb);
            }
        }

        // Knight attacks
        Bitboard our_knights = pos.pieces(us, KNIGHT);
        Bitboard knight_attacks = 0;
        while (our_knights) {
            Square ksq = pop_lsb(our_knights);
            knight_attacks |= knight_attacks_bb(ksq);
        }

        // Total controlled squares in space area
        Bitboard our_control = (pawn_attacks | knight_attacks) & space_area;

        // Subtract squares occupied by enemy pawns (they block our control)
        our_control &= ~pos.pieces(them, PAWN);

        controlled_squares = popcount(our_control);

        // Bonus for space advantage
        if (controlled_squares >= 5) {
            mg_score += sign * 50;
            eg_score += sign * 10;
        } else if (controlled_squares >= 3) {
            mg_score += sign * 15;
            eg_score += sign * 5;
        }
    }

    // Center control bonus: extra value for controlling center squares (d4, d5, e4, e5)
    int game_ply = pos.game_ply();  // Get once for use in multiple sections

    Bitboard center_squares = 0;
    center_squares |= square_bb(D4);
    center_squares |= square_bb(D5);
    center_squares |= square_bb(E4);
    center_squares |= square_bb(E5);

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Sign sign = (us == WHITE) ? 1 : -1;

        Bitboard our_center_control = 0;

        // Check pawn attacks on center
        Bitboard our_pawns = pos.pieces(us, PAWN);
        while (our_pawns) {
            Square psq = pop_lsb(our_pawns);
            Bitboard pb = square_bb(psq);
            if (us == WHITE) {
                if (file_of(psq) > FILE_A) our_center_control |= shift_nw(pb);
                if (file_of(psq) < FILE_H) our_center_control |= shift_ne(pb);
            } else {
                if (file_of(psq) > FILE_A) our_center_control |= shift_sw(pb);
                if (file_of(psq) < FILE_H) our_center_control |= shift_se(pb);
            }
        }

        // Check knight attacks on center
        Bitboard our_knights = pos.pieces(us, KNIGHT);
        while (our_knights) {
            Square ksq = pop_lsb(our_knights);
            our_center_control |= knight_attacks_bb(ksq);
        }

        // Check if we have pieces on center squares
        Bitboard our_pieces_on_center = pos.pieces(us) & center_squares;

        int center_control_count = popcount(our_center_control & center_squares);
        int center_occupation_count = popcount(our_pieces_on_center);

        mg_score += sign * (center_control_count * 10 + center_occupation_count * 20);
        eg_score += sign * (center_control_count * 5 + center_occupation_count * 10);

        // Penalty for not having center pawns (d4/e4 for white, d5/e5 for black) in opening
        // But allow variety - reduce penalty at very start to avoid always playing d4/e4 immediately
        if (game_ply < 20) {  // Before move 10
            Bitboard our_pawns = pos.pieces(us, PAWN);
            bool has_d4 = (us == WHITE && (our_pawns & square_bb(D4))) ||
                           (us == BLACK && (our_pawns & square_bb(D5)));
            bool has_e4 = (us == WHITE && (our_pawns & square_bb(E4))) ||
                           (us == BLACK && (our_pawns & square_bb(E5)));

            int center_pawns = (has_d4 ? 1 : 0) + (has_e4 ? 1 : 0);

            // Penalty for missing center pawns - but reduced at game start for variety
            int missing = 2 - center_pawns;
            if (missing > 0) {
                // Reduced penalty at start (game_ply < 6) to allow opening variety
                // Full penalty kicks in after move 3
                int base_penalty = (game_ply < 6) ? 40 : 120;
                int scaling_penalty = (game_ply < 6) ? 10 : 30;

                mg_score -= sign * missing * (base_penalty + game_ply * scaling_penalty);
                eg_score -= sign * missing * ((base_penalty / 2) + game_ply * (scaling_penalty / 2));
            }

            // Penalty for early e-pawn advance before castling - opens e-file against king
            // This is a critical opening principle: don't open the e-file early if king is in center
            Square ksq = pos.king_sq(us);
            Rank krank = rank_of(ksq);
            File kfile = file_of(ksq);

            // Check if king has castled or is on safe back rank square
            bool king_safe = (us == WHITE && krank == RANK_1 && (kfile == FILE_G || kfile == FILE_C)) ||
                             (us == BLACK && krank == RANK_8 && (kfile == FILE_G || kfile == FILE_C));

            // Also consider king still in center as unsafe
            bool king_in_center = (us == WHITE && krank >= RANK_1 && krank <= RANK_2) ||
                                  (us == BLACK && krank >= RANK_7 && krank <= RANK_8);

            if (!king_safe && king_in_center) {
                // Check if e-pawn has moved from starting square
                bool e_pawn_advanced = false;
                if (us == WHITE) {
                    // White e-pawn starts on e2
                    e_pawn_advanced = !(our_pawns & square_bb(E2)) && (our_pawns & (BB_RANK_3 | BB_RANK_4 | BB_RANK_5));
                } else {
                    // Black e-pawn starts on e7
                    e_pawn_advanced = !(our_pawns & square_bb(E7)) && (our_pawns & (BB_RANK_3 | BB_RANK_4 | BB_RANK_5 | BB_RANK_6));
                }

                if (e_pawn_advanced) {
                    // Penalty for opening e-file before king is safe
                    // This is VERY dangerous - classic blunder
                    int e_file_penalty = 150 + (game_ply * 20);
                    mg_score -= sign * e_file_penalty;
                    eg_score -= sign * (e_file_penalty / 2);
                }
            }
        }
    }

    // Development and opening principles evaluation
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Sign sign = (us == WHITE) ? 1 : -1;

        // Knights on rim penalty (a/h and b/g files are suboptimal in opening)
        Bitboard our_knights = pos.pieces(us, KNIGHT);
        while (our_knights) {
            Square sq = pop_lsb(our_knights);
            File f = file_of(sq);
            Rank r = rank_of(sq);

            // Heavy penalty for a/h files (rim)
            if (f == FILE_A || f == FILE_H) {
                mg_score -= sign * 30;
                eg_score -= sign * 15;
            }
            // Lighter penalty for b/g files - discouraged in opening/middlegame
            else if (f == FILE_B || f == FILE_G) {
                // Strong penalty in early game when pieces should go to center
                if (game_ply < 20) {  // Before move 10
                    mg_score -= sign * 50;
                    eg_score -= sign * 30;
                } else if (game_ply < 40) {  // Before move 20
                    mg_score -= sign * 30;
                    eg_score -= sign * 15;
                }
            }

            // Bonus for knights on center files (d/e) in center ranks
            if ((f == FILE_D || f == FILE_E) && r >= RANK_3 && r <= RANK_6) {
                mg_score += sign * 15;
                eg_score += sign * 10;
            }
        }

        // Bishop development penalty: bishops still on starting squares after move 6
        if (game_ply > 12) {  // After move 6 (12 plies)
            Bitboard our_bishops = pos.pieces(us, BISHOP);
            while (our_bishops) {
                Square sq = pop_lsb(our_bishops);
                Rank r = rank_of(sq);
                File f = file_of(sq);

                // Check if bishop is on starting square
                bool on_start = false;
                if (us == WHITE && r == RANK_1) {
                    if ((f == FILE_C) || (f == FILE_F)) on_start = true;
                } else if (us == BLACK && r == RANK_8) {
                    if ((f == FILE_C) || (f == FILE_F)) on_start = true;
                }

                if (on_start) {
                    mg_score -= sign * 15;
                    eg_score -= sign * 10;
                }
            }
        }

        // Undeveloped minor piece penalty after move 10
        if (game_ply > 20) {  // After move 10 (20 plies)
            int undeveloped_minors = 0;

            // Check knights on back rank
            our_knights = pos.pieces(us, KNIGHT);
            while (our_knights) {
                Square sq = pop_lsb(our_knights);
                Rank r = rank_of(sq);
                if ((us == WHITE && r == RANK_1) || (us == BLACK && r == RANK_8)) {
                    undeveloped_minors++;
                }
            }

            // Check bishops on back rank
            Bitboard our_bishops = pos.pieces(us, BISHOP);
            while (our_bishops) {
                Square sq = pop_lsb(our_bishops);
                Rank r = rank_of(sq);
                if ((us == WHITE && r == RANK_1) || (us == BLACK && r == RANK_8)) {
                    undeveloped_minors++;
                }
            }

            if (undeveloped_minors > 0) {
                mg_score -= sign * undeveloped_minors * 20;
                eg_score -= sign * undeveloped_minors * 10;
            }
        }

        // Early queen development penalty - critical opening principle
        // Developing queen early loses tempo and exposes it to attack
        if (game_ply < 20) {  // Before move 10
            Bitboard our_queens = pos.pieces(us, QUEEN);
            while (our_queens) {
                Square qsq = pop_lsb(our_queens);
                Rank qrank = rank_of(qsq);

                // Check if queen has left starting square (developed)
                // White queen starts on d8 (which is d1 in relative terms, square index 3)
                // Black queen starts on d0 (which is d8 in relative terms, square index 59)
                bool queen_developed = false;
                if (us == WHITE && qsq != D1) queen_developed = true;
                if (us == BLACK && qsq != D8) queen_developed = true;

                if (queen_developed) {
                    // Count undeveloped minor pieces
                    int undeveloped_minors_for_queen = 0;

                    // White knights start on b1(g1) = squares 1, 6
                    // Black knights start on b8(g8) = squares 57, 62
                    Bitboard our_knights = pos.pieces(us, KNIGHT);
                    while (our_knights) {
                        Square nsq = pop_lsb(our_knights);
                        bool knight_on_start = false;
                        if (us == WHITE) {
                            if (nsq == B1 || nsq == G1) knight_on_start = true;
                        } else {
                            if (nsq == B8 || nsq == G8) knight_on_start = true;
                        }
                        if (knight_on_start) undeveloped_minors_for_queen++;
                    }

                    // White bishops start on c1(f1) = squares 2, 5
                    // Black bishops start on c8(f8) = squares 58, 61
                    Bitboard our_bishops = pos.pieces(us, BISHOP);
                    while (our_bishops) {
                        Square bsq = pop_lsb(our_bishops);
                        bool bishop_on_start = false;
                        if (us == WHITE) {
                            if (bsq == C1 || bsq == F1) bishop_on_start = true;
                        } else {
                            if (bsq == C8 || bsq == F8) bishop_on_start = true;
                        }
                        if (bishop_on_start) undeveloped_minors_for_queen++;
                    }

                    // Heavy penalty if queen developed before minor pieces
                    // This is a classic opening mistake
                    if (undeveloped_minors_for_queen >= 2) {
                        int queen_penalty = 200;  // Very strong penalty
                        // Scale by how early in the game
                        int early_bonus = (20 - game_ply) * 10;
                        mg_score -= sign * (queen_penalty + early_bonus);
                        eg_score -= sign * (queen_penalty / 2 + early_bonus / 2);
                    } else if (undeveloped_minors_for_queen == 1) {
                        // Still bad, but less severe
                        mg_score -= sign * 100;
                        eg_score -= sign * 50;
                    }

                    // Extra penalty for queen on d3/d6 (common but usually bad early)
                    File qfile = file_of(qsq);
                    if ((us == WHITE && qrank == RANK_3 && qfile == FILE_D) ||
                        (us == BLACK && qrank == RANK_6 && qfile == FILE_D)) {
                        mg_score -= sign * 80;
                        eg_score -= sign * 40;
                    }
                }
            }
        }

        // Castling bonus/penalty
        Square ksq = king_sq[int(us)];
        Rank krank = rank_of(ksq);
        File kfile = file_of(ksq);

        // Castled king bonus (king on g1/g8 or c1/c8 with rook nearby)
        bool has_castled = false;
        if (us == WHITE) {
            if ((krank == RANK_1) && ((kfile == FILE_G) || (kfile == FILE_C))) {
                has_castled = true;
            }
        } else {
            if ((krank == RANK_8) && ((kfile == FILE_G) || (kfile == FILE_C))) {
                has_castled = true;
            }
        }

        if (has_castled) {
            mg_score += sign * 50;
            eg_score += sign * 30;
        }

        // Lost castling rights without castling penalty
        CastlingRight kingside = us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight queenside = us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;

        bool can_kingside = pos.castling_allowed(us, kingside);
        bool can_queenside = pos.castling_allowed(us, queenside);

        if (!can_kingside && !can_queenside && !has_castled) {
            // Lost all castling rights but didn't castle
            mg_score -= sign * 40;
            eg_score -= sign * 20;
        }

        // King in center penalty - apply earlier and stronger
        if (game_ply > 4) {  // After just 2 moves - apply earlier!
            bool king_in_center = (us == WHITE && krank >= RANK_3 && krank <= RANK_5) ||
                                  (us == BLACK && krank >= RANK_4 && krank <= RANK_6);

            if (king_in_center) {
                // King in center is VERY dangerous - especially early
                // Massive penalty that scales with how early it is
                int base_penalty = 200;  // Increased from 120
                int early_bonus = (40 - game_ply) * 10;  // More penalty for early king moves
                int extra_penalty = (game_ply - 10) * 5;  // Grows as game progresses
                mg_score -= sign * (base_penalty + early_bonus + extra_penalty);
                eg_score -= sign * (100 + extra_penalty / 2);

                // Extra penalty when enemy queen/rook is in our half
                Color them = Color(us ^ 1);
                Bitboard their_queens = pos.pieces(them, QUEEN);
                Bitboard their_rooks = pos.pieces(them, ROOK);

                // Check if enemy queen is in our half
                if (us == WHITE) {
                    // Our half is ranks 1-4
                    if (their_queens & (BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4)) {
                        mg_score -= sign * 60;  // Enemy queen in our half = very bad
                    }
                    if (their_rooks & (BB_RANK_1 | BB_RANK_2 | BB_RANK_3 | BB_RANK_4)) {
                        mg_score -= sign * 30;  // Enemy rooks in our half = bad
                    }
                } else {
                    // Our half is ranks 5-8
                    if (their_queens & (BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8)) {
                        mg_score -= sign * 60;
                    }
                    if (their_rooks & (BB_RANK_5 | BB_RANK_6 | BB_RANK_7 | BB_RANK_8)) {
                        mg_score -= sign * 30;
                    }
                }

                // Count open files near king (existing code)
                Bitboard all_pawns = pos.pieces(PAWN);
                int open_files_near_king = 0;

                if (kfile > FILE_A && !(all_pawns & file_bb(File(kfile - 1)))) {
                    open_files_near_king++;
                }
                if (!(all_pawns & file_bb(kfile))) {
                    open_files_near_king++;
                }
                if (kfile < FILE_H && !(all_pawns & file_bb(File(kfile + 1)))) {
                    open_files_near_king++;
                }

                mg_score -= sign * open_files_near_king * 40;
                eg_score -= sign * open_files_near_king * 20;
            }
        }

        // Unsafe king squares: d7/e7 for Black, d2/e2 for White
        // These squares expose the king to central files without castling safety
        // Also applies to d8/e8 for Black (king hasn't moved from back rank but is blocking)
        if (!has_castled && game_ply > 4) {  // After move 2 - earlier to catch Kd7 issues
            bool on_unsafe_square = false;
            bool king_moved_from_back = false;

            if (us == WHITE) {
                // d2 or e2 is dangerous - king blocks center pawns
                if ((krank == RANK_2) && ((kfile == FILE_D) || (kfile == FILE_E))) {
                    on_unsafe_square = true;
                }
                // King moved from e1 (hasn't castled but moved)
                if (krank == RANK_1 && kfile != FILE_E) {
                    king_moved_from_back = true;
                }
            } else {
                // d7 or e7 is dangerous - same issue for Black
                if ((krank == RANK_7) && ((kfile == FILE_D) || (kfile == FILE_E))) {
                    on_unsafe_square = true;
                }
                // King moved from e8 (hasn't castled but moved)
                // This catches Kd7 which is especially dangerous
                if (krank != RANK_8) {
                    king_moved_from_back = true;
                }
            }

            if (on_unsafe_square || king_moved_from_back) {
                // Check if enemy has developed pieces that can attack
                Color them = Color(us ^ 1);
                bool enemy_has_queen = pos.pieces(them, QUEEN) != 0;
                bool enemy_has_rook = pos.pieces(them, ROOK) != 0;
                bool enemy_has_bishop = pos.pieces(them, BISHOP) != 0;

                int unsafe_penalty = 100;  // Base penalty
                if (king_moved_from_back) unsafe_penalty += 50;  // Extra for moved king
                if (enemy_has_queen) unsafe_penalty += 80;
                if (enemy_has_rook) unsafe_penalty += 40;
                if (enemy_has_bishop) unsafe_penalty += 20;

                // Extra penalty when game is still very young
                if (game_ply < 10) unsafe_penalty += 50;

                mg_score -= sign * unsafe_penalty;
                eg_score -= sign * (unsafe_penalty / 2);
            }
        }
    }

    // King danger and queen tropism evaluation
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Square their_king = king_sq[int(them)];
        Square our_king = king_sq[int(us)];

        // Pawn shield evaluation for our king
        Score shield_bonus = evaluate_pawn_shield(pos, us, our_king);
        mg_score += shield_bonus;
        eg_score += shield_bonus;

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

        // King danger: count attackers and their weight - INCREASED FOR BETTER KING SAFETY
        Bitboard danger_zone = king_danger_zone(their_king);
        int danger = 0;

        // Count pieces attacking king zone
        Bitboard our_knights = pos.pieces(us, KNIGHT);
        while (our_knights) {
            Square sq = pop_lsb(our_knights);
            if (knight_attacks_bb(sq) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 80;  // Knight attacks (increased from 40)
            }
        }

        Bitboard our_bishops = pos.pieces(us, BISHOP);
        while (our_bishops) {
            Square sq = pop_lsb(our_bishops);
            if (bb_diag_attacks(sq, pos.pieces()) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 100;  // Bishop attacks (increased from 50)
            }
        }

        Bitboard our_rooks = pos.pieces(us, ROOK);
        while (our_rooks) {
            Square sq = pop_lsb(our_rooks);
            Bitboard attacks = (bb_rank_attacks(sq, pos.pieces()) | bb_file_attacks(sq, pos.pieces()));
            if (attacks & danger_zone) {
                king_attackers[int(us)]++;
                danger += 140;  // Rook attacks are dangerous (increased from 70)
            }
        }

        Bitboard our_queens = pos.pieces(us, QUEEN);
        while (our_queens) {
            Square sq = pop_lsb(our_queens);
            if (queen_attacks_bb(sq, pos.pieces()) & danger_zone) {
                king_attackers[int(us)]++;
                danger += 250;  // Queen attacks are very dangerous (increased from 120)
            }
        }

        // Bonus for multiple attackers (coordination)
        if (king_attackers[int(us)] >= 2) {
            danger += (king_attackers[int(us)] - 1) * 60;  // Increased from 30
        }

        // Penalty when we have no safe king position (king in center) - INCREASED
        Rank krank = rank_of(our_king);
        if ((us == WHITE && krank >= RANK_3 && krank <= RANK_5) ||
            (us == BLACK && krank >= RANK_4 && krank <= RANK_6)) {
            // King in center - very dangerous in middle game
            mg_score -= (us == WHITE ? 1 : -1) * 80;  // Increased from 40
            // Even worse when under attack
            if (king_attackers[int(them)] > 0) {
                mg_score -= (us == WHITE ? 1 : -1) * 150 * king_attackers[int(them)];  // Increased from 60
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

    // Endgame evaluation: king activity becomes important when pieces are few
    if (phase <= 8) {  // Late endgame
        for (int c_idx = 0; c_idx < 2; ++c_idx) {
            Color us = Color(c_idx);
            Sign sign = (us == WHITE) ? 1 : -1;
            Square ksq = king_sq[int(us)];

            // Centralized king is good in endgame
            File kf = file_of(ksq);
            Rank kr = rank_of(ksq);

            // Bonus for king in center (files D-F, ranks 3-6)
            bool king_centered = (kf >= FILE_D && kf <= FILE_F && kr >= RANK_3 && kr <= RANK_6);
            if (king_centered) {
                eg_score += sign * 30;
            }

            // Distance of king to center squares (d4, e4, d5, e5)
            int center_dist = 10;
            Bitboard center = square_bb(D4) | square_bb(E4) | square_bb(D5) | square_bb(E5);
            while (center) {
                Square csq = pop_lsb(center);
                File cf = file_of(csq);
                Rank cr = rank_of(csq);
                int dist = std::max(int(std::abs(int(kf) - int(cf))), int(std::abs(int(kr) - int(cr))));
                center_dist = std::min(center_dist, dist);
            }
            eg_score += sign * (4 - center_dist) * 10;

            // King's distance to enemy pawns (closer is better for promoting)
            Bitboard enemy_pawns = pos.pieces(Color(us ^ 1), PAWN);
            if (enemy_pawns) {
                int min_pawn_dist = 10;
                while (enemy_pawns) {
                    Square psq = pop_lsb(enemy_pawns);
                    File pf = file_of(psq);
                    Rank pr = rank_of(psq);
                    int dist = std::max(int(std::abs(int(kf) - int(pf))), int(std::abs(int(kr) - int(pr))));
                    min_pawn_dist = std::min(min_pawn_dist, dist);
                }
                eg_score += sign * (8 - min_pawn_dist) * 5;
            }
        }

        // Opposition: if kings face each other with one square between and it's our turn, we have advantage
        Square white_king = king_sq[int(WHITE)];
        Square black_king = king_sq[int(BLACK)];
        int king_file_dist = std::abs(int(file_of(white_king)) - int(file_of(black_king)));
        int king_rank_dist = std::abs(int(rank_of(white_king)) - int(rank_of(black_king)));

        if (king_file_dist <= 1 && king_rank_dist == 1) {
            // Kings are in opposition - bonus for the side that's not in check
            if (!pos.is_check()) {
                // The side to move gains from opposition
                mg_score += pos.side_to_move() == WHITE ? 20 : -20;
            }
        }
    }

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

    // Simplified threat detection - just check king proximity for tactical awareness
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Sign sign = (us == WHITE) ? 1 : -1;

        Square our_king = pos.king_sq(us);

        // Penalty if our king is close to enemy pieces (danger)
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, PAWN);
        int danger = 0;
        while (their_pieces) {
            Square sq = pop_lsb(their_pieces);
            int d = distance(our_king, sq);
            if (d <= 2) danger += (3 - d) * 20;
        }
        mg_score -= sign * danger;
        eg_score -= sign * danger;
    }

    // Connected rooks bonus: two rooks on the same rank or file without pieces between them
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Sign sign = (us == WHITE) ? 1 : -1;

        Bitboard our_rooks = pos.pieces(us, ROOK);
        if (popcount(our_rooks) >= 2) {
            Square rook_squares[2];
            int idx = 0;
            Bitboard tmp = our_rooks;
            while (tmp && idx < 2) {
                rook_squares[idx++] = pop_lsb(tmp);
            }

            Square r1 = rook_squares[0];
            Square r2 = rook_squares[1];

            // Check if rooks are connected (same rank or file with no pieces between)
            bool connected = false;
            if (rank_of(r1) == rank_of(r2)) {
                // Same rank - check if squares between are empty
                Bitboard between = 0;
                for (int s = int(std::min(int(r1), int(r2))) + 1; s < int(std::max(int(r1), int(r2))); ++s) {
                    between |= square_bb(Square(s));
                }
                if (!(pos.pieces() & between)) {
                    connected = true;
                }
            } else if (file_of(r1) == file_of(r2)) {
                // Same file - check if squares between are empty
                Bitboard between = 0;
                int f1 = int(std::min(int(r1), int(r2)));
                int f2 = int(std::max(int(r1), int(r2)));
                for (int r = f1 + 8; r < f2; r += 8) {
                    between |= square_bb(Square(r));
                }
                if (!(pos.pieces() & between)) {
                    connected = true;
                }
            }

            if (connected) {
                mg_score += sign * 30;  // Significant bonus in middlegame
                eg_score += sign * 30;
            }
        }
    }

    // Knight outpost bonus: knight on square where it can't be attacked by enemy pawns
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Color them = Color(us ^ 1);
        Sign sign = (us == WHITE) ? 1 : -1;

        Bitboard our_knights = pos.pieces(us, KNIGHT);
        Bitboard their_pawns = pos.pieces(them, PAWN);

        while (our_knights) {
            Square knight_sq = pop_lsb(our_knights);

            // Check if knight is in enemy territory (rank 4-6 for white, 2-4 for black)
            bool in_enemy_territory = (us == WHITE && rank_of(knight_sq) >= RANK_4 && rank_of(knight_sq) <= RANK_6) ||
                                      (us == BLACK && rank_of(knight_sq) >= RANK_3 && rank_of(knight_sq) <= RANK_5);

            if (!in_enemy_territory) continue;

            // Check if enemy pawns can attack this square
            Bitboard pawn_attacks = 0;
            Bitboard tmp = their_pawns;
            while (tmp) {
                Square pawn_sq = pop_lsb(tmp);
                Bitboard pb = square_bb(pawn_sq);
                if (them == WHITE) {
                    if (file_of(pawn_sq) > FILE_A) pawn_attacks |= shift_nw(pb);
                    if (file_of(pawn_sq) < FILE_H) pawn_attacks |= shift_ne(pb);
                } else {
                    if (file_of(pawn_sq) > FILE_A) pawn_attacks |= shift_sw(pb);
                    if (file_of(pawn_sq) < FILE_H) pawn_attacks |= shift_se(pb);
                }
            }

            if (!(pawn_attacks & square_bb(knight_sq))) {
                // Knight is safe from pawn attacks - it's an outpost
                // Extra bonus if supported by our pawn
                bool supported = false;
                Bitboard our_pawns = pos.pieces(us, PAWN);
                Bitboard pawn_support = 0;
                tmp = our_pawns;
                while (tmp) {
                    Square pawn_sq = pop_lsb(tmp);
                    Bitboard pb = square_bb(pawn_sq);
                    if (us == WHITE) {
                        if (file_of(pawn_sq) > FILE_A) pawn_support |= shift_nw(pb);
                        if (file_of(pawn_sq) < FILE_H) pawn_support |= shift_ne(pb);
                    } else {
                        if (file_of(pawn_sq) > FILE_A) pawn_support |= shift_sw(pb);
                        if (file_of(pawn_sq) < FILE_H) pawn_support |= shift_se(pb);
                    }
                }

                if (pawn_support & square_bb(knight_sq)) {
                    supported = true;
                }

                // Outpost bonus - higher if supported
                mg_score += sign * (supported ? 40 : 25);
                eg_score += sign * (supported ? 30 : 15);
            }
        }
    }

    // Tempo bonus: small advantage for having the move
    // In middle game, tempo is more valuable; in endgame, less so
    mg_score += 15;
    eg_score += 5;

    // Material imbalance penalty: losing material in opening is very bad
    int current_ply = pos.game_ply();
    if (current_ply < 30) {
        // Count material value for each side (excluding pawns)
        int white_material = 0;
        int black_material = 0;

        Bitboard white_pieces = pos.pieces(WHITE);
        while (white_pieces) {
            Square sq = pop_lsb(white_pieces);
            PieceType pt = pos.piece_type_on(sq);
            if (pt == KNIGHT) white_material += 300;
            else if (pt == BISHOP) white_material += 320;
            else if (pt == ROOK) white_material += 500;
            else if (pt == QUEEN) white_material += 900;
        }

        Bitboard black_pieces = pos.pieces(BLACK);
        while (black_pieces) {
            Square sq = pop_lsb(black_pieces);
            PieceType pt = pos.piece_type_on(sq);
            if (pt == KNIGHT) black_material += 300;
            else if (pt == BISHOP) black_material += 320;
            else if (pt == ROOK) black_material += 500;
            else if (pt == QUEEN) black_material += 900;
        }

        // Large penalty for being down material in opening
        int material_diff = white_material - black_material;
        if (material_diff < 0) {
            // White is down - penalize heavily
            int penalty = -material_diff * (30 - current_ply) / 10;
            mg_score -= penalty * 2;
            eg_score -= penalty;
        } else if (material_diff > 0) {
            // White is up - bonus
            int bonus = material_diff * (30 - current_ply) / 15;
            mg_score += bonus;
            eg_score += bonus / 2;
        }
    }

    // Interpolate between middle game and endgame
    Score score = (mg_score * phase + eg_score * (24 - phase)) / 24;

    // Apply contempt from the root player's perspective
    // This is passed through the search params and affects draw decisions
    // Positive contempt = avoid draws, play for wins
    // The actual contempt adjustment is applied at search level
    // This is just a placeholder for where contempt could be evaluated

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
