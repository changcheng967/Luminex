#include "luminex.h"

namespace luminex {

// PeSTO piece values (MG and EG)
static constexpr int PieceValueMG[8] = { 82, 337, 365, 477, 1025, 0, 0, 0 };
static constexpr int PieceValueEG[8] = { 94, 281, 297, 512, 936, 0, 0, 0 };

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
        // For each square, check if an adjacent-file friendly pawn can support it
        Bitboard supported_by_adj = pawn_attacks_bb(c, our_pawns);

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
                mg_score += sign * (4 + r * 2);
                eg_score += sign * (2 + r);
            }

            // Backward pawn detection: a pawn that cannot advance safely
            // and has no friendly pawn on adjacent files behind or beside it
            {
                // Squares this pawn could advance to (one or two squares)
                Square push = relative_square(c, make_square(f, Rank(r + 1)));
                // Check if the push square is attacked by enemy pawns
                bool push_attacked = (push < SQUARE_NONE) &&
                    (pawn_attacks_bb(them, their_pawns) & square_bb(push));
                // Check if there's a friendly pawn on adjacent files that can support the push square
                bool has_support = false;
                if (f > FILE_A && file_count[f - 1] > 0) has_support = true;
                if (f < FILE_H && file_count[f + 1] > 0) has_support = true;

                if (push_attacked && !has_support && !left && !right && r >= RANK_2 && r <= RANK_5) {
                    // Weakened backward detection: only flag if also somewhat isolated
                    // Full isolation is already caught above, so check partial isolation
                    bool left_partial = (f > FILE_A && file_count[f - 1] > 0);
                    bool right_partial = (f < FILE_H && file_count[f + 1] > 0);
                    if (!left_partial || !right_partial) {
                        mg_score -= sign * 8;
                        eg_score -= sign * 10;
                    }
                }
            }

            // Passed pawn bonus with blocker and king proximity
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
                int mg_passer = 15 + r * 10;
                int eg_passer = 30 + r * 25;

                // Blocker penalty: enemy pieces blocking the path
                if (ahead_file & pos.pieces(them)) {
                    mg_passer -= 10;
                    eg_passer -= 15;
                }

                // Rook behind passed pawn bonus
                Bitboard behind = file_bb(f) & pos.pieces(c, ROOK);
                if (behind && c == WHITE && rank_of(sq) > rank_of(lsb(behind))) {
                    mg_passer += 15;
                    eg_passer += 20;
                } else if (behind && c == BLACK && rank_of(sq) < rank_of(lsb(behind))) {
                    mg_passer += 15;
                    eg_passer += 20;
                }

                // King proximity (EG only - king becomes relevant in endgame)
                Square our_ksq = ksq_arr[c_idx];
                Square their_ksq = ksq_arr[c_idx ^ 1];
                Square promo_sq = relative_square(c, make_square(f, RANK_8));
                int our_kdist = distance(our_ksq, promo_sq);
                int their_kdist = distance(their_ksq, promo_sq);
                eg_passer += (their_kdist - our_kdist) * 5;

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
            int mob = popcount(attacks & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 3;

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
            int mob = popcount(bb_diag_attacks(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 3;
            eg_score += sign * mob * 4;

            // Bad bishop penalty: bishop hemmed in by own pawns on same color complex
            {
                // Count own pawns on the bishop's color complex
                bool is_light_sq = ((int(sq) + (sq / 8)) % 2) == 0;
                Bitboard same_color_pawns = our_pawns;
                int pawns_on_color = 0;
                while (same_color_pawns) {
                    Square psq = pop_lsb(same_color_pawns);
                    bool psq_light = ((int(psq) + (psq / 8)) % 2) == 0;
                    if (psq_light == is_light_sq) pawns_on_color++;
                }
                if (pawns_on_color >= 3) {
                    mg_score -= sign * 8;
                    eg_score -= sign * 5;
                }
            }
        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (PieceValueMG[ROOK] + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (PieceValueEG[ROOK] + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);

            int mob = popcount(rook_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 4;

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

            int mob = popcount(queen_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob;
            eg_score += sign * mob * 2;
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
            Square front = relative_square(c, make_square(kfile, RANK_2));
            Bitboard shield = square_bb(front);
            if (kfile > FILE_A) shield |= square_bb(relative_square(c, make_square(File(kfile - 1), RANK_2)));
            if (kfile < FILE_H) shield |= square_bb(relative_square(c, make_square(File(kfile + 1), RANK_2)));
            int count = popcount(our_pawns & shield);
            mg_score += sign * count * 15;

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

        // Threat evaluation: count hanging pieces (attacked by enemy, not defended)
        {
            Bitboard our_pieces = pos.pieces(c) ^ pos.pieces(c, PAWN) ^ pos.pieces(c, KING);
            while (our_pieces) {
                Square sq = pop_lsb(our_pieces);
                // Is this piece attacked by enemy?
                Bitboard enemy_attackers = pos.attackers_to(sq) & pos.pieces(them);
                if (enemy_attackers) {
                    // Is it defended by us?
                    Bitboard our_defenders = pos.attackers_to(sq) & pos.pieces(c) & ~square_bb(sq);
                    if (!our_defenders) {
                        // Undefended piece under attack - penalty
                        PieceType pt = piece_type_of(pos.piece_on(sq));
                        int threat_penalty = 0;
                        switch (pt) {
                            case KNIGHT: threat_penalty = 30; break;
                            case BISHOP: threat_penalty = 30; break;
                            case ROOK:   threat_penalty = 40; break;
                            case QUEEN:  threat_penalty = 60; break;
                            default: break;
                        }
                        mg_score -= sign * threat_penalty;
                        eg_score -= sign * (threat_penalty / 2);
                    }
                }
            }
        }

        // Minor pieces threatened by enemy pawns
        {
            Bitboard enemy_pawn_attacks = pawn_attacks_bb(them, their_pawns);
            Bitboard minors = pos.pieces(c, KNIGHT, BISHOP);
            Bitboard threatened = minors & enemy_pawn_attacks;
            while (threatened) {
                pop_lsb(threatened);
                mg_score -= sign * 10;
                eg_score -= sign * 5;
            }
        }

        // Connected rooks bonus: two rooks on the same file or rank
        {
            Bitboard our_rooks = pos.pieces(c, ROOK);
            if (popcount(our_rooks) >= 2) {
                Bitboard r1 = our_rooks;
                while (r1) {
                    Square sq1 = pop_lsb(r1);
                    Bitboard rook_attacks_from_sq1 = rook_attacks_bb(sq1, occupied);
                    // Check if another rook is connected (same rank or file, no pieces between)
                    Bitboard connected = rook_attacks_from_sq1 & (our_rooks ^ square_bb(sq1));
                    if (connected) {
                        mg_score += sign * 15;
                        eg_score += sign * 10;
                    }
                }
            }
        }
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) { mg_score += 60; eg_score += 80; }
    if (bishop_count[BLACK] >= 2) { mg_score -= 60; eg_score -= 80; }

    // King safety using attack maps
    for (int c = 0; c < 2; ++c) {
        Color them = Color(c ^ 1);
        Sign sign = (c == 0) ? 1 : -1;
        Square our_ksq = ksq_arr[c];

        // King zone: 3x3 area
        Bitboard king_zone = king_attacks_bb(our_ksq) | square_bb(our_ksq);

        int attack_units = 0;
        int attacker_count = 0;

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

        // Knight attacks on king zone
        Bitboard enemy_kn = pos.pieces(them, KNIGHT);
        while (enemy_kn) {
            if (knight_attacks_bb(pop_lsb(enemy_kn)) & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
        }

        // Bishop attacks on king zone
        Bitboard enemy_bi = pos.pieces(them, BISHOP);
        while (enemy_bi) {
            if (bishop_attacks_bb(pop_lsb(enemy_bi), occupied) & king_zone) {
                attack_units += 3;
                attacker_count++;
            }
        }

        // Rook attacks on king zone
        Bitboard enemy_ro = pos.pieces(them, ROOK);
        while (enemy_ro) {
            if (rook_attacks_bb(pop_lsb(enemy_ro), occupied) & king_zone) {
                attack_units += 4;
                attacker_count++;
            }
        }

        // Queen attacks on king zone
        Bitboard enemy_qu = pos.pieces(them, QUEEN);
        while (enemy_qu) {
            if (queen_attacks_bb(pop_lsb(enemy_qu), occupied) & king_zone) {
                attack_units += 6;
                attacker_count++;
            }
        }

        // Non-linear danger
        int danger = 0;
        if (attacker_count >= 2) {
            danger = attack_units * attack_units / 2;
        } else if (attacker_count == 1) {
            danger = attack_units;
        }

        // Scale by attacking material
        int attacking_material = popcount(enemy_qu) * 4 + popcount(enemy_ro) * 2;
        if (attacking_material < 4) danger = danger * attacking_material / 4;

        danger = std::min(danger, 400);

        mg_score -= sign * danger;
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
