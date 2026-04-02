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
    // BLACK (mirrored from WHITE - flip vertically: sq ^ 56)
    {
        // PAWN (rank 7 corresponds to white rank 2)
        {0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10,-20,-20, 10, 10,  5,
         5, -5,-10,  0,  0,-10, -5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5,  5, 10, 25, 25, 10,  5,  5,
         10, 10, 20, 30, 30, 20, 10, 10,
         50, 50, 50, 50, 50, 50, 50, 50,
         0, 0, 0, 0, 0, 0, 0, 0
        },
        // KNIGHT
        {
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        },
        // BISHOP
        {
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        },
        // ROOK
        {
            0,  0,  0,  5,  5,  0,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            5, 10, 10, 10, 10, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        },
        // QUEEN
        {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        },
        // KING
        {
            -50,-50,-50,-50,-50,-50,-50,-50,
            -50,-50,-50,-50,-50,-50,-50,-50,
            -40,-40,-40,-50,-50,-40,-40,-40,
            -30,-30,-30,-40,-40,-30,-30,-30,
            -20,-10,-10,-20,-20,-10,-10,-20,
            10, 20, 10,  0,  0, 10, 20, 10,
            40, 50, 30, 20, 20, 30, 50, 40,
            50, 60, 40, 30, 30, 40, 60, 50
        },
        {0},
        {0}
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
    // BLACK (mirrored from WHITE)
    {
        // PAWN
        {
            0,  0,  0,  0,  0,  0,  0,  0,
            10, 10, 10, 10, 10, 10, 10, 10,
            30, 30, 30, 30, 30, 30, 30, 30,
            50, 50, 50, 50, 50, 50, 50, 50,
            70, 70, 70, 70, 70, 70, 70, 70,
            90, 90, 90, 90, 90, 90, 90, 90,
            100, 100, 100, 100, 100, 100, 100, 100,
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
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 15, 15, 15, 15, 10,-10,
            -10, 10, 15, 20, 20, 15, 10,-10,
            -10, 10, 15, 20, 20, 15, 10,-10,
            -10, 10, 15, 15, 15, 15, 10,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        },
        // ROOK
        {
            0,  0,  0,  5,  5,  0,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            5, 10, 10, 10, 10, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        },
        // QUEEN
        {
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        },
        // KING (endgame - mirrored)
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
        {0},
        {0}
    }
};

using Score = Value;

// King danger zone: 3x3 around king expanded by 1 rank
inline Bitboard king_danger_zone(Square ksq) {
    Bitboard zone = 0;
    Bitboard kbb = square_bb(ksq);
    zone |= kbb;
    if (file_of(ksq) > FILE_A) zone |= shift_w(kbb) | shift_nw(kbb) | shift_sw(kbb);
    if (file_of(ksq) < FILE_H) zone |= shift_e(kbb) | shift_ne(kbb) | shift_se(kbb);
    zone |= shift_n(kbb) | shift_s(kbb);
    if (rank_of(ksq) > RANK_1) zone |= shift_s(zone);
    if (rank_of(ksq) < RANK_8) zone |= shift_n(zone);
    return zone;
}

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

            mg_score += sign * (PAWN_VALUE + PST_MG_TABLE[int(c)][int(PAWN)][int(sq)]);
            eg_score += sign * (PAWN_VALUE + PST_EG_TABLE[int(c)][int(PAWN)][int(sq)]);

            // Doubled pawn
            if (file_count[f] > 1) {
                mg_score -= sign * 10;
                eg_score -= sign * 20;
            }

            // Isolated pawn
            bool left = (f > FILE_A && file_count[f - 1] > 0);
            bool right = (f < FILE_H && file_count[f + 1] > 0);
            if (!left && !right) {
                mg_score -= sign * 15;
                eg_score -= sign * 20;
            }

            // Passed pawn
            Bitboard ahead = 0;
            for (int rr = r + 1; rr <= RANK_7; ++rr) {
                Square rsq = relative_square(c, make_square(f, Rank(rr)));
                ahead |= square_bb(rsq);
                if (f > FILE_A) ahead |= square_bb(relative_square(c, make_square(File(f - 1), Rank(rr))));
                if (f < FILE_H) ahead |= square_bb(relative_square(c, make_square(File(f + 1), Rank(rr))));
            }
            if (!(ahead & their_pawns)) {
                mg_score += sign * (15 + r * 8);
                eg_score += sign * (30 + r * 20);
            }
        }

        // Knights
        Bitboard knights = pos.pieces(c, KNIGHT);
        while (knights) {
            Square sq = pop_lsb(knights);
            mg_score += sign * (KNIGHT_VALUE + PST_MG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            eg_score += sign * (KNIGHT_VALUE + PST_EG_TABLE[int(c)][int(KNIGHT)][int(sq)]);
            int mob = popcount(knight_attacks_bb(sq) & ~pos.pieces(c));
            mg_score += sign * mob * 4;
            eg_score += sign * mob * 8;
        }

        // Bishops
        Bitboard bishops = pos.pieces(c, BISHOP);
        bishop_count[c_idx] = popcount(bishops);
        while (bishops) {
            Square sq = pop_lsb(bishops);
            mg_score += sign * (BISHOP_VALUE + PST_MG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            eg_score += sign * (BISHOP_VALUE + PST_EG_TABLE[int(c)][int(BISHOP)][int(sq)]);
            int mob = popcount(bb_diag_attacks(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 5;
            eg_score += sign * mob * 10;
        }

        // Rooks
        Bitboard rooks = pos.pieces(c, ROOK);
        while (rooks) {
            Square sq = pop_lsb(rooks);
            mg_score += sign * (ROOK_VALUE + PST_MG_TABLE[int(c)][int(ROOK)][int(sq)]);
            eg_score += sign * (ROOK_VALUE + PST_EG_TABLE[int(c)][int(ROOK)][int(sq)]);
            int mob = popcount((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 6;

            // Open/semi-open file
            File f = file_of(sq);
            if (!(pos.pieces(PAWN) & file_bb(f))) {
                mg_score += sign * 40;
                eg_score += sign * 40;
            } else if (!(pos.pieces(c, PAWN) & file_bb(f))) {
                mg_score += sign * 15;
                eg_score += sign * 20;
            }
        }

        // Queens
        Bitboard queens = pos.pieces(c, QUEEN);
        while (queens) {
            Square sq = pop_lsb(queens);
            mg_score += sign * (QUEEN_VALUE + PST_MG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            eg_score += sign * (QUEEN_VALUE + PST_EG_TABLE[int(c)][int(QUEEN)][int(sq)]);
            int mob = popcount(queen_attacks_bb(sq, occupied) & ~pos.pieces(c));
            mg_score += sign * mob * 2;
            eg_score += sign * mob * 4;
        }

        // King PST
        Square ksq = king_sq[c_idx];
        mg_score += sign * PST_MG_TABLE[int(c)][int(KING)][int(ksq)];
        eg_score += sign * PST_EG_TABLE[int(c)][int(KING)][int(ksq)];

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

        CastlingRight ks = c == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        CastlingRight qs = c == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        if (!pos.castling_allowed(c, ks) && !pos.castling_allowed(c, qs) && !castled) {
            mg_score -= sign * 25;
            eg_score -= sign * 10;
        }
    }

    // Bishop pair bonus
    if (bishop_count[WHITE] >= 2) { mg_score += 60; eg_score += 80; }
    if (bishop_count[BLACK] >= 2) { mg_score -= 60; eg_score -= 80; }

    // King danger: count pieces attacking enemy king zone
    int king_danger_val[2] = {0, 0};
    int attackers[2] = {0, 0};
    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Color us = Color(c_idx);
        Bitboard dz = king_danger_zone(king_sq[c_idx ^ 1]);

        Bitboard kn = pos.pieces(us, KNIGHT);
        while (kn) { if (knight_attacks_bb(pop_lsb(kn)) & dz) { attackers[c_idx]++; king_danger_val[c_idx] += 4; } }

        Bitboard bi = pos.pieces(us, BISHOP);
        while (bi) { if (bb_diag_attacks(pop_lsb(bi), occupied) & dz) { attackers[c_idx]++; king_danger_val[c_idx] += 5; } }

        Bitboard ro = pos.pieces(us, ROOK);
        while (ro) { Square sq = pop_lsb(ro); if ((bb_rank_attacks(sq, occupied) | bb_file_attacks(sq, occupied)) & dz) { attackers[c_idx]++; king_danger_val[c_idx] += 7; } }

        Bitboard qu = pos.pieces(us, QUEEN);
        while (qu) { if (queen_attacks_bb(pop_lsb(qu), occupied) & dz) { attackers[c_idx]++; king_danger_val[c_idx] += 12; } }

        if (attackers[c_idx] >= 2) king_danger_val[c_idx] += attackers[c_idx] * attackers[c_idx];
    }

    for (int c_idx = 0; c_idx < 2; ++c_idx) {
        Sign sign = (c_idx == 0) ? 1 : -1;
        if (attackers[c_idx] >= 2) mg_score += sign * king_danger_val[c_idx] * 8;
    }

    // Phase calculation
    int phase = popcount(pos.pieces(KNIGHT)) + popcount(pos.pieces(BISHOP))
              + popcount(pos.pieces(ROOK)) * 2 + popcount(pos.pieces(QUEEN)) * 4;
    phase = std::min(24, phase);

    // Endgame king activity
    if (phase <= 8) {
        for (int c_idx = 0; c_idx < 2; ++c_idx) {
            Sign sign = (c_idx == 0) ? 1 : -1;
            Square ksq = king_sq[c_idx];
            File kf = file_of(ksq);
            Rank kr = rank_of(ksq);
            int center_dist = std::min({
                std::max(std::abs(int(kf) - 3), std::abs(int(kr) - 3)),
                std::max(std::abs(int(kf) - 3), std::abs(int(kr) - 4)),
                std::max(std::abs(int(kf) - 4), std::abs(int(kr) - 3)),
                std::max(std::abs(int(kf) - 4), std::abs(int(kr) - 4))
            });
            eg_score += sign * (4 - center_dist) * 15;
        }
    }

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
