#include "luminex.h"

namespace luminex {

// Move generation implementation

template<GenType T>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList) {
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);  // Switch color by XORing with 1 (was ~us which is wrong!)
    Square ksq = pos.king_sq(us);

    [[maybe_unused]] const Bitboard pinned = pos.pinned();
    const Bitboard checkers = pos.checkers();

    // Generate moves for each piece type
    // Pawns
    Bitboard pawns = pos.pieces(us, PAWN);
    Direction NORTH = us == WHITE ? 8 : -8;

    while (pawns) {
        Square from = pop_lsb(pawns);
        Square to = Square(from + NORTH);

        // Single pawn push
        if (!(pos.pieces() & square_bb(to))) {
            if (relative_rank(us, to) == RANK_8) {
                // Promotion
                if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_QUEEN_PROMO);
                    *moveList++ = Move(from, to, MF_ROOK_PROMO);
                    *moveList++ = Move(from, to, MF_BISHOP_PROMO);
                    *moveList++ = Move(from, to, MF_KNIGHT_PROMO);
                }
            } else {
                if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_QUIET);
                }

                // Double pawn push
                if (relative_rank(us, from) == RANK_2) {
                    Square to2 = Square(from + NORTH * 2);
                    if (!(pos.pieces() & square_bb(to2))) {
                        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                            *moveList++ = Move(from, to2, MF_DOUBLE_PAWN);
                        }
                    }
                }
            }
        }

        // Pawn captures
        Bitboard targets = pos.pieces(them) | (T == GEN_EVASION ? 0 : square_bb(pos.ep_square()));
        Bitboard attacks = pawn_attacks_bb(us, from) & targets;

        while (attacks) {
            Square to = pop_lsb(attacks);

            if (relative_rank(us, to) == RANK_8) {
                // Promotion capture
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_QUEEN_PROMO | MF_CAPTURE);
                    *moveList++ = Move(from, to, MF_ROOK_PROMO | MF_CAPTURE);
                    *moveList++ = Move(from, to, MF_BISHOP_PROMO | MF_CAPTURE);
                    *moveList++ = Move(from, to, MF_KNIGHT_PROMO | MF_CAPTURE);
                }
            } else if (to == pos.ep_square()) {
                // En passant
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_EN_PASSANT | MF_CAPTURE);
                }
            } else {
                // Normal capture
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_CAPTURE);
                }
            }
        }
    }

    // Knights
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard attacks = knight_attacks_bb(from) & pos.pieces(them);
        Bitboard quiets = knight_attacks_bb(from) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
            quiets &= ~pos.pieces(us);
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // Bishops
    Bitboard bishops = pos.pieces(us, BISHOP);
    while (bishops) {
        Square from = pop_lsb(bishops);
        Bitboard attacks = bb_diag_attacks(from, pos.pieces()) & pos.pieces(them);
        Bitboard quiets = bb_diag_attacks(from, pos.pieces()) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
            quiets &= ~pos.pieces(us);
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // Rooks
    Bitboard rooks = pos.pieces(us, ROOK);
    while (rooks) {
        Square from = pop_lsb(rooks);
        Bitboard attacks = (bb_rank_attacks(from, pos.pieces()) | bb_file_attacks(from, pos.pieces())) & pos.pieces(them);
        Bitboard quiets = (bb_rank_attacks(from, pos.pieces()) | bb_file_attacks(from, pos.pieces())) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
            quiets &= ~pos.pieces(us);
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // Queens
    Bitboard queens = pos.pieces(us, QUEEN);
    while (queens) {
        Square from = pop_lsb(queens);
        Bitboard attacks = queen_attacks_bb(from) & pos.pieces(them);
        Bitboard quiets = queen_attacks_bb(from) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
            quiets &= ~pos.pieces(us);
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // King
    Bitboard king_attacks = king_attacks_bb(ksq) & pos.pieces(them);
    Bitboard king_quiets = king_attacks_bb(ksq) & ~pos.pieces();

    // King captures
    if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
        while (king_attacks) {
            Square to = pop_lsb(king_attacks);
            *moveList++ = Move(ksq, to, MF_CAPTURE);
        }
    }

    // King quiets
    if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
        king_quiets &= ~pos.pieces(us);
        while (king_quiets) {
            Square to = pop_lsb(king_quiets);
            *moveList++ = Move(ksq, to, MF_QUIET);
        }
    }

    // Castling (only when not in check and not generating evasions)
    if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
        if (!checkers) {
            // Kingside
            CastlingRight king_side = us == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
            if (pos.castling_allowed(us, king_side)) {
                Square to = Square(us == WHITE ? G1 : G8);
                *moveList++ = Move(ksq, to, MF_CASTLING_KING);
            }
            // Queenside
            CastlingRight queen_side = us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
            if (pos.castling_allowed(us, queen_side)) {
                Square to = Square(us == WHITE ? C1 : C8);
                *moveList++ = Move(ksq, to, MF_CASTLING_QUEEN);
            }
        }
    }

    return moveList;
}

template<GenType T>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    if constexpr (T == GEN_LEGAL) {
        ExtMove* start = moveList;
        ExtMove* end = generate_moves<GEN_NON_EVASION>(pos, moveList);

        // Filter for legal moves
        ExtMove* legal_end = start;
        for (ExtMove* it = start; it != end; ++it) {
            bool legal = pos.legal(it->move);
            if (legal) {
                *(legal_end++) = *it;
            }
        }
        return legal_end;
    } else {
        return generate_moves<T>(pos, moveList);
    }
}

// Explicit template instantiations
template ExtMove* generate<GEN_LEGAL>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_CAPTURE>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_QUIET>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_EVASION>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_ALL>(const Position& pos, ExtMove* moveList);

} // namespace luminex
