#include "luminex.h"
#include <iostream>

namespace luminex {

// Move generation implementation

template<GenType T>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList) {
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);  // Switch color by XORing with 1
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
                    *moveList++ = Move(from, to, MF_PROMO_QUEEN);
                    *moveList++ = Move(from, to, MF_PROMO_ROOK);
                    *moveList++ = Move(from, to, MF_PROMO_BISHOP);
                    *moveList++ = Move(from, to, MF_PROMO_KNIGHT);
                }
            } else {
                if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_QUIET);
                }

                // Double pawn push
                if (relative_rank(us, from) == RANK_2) {
                    Square to2 = Square(from + NORTH * 2);
                    if (!(pos.pieces() & square_bb(to2))) {
                        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                            *moveList++ = Move(from, to2, MF_DOUBLE_PAWN);
                        }
                    }
                }
            }
        }

        // Pawn captures
        // Exclude enemy king from captures
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
        Bitboard targets = their_pieces | (T == GEN_EVASION ? 0 : square_bb(pos.ep_square()));
        Bitboard attacks = pawn_attacks_bb(us, from) & targets;

        while (attacks) {
            Square to = pop_lsb(attacks);

            if (relative_rank(us, to) == RANK_8) {
                // Promotion capture
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_CAPTURE_PROMO_QUEEN);
                    *moveList++ = Move(from, to, MF_CAPTURE_PROMO_ROOK);
                    *moveList++ = Move(from, to, MF_CAPTURE_PROMO_BISHOP);
                    *moveList++ = Move(from, to, MF_CAPTURE_PROMO_KNIGHT);
                }
            } else if (to == pos.ep_square()) {
                // En passant
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, to, MF_EN_PASSANT);
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
        // Captures: enemy pieces except king
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
        Bitboard attacks = knight_attacks_bb(from) & their_pieces;
        // Quiets: empty squares only
        Bitboard quiets = knight_attacks_bb(from) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
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
        // Captures: enemy pieces except king
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
        Bitboard attacks = bb_diag_attacks(from, pos.pieces()) & their_pieces;
        // Quiets: empty squares only
        Bitboard quiets = bb_diag_attacks(from, pos.pieces()) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
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
        // Captures: enemy pieces except king (kings cannot be captured)
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
        Bitboard attacks = (bb_rank_attacks(from, pos.pieces()) | bb_file_attacks(from, pos.pieces())) & their_pieces;
        // Quiets: empty squares only (king squares are NOT valid destinations)
        Bitboard quiets = (bb_rank_attacks(from, pos.pieces()) | bb_file_attacks(from, pos.pieces())) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
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
        // Captures: enemy pieces except king
        Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
        Bitboard attacks = queen_attacks_bb(from, pos.pieces()) & their_pieces;
        // Quiets: empty squares only
        Bitboard quiets = queen_attacks_bb(from, pos.pieces()) & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            quiets &= ~pos.pieces(us);
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // King
    // Exclude enemy king from captures (king cannot capture enemy king)
    Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);
    Bitboard king_attacks = king_attacks_bb(ksq) & their_pieces;
    Bitboard king_quiets = king_attacks_bb(ksq) & ~pos.pieces();

    // King captures
    if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
        while (king_attacks) {
            Square to = pop_lsb(king_attacks);
            *moveList++ = Move(ksq, to, MF_CAPTURE);
        }
    }

    // King quiets
    if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
        king_quiets &= ~pos.pieces(us);
        while (king_quiets) {
            Square to = pop_lsb(king_quiets);
            *moveList++ = Move(ksq, to, MF_QUIET);
        }
    }

    // Castling (only when not in check and not generating evasions)
    if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
        if (!checkers) {
            // CRITICAL: Only generate castling if king is on starting square
            // White king must be on e1, black king must be on e8
            bool king_on_start = (us == WHITE && ksq == E1) || (us == BLACK && ksq == E8);
            if (!king_on_start) {
                // King has moved - cannot castle, even if castling rights aren't revoked
                // This prevents generating illegal castling moves like b1g1
            } else {
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
    }

    return moveList;
}

template<GenType T>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    if constexpr (T == GEN_LEGAL) {
        ExtMove* start = moveList;
        ExtMove* end;

        // When in check, use evasion generation; otherwise use non-evasion
        if (pos.is_check()) {
            end = generate_moves<GEN_EVASION>(pos, moveList);
        } else {
            end = generate_moves<GEN_NON_EVASION>(pos, moveList);
        }

        // Filter for legal moves (evasion moves should already be legal, but verify anyway)
        ExtMove* legal_end = start;
        for (ExtMove* it = start; it != end; ++it) {
            bool is_legal = pos.legal(it->move);
            if (is_legal) {
                *(legal_end++) = *it;
            } else {
                // DEBUG: log filtered move
                std::cerr << "FILTERED: from=" << int(it->move.from()) << " to=" << int(it->move.to()) << "\n";
                std::cerr.flush();
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
