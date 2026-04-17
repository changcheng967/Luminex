#include "luminex.h"

namespace luminex {

// Move generation implementation

template<GenType T>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList) {
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);  // Switch color by XORing with 1
    Square ksq = pos.king_sq(us);

    [[maybe_unused]] const Bitboard pinned = pos.pinned();
    const Bitboard checkers = pos.checkers();
    const Bitboard their_pieces = pos.pieces(them) & ~pos.pieces(them, KING);

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
        // Only include EP square if it's actually set (not SQUARE_NONE)
        // square_bb(SQUARE_NONE) where SQUARE_NONE=64 causes UB: 1ULL << 64
        Bitboard ep_bb = BB_EMPTY;
        if (pos.ep_square() != SQUARE_NONE) {
            ep_bb = square_bb(pos.ep_square());
        }
        Bitboard targets = their_pieces | ep_bb;
        Bitboard attacks = pawn_attacks_bb(us, from) & targets;

        while (attacks) {
            Square cap_to = pop_lsb(attacks);

            if (relative_rank(us, cap_to) == RANK_8) {
                // Promotion capture
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, cap_to, MF_CAPTURE_PROMO_QUEEN);
                    *moveList++ = Move(from, cap_to, MF_CAPTURE_PROMO_ROOK);
                    *moveList++ = Move(from, cap_to, MF_CAPTURE_PROMO_BISHOP);
                    *moveList++ = Move(from, cap_to, MF_CAPTURE_PROMO_KNIGHT);
                }
            } else if (cap_to == pos.ep_square()) {
                // En passant
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, cap_to, MF_EN_PASSANT);
                }
            } else {
                // Normal capture
                if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                    *moveList++ = Move(from, cap_to, MF_CAPTURE);
                }
            }
        }
    }

    // Knights
    Bitboard knights = pos.pieces(us, KNIGHT);
    while (knights) {
        Square from = pop_lsb(knights);
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
        Bitboard all_attacks = bishop_attacks_bb(from, pos.pieces());
        Bitboard attacks = all_attacks & their_pieces;
        // Quiets: empty squares only
        Bitboard quiets = all_attacks & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
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
        Bitboard all_attacks = rook_attacks_bb(from, pos.pieces());
        Bitboard attacks = all_attacks & their_pieces;
        // Quiets: empty squares only (king squares are NOT valid destinations)
        Bitboard quiets = all_attacks & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
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
        Bitboard all_attacks = queen_attacks_bb(from, pos.pieces());
        Bitboard attacks = all_attacks & their_pieces;
        // Quiets: empty squares only
        Bitboard quiets = all_attacks & ~pos.pieces();

        // Captures
        if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (attacks) {
                Square to = pop_lsb(attacks);
                *moveList++ = Move(from, to, MF_CAPTURE);
            }
        }

        // Quiets
        if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
            while (quiets) {
                Square to = pop_lsb(quiets);
                *moveList++ = Move(from, to, MF_QUIET);
            }
        }
    }

    // King
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
                    // CRITICAL: Verify rook is present before generating castling
                    Square rook_sq = Square(us == WHITE ? H1 : H8);
                    if (pos.piece_on(rook_sq) == make_piece(us, ROOK)) {
                        Square to = Square(us == WHITE ? G1 : G8);
                        *moveList++ = Move(ksq, to, MF_CASTLING_KING);
                    }
                }
                // Queenside
                CastlingRight queen_side = us == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
                if (pos.castling_allowed(us, queen_side)) {
                    // CRITICAL: Verify rook is present before generating castling
                    Square rook_sq = Square(us == WHITE ? A1 : A8);
                    if (pos.piece_on(rook_sq) == make_piece(us, ROOK)) {
                        Square to = Square(us == WHITE ? C1 : C8);
                        *moveList++ = Move(ksq, to, MF_CASTLING_QUEEN);
                    }
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
            }
        }
        return legal_end;
    } else if constexpr (T == GEN_QUIET_CHECK) {
        // Generate quiet moves that give check: generate all quiets, filter for checks
        ExtMove* start = moveList;
        ExtMove* end = generate_moves<GEN_QUIET>(pos, moveList);
        Color us = pos.side_to_move();
        Square opp_ksq = pos.king_sq(Color(us ^ 1));
        Bitboard occ = pos.pieces();

        ExtMove* check_end = start;
        for (ExtMove* it = start; it != end; ++it) {
            Move m = it->move;
            Square from = m.from();
            Square to = m.to();
            PieceType pt = piece_type_of(pos.piece_on(from));
            bool gives_chk = false;

            // Check if piece on 'to' attacks opponent king
            // Use occupancy with piece moved (remove from, add to)
            Bitboard occ_after = (occ ^ square_bb(from)) | square_bb(to);
            switch (pt) {
                case PAWN:
                    gives_chk = (pawn_attacks_bb(us, square_bb(to)) & square_bb(opp_ksq)) != 0;
                    // Also check discovered check
                    if (!gives_chk) {
                        Bitboard occ_no_pawn = occ ^ square_bb(from);
                        gives_chk = (rook_attacks_bb(opp_ksq, occ_no_pawn) | bishop_attacks_bb(opp_ksq, occ_no_pawn))
                                    & (pos.pieces(us, ROOK, QUEEN) | pos.pieces(us, BISHOP, QUEEN)) & ~square_bb(from);
                    }
                    break;
                case KNIGHT:
                    gives_chk = (knight_attacks_bb(to) & square_bb(opp_ksq)) != 0;
                    break;
                case BISHOP:
                    gives_chk = (bishop_attacks_bb(to, occ_after) & square_bb(opp_ksq)) != 0;
                    break;
                case ROOK:
                    gives_chk = (rook_attacks_bb(to, occ_after) & square_bb(opp_ksq)) != 0;
                    break;
                case QUEEN:
                    gives_chk = ((bishop_attacks_bb(to, occ_after) | rook_attacks_bb(to, occ_after)) & square_bb(opp_ksq)) != 0;
                    break;
                case KING:
                    // King can't give direct check, only discovered check
                    {
                        Bitboard occ_no_king = occ ^ square_bb(from);
                        gives_chk = (rook_attacks_bb(opp_ksq, occ_no_king) | bishop_attacks_bb(opp_ksq, occ_no_king))
                                    & (pos.pieces(us, ROOK, QUEEN) | pos.pieces(us, BISHOP, QUEEN)) & ~square_bb(from);
                    }
                    break;
                default: break;
            }

            if (gives_chk && pos.legal(m)) {
                *(check_end++) = *it;
            }
        }
        return check_end;
    } else {
        return generate_moves<T>(pos, moveList);
    }
}

// Explicit template instantiations
template ExtMove* generate<GEN_LEGAL>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_CAPTURE>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_QUIET>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_QUIET_CHECK>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_EVASION>(const Position& pos, ExtMove* moveList);
template ExtMove* generate<GEN_ALL>(const Position& pos, ExtMove* moveList);

} // namespace luminex
