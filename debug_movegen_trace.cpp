#include "src/luminex.h"
#include <iostream>

using namespace luminex;

// Override to add debug
namespace luminex {
template<GenType T>
ExtMove* generate_moves_debug(const Position& pos, ExtMove* moveList) {
    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);

    // Pawns
    Bitboard pawns = pos.pieces(us, PAWN);
    Direction NORTH = us == WHITE ? 8 : -8;

    int move_num = 0;
    while (pawns) {
        Square from = pop_lsb(pawns);
        Square to = Square(from + NORTH);

        // Single pawn push
        if (!(pos.pieces() & square_bb(to))) {
            if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                *moveList++ = Move(from, to, MF_QUIET);
                move_num++;
                std::cout << "Generated quiet pawn: " << int(from) << " -> " << int(to) << std::endl;
            }

            // Double pawn push
            if (relative_rank(us, from) == RANK_2) {
                Square to2 = Square(from + NORTH * 2);
                if (!(pos.pieces() & square_bb(to2))) {
                    if constexpr (T == GEN_QUIET || T == GEN_ALL || T == GEN_LEGAL || T == GEN_NON_EVASION) {
                        *moveList++ = Move(from, to2, MF_DOUBLE_PAWN);
                        move_num++;
                        std::cout << "Generated double pawn: " << int(from) << " -> " << int(to2) << std::endl;
                    }
                }
            }
        }

        // Pawn captures
        Bitboard targets = pos.pieces(them);
        Bitboard attacks = pawn_attacks_bb(us, from) & targets;

        std::cout << "Pawn at " << int(from) << ": attacks = 0x" << std::hex << attacks << std::dec
                  << ", popcount = " << popcount(attacks) << std::endl;

        while (attacks) {
            Square to = pop_lsb(attacks);
            if constexpr (T == GEN_CAPTURE || T == GEN_ALL || T == GEN_LEGAL || T == GEN_EVASION || T == GEN_NON_EVASION) {
                *moveList++ = Move(from, to, MF_CAPTURE);
                move_num++;
                std::cout << "Generated capture: " << int(from) << " -> " << int(to) << std::endl;
            }
        }
    }

    std::cout << "Total pawn moves: " << move_num << std::endl;
    return moveList;
}
}

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "=== Debug Move Generation ===" << std::endl;
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate_moves_debug<GEN_LEGAL>(pos, moves);

    std::cout << "\nTotal moves generated: " << (end - moves) << std::endl;

    return 0;
}
