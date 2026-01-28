#include "src/luminex.h"
#include <iostream>
#include <iomanip>

using namespace luminex;

// Manually generate moves to trace
int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    Color us = pos.side_to_move();
    Color them = Color(us ^ 1);

    std::cout << "us = " << int(us) << " (WHITE)" << std::endl;
    std::cout << "them = " << int(them) << " (BLACK)" << std::endl;
    std::cout << "pieces(BLACK) = 0x" << std::hex << pos.pieces(BLACK) << std::dec << std::endl;
    std::cout << "ep_square = " << int(pos.ep_square()) << std::endl;
    std::cout << std::endl;

    // Check pawn at b2
    Square from = Square(9); // b2
    std::cout << "=== Pawn at b2 (Square " << int(from) << ") ===" << std::endl;

    Bitboard attacks = pawn_attacks_bb(us, from);
    std::cout << "pawn_attacks_bb(WHITE, b2) = 0x" << std::hex << attacks << std::dec << std::endl;

    Bitboard targets = pos.pieces(them) | square_bb(pos.ep_square());
    std::cout << "targets = pieces(BLACK) | ep_square = 0x" << std::hex << targets << std::dec << std::endl;

    Bitboard capture_squares = attacks & targets;
    std::cout << "attacks & targets = 0x" << std::hex << capture_squares << std::dec << std::endl;

    std::cout << std::endl;
    std::cout << "Now generating all legal moves..." << std::endl;

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    std::cout << "Generated " << (end - moves) << " moves" << std::endl;

    for (ExtMove* it = moves; it != end; ++it) {
        Move m = it->move;
        Square f = m.from();
        Square t = m.to();

        char fFile = 'a' + file_of(f);
        char fRank = '1' + rank_of(f);
        char tFile = 'a' + file_of(t);
        char tRank = '1' + rank_of(t);

        std::cout << fFile << fRank << tFile << tRank << " flags=0x" << std::hex << m.flags() << std::dec;

        if (f == Square(9)) { // b2
            std::cout << " <-- from b2!";
            if (m.is_capture()) std::cout << " IS_CAPTURE!";
        }

        std::cout << std::endl;
    }

    return 0;
}
