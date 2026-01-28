#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Check BLACK pieces
    Bitboard black_pieces = pos.pieces(BLACK);
    std::cout << "black_pieces = 0x" << std::hex << black_pieces << std::dec << std::endl;
    std::cout << "popcount(black_pieces) = " << popcount(black_pieces) << std::endl;

    // Check pawn attacks for A2 (square 8)
    Bitboard attacks = pawn_attacks_bb(WHITE, Square(8));
    std::cout << "\npawn_attacks_bb(WHITE, 8) = 0x" << std::hex << attacks << std::dec << std::endl;

    // Check which bits are set
    std::cout << "Attack squares:" << std::endl;
    Bitboard temp = attacks;
    while (temp) {
        Square s = pop_lsb(temp);
        std::cout << "  Square " << int(s) << " (0x" << std::hex << (1ULL << int(s)) << std::dec << ")" << std::endl;
    }

    // Check AND with BLACK pieces
    Bitboard captures = attacks & black_pieces;
    std::cout << "\nattacks & black_pieces = 0x" << std::hex << captures << std::dec << std::endl;
    std::cout << "popcount(captures) = " << popcount(captures) << std::endl;

    // Check specific square
    std::cout << "\nIs there a BLACK piece on square 17 (B3)?" << std::endl;
    std::cout << "  pos.pieces(BLACK) has bit 17: " << ((black_pieces & (1ULL << 17)) ? "YES" : "NO") << std::endl;

    return 0;
}
