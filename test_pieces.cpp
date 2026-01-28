#include "src/luminex.h"
#include <iostream>
#include <iomanip>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "pieces(WHITE) = 0x" << std::hex << pos.pieces(WHITE) << std::dec << std::endl;
    std::cout << "pieces(BLACK) = 0x" << std::hex << pos.pieces(BLACK) << std::dec << std::endl;
    std::cout << std::endl;

    // Check specific squares
    std::cout << "piece_on(a3) = " << int(pos.piece_on(A3)) << " (expected " << int(NO_PIECE) << " = NO_PIECE)" << std::endl;
    std::cout << "piece_on(b2) = " << int(pos.piece_on(Square(9))) << " (expected 0 = WHITE_PAWN)" << std::endl;
    std::cout << std::endl;

    // Check pawn attacks for pawn at b2
    Bitboard attacks = pawn_attacks_bb(WHITE, Square(9));
    std::cout << "pawn_attacks_bb(WHITE, b2) = 0x" << std::hex << attacks << std::dec << std::endl;

    // Squares attacked by pawn at b2
    std::cout << "Pawn at b2 attacks:" << std::endl;
    Bitboard temp = attacks;
    while (temp) {
        Square s = pop_lsb(temp);
        char file = 'a' + file_of(s);
        char rank = '1' + rank_of(s);
        std::cout << "  " << file << rank << " (piece=" << int(pos.piece_on(s)) << ")" << std::endl;
    }
    std::cout << std::endl;

    // Check if b2a3 is a valid capture according to move generation
    Bitboard black_pieces = pos.pieces(BLACK);
    Bitboard capture_targets = attacks & black_pieces;
    std::cout << "pawn_attacks & pieces(BLACK) = 0x" << std::hex << capture_targets << std::dec << std::endl;

    return 0;
}
