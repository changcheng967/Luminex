#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "=== Board State ===" << std::endl;
    for (int i = 0; i < 64; ++i) {
        Square s = Square(i);
        Piece pc = pos.piece_on(s);
        if (pc != NO_PIECE) {
            std::cout << "Square " << i << ": piece=" << int(pc)
                      << " (" << (color_of_piece(pc) == WHITE ? "WHITE" : "BLACK")
                      << " " << int(piece_type_of(pc)) << ")" << std::endl;
        }
    }

    std::cout << "\n=== WHITE pieces on rank 1 ===" << std::endl;
    for (int i = 0; i < 8; ++i) {
        Square s = Square(i);
        Piece pc = pos.piece_on(s);
        std::cout << "Square " << i << ": piece=" << int(pc) << std::endl;
    }

    std::cout << "\n=== BLACK pieces (pieces(BLACK)) ===" << std::endl;
    Bitboard black_pieces = pos.pieces(BLACK);
    std::cout << "black_pieces = " << black_pieces << " (0x" << std::hex << black_pieces << std::dec << ")" << std::endl;
    std::cout << "popcount = " << popcount(black_pieces) << std::endl;

    return 0;
}
