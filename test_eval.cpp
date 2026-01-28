#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();
    
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    Value eval = evaluate(pos);
    std::cout << "Eval from startpos (WHITE to move): " << eval << " cp" << std::endl;
    std::cout << "Expected: ~0 cp" << std::endl;
    
    // Check piece counts
    std::cout << "\nWhite pieces:" << std::endl;
    for (PieceType pt = PAWN; pt <= KING; ++pt) {
        Bitboard b = pos.pieces(WHITE, pt);
        std::cout << "  " << pt << ": " << popcount(b) << std::endl;
    }
    
    return 0;
}
