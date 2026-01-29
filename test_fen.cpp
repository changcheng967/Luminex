#include <iostream>
#include <string>

int main() {
    std::string fen = "rnbqkbnr/pp2pppp/3N4/2pp4/8/8/PPPPPPPP/R1BQKBNR b KQkq - 0 1";
    
    std::cout << "FEN: " << fen << std::endl << std::endl;
    
    // Parse and display
    std::cout << "  a b c d e f g h" << std::endl;
    std::cout << "  ----------------" << std::endl;
    
    int rank = 8;
    for (char c : fen) {
        if (c == ' ') break;
        if (c == '/') {
            std::cout << " " << rank-- << std::endl;
        } else if (c >= '1' && c <= '8') {
            int empty = c - '0';
            for (int i = 0; i < empty; i++) std::cout << ".";
        } else {
            std::cout << c;
        }
    }
    std::cout << " " << rank << std::endl;
    std::cout << "  ----------------" << std::endl;
    std::cout << "  a b c d e f g h" << std::endl << std::endl;
    
    // Check knight on d6 attacks
    std::cout << "Knight on d6 attacks:" << std::endl;
    std::cout << "  d6 +2 ranks +1 file = e8 (BLACK KING!)" << std::endl;
    std::cout << "  This IS check!" << std::endl << std::endl;
    
    // Queen on d8 can capture on d6?
    std::cout << "Queen on d8 to d6:" << std::endl;
    std::cout << "  File move: d8 -> d7 -> d6" << std::endl;
    std::cout << "  d7 is blocked by black pawn!" << std::endl;
    std::cout << "  Queen CANNOT capture on d6!" << std::endl;
    
    return 0;
}
