#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "Starting position evaluation:" << std::endl;
    Value eval = evaluate(pos);
    std::cout << "Raw eval: " << eval << std::endl;
    std::cout << "Eval in centipawns: " << (eval * 100 / PAWN_VALUE) << std::endl;
    std::cout << "FEN: " << pos.fen() << std::endl;

    return 0;
}
