#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Test single move do/undo
    Move m(A2, A3, MF_QUIET);

    std::cout << "Doing move a2a3..." << std::endl;
    pos.do_move(m);
    std::cout << "After move FEN: " << pos.fen() << std::endl;

    std::cout << "Undoing move..." << std::endl;
    pos.undo_move(m);
    std::cout << "After undo FEN: " << pos.fen() << std::endl;

    return 0;
}
