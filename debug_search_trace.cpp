#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "=== Direct Evaluation ===" << std::endl;
    Value direct_eval = evaluate(pos);
    std::cout << "Direct eval: " << direct_eval << std::endl;
    std::cout << "Side to move: " << (pos.side_to_move() == WHITE ? "WHITE" : "BLACK") << std::endl;

    std::cout << "\n=== Test Search at Different Depths ===" << std::endl;
    for (int d = 1; d <= 6; d++) {
        std::cout << "\n--- Depth " << d << " ---" << std::endl;
        Position fresh_pos;
        fresh_pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        Limits limits;
        limits.depth = d;
        search(fresh_pos, limits);
    }

    // Now test from black's perspective
    std::cout << "\n=== Same position but pretend it's black's turn ===" << std::endl;
    Position pos2;
    pos2.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    Value eval_black = evaluate(pos2);
    std::cout << "Eval (black to move): " << eval_black << std::endl;

    return 0;
}
