#include "src/luminex.h"
#include <iostream>

using namespace luminex;

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "=== Search to Depth 6 ===" << std::endl;
    Limits limits;
    limits.depth = 6;
    search(pos, limits);

    std::cout << "\n=== Final Evaluation ===" << std::endl;
    Value eval = evaluate(pos);
    std::cout << "Direct eval: " << eval << std::endl;

    return 0;
}
