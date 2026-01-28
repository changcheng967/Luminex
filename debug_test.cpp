#include "src/luminex.h"
#include <iostream>

// Copy of global variables from search.cpp
namespace luminex {
    Limits limits;
    SearchParams params;
    std::atomic<uint64_t> nodes;
    std::atomic<bool> stop;
    int root_depth;
    Value root_score;

// Copy of init() from main.cpp
void my_init() {
    init_zobrist();
    init_evaluation();
    TT.resize(256);
    TT.clear();
}

} // namespace luminex

int main() {
    using namespace luminex;

    my_init();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cout << "FEN: " << pos.fen() << std::endl;
    std::cout << "side_to_move: " << (pos.side_to_move() == WHITE ? "WHITE" : "BLACK") << std::endl;

    // Generate legal moves
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    int count = end - moves;
    std::cout << "Legal moves generated: " << count << std::endl;

    if (count > 0) {
        std::cout << "First move: " << moves[0].move.from() << moves[0].move.to() << std::endl;
    }

    return 0;
}
