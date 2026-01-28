#include "src/luminex.h"
#include <iostream>

using namespace luminex;

uint64_t test_perft(Position& pos, Depth depth) {
    if (depth == 0) return 1;

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    uint64_t count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        pos.do_move(it->move);
        uint64_t subcount = test_perft(pos, depth - 1);
        pos.undo_move(it->move);
        if (subcount == 0 && depth > 1) {
            std::cerr << "ERROR: subcount=0 at depth " << depth << std::endl;
            return 0;
        }
        count += subcount;
    }

    return count;
}

int main() {
    init_zobrist();
    init_evaluation();

    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    for (int depth = 1; depth <= 5; depth++) {
        std::cout << "Perft(" << depth << "): ";
        std::cout.flush();
        uint64_t result = test_perft(pos, depth);
        std::cout << result << std::endl;
    }

    return 0;
}
