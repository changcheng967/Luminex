#include "luminex.h"
#include <chrono>
#include <iostream>

namespace luminex {

void init() {
    init_zobrist();
    init_evaluation();
    TT.resize(256); // 256 MB transposition table
    TT.clear();
}

// Perft for testing
uint64_t perft(Position& pos, Depth depth) {
    if (depth == 0) return 1;

    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    uint64_t count = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        StateInfo st;
        pos.do_move(it->move, st);
        count += perft(pos, depth - 1);
        pos.undo_move(it->move);
    }

    return count;
}

} // namespace luminex

int main(int argc, char* argv[]) {
    using namespace luminex;

    // Initialize engine
    init();

    // Check for bench mode
    if (argc > 1 && std::string(argv[1]) == "bench") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        auto start = std::chrono::steady_clock::now();
        uint64_t nodes = perft(pos, 5);
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Perft 5: " << nodes << " nodes in " << duration << " ms"
                  << " (" << nodes * 1000 / duration << " nps)" << std::endl;
        return 0;
    }

    // Enter UCI loop
    uci_loop();

    return 0;
}
