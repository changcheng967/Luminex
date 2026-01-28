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
        pos.do_move(it->move);
        count += perft(pos, depth - 1);
        pos.undo_move(it->move);
    }

    return count;
}

} // namespace luminex

int main(int argc, char* argv[]) {
    using namespace luminex;

    // Force unbuffered stdout for immediate UCI output
    std::cout << std::unitbuf;

    // Initialize engine
    init();

    // Check for debug mode
    if (argc > 1 && std::string(argv[1]) == "debug") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        std::cout << "FEN: " << pos.fen() << std::endl;
        std::cout << "piece_on(A1): " << int(pos.piece_on(A1)) << std::endl;
        std::cout << "piece_on(A8): " << int(pos.piece_on(A8)) << std::endl;
        std::cout << "piece_on(A2): " << int(pos.piece_on(A2)) << std::endl;
        std::cout << "piece_on(H8): " << int(pos.piece_on(H8)) << std::endl;
        return 0;
    }

    // Check for bench mode
    if (argc > 1 && std::string(argv[1]) == "bench") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        auto start = std::chrono::steady_clock::now();
        uint64_t nodes = perft(pos, 1);
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
