#include "luminex.h"
#include <chrono>
#include <iostream>
#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

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
        if (!pos.do_move(it->move)) {
            // Should never happen with legal move generation
            pos.undo_move(it->move);
            continue;
        }
        count += perft(pos, depth - 1);
        pos.undo_move(it->move);
    }

    return count;
}

// Perft divide - shows perft result for each move
uint64_t perft_divide(Position& pos, Depth depth) {
    ExtMove moves[MAX_MOVES];
    ExtMove* end = generate<GEN_LEGAL>(pos, moves);

    uint64_t total = 0;
    for (ExtMove* it = moves; it != end; ++it) {
        if (!pos.do_move(it->move)) {
            // Should never happen with legal move generation
            pos.undo_move(it->move);
            continue;
        }
        uint64_t count = (depth == 1) ? 1 : perft(pos, depth - 1);
        pos.undo_move(it->move);

        std::cout << it->move << ": " << count << std::endl;
        total += count;
    }
    return total;
}

} // namespace luminex

int main(int argc, char* argv[]) {
    using namespace luminex;

#ifdef _WIN32
    // Set stdin/stdout to binary mode for Windows pipe compatibility
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    // Disable all buffering for immediate UCI output (critical for cutechess-cli)
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    std::ios_base::sync_with_stdio(false);
    std::cout.setf(std::ios::unitbuf);

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
        uint64_t nodes = perft(pos, 1);
        std::cout << "Perft 1: " << nodes << " nodes" << std::endl;
        return 0;
    }

    // Check for divide mode
    if (argc > 1 && std::string(argv[1]) == "divide") {
        int depth = (argc > 2) ? std::atoi(argv[2]) : 3;
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        std::cout << "Divide perft depth " << depth << ":" << std::endl;
        uint64_t total = perft_divide(pos, depth);
        std::cout << "Total: " << total << " nodes" << std::endl;
        return 0;
    }

    // Enter UCI loop
    uci_loop();

    return 0;
}
