#include "luminex.h"
#include <chrono>
#include <iostream>
#include <cstdio>
#include <fstream>

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
            // Do NOT call undo_move - do_move guarantees atomic failure
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
            // Do NOT call undo_move - do_move guarantees atomic failure
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

    // Disable output buffering for immediate UCI response
    // Keep sync_with_stdio enabled for proper text mode handling
    setvbuf(stdout, NULL, _IONBF, 0);
    std::ios_base::sync_with_stdio(true);

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

    // Check for eval mode
    if (argc > 1 && std::string(argv[1]) == "eval") {
        Position pos;
        if (argc > 2) {
            // Use provided FEN
            std::string fen;
            for (int i = 2; i < argc; ++i) {
                if (i > 2) fen += " ";
                fen += argv[i];
            }
            pos.set(fen);
        } else {
            pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }
        std::cout << "FEN: " << pos.fen() << std::endl;
        std::cout << "Static eval: " << evaluate(pos) << std::endl;
        return 0;
    }

    // Check for bench mode
    if (argc > 1 && std::string(argv[1]) == "bench") {
        Position pos;
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        using namespace std::chrono;
        auto start = high_resolution_clock::now();

        uint64_t p1 = perft(pos, 1);
        std::cout << "Perft 1: " << p1 << " (expected 20)" << std::endl;

        uint64_t p2 = perft(pos, 2);
        std::cout << "Perft 2: " << p2 << " (expected 400)" << std::endl;

        uint64_t p3 = perft(pos, 3);
        std::cout << "Perft 3: " << p3 << " (expected 8902)" << std::endl;

        uint64_t p4 = perft(pos, 4);
        std::cout << "Perft 4: " << p4 << " (expected 197281)" << std::endl;

        auto end = high_resolution_clock::now();
        auto ms = duration_cast<milliseconds>(end - start).count();
        std::cout << "Time: " << ms << "ms" << std::endl;

        if (p1 == 20 && p2 == 400 && p3 == 8902 && p4 == 197281) {
            std::cout << "ALL PERFT TESTS PASSED" << std::endl;
        } else {
            std::cout << "PERFT TESTS FAILED!" << std::endl;
        }
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
