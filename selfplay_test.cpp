#include "src/luminex.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace luminex;

// Simple UCI engine wrapper for self-play
class Engine {
public:
    Position pos;
    bool is_white;

    Engine(bool white) : is_white(white) {
        pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    Move search(int depth) {
        Limits limits;
        limits.depth = depth;
        limits.movetime = 0;
        limits.nodes = 0;
        limits.infinite = false;

        return ::luminex::search(pos, limits);
    }

    void do_move(Move m) {
        if (pos.legal(m)) {
            pos.do_move(m);
        }
    }

    std::string get_fen() const {
        return pos.fen();
    }

    bool is_checkmate() {
        if (pos.is_check()) {
            ExtMove moves[MAX_MOVES];
            ExtMove* end = generate<GEN_LEGAL>(pos, moves);
            if (moves == end) return true;  // Checkmate
        }
        return false;
    }

    bool is_stalemate() {
        if (!pos.is_check()) {
            ExtMove moves[MAX_MOVES];
            ExtMove* end = generate<GEN_LEGAL>(pos, moves);
            if (moves == end) return true;  // Stalemate
        }
        return false;
    }

    bool is_draw() {
        return pos.is_draw();
    }
};

int main() {
    std::cout << "Luminex Self-Play Test" << std::endl;
    std::cout << "======================" << std::endl;

    // Initialize engine (inline init from main.cpp)
    init_zobrist();
    init_evaluation();
    TT.resize(256);
    TT.clear();

    Engine white(true);
    Engine black(false);

    // Sync positions
    black.pos = white.pos;

    int move_count = 0;
    int max_moves = 100;  // Prevent infinite loops

    while (move_count < max_moves) {
        std::cout << "\n--- Move " << (move_count + 1) << " ---" << std::endl;
        std::cout << "FEN: " << white.get_fen() << std::endl;

        // White to move
        std::cout << "White thinking..." << std::endl;
        Move white_move = white.search(6);

        if (!white_move) {
            std::cout << "White has no legal moves!" << std::endl;
            if (white.pos.is_check()) {
                std::cout << "CHECKMATE! Black wins!" << std::endl;
            } else {
                std::cout << "STALEMATE! Draw!" << std::endl;
            }
            break;
        }

        std::cout << "White plays: " << white_move << std::endl;
        white.do_move(white_move);
        black.pos.do_move(white_move);
        move_count++;

        if (white.is_draw() || white.is_checkmate() || white.is_stalemate()) {
            if (white.is_checkmate()) {
                std::cout << "CHECKMATE after white's move!" << std::endl;
            } else if (white.is_stalemate()) {
                std::cout << "STALEMATE after white's move!" << std::endl;
            } else {
                std::cout << "DRAW (repetition/50-move) after white's move!" << std::endl;
            }
            break;
        }

        if (move_count >= max_moves) break;

        // Black to move
        std::cout << "Black thinking..." << std::endl;
        Move black_move = black.search(6);

        if (!black_move) {
            std::cout << "Black has no legal moves!" << std::endl;
            if (black.pos.is_check()) {
                std::cout << "CHECKMATE! White wins!" << std::endl;
            } else {
                std::cout << "STALEMATE! Draw!" << std::endl;
            }
            break;
        }

        std::cout << "Black plays: " << black_move << std::endl;
        black.do_move(black_move);
        white.pos.do_move(black_move);
        move_count++;

        if (black.is_draw() || black.is_checkmate() || black.is_stalemate()) {
            if (black.is_checkmate()) {
                std::cout << "CHECKMATE after black's move!" << std::endl;
            } else if (black.is_stalemate()) {
                std::cout << "STALEMATE after black's move!" << std::endl;
            } else {
                std::cout << "DRAW (repetition/50-move) after black's move!" << std::endl;
            }
            break;
        }
    }

    if (move_count >= max_moves) {
        std::cout << "\nGame ended after " << max_moves << " moves (limit reached)" << std::endl;
    }

    std::cout << "\nFinal FEN: " << white.get_fen() << std::endl;
    std::cout << "\nTest completed successfully!" << std::endl;

    return 0;
}
