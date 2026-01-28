#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>

namespace luminex {

// Position for UCI
static Position pos;

void handle_uci() {
    std::cout << "id name " << ENGINE_NAME << " " << ENGINE_VERSION << std::endl;
    std::cout << "id author " << ENGINE_AUTHOR << std::endl;
    std::cout << "uciok" << std::endl;
}

void handle_isready() {
    std::cout << "readyok" << std::endl;
}

void handle_ucinewgame() {
    // Clear transposition table
    TT.clear();
    TT.new_search();
}

void handle_position(Position& pos, const std::string& cmd) {
    size_t pos_idx = cmd.find("position");
    size_t moves_idx = cmd.find("moves");

    std::string fen;

    if (moves_idx != std::string::npos) {
        fen = cmd.substr(pos_idx + 8, moves_idx - pos_idx - 9);
    } else {
        fen = cmd.substr(pos_idx + 8);
    }

    // Trim leading/trailing whitespace
    size_t start = fen.find_first_not_of(" \t");
    if (start != std::string::npos) {
        fen = fen.substr(start);
    }

    // Remove "startpos" prefix
    if (fen == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    }

    pos.set(fen);

    // Apply moves
    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 6);
        std::istringstream ss(moves_str);
        std::string move_str;

        while (ss >> move_str) {
            // Parse move
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            uint16_t flags = MF_QUIET;

            // Check for promotion
            if (move_str.length() > 4) {
                switch (move_str[4]) {
                    case 'n': flags = MF_KNIGHT_PROMO; break;
                    case 'b': flags = MF_BISHOP_PROMO; break;
                    case 'r': flags = MF_ROOK_PROMO; break;
                    case 'q': flags = MF_QUEEN_PROMO; break;
                }
                // Promotions are also captures
                PieceType captured = pos.piece_type_on(to);
                if (captured != PT_NONE) {
                    flags |= MF_CAPTURE;
                }
            } else if (pos.piece_type_on(to) != PT_NONE) {
                flags = MF_CAPTURE;
            }

            // Check for castling
            if (pos.piece_type_on(from) == KING && std::abs(file_of(to) - file_of(from)) > 1) {
                flags = (file_of(to) > file_of(from)) ? MF_CASTLING_KING : MF_CASTLING_QUEEN;
            }

            // Check for en passant
            if (pos.piece_type_on(from) == PAWN && pos.piece_type_on(to) == PT_NONE &&
                file_of(from) != file_of(to)) {
                flags = MF_EN_PASSANT | MF_CAPTURE;
            }

            Move m(from, to, flags);

            if (pos.legal(m)) {
                pos.do_move(m);
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    std::istringstream ss(cmd);
    std::string token;

    limits = Limits();

    while (ss >> token) {
        if (token == "depth") {
            ss >> limits.depth;
        } else if (token == "nodes") {
            ss >> limits.nodes;
        } else if (token == "movetime") {
            ss >> limits.movetime;
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "wtime") {
            ss >> limits.time[WHITE];
        } else if (token == "btime") {
            ss >> limits.time[BLACK];
        } else if (token == "winc") {
            ss >> limits.inc[WHITE];
        } else if (token == "binc") {
            ss >> limits.inc[BLACK];
        } else if (token == "movestogo") {
            int movestogo;
            ss >> movestogo;
        }
    }

    if (limits.depth == 0) {
        limits.depth = MAX_PLY - 1;
    }

    // Run search
    auto start = std::chrono::steady_clock::now();
    Move best_move = search(pos, limits);
    auto end = std::chrono::steady_clock::now();

    int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Send final info (use actual depth searched, not incremented root_depth)
    int actual_depth = (root_depth > 1) ? root_depth - 1 : 1;
    uci_info(pos, actual_depth, root_score, nodes.load(), time_ms);

    // Send best move
    if (best_move) {
        std::cout << "bestmove " << best_move;
        // TODO: Add ponder move
        std::cout << std::endl;
    } else {
        std::cout << "bestmove 0000" << std::endl;
    }
}

void handle_setoption(const std::string&) {
    // TODO: Implement options
}

void handle_stop() {
    stop = true;
}

void handle_quit() {
    stop = true;
}

void uci_loop() {
    std::string line;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            handle_uci();
        } else if (cmd == "isready") {
            handle_isready();
        } else if (cmd == "ucinewgame") {
            handle_ucinewgame();
        } else if (cmd == "position") {
            handle_position(pos, line);
        } else if (cmd == "go") {
            handle_go(pos, line);
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "stop") {
            handle_stop();
        } else if (cmd == "quit") {
            handle_quit();
            break;
        } else if (cmd == "d") {
            // Debug: print board
            std::cout << pos.fen() << std::endl;
        }
    }
}

void uci_send(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    std::vprintf(msg, args);
    va_end(args);
}

} // namespace luminex
