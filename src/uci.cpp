#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>

namespace luminex {

// Position for UCI
static Position pos;

void handle_uci() {
    std::cout << "id name " << ENGINE_NAME << " " << ENGINE_VERSION << "\n";
    std::cout.flush();
    std::cout << "id author " << ENGINE_AUTHOR << "\n";
    std::cout.flush();
    // Declare UCI options
    std::cout << "option name Hash type spin default 128 min 1 max 1048576\n";
    std::cout.flush();
    std::cout << "option name Contempt type spin default 0 min -1000 max 1000\n";
    std::cout.flush();
    std::cout << "option name Clear Hash type button\n";
    std::cout.flush();
    std::cout << "uciok\n";
    std::cout.flush();
}

void handle_isready() {
    std::cout << "readyok\n";
    std::cout.flush();
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

    // Debug: print the position command (truncated)
    // std::cout << "info string pos_cmd: " << cmd.substr(0, 50) << "\n";
    // std::cout.flush();

    // Trim leading/trailing whitespace
    size_t start = fen.find_first_not_of(" \t");
    if (start != std::string::npos) {
        fen = fen.substr(start);
    }

    // Remove "startpos" prefix
    if (fen == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    }
    // Remove "fen" prefix if present
    else if (fen.size() >= 3 && fen.substr(0, 3) == "fen") {
        size_t fen_start = fen.find_first_not_of(" \t", 3);
        if (fen_start != std::string::npos) {
            fen = fen.substr(fen_start);
        } else {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
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
                bool isCapture = pos.piece_type_on(to) != PT_NONE;
                switch (move_str[4]) {
                    case 'n': flags = isCapture ? MF_CAPTURE_PROMO_KNIGHT : MF_PROMO_KNIGHT; break;
                    case 'b': flags = isCapture ? MF_CAPTURE_PROMO_BISHOP : MF_PROMO_BISHOP; break;
                    case 'r': flags = isCapture ? MF_CAPTURE_PROMO_ROOK : MF_PROMO_ROOK; break;
                    case 'q': flags = isCapture ? MF_CAPTURE_PROMO_QUEEN : MF_PROMO_QUEEN; break;
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
                flags = MF_EN_PASSANT;
            }

            Move m(from, to, flags);

            // Trust the GUI - always make the move
            // If there's a bug in our move generation, we want to play what GUI says
            pos.do_move(m);
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    // Debug log file - use process ID to separate engines
    static FILE* uci_log = nullptr;
    static int log_init = 0;
    if (!log_init) {
        // Use different files for different processes based on side to move
        const char* fname = (pos.side_to_move() == WHITE) ?
            "C:/Users/chang/Downloads/Luminex/luminex_white_log.txt" :
            "C:/Users/chang/Downloads/Luminex/luminex_black_log.txt";
        uci_log = fopen(fname, "w");
        log_init = 1;
    }
    if (uci_log) {
        fprintf(uci_log, "handle_go called: %s\n", cmd.c_str());
        fflush(uci_log);
    }

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

    // If no depth specified and not using time control, set max depth
    // (prevents infinite search when "go" is sent with no params)
    if (limits.depth == 0 && (limits.time[WHITE] == 0 && limits.time[BLACK] == 0)) {
        limits.depth = MAX_PLY - 1;
    }

    // Check for terminal positions (checkmate/stalemate) before searching
    ExtMove move_list[256];
    ExtMove* move_end = generate_legals(pos, move_list);
    int legal_count = int(move_end - move_list);

    if (legal_count == 0) {
        // No legal moves - game over
        bool is_check = pos.is_check();

        if (is_check) {
            // Checkmate - send mate in 0 (we are mated)
            std::cout << "info depth 0 score mate 0 nodes 0\n";
        } else {
            // Stalemate
            std::cout << "info depth 0 score cp 0 nodes 0\n";
        }
        std::cout.flush();

        // For terminal positions, send bestmove without argument
        // The GUI should recognize game over from the score info
        std::cout << "bestmove\n";
        std::cout.flush();

        if (uci_log) {
            fprintf(uci_log, "TERMINAL: %s legal_count=0\n", is_check ? "checkmate" : "stalemate");
            fflush(uci_log);
        }
        return;
    }

    // Run search
    auto start = std::chrono::steady_clock::now();
    std::cout << "info string DEBUG_BEFORE_SEARCH\n";
    std::cout.flush();
    Move best_move = search(pos, limits);
    auto end = std::chrono::steady_clock::now();
    std::cout << "info string DEBUG_AFTER_SEARCH valid=" << (best_move ? "YES" : "NO") << "\n";
    std::cout.flush();

    int time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Send final info
    uci_info(pos, root_depth, root_score, nodes.load(), time_ms);

    // DEBUG: Log what we're about to send
    std::cerr << "UCI: Sending bestmove, valid=" << (best_move ? "YES" : "NO") << "\n";
    if (best_move) {
        std::cerr << "  from=" << int(best_move.from()) << " to=" << int(best_move.to()) << "\n";
    }
    std::cerr.flush();

    // Send best move with explicit flush
    if (best_move) {
        std::cout << "info string DEBUG_PRINTING_MOVE from=" << int(best_move.from()) << " to=" << int(best_move.to()) << " raw=" << best_move.raw() << "\n";
        std::cout.flush();
        std::string move_str;
        std::ostringstream oss;
        oss << best_move;
        move_str = oss.str();
        std::cout << "info string DEBUG_MOVE_STRING=" << move_str << "\n";
        std::cout.flush();

        // Also log to file
        if (uci_log) {
            fprintf(uci_log, "BESTMOVE valid=1 from=%d to=%d raw=%d str=%s\n",
                    int(best_move.from()), int(best_move.to()), best_move.raw(), move_str.c_str());
            fflush(uci_log);
        }

        std::cout << "bestmove " << best_move << "\n";
    } else {
        std::cout << "info string DEBUG_PRINTING_MOVE INVALID\n";
        std::cout.flush();

        // Also log to file
        if (uci_log) {
            fprintf(uci_log, "BESTMOVE valid=0 (will send 0000)\n");
            fflush(uci_log);
        }

        std::cout << "bestmove 0000\n";
    }
    std::cout.flush();

    // Log to file
    if (uci_log) {
        fprintf(uci_log, "SENT bestmove\n");
        fflush(uci_log);
    }

    // EXTRA DEBUG: If we sent 0000, this is critical - dump more info
    if (!best_move) {
        fprintf(uci_log, "CRITICAL: best_move is 0! Checking position...\n");
        fflush(uci_log);

        // Try to generate moves and see what we get
        ExtMove check_moves[MAX_MOVES];
        ExtMove* check_end = generate<GEN_LEGAL>(pos, check_moves);
        int check_count = int(check_end - check_moves);
        fprintf(uci_log, "CRITICAL: Generated %d legal moves\n", check_count);
        fflush(uci_log);

        if (check_count > 0) {
            fprintf(uci_log, "CRITICAL: First move would be: from=%d to=%d raw=%d\n",
                    int(check_moves[0].move.from()), int(check_moves[0].move.to()),
                    check_moves[0].move.raw());
            fflush(uci_log);
        }
    }
}

void handle_setoption(const std::string& cmd) {
    std::istringstream ss(cmd);
    std::string token;
    ss >> token; // "setoption"
    ss >> token; // "name"

    std::string name;
    std::string value;

    // Get the option name (may contain spaces)
    while (ss >> token && token != "value") {
        if (!name.empty()) name += " ";
        name += token;
    }

    // Get the option value
    while (ss >> token) {
        if (!value.empty()) value += " ";
        value += token;
    }

    // Handle known options
    if (name == "Contempt" || name == "UCI_AnalyseMode") {
        if (name == "Contempt") {
            params.contempt = std::stoi(value);
        }
    } else if (name == "Clear Hash") {
        TT.clear();
    } else if (name == "Hash" || name == "Threads") {
        // These are typically set before initialization
        // We don't dynamically resize TT during play
    }
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
            std::cout << pos.fen() << "\n";
            std::cout.flush();
        }
    }
}

void uci_send(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    std::vprintf(msg, args);
    va_end(args);
    std::cout.flush();
}

} // namespace luminex
