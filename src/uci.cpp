#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

namespace luminex {

// Position for UCI
static Position pos;

// Search thread management
static std::thread search_thread;
static std::mutex io_mutex;  // Mutex for stdout operations

// Check for stop command - simply check the atomic flag
// The main UCI loop handles receiving "stop" and setting this flag
bool check_for_stop_command() {
    return stop.load(std::memory_order_relaxed);
}

// Thread-safe output
static void safe_output(const std::string& msg) {
    std::lock_guard<std::mutex> lock(io_mutex);
    std::cout << msg << std::flush;
}

// Thread-safe output for search.cpp (exports the mutex functionality)
void uci_safe_output(const std::string& msg) {
    safe_output(msg);
}

// Worker function that runs search in a separate thread
static void search_worker(Position pos_copy, Limits lim) {
    Move best_move = search(pos_copy, lim);

    // Output best move directly from search thread (thread-safe)
    std::ostringstream oss;
    if (best_move != MOVE_NONE) {
        oss << "bestmove " << best_move << "\n";
    } else {
        oss << "bestmove 0000\n";
    }
    safe_output(oss.str());
}

void handle_uci() {
    // CRITICAL FIX: Initialize TT with default size (128MB)
    // Without this, table.empty() is true and all probes return dummy entry
    TT.resize(128);

    // CRITICAL: Initialize evaluation tables (PST mirroring)
    init_evaluation();

    // CRITICAL: Send each line separately with immediate flush (thread-safe)
    safe_output("id name " + std::string(ENGINE_NAME) + " " + ENGINE_VERSION + "\n");
    safe_output("id author " + std::string(ENGINE_AUTHOR) + "\n");
    safe_output("option name Hash type spin default 128 min 1 max 1048576\n");
    safe_output("option name Contempt type spin default 0 min -1000 max 1000\n");
    safe_output("option name Clear Hash type button\n");
    safe_output("uciok\n");
}

void handle_isready() {
    safe_output("readyok\n");
}

void handle_ucinewgame() {
    // Clear transposition table
    TT.clear();
    TT.new_search();

    // Reset position to starting position
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
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

    // Trim leading whitespace
    size_t start = fen.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        fen = fen.substr(start);
    }

    // Trim trailing whitespace (including \r from Windows line endings)
    size_t end = fen.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
        fen = fen.substr(0, end + 1);
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

    // Replay moves using legal move generation
    // This ensures correct move flags and prevents position corruption
    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 6);
        std::istringstream ss(moves_str);
        std::string move_str;

        while (ss >> move_str) {
            // Parse move directly from UCI string
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            // Determine promotion type if present
            PieceType promo_pt = PT_NONE;
            if (move_str.length() > 4) {
                switch (move_str[4]) {
                    case 'q': promo_pt = QUEEN; break;
                    case 'r': promo_pt = ROOK; break;
                    case 'b': promo_pt = BISHOP; break;
                    case 'n': promo_pt = KNIGHT; break;
                }
            }

            // Match the UCI move against the legal move list to get correct flags
            // This is the standard approach (used by Stockfish) and avoids all
            // flag-construction bugs for castling, en passant, captures, promotions, etc.
            ExtMove legal_moves[MAX_MOVES];
            ExtMove* legal_end = generate<GEN_LEGAL>(pos, legal_moves);
            Move matched = MOVE_NONE;
            for (ExtMove* lit = legal_moves; lit != legal_end; ++lit) {
                if (lit->move.from() == from && lit->move.to() == to) {
                    if (promo_pt != PT_NONE) {
                        if (lit->move.is_promotion() && lit->move.promotion_type() == promo_pt) {
                            matched = lit->move;
                            break;
                        }
                    } else if (!lit->move.is_promotion()) {
                        matched = lit->move;
                        break;
                    }
                }
            }

            if (matched == MOVE_NONE) {
                // Move not found in legal moves - position may be corrupt
                // Reset to startpos as a safety measure
                pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                return;
            }
            if (!pos.do_move(matched)) {
                pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                return;
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    // Wait for any previous search thread to finish and clean up
    if (search_thread.joinable()) {
        search_thread.join();
    }

    TT.new_search();

    std::istringstream ss(cmd);
    std::string token;
    limits = Limits();

    while (ss >> token) {
        if (token == "depth") ss >> limits.depth;
        else if (token == "nodes") ss >> limits.nodes;
        else if (token == "movetime") ss >> limits.movetime;
        else if (token == "infinite") limits.infinite = true;
        else if (token == "wtime") ss >> limits.time[WHITE];
        else if (token == "btime") ss >> limits.time[BLACK];
        else if (token == "winc") ss >> limits.inc[WHITE];
        else if (token == "binc") ss >> limits.inc[BLACK];
        else if (token == "movestogo") ss >> limits.movestogo;
    }

    if (limits.depth == 0 && !limits.infinite && limits.movetime == 0 &&
        limits.time[WHITE] == 0 && limits.time[BLACK] == 0) {
        limits.depth = 6;
    }

    // Launch search thread - it will output bestmove when done
    search_thread = std::thread(search_worker, pos, limits);

    // Return immediately - main loop continues processing commands
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
    } else if (name == "Hash") {
        // CRITICAL FIX: Actually resize the TT when Hash option is set
        if (!value.empty()) {
            size_t hash_size = std::stoi(value);
            TT.resize(hash_size);
        }
    } else if (name == "Threads") {
        // Threads not implemented yet
    }
}

void handle_stop() {
    stop = true;
    // With synchronous search, the stop flag will be checked on next check_time()
    // No need to wait for search or send bestmove - search() handles that
}

void handle_quit() {
    stop = true;
    // Exit will be handled by uci_loop breaking
}

void uci_loop() {
    std::string line;

    while (std::getline(std::cin, line)) {
        // Strip trailing \r (Windows line endings from GUIs)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            handle_uci();
        } else if (cmd == "isready") {
            // Always respond to isready immediately, even during search (thread-safe)
            safe_output("readyok\n");
        } else if (cmd == "ucinewgame") {
            // Stop any running search
            stop = true;
            if (search_thread.joinable()) {
                search_thread.join();
            }
            stop = false;
            handle_ucinewgame();
        } else if (cmd == "position") {
            // If search is running, stop it first
            stop = true;
            if (search_thread.joinable()) {
                search_thread.join();
            }
            stop = false;
            handle_position(pos, line);
        } else if (cmd == "go") {
            handle_go(pos, line);
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "stop") {
            stop = true;
            // Don't wait - let search thread finish and output bestmove on its own
        } else if (cmd == "quit") {
            stop = true;
            if (search_thread.joinable()) {
                search_thread.join();
            }
            break;
        } else if (cmd == "d") {
            // Debug: print board
            safe_output(pos.fen() + "\n");
        }
    }

    // Clean up search thread on exit
    if (search_thread.joinable()) {
        search_thread.join();
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
