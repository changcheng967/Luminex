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
#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif

namespace luminex {

// Global volatile stop flag - defined in search.cpp
extern volatile bool g_stop_requested;

// Position for UCI
static Position pos;

// Search thread management
static std::thread search_thread;
static std::mutex io_mutex;

// Debug logging
static FILE* dbglog = nullptr;

// Check for stop command - simply check the atomic flag
bool check_for_stop_command() {
    return stop.load(std::memory_order_seq_cst);
}

// Thread-safe output
static void safe_output(const std::string& msg) {
    std::lock_guard<std::mutex> lock(io_mutex);
    if (dbglog) { fprintf(dbglog, "SEND: [%s]\n", msg.c_str()); fflush(dbglog); }
    std::cout << msg << std::flush;
}

// Thread-safe output for search.cpp (exports the mutex functionality)
void uci_safe_output(const std::string& msg) {
    safe_output(msg);
}

// Debug log function for search.cpp
void uci_debug_log(const char* format, ...) {
    if (dbglog) {
        va_list args;
        va_start(args, format);
        vfprintf(dbglog, format, args);
        va_end(args);
        fflush(dbglog);
    }
}

// Worker function that runs search in a separate thread
static void search_worker(Position pos_copy, Limits lim) {
    if (dbglog) { fprintf(dbglog, "SEARCH_START: depth=%d time[W]=%d time[B]=%d movetime=%d\n",
        lim.depth, lim.time[0], lim.time[1], lim.movetime); fflush(dbglog); }

    Move best_move = search(pos_copy, lim);

    if (dbglog) { fprintf(dbglog, "SEARCH_DONE: about to send bestmove\n"); fflush(dbglog); }

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
    TT.resize(128);
    init_evaluation();
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
    TT.clear();
    TT.new_search();
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

    size_t start = fen.find_first_not_of(" \t\r\n");
    if (start != std::string::npos) {
        fen = fen.substr(start);
    }

    size_t end = fen.find_last_not_of(" \t\r\n");
    if (end != std::string::npos) {
        fen = fen.substr(0, end + 1);
    }

    if (fen == "startpos") {
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    } else if (fen.size() >= 3 && fen.substr(0, 3) == "fen") {
        size_t fen_start = fen.find_first_not_of(" \t", 3);
        if (fen_start != std::string::npos) {
            fen = fen.substr(fen_start);
        } else {
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
    }

    pos.set(fen);

    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 6);
        std::istringstream ss(moves_str);
        std::string move_str;

        while (ss >> move_str) {
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            PieceType promo_pt = PT_NONE;
            if (move_str.length() > 4) {
                switch (move_str[4]) {
                    case 'q': promo_pt = QUEEN; break;
                    case 'r': promo_pt = ROOK; break;
                    case 'b': promo_pt = BISHOP; break;
                    case 'n': promo_pt = KNIGHT; break;
                }
            }

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
    // Wait for any previous search thread to finish
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

    // Launch search thread
    search_thread = std::thread(search_worker, pos, limits);
}

void handle_setoption(const std::string& cmd) {
    std::istringstream ss(cmd);
    std::string token;
    ss >> token;
    ss >> token;

    std::string name;
    std::string value;

    while (ss >> token && token != "value") {
        if (!name.empty()) name += " ";
        name += token;
    }

    while (ss >> token) {
        if (!value.empty()) value += " ";
        value += token;
    }

    if (name == "Contempt" || name == "UCI_AnalyseMode") {
        if (name == "Contempt") {
            params.contempt = std::stoi(value);
        }
    } else if (name == "Clear Hash") {
        TT.clear();
    } else if (name == "Hash") {
        if (!value.empty()) {
            size_t hash_size = std::stoi(value);
            TT.resize(hash_size);
        }
    }
}

void handle_quit() {
    g_stop_requested = true;
    stop.store(true, std::memory_order_seq_cst);
}

void uci_loop() {
    // Debug logging disabled - remove comment to enable
    // dbglog = fopen("C:\\Users\\chang\\Downloads\\Luminex\\luminex_debug.log", "w");

    std::string line;

    while (std::getline(std::cin, line)) {
        // Log received command
        if (dbglog) { fprintf(dbglog, "RECV: [%s]\n", line.c_str()); fflush(dbglog); }

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
            safe_output("readyok\n");
        } else if (cmd == "ucinewgame") {
            stop.store(true, std::memory_order_seq_cst);
            if (search_thread.joinable()) {
                search_thread.join();
            }
            stop.store(false, std::memory_order_seq_cst);
            handle_ucinewgame();
        } else if (cmd == "position") {
            stop.store(true, std::memory_order_seq_cst);
            if (search_thread.joinable()) {
                search_thread.join();
            }
            stop.store(false, std::memory_order_seq_cst);
            handle_position(pos, line);
        } else if (cmd == "go") {
            handle_go(pos, line);
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "stop") {
            if (dbglog) { fprintf(dbglog, "STOP_HANDLER: setting stop=true\n"); fflush(dbglog); }
            g_stop_requested = true;  // Set volatile flag first for immediate visibility
            stop.store(true, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);  // Ensure visibility
            // Do NOT join here - search thread will see flag and terminate
        } else if (cmd == "quit") {
            g_stop_requested = true;
            stop.store(true, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (search_thread.joinable()) {
                search_thread.join();
            }
            break;
        } else if (cmd == "d") {
            safe_output(pos.fen() + "\n");
        }
    }

    if (search_thread.joinable()) {
        search_thread.join();
    }

    if (dbglog) { fclose(dbglog); }
}

void uci_send(const char* msg, ...) {
    va_list args;
    va_start(args, msg);
    std::vprintf(msg, args);
    va_end(args);
    std::cout.flush();
}

} // namespace luminex
