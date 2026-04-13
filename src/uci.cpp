#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4459) // variable shadowing - intentional for pos parameter
#include "luminex.h"
#include "evaluation.h"
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
#ifdef _WIN32
static HANDLE native_thread_handle = nullptr;
#endif
static std::mutex io_mutex;

// Debug logging
static FILE* dbglog = nullptr;

// Check for stop command - simply check the atomic flag
bool check_for_stop_command() {
    return stop.load(std::memory_order_seq_cst);
}

// Thread-safe output
static void safe_output(const std::string& msg) {
#ifdef WASM_BUILD
    // WASM: use printf (captured by Emscripten print callback), no mutex needed
    printf("%s", msg.c_str());
    fflush(stdout);
#else
    std::lock_guard<std::mutex> lock(io_mutex);
    if (dbglog) { fprintf(dbglog, "SEND: [%s]\n", msg.c_str()); fflush(dbglog); }
    std::cout << msg << std::flush;
#endif
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

// Forward declaration
static void search_worker(Position pos_copy, Limits lim);

// Wait for search thread to finish (handles both std::thread and native thread)
static void wait_for_search_thread() {
    if (search_thread.joinable()) {
        search_thread.join();
    }
#ifdef _WIN32
    if (native_thread_handle) {
        WaitForSingleObject(native_thread_handle, INFINITE);
        CloseHandle(native_thread_handle);
        native_thread_handle = nullptr;
    }
#endif
}

// Launch search with large stack (4MB on Windows, default on other platforms)
static void launch_search_thread(Position pos_copy, Limits lim) {
#ifdef _WIN32
    // Wait for previous native thread if any
    if (native_thread_handle) {
        WaitForSingleObject(native_thread_handle, INFINITE);
        CloseHandle(native_thread_handle);
        native_thread_handle = nullptr;
    }
    auto* thread_args = new std::pair<Position, Limits>(pos_copy, lim);
    native_thread_handle = reinterpret_cast<HANDLE>(_beginthreadex(
        nullptr, 8 * 1024 * 1024,  // 8MB stack
        [](void* arg) -> unsigned {
            auto* p = static_cast<std::pair<Position, Limits>*>(arg);
            search_worker(p->first, p->second);
            delete p;
            return 0;
        },
        thread_args, 0, nullptr));
    if (!native_thread_handle) {
        delete thread_args;
        // Fallback to std::thread
        search_thread = std::thread(search_worker, pos_copy, lim);
    }
#else
    search_thread = std::thread(search_worker, pos_copy, lim);
#endif
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
    init_evaluation();
    safe_output("id name " + std::string(ENGINE_NAME) + " " + ENGINE_VERSION + "\n");
    safe_output("id author " + std::string(ENGINE_AUTHOR) + "\n");
    safe_output("option name Hash type spin default 128 min 1 max 1048576\n");
    safe_output("option name Threads type spin default 1 min 1 max 64\n");
    safe_output("option name Contempt type spin default 0 min -1000 max 1000\n");
    safe_output("option name Clear Hash type button\n");
    // Eval parameters (self-engineered defaults)
    safe_output("option name BishopPairMG type spin default 40 min -100 max 200\n");
    safe_output("option name BishopPairEG type spin default 100 min -100 max 300\n");
    safe_output("option name RookOpenMG type spin default 25 min -50 max 150\n");
    safe_output("option name RookOpenEG type spin default 40 min -50 max 150\n");
    safe_output("option name RookSemiOpenMG type spin default 15 min -50 max 100\n");
    safe_output("option name RookSemiOpenEG type spin default 20 min -50 max 100\n");
    safe_output("option name Rook7thMG type spin default 30 min -50 max 100\n");
    safe_output("option name Rook7thEG type spin default 25 min -50 max 150\n");
    safe_output("option name PawnShieldCenter type spin default 12 min -20 max 40\n");
    safe_output("option name PawnShieldKnight type spin default 15 min -20 max 40\n");
    safe_output("option name PawnShieldRook type spin default 8 min -20 max 30\n");
    safe_output("option name PawnStorm type spin default 10 min 0 max 30\n");
    safe_output("option name OpenFilePenaltyMG type spin default 20 min -50 max 100\n");
    safe_output("option name OpenFilePenaltyEG type spin default 15 min -50 max 100\n");
    safe_output("option name OutpostKnightMG type spin default 30 min -20 max 60\n");
    safe_output("option name OutpostKnightEG type spin default 15 min -20 max 40\n");
    safe_output("option name OutpostBishopMG type spin default 40 min -20 max 100\n");
    safe_output("option name OutpostBishopEG type spin default 25 min -20 max 60\n");
    safe_output("option name HangingPawnMG type spin default 10 min -20 max 50\n");
    safe_output("option name HangingPawnEG type spin default 35 min -20 max 100\n");
    safe_output("option name FarKnightMG type spin default 15 min -50 max 60\n");
    safe_output("option name FarKnightEG type spin default 5 min -50 max 60\n");
    safe_output("option name FarBishopMG type spin default 10 min -50 max 40\n");
    safe_output("option name FarBishopEG type spin default 5 min -50 max 40\n");
    safe_output("uciok\n");
}

void handle_isready() {
    safe_output("readyok\n");
}

void handle_ucinewgame() {
    TT.clear();
    TT.new_search();
    clear_correction_history();
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
    wait_for_search_thread();

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

    // "go" without any time controls or depth: search indefinitely until "stop"
    if (limits.depth == 0 && !limits.infinite && limits.movetime == 0 &&
        limits.time[WHITE] == 0 && limits.time[BLACK] == 0 && limits.nodes == 0) {
        limits.infinite = true;
    }

#ifdef WASM_BUILD
    // WASM: run search synchronously (we're already in a web worker, no threading needed)
    g_stop_requested = false;
    stop.store(false, std::memory_order_seq_cst);
    Move best_move = search(pos, limits);
    std::ostringstream oss;
    if (best_move != MOVE_NONE) {
        oss << "bestmove " << best_move << "\n";
    } else {
        oss << "bestmove 0000\n";
    }
    safe_output(oss.str());
#else
    // Native: launch search thread with large stack to prevent stack overflow
    launch_search_thread(pos, limits);
#endif
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
    } else if (name == "BishopPairMG") { g_eval_params.bishop_pair_mg = std::stoi(value);
    } else if (name == "BishopPairEG") { g_eval_params.bishop_pair_eg = std::stoi(value);
    } else if (name == "RookOpenMG") { g_eval_params.rook_open_mg = std::stoi(value);
    } else if (name == "RookOpenEG") { g_eval_params.rook_open_eg = std::stoi(value);
    } else if (name == "RookSemiOpenMG") { g_eval_params.rook_semi_open_mg = std::stoi(value);
    } else if (name == "RookSemiOpenEG") { g_eval_params.rook_semi_open_eg = std::stoi(value);
    } else if (name == "Rook7thMG") { g_eval_params.rook_7th_mg = std::stoi(value);
    } else if (name == "Rook7thEG") { g_eval_params.rook_7th_eg = std::stoi(value);
    } else if (name == "PawnShieldCenter") { g_eval_params.pawn_shield_center = std::stoi(value);
    } else if (name == "PawnShieldKnight") { g_eval_params.pawn_shield_knight = std::stoi(value);
    } else if (name == "PawnShieldRook") { g_eval_params.pawn_shield_rook = std::stoi(value);
    } else if (name == "PawnStorm") { g_eval_params.pawn_storm = std::stoi(value);
    } else if (name == "OpenFilePenaltyMG") { g_eval_params.open_file_penalty_mg = std::stoi(value);
    } else if (name == "OpenFilePenaltyEG") { g_eval_params.open_file_penalty_eg = std::stoi(value);
    } else if (name == "OutpostKnightMG") { g_eval_params.outpost_knight_mg = std::stoi(value);
    } else if (name == "OutpostKnightEG") { g_eval_params.outpost_knight_eg = std::stoi(value);
    } else if (name == "OutpostBishopMG") { g_eval_params.outpost_bishop_mg = std::stoi(value);
    } else if (name == "OutpostBishopEG") { g_eval_params.outpost_bishop_eg = std::stoi(value);
    } else if (name == "HangingPawnMG") { g_eval_params.hanging_pawn_mg = std::stoi(value);
    } else if (name == "HangingPawnEG") { g_eval_params.hanging_pawn_eg = std::stoi(value);
    } else if (name == "FarKnightMG") { g_eval_params.far_knight_mg = std::stoi(value);
    } else if (name == "FarKnightEG") { g_eval_params.far_knight_eg = std::stoi(value);
    } else if (name == "FarBishopMG") { g_eval_params.far_bishop_mg = std::stoi(value);
    } else if (name == "FarBishopEG") { g_eval_params.far_bishop_eg = std::stoi(value);
    } else if (name == "Threads") {
        int t = std::stoi(value);
        if (t < 1) t = 1;
        if (t > 64) t = 64;
        num_threads = t;
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
    // Debug logging disabled
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
            wait_for_search_thread();
            stop.store(false, std::memory_order_seq_cst);
            handle_ucinewgame();
        } else if (cmd == "position") {
            stop.store(true, std::memory_order_seq_cst);
            wait_for_search_thread();
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
            wait_for_search_thread();
            break;
        } else if (cmd == "d") {
            safe_output(pos.fen() + "\n");
        }
    }

    wait_for_search_thread();

    if (dbglog) { fclose(dbglog); }
}

// Process a single UCI command (for WASM/web use)
// This is the same logic as the while loop in uci_loop() but processes one command
void process_uci_command(const std::string& line) {
    if (line.empty()) return;

    std::string cmd_line = line;
    if (!cmd_line.empty() && cmd_line.back() == '\r') {
        cmd_line.pop_back();
    }
    if (cmd_line.empty()) return;

    std::istringstream ss(cmd_line);
    std::string cmd;
    ss >> cmd;

    if (cmd == "uci") {
        handle_uci();
    } else if (cmd == "isready") {
        safe_output("readyok\n");
    } else if (cmd == "ucinewgame") {
        stop.store(true, std::memory_order_seq_cst);
        wait_for_search_thread();
        stop.store(false, std::memory_order_seq_cst);
        handle_ucinewgame();
    } else if (cmd == "position") {
        stop.store(true, std::memory_order_seq_cst);
        wait_for_search_thread();
        stop.store(false, std::memory_order_seq_cst);
        handle_position(pos, cmd_line);
    } else if (cmd == "go") {
        handle_go(pos, cmd_line);
    } else if (cmd == "setoption") {
        handle_setoption(cmd_line);
    } else if (cmd == "stop") {
        g_stop_requested = true;
        stop.store(true, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    } else if (cmd == "quit") {
        g_stop_requested = true;
        stop.store(true, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        wait_for_search_thread();
    } else if (cmd == "d") {
        safe_output(pos.fen() + "\n");
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
