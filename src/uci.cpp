#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace luminex {

// Position for UCI
static Position pos;

// Check if there's input available on stdin without blocking
// Returns true if "stop" or "quit" command was received
// This allows the engine to respond to stop commands during search
//
// IMPORTANT: We only peek at stdin and consume stop/quit commands.
// All other commands must be left for the main loop to process.
// Since we can't easily "put back" input we've read, we use a
// platform-specific approach to only read when we're sure it's stop/quit.
bool check_for_stop_command() {
#ifdef _WIN32
    static HANDLE hStdin = INVALID_HANDLE_VALUE;
    if (hStdin == INVALID_HANDLE_VALUE) {
        hStdin = GetStdHandle(STD_INPUT_HANDLE);
    }
    DWORD bytesAvailable = 0;
    if (PeekNamedPipe(hStdin, nullptr, 0, nullptr, &bytesAvailable, nullptr)) {
        if (bytesAvailable > 0) {
            // Input available - peek at it without consuming
            // We need to check if it starts with 's' (stop) or 'q' (quit)
            // If it's neither, we leave it for the main loop

            // Read the first few bytes to check the command
            char buffer[32];
            DWORD bytesRead = 0;
            if (PeekNamedPipe(hStdin, buffer, sizeof(buffer) - 1, &bytesRead, nullptr, nullptr)) {
                if (bytesRead > 0) {
                    buffer[bytesRead] = '\0';

                    // Check if it starts with "stop" or "quit"
                    // Commands are newline-terminated, so look for that too
                    bool is_stop = (bytesRead >= 4 && strncmp(buffer, "stop", 4) == 0);
                    bool is_quit = (bytesRead >= 4 && strncmp(buffer, "quit", 4) == 0);

                    if (is_stop || is_quit) {
                        // This is a stop/quit command - consume and process it
                        std::string line;
                        if (std::getline(std::cin, line)) {
                            if (!line.empty() && line.back() == '\r') {
                                line.pop_back();
                            }
                            if (line == "stop" || line == "quit") {
                                stop = true;
                                return true;
                            }
                        }
                    }
                    // Not a stop/quit command - leave it for main loop
                    // Don't consume anything, just return
                }
            }
        }
    }
    return false;
#else
    // Non-Windows platforms: could use select() or poll()
    // For now, just return false
    return false;
#endif
}

void handle_uci() {
    // CRITICAL FIX: Initialize TT with default size (128MB)
    // Without this, table.empty() is true and all probes return dummy entry
    TT.resize(128);

    // CRITICAL: Initialize evaluation tables (PST mirroring)
    init_evaluation();

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

            if (matched == MOVE_NONE || !pos.do_move(matched)) {
                break;  // No matching legal move found or do_move failed
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    TT.new_search();

    // Verify position consistency before searching
    pos.assert_consistency("handle_go entry");

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
            ss >> limits.movestogo;
        }
    }

    // If no depth, time control, movetime, or infinite is specified, set reasonable default depth
    // This prevents infinite search when "go" is sent with no params
    if (limits.depth == 0 && !limits.infinite && limits.movetime == 0 &&
        limits.time[WHITE] == 0 && limits.time[BLACK] == 0) {
        limits.depth = 6;  // Safe default depth for bare "go" command
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

        return;
    }

    // Save FEN before search to restore later
    std::string fen_before_search = pos.fen();

    // Run search
    Move best_move = search(pos, limits);

    // CRITICAL: Restore position from saved FEN
    // Search modifies pos directly via do_move/undo_move, and any failure
    // or exception could corrupt the state. Re-parsing from FEN ensures clean state.
    pos.set(fen_before_search);

    // NUCLEAR SAFETY CHECK: Validate best_move against complete GEN_LEGAL list
    // We must generate moves TWICE to work around potential state corruption:
    // 1. First generation: check if best_move is in legal list
    // 2. If not valid: second generation for fallback
    // This uses fresh state for each generation to minimize corruption risk.
    bool best_move_is_valid = false;
    if (best_move) {
        // First, verify basic move geometry
        Square from = best_move.from();
        Square to = best_move.to();

        if (from < SQUARE_NONE && to < SQUARE_NONE) {
            Piece pc = pos.piece_on(from);
            if (pc != NO_PIECE && pos.color_of_piece(pc) == pos.side_to_move()) {
                // Basic geometry passes, now check against full legal list
                // We re-parse position to get absolutely fresh state
                Position fresh_pos;
                fresh_pos.set(fen_before_search);
                ExtMove moves[MAX_MOVES];
                ExtMove* end = generate<GEN_LEGAL>(fresh_pos, moves);
                uint32_t best_raw = best_move.raw();

                for (ExtMove* it = moves; it != end; ++it) {
                    if (it->move.raw() == best_raw) {
                        best_move_is_valid = true;
                        break;
                    }
                }
            }
        }
    }

    // If best_move failed validation, use first legal move from FRESH position
    if (!best_move_is_valid) {
        Position fresh_pos;
        fresh_pos.set(fen_before_search);
        ExtMove moves[MAX_MOVES];
        ExtMove* end = generate<GEN_LEGAL>(fresh_pos, moves);
        if (end > moves) {
            best_move = moves[0].move;
        } else {
            best_move = MOVE_NONE;  // Terminal position
        }
    }

    // Send final info - use root_depth-1 since loop incremented it
    // Note: This is redundant as search() already outputs info, but kept for compatibility
    int final_depth = (root_depth > 1) ? root_depth - 1 : 1;
    uci_info(pos, final_depth, root_score, nodes.load(), 0);

    if (best_move) {
        std::cout << "bestmove " << best_move << "\n";
    } else {
        std::cout << "bestmove 0000\n";
    }
    std::cout.flush();
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
}

void handle_quit() {
    stop = true;
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
