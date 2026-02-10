#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

namespace luminex {

// Position for UCI
static Position pos;

// Simple check for stop - returns current stop flag state
bool check_for_stop_command() {
    return stop.load(std::memory_order_relaxed);
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

            if (matched == MOVE_NONE) {
                std::cout << "info string MOVE_NOT_FOUND: " << move_str << " fen=" << pos.fen() << std::endl;
                std::cout.flush();
                // CRITICAL FIX: Don't reset position - just skip this move
                // The game state might have drifted, but resetting to start makes it worse
                break;
            }
            if (!pos.do_move(matched)) {
                std::cout << "info string DO_MOVE_FAILED: " << move_str << " fen=" << pos.fen() << std::endl;
                std::cout.flush();
                // CRITICAL FIX: Don't reset position - just skip this move
                break;
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    TT.new_search();

    // Verify position consistency before searching
    pos.assert_consistency("handle_go entry");

    // Clear TT on time management to prevent stale entries from causing illegal moves
    // This is a workaround for the TT returning moves from wrong positions
    // TODO: Fix root cause of TT pollution
    TT.clear();

    // Check for terminal positions early (no legal moves)
    ExtMove check_moves[MAX_MOVES];
    ExtMove* check_end = generate<GEN_LEGAL>(pos, check_moves);
    int check_count = int(check_end - check_moves);

    if (check_count == 0) {
        // Terminal position - no legal moves
        std::cout << "info depth 0 score cp 0 nodes 0\n";
        std::cout << "bestmove 0000\n";
        std::cout.flush();
        return;
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

        // CRITICAL FIX: Send bestmove 0000 for terminal positions
        // Sending bare "bestmove" causes UCI protocol desync - CuteChess
        // will read the next line as the move, shifting all commands
        std::cout << "bestmove 0000\n";
        std::cout.flush();

        return;
    }

    // Save FEN before search to restore later
    std::string fen_before_search = pos.fen();
    std::cout << "info string SAVING_FEN before_search=" << fen_before_search << std::endl;
    std::cout << "info string SEARCH_SIDE_TO_MOVE=" << (pos.side_to_move() == WHITE ? "WHITE" : "BLACK") << std::endl;
    std::cout.flush();

    // SIMPLE SYNCHRONOUS SEARCH - no threading
    Move best_move = search(pos, limits);

    // CRITICAL: Restore position from saved FEN
    // Search modifies pos directly via do_move/undo_move. Re-parsing ensures clean state.
    pos.set(fen_before_search);

    // DEBUG: Check if restored position matches expected
    std::string fen_after_restore = pos.fen();
    if (fen_before_search != fen_after_restore) {
        std::cout << "info string FEN_MISMATCH before=" << fen_before_search << " after=" << fen_after_restore << std::endl;
        std::cout.flush();
    }

    // Validate and send bestmove
    if (best_move) {
        // DEBUG: Print position state BEFORE validation
        std::cout << "info string BEFORE_VALIDATION fen=" << pos.fen() << " side_to_move=" << (pos.side_to_move() == WHITE ? "WHITE" : "BLACK") << " best_move=" << best_move << std::endl;
        std::cout.flush();

        // Validate against legal moves
        ExtMove validate_moves[MAX_MOVES];
        ExtMove* validate_end = generate<GEN_LEGAL>(pos, validate_moves);

        // DEBUG: Show what piece is on the from square of best_move
        Square from_sq = best_move.from();
        Piece pc = pos.piece_on(from_sq);
        std::cout << "info string PIECE_ON_FROM from=" << from_sq << " piece=" << int(pc) << " color=" << (pc != NO_PIECE ? (color_of_piece(pc) == WHITE ? "WHITE" : "BLACK") : "NONE") << " side_to_move=" << (pos.side_to_move() == WHITE ? "WHITE" : "BLACK") << std::endl;

        // DEBUG: Count pieces by color to detect color flip
        int white_pieces = 0;
        int black_pieces = 0;
        int white_kings = 0;
        int black_kings = 0;
        for (int sq = 0; sq < 64; ++sq) {
            Piece p = pos.piece_on(Square(sq));
            if (p != NO_PIECE) {
                if (color_of_piece(p) == WHITE) {
                    white_pieces++;
                    if (piece_type_of(p) == KING) white_kings++;
                } else {
                    black_pieces++;
                    if (piece_type_of(p) == KING) black_kings++;
                }
            }
        }
        std::cout << "info string PIECE_COUNT white=" << white_pieces << " black=" << black_pieces << " white_kings=" << white_kings << " black_kings=" << black_kings << std::endl;
        std::cout.flush();

        // DEBUG: Print first few legal moves for sanity check
        std::cout << "info string FIRST_LEGAL_MOVES count=" << (validate_end - validate_moves) << ":";
        int n = (validate_end - validate_moves);
        for (int i = 0; i < (n < 5 ? n : 5); ++i) {
            Square f = validate_moves[i].move.from();
            Piece p = pos.piece_on(f);
            std::cout << " " << validate_moves[i].move << "(piece=" << int(p) << "," << (p != NO_PIECE ? (color_of_piece(p) == WHITE ? "W" : "B") : "N") << ")";
        }
        std::cout << std::endl;
        std::cout.flush();

        bool valid = false;
        for (ExtMove* it = validate_moves; it != validate_end; ++it) {
            if (it->move.raw() == best_move.raw()) {
                valid = true;
                break;
            }
        }

        // CRITICAL SANITY CHECK: Explicitly verify the piece belongs to side to move
        if (valid && pc != NO_PIECE) {
            Color piece_color = color_of_piece(pc);
            Color stm = pos.side_to_move();
            if (piece_color != stm) {
                std::cout << "info string SANITY_CHECK_FAIL piece_color=" << (piece_color == WHITE ? "WHITE" : "BLACK") << " stm=" << (stm == WHITE ? "WHITE" : "BLACK") << std::endl;
                std::cout.flush();
                // Force fallback - this should never happen if everything is working correctly
                valid = false;
            }
        }

        // DEBUG: Print board state if we're about to send a potentially problematic move
        if (valid) {
            std::cout << "info string BOARD_STATE stm=" << (pos.side_to_move() == WHITE ? "W" : "B") << " from=" << from_sq << " piece=" << int(pc) << std::endl;
            // Print pieces on key squares
            Piece p_g8 = pos.piece_on(G8);
            Piece p_f6 = pos.piece_on(F6);
            Piece p_d3 = pos.piece_on(D3);
            std::cout << "info string KEY_SQUARES g8=" << int(p_g8) << " f6=" << int(p_f6) << " d3=" << int(p_d3) << std::endl;
            std::cout.flush();
        }
        // DEBUG: Always log the move being sent
        std::cout << "info string SENDING_MOVE move=" << best_move << " valid=" << valid << " legal_count=" << (validate_end - validate_moves) << " fen=" << pos.fen() << std::endl;
        std::cout.flush();
        if (valid) {
            std::cout << "bestmove " << best_move << "\n";
        } else {
            // DEBUG: Illegal move detected - print all legal moves
            std::cout << "info string ILLEGAL_MOVE_DETECTED move=" << best_move << " fen=" << pos.fen() << " legal_count=" << (validate_end - validate_moves) << std::endl;
            std::cout << "info string LEGAL_MOVES:";
            for (ExtMove* it = validate_moves; it != validate_end; ++it) {
                std::cout << " " << it->move;
            }
            std::cout << std::endl;
            std::cout.flush();
            // Fallback to first legal move
            if (validate_end > validate_moves) {
                std::cout << "info string FALLBACK_MOVE move=" << validate_moves[0].move << std::endl;
                std::cout << "bestmove " << validate_moves[0].move << "\n";
            } else {
                std::cout << "bestmove 0000\n";
            }
        }
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
            handle_isready();
        } else if (cmd == "ucinewgame") {
            handle_ucinewgame();
        } else if (cmd == "position") {
            handle_position(pos, line);
            std::cout << "info string AFTER_POSITION fen=" << pos.fen() << std::endl;
            std::cout.flush();
        } else if (cmd == "go") {
            std::cout << "info string BEFORE_GO fen=" << pos.fen() << std::endl;
            std::cout.flush();
            handle_go(pos, line);
        } else if (cmd == "setoption") {
            handle_setoption(line);
        } else if (cmd == "stop") {
            stop = true;
        } else if (cmd == "quit") {
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
