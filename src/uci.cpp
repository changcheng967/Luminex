#define _CRT_SECURE_NO_WARNINGS
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
    // Debug logging to verify this is being called
    std::cerr << "DEBUG: ucinewgame received, clearing state" << std::endl;

    // Clear transposition table
    TT.clear();
    TT.new_search();

    // CRITICAL: Reset position to starting position
    // This prevents state from previous game from leaking into new game
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::cerr << "DEBUG: State cleared, position reset" << std::endl;
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

    // CRITICAL FIX: Use legal move generation for position replay
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

            // CRITICAL: Proper move flag detection including En Passant and Capture-Promotion
            PieceType piece_type_from = pos.piece_type_on(from);
            Piece piece_at_to = pos.piece_on(to);
            bool is_capture = (piece_at_to != NO_PIECE);

            // Check for En Passant: pawn moves diagonally to empty ep_square
            bool is_en_passant = (piece_type_from == PAWN &&
                                  file_of(from) != file_of(to) &&  // Diagonal move
                                  to == pos.ep_square());           // To is ep_square

            Move m;
            if (promo_pt != PT_NONE) {
                // Promotion: Use correct flag based on whether it's also a capture
                if (is_capture) {
                    // Capture-Promotion (MF_CAPTURE_PROMO_*)
                    switch (promo_pt) {
                        case QUEEN: m = Move(from, to, MF_CAPTURE_PROMO_QUEEN); break;
                        case ROOK: m = Move(from, to, MF_CAPTURE_PROMO_ROOK); break;
                        case BISHOP: m = Move(from, to, MF_CAPTURE_PROMO_BISHOP); break;
                        case KNIGHT: m = Move(from, to, MF_CAPTURE_PROMO_KNIGHT); break;
                        default: m = Move(from, to, MF_CAPTURE_PROMO_QUEEN); break;
                    }
                } else {
                    // Quiet Promotion (MF_PROMO_*)
                    switch (promo_pt) {
                        case QUEEN: m = Move(from, to, MF_PROMO_QUEEN); break;
                        case ROOK: m = Move(from, to, MF_PROMO_ROOK); break;
                        case BISHOP: m = Move(from, to, MF_PROMO_BISHOP); break;
                        case KNIGHT: m = Move(from, to, MF_PROMO_KNIGHT); break;
                        default: m = Move(from, to, MF_PROMO_QUEEN); break;
                    }
                }
            } else if (is_en_passant) {
                // En Passant capture
                m = Move(from, to, MF_EN_PASSANT);
            } else {
                // Normal move: quiet or capture
                int flag = is_capture ? MF_CAPTURE : MF_QUIET;
                m = Move(from, to, flag);
            }

            // DIAGNOSTIC: Check board state before and after do_move during replay
            static int replay_count = 0;
            if (replay_count < 10) {
                Piece p_before = pos.piece_on(from);
                std::cerr << "REPLAY #" << replay_count << ": " << move_str << " (from=" << from << " to=" << to << ") ";
                std::cerr << "piece at from=" << int(p_before);
                if (p_before != NO_PIECE) {
                    std::cerr << " (" << (pos.color_of_piece(p_before) == WHITE ? "W" : "B") << ")";
                }
                std::cerr << " side_to_move=" << (pos.side_to_move() == WHITE ? "W" : "B") << std::endl;
                replay_count++;
            }

            pos.do_move(m);

            if (replay_count <= 10) {
                Piece p_after = pos.piece_on(to);
                std::cerr << "  after do_move: piece at to=" << int(p_after);
                if (p_after != NO_PIECE) {
                    std::cerr << " (" << (pos.color_of_piece(p_after) == WHITE ? "W" : "B") << ")";
                }
                std::cerr << " side_to_move=" << (pos.side_to_move() == WHITE ? "W" : "B") << std::endl;
            }
        }
    }
}

void handle_go(Position& pos, const std::string& cmd) {
    // CRITICAL: Clear TT before each search to prevent pollution from previous games
    // This is especially important after fixing Zobrist key bugs
    TT.clear();
    TT.new_search();

    // CRITICAL: Verify position consistency before searching
    // This catches board corruption early
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

        return;
    }

    // CRITICAL: Save FEN before search to restore later
    // This protects against position corruption during search
    std::string fen_before_search = pos.fen();
    std::cerr << "BEFORE SEARCH: " << fen_before_search << std::endl;

    // Run search
    Move best_move = search(pos, limits);

    std::cerr << "AFTER SEARCH:  " << pos.fen() << std::endl;

    if (pos.fen() != fen_before_search) {
        std::cerr << "*** POSITION CORRUPTED BY SEARCH ***" << std::endl;
    }

    // CRITICAL: Restore position from saved FEN (in case search corrupted it)
    // This ensures validation happens against correct position state
    pos.set(fen_before_search);

    // DEBUG: Verify position was restored correctly
    std::string fen_after_restore = pos.fen();
    if (fen_before_search != fen_after_restore) {
        std::cerr << "\n=== FEN RESTORE FAILED ===\n";
        std::cerr << "Expected: " << fen_before_search << "\n";
        std::cerr << "Got:      " << fen_after_restore << "\n";
        std::cerr << "===========================\n";
    }

    // CRITICAL: Check board consistency after search
    // Search may have corrupted the position if do_move/undo aren't balanced
    pos.assert_consistency("handle_go after search");

    // Send final info
    uci_info(pos, root_depth, root_score, nodes.load(), 0);

    // Send best move
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
