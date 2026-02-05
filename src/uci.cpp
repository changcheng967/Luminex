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
    // CRITICAL FIX: Initialize TT with default size (128MB)
    // Without this, table.empty() is true and all probes return dummy entry
    TT.resize(128);

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

            // CRITICAL: Proper move flag detection including En Passant, Castling, and Capture-Promotion
            PieceType piece_type_from = pos.piece_type_on(from);
            Piece piece_at_to = pos.piece_on(to);
            bool is_capture = (piece_at_to != NO_PIECE);

            // Check for Castling: king moves two squares horizontally
            bool is_castling = (piece_type_from == KING && std::abs(int(from) - int(to)) == 2);

            // Check for En Passant: pawn moves diagonally to empty ep_square
            bool is_en_passant = (piece_type_from == PAWN &&
                                  file_of(from) != file_of(to) &&  // Diagonal move
                                  to == pos.ep_square());           // To is ep_square

            Move m;
            if (is_castling) {
                // Castling: determine kingside or queenside based on destination file
                if (file_of(to) > FILE_E) {
                    m = Move(from, to, MF_CASTLING_KING);  // Kingside (e1g1 or e8g8)
                } else {
                    m = Move(from, to, MF_CASTLING_QUEEN); // Queenside (e1c1 or e8c8)
                }
            } else if (promo_pt != PT_NONE) {
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

            // Execute move - skip if do_move fails
            if (!pos.do_move(m)) {
                break;  // Position desynchronized, skip remaining moves
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

    // Save FEN before search to restore later
    std::string fen_before_search = pos.fen();

    // Run search
    Move best_move = search(pos, limits);

    // Restore position from saved FEN
    pos.set(fen_before_search);

#ifndef NDEBUG
    // Check board consistency after search (debug builds only)
    pos.assert_consistency("handle_go after search");
#endif

    // Validate best_move before sending - multiple layers of defense
    if (best_move) {
        Square from = best_move.from();
        Square to = best_move.to();

        // CRITICAL: Validate square bounds first
        // board[] has 64 elements (0-63 valid), SQUARE_NONE = 64 is invalid
        if (from >= SQUARE_NONE || to >= SQUARE_NONE) {
            // Invalid squares - generate a legal move as fallback
            ExtMove moves[MAX_MOVES];
            ExtMove* end = generate<GEN_LEGAL>(pos, moves);
            if (end > moves) {
                best_move = moves[0].move;
            } else {
                best_move = MOVE_NONE;
            }
        } else {
            // Squares are valid - check legality and side match
            Piece pc = pos.piece_on(from);
            Color piece_color = (pc != NO_PIECE) ? pos.color_of_piece(pc) : NO_COLOR;
            bool is_legal = pos.legal(best_move);
            bool side_match = (piece_color == pos.side_to_move());

            if (!side_match || !is_legal) {
                // Generate a legal move as fallback
                ExtMove moves[MAX_MOVES];
                ExtMove* end = generate<GEN_LEGAL>(pos, moves);
                if (end > moves) {
                    best_move = moves[0].move;
                } else {
                    best_move = MOVE_NONE;
                }
            }
        }
    }

    // Send final info
    uci_info(pos, root_depth, root_score, nodes.load(), 0);

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
