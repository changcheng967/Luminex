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

    // DEBUG: Log position replay for debugging
    static int move_num = 0;
    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 6);
        std::istringstream ss(moves_str);
        std::string move_str;

        move_num = 0;
        while (ss >> move_str) {
            move_num++;
            // Parse move directly from UCI string
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            // Determine move flags based on current position
            Color us = pos.side_to_move();
            Piece piece_from = pos.piece_on(from);
            uint16_t flags = MF_QUIET;

            // Check if it's a capture
            Piece piece_to = pos.piece_on(to);
            if (piece_to != NO_PIECE && pos.color_of_piece(piece_to) != us) {
                flags = MF_CAPTURE;
            }

            // Check for promotion
            PieceType promo_pt = PT_NONE;
            if (move_str.length() > 4) {
                switch (move_str[4]) {
                    case 'q': promo_pt = QUEEN; break;
                    case 'r': promo_pt = ROOK; break;
                    case 'b': promo_pt = BISHOP; break;
                    case 'n': promo_pt = KNIGHT; break;
                }
                if (flags == MF_CAPTURE) {
                    if (promo_pt == KNIGHT) flags = MF_CAPTURE_PROMO_KNIGHT;
                    else if (promo_pt == BISHOP) flags = MF_CAPTURE_PROMO_BISHOP;
                    else if (promo_pt == ROOK) flags = MF_CAPTURE_PROMO_ROOK;
                    else if (promo_pt == QUEEN) flags = MF_CAPTURE_PROMO_QUEEN;
                } else {
                    if (promo_pt == KNIGHT) flags = MF_PROMO_KNIGHT;
                    else if (promo_pt == BISHOP) flags = MF_PROMO_BISHOP;
                    else if (promo_pt == ROOK) flags = MF_PROMO_ROOK;
                    else if (promo_pt == QUEEN) flags = MF_PROMO_QUEEN;
                }
            }

            // Check for castling (king moving two squares)
            if (piece_from == make_piece(us, KING)) {
                int file_diff = std::abs(int(file_of(from)) - int(file_of(to)));
                if (file_diff == 2) {
                    flags = (to > from) ? MF_CASTLING_KING : MF_CASTLING_QUEEN;
                }
            }

            // Check for double pawn push
            if (flags == MF_QUIET && piece_from == make_piece(us, PAWN)) {
                Rank from_rank = relative_rank(us, from);
                Rank to_rank = relative_rank(us, to);
                if (from_rank == RANK_2 && to_rank == RANK_4) {
                    flags = MF_DOUBLE_PAWN;
                }
            }

            // Check for en passant (diagonal pawn move to empty square)
            if (flags == MF_QUIET && piece_from == make_piece(us, PAWN)) {
                if (file_of(from) != file_of(to) && piece_to == NO_PIECE) {
                    if (to == pos.ep_square()) {
                        flags = MF_EN_PASSANT;
                    }
                }
            }

            // DEBUG: Verify piece exists on source square
            if (piece_from == NO_PIECE || pos.color_of_piece(piece_from) != us) {
                std::cerr << "\n=== REPLAY ERROR at move " << move_num << " ===\n";
                std::cerr << "Move: " << move_str << "\n";
                std::cerr << "From square: " << from << "\n";
                std::cerr << "Piece on from: " << int(piece_from) << "\n";
                std::cerr << "Side to move: " << (us == WHITE ? "WHITE" : "BLACK") << "\n";
                std::cerr << "FEN before: " << pos.fen() << "\n";
                std::cerr << "===================================\n";
            }

            Move m(from, to, flags);
            pos.do_move(m);
        }

        // DEBUG: Log final position after replay
        std::cerr << "Replay completed: " << move_num << " moves, FEN: " << pos.fen() << "\n";
    }
}

void handle_go(Position& pos, const std::string& cmd) {
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

    // Run search
    Move best_move = search(pos, limits);

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

    // TARGETED DEBUG: Check if d8e8 or e8d8 moves (king position issues)
    if ((best_move.from() == Square(3) && best_move.to() == Square(4)) ||  // d8=3, e8=4
        (best_move.from() == Square(4) && best_move.to() == Square(3))) {
        std::cerr << "\n=== D8/E8 MOVE DEBUG ===\n";
        std::cerr << "Best move: " << best_move << "\n";
        std::cerr << "FEN: " << pos.fen() << "\n";
        std::cerr << "White king at: " << pos.king_sq(WHITE) << " (should be 0=e1)\n";
        std::cerr << "Black king at: " << pos.king_sq(BLACK) << " (should be 4=e8)\n";
        std::cerr << "Piece on d8(3): " << int(pos.piece_on(Square(3))) << "\n";
        std::cerr << "Piece on e8(4): " << int(pos.piece_on(Square(4))) << "\n";
        std::cerr << "========================\n";
    }

    // CRITICAL FIX: Match by from/to, then use the legal move object with correct flags
    // This prevents illegal moves caused by wrong flags from TT or stale position data
    // For promotions, also match promotion piece type
    if (best_move != MOVE_NONE) {
        ExtMove legal_moves[256];
        ExtMove* legal_end = generate<GEN_LEGAL>(pos, legal_moves);

        // Find matching move by from/to (and promotion type for promotions)
        bool found_match = false;
        for (ExtMove* it = legal_moves; it != legal_end; ++it) {
            bool match = (it->move.from() == best_move.from() &&
                          it->move.to() == best_move.to());

            // For promotions, also verify promotion type matches
            if (match && (best_move.flags() & 0xF)) {
                // Check if this is a promotion (flags 0x8-0xF are promotions)
                bool is_promotion = (best_move.flags() >= 0x8 && best_move.flags() <= 0xF);
                if (is_promotion) {
                    match = (it->move.flags() == best_move.flags());
                }
            }

            if (match) {
                // Use the legal move object instead - ensures correct flags!
                best_move = it->move;
                found_match = true;
                break;
            }
        }

        // If no match found, fall back to first legal move
        if (!found_match && legal_end != legal_moves) {
            best_move = legal_moves[0].move;
        }

        // Final sanity check
        if (best_move && !pos.legal(best_move)) {
            if (legal_end != legal_moves) {
                best_move = legal_moves[0].move;
            } else {
                best_move = MOVE_NONE;
            }
        }
    }

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
