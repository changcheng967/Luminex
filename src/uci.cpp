#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <iostream>
#include <sstream>
#include <string>

namespace luminex {

// Debug flag for detailed move replay logging (set via environment variable or debug option)
static bool DEBUG_REPLAY = false;  // Disabled for normal testing

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
    // Debug: Log every position command
    static int pos_cmd_count = 0;
    std::cerr << "\n=== POSITION COMMAND #" << (++pos_cmd_count) << " ===\n";
    std::cerr << "CMD: " << cmd << "\n";
    std::cerr << "Current FEN before set: " << pos.fen() << "\n";
    std::cerr << "=====================================\n";

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

    // Debug: Log FEN after set
    std::cerr << "After set, FEN: " << pos.fen() << "\n";

    // CRITICAL: Verify initial position was set correctly
    // Log to stdout so we can see it in cutechess output
    static int verify_mode = 0;  // Disabled for performance, enable to debug
    if (verify_mode) {
        std::cout << "info string FEN after set: " << pos.fen() << "\n";
        std::cout.flush();
    }

    // DEBUG: Log initial position
    static int replay_count = 0;
    if (DEBUG_REPLAY) {
        std::cerr << "\n=== REPLAY #" << (++replay_count) << " ===\n";
        std::cerr << "Initial FEN: " << fen << "\n";
        std::cerr << "============================\n";
    }

    // Apply moves
    if (moves_idx != std::string::npos) {
        std::string moves_str = cmd.substr(moves_idx + 6);
        std::istringstream ss(moves_str);
        std::string move_str;

        while (ss >> move_str) {
            // Parse move - use legal move generation for complex moves (promotions, castling)
            if (move_str.length() < 4) continue;

            Square from = Square((move_str[1] - '1') * 8 + (move_str[0] - 'a'));
            Square to = Square((move_str[3] - '1') * 8 + (move_str[2] - 'a'));

            // DEBUG: Log move before applying
            std::string fen_before = DEBUG_REPLAY ? pos.fen() : "";
            if (DEBUG_REPLAY) {
                std::cerr << "\nMove: " << move_str << " (from " << from << " to " << to << ")\n";
                std::cerr << "Before: " << fen_before << "\n";
                std::cerr << "Piece on " << from << ": " << int(pos.piece_on(from)) << "\n";
                std::cerr << "Piece on " << to << ": " << int(pos.piece_on(to)) << "\n";
            }

            // For complex moves (promotions, castling), use legal move generation to find correct flags
            if (move_str.length() > 4 || pos.piece_on(from) == make_piece(pos.side_to_move(), KING)) {
                bool found = false;
                Move matched_move = MOVE_NONE;

                ExtMove moves[256];
                ExtMove* end = generate_legals(pos, moves);
                if (DEBUG_REPLAY) {
                    std::cerr << "Legal moves from " << from << ": ";
                    for (ExtMove* it = moves; it != end; ++it) {
                        if (it->move.from() == from) {
                            std::cerr << it->move.to() << " ";
                        }
                    }
                    std::cerr << "\n";
                }
                for (ExtMove* it = moves; it != end; ++it) {
                    if (it->move.from() == from && it->move.to() == to) {
                        // Check promotion piece if specified
                        if (move_str.length() > 4) {
                            PieceType promo_pt = PT_NONE;
                            switch (move_str[4]) {
                                case 'n': promo_pt = KNIGHT; break;
                                case 'b': promo_pt = BISHOP; break;
                                case 'r': promo_pt = ROOK; break;
                                case 'q': promo_pt = QUEEN; break;
                            }
                            uint16_t flags = it->move.flags();
                            bool matches = false;
                            if (promo_pt == KNIGHT && (flags == MF_PROMO_KNIGHT || flags == MF_CAPTURE_PROMO_KNIGHT)) matches = true;
                            else if (promo_pt == BISHOP && (flags == MF_PROMO_BISHOP || flags == MF_CAPTURE_PROMO_BISHOP)) matches = true;
                            else if (promo_pt == ROOK && (flags == MF_PROMO_ROOK || flags == MF_CAPTURE_PROMO_ROOK)) matches = true;
                            else if (promo_pt == QUEEN && (flags == MF_PROMO_QUEEN || flags == MF_CAPTURE_PROMO_QUEEN)) matches = true;
                            if (!matches) continue;
                        }
                        matched_move = it->move;
                        found = true;
                        if (DEBUG_REPLAY) {
                            std::cerr << "Matched with flags: 0x" << std::hex << matched_move.flags() << std::dec << "\n";
                        }
                        break;
                    }
                }

                if (found) {
                    pos.do_move(matched_move);
                } else {
                    if (DEBUG_REPLAY) {
                        std::cerr << "ERROR: Move not found in legal moves!\n";
                    }
                    // Skip this move - don't corrupt board state
                    continue;
                }
            } else {
                // Simple move parsing for normal moves
                uint16_t flags = MF_QUIET;
                Color us = pos.side_to_move();

                // Check for promotion (shouldn't reach here due to above check, but just in case)
                if (move_str.length() > 4) {
                    bool isCapture = (pos.piece_on(to) != NO_PIECE);
                    switch (move_str[4]) {
                        case 'n': flags = isCapture ? MF_CAPTURE_PROMO_KNIGHT : MF_PROMO_KNIGHT; break;
                        case 'b': flags = isCapture ? MF_CAPTURE_PROMO_BISHOP : MF_PROMO_BISHOP; break;
                        case 'r': flags = isCapture ? MF_CAPTURE_PROMO_ROOK : MF_PROMO_ROOK; break;
                        case 'q': flags = isCapture ? MF_CAPTURE_PROMO_QUEEN : MF_PROMO_QUEEN; break;
                    }
                }

                // Capture check
                Piece target = pos.piece_on(to);
                if (target != NO_PIECE && pos.color_of_piece(target) != us) {
                    if (move_str.length() <= 4) flags = MF_CAPTURE;
                }

                // Double pawn push
                if (flags == MF_QUIET && pos.piece_on(from) == make_piece(us, PAWN) &&
                    file_of(from) == file_of(to)) {
                    Rank from_rank = relative_rank(us, from);
                    Rank to_rank = relative_rank(us, to);
                    if (from_rank == RANK_2 && to_rank == RANK_4) {
                        flags = MF_DOUBLE_PAWN;
                    }
                }

                // En passant
                if (flags == MF_QUIET && pos.piece_on(from) == make_piece(us, PAWN) &&
                    file_of(from) != file_of(to)) {
                    flags = MF_EN_PASSANT;
                }

                if (DEBUG_REPLAY) {
                    std::cerr << "Flags: 0x" << std::hex << flags << std::dec << "\n";
                }

                Move m(from, to, flags);
                pos.do_move(m);
            }

            // DEBUG: Check board state after move
            if (DEBUG_REPLAY) {
                std::cerr << "After:  " << pos.fen() << "\n";
                pos.assert_consistency("handle_position after move");

                // Check king positions
                Square wk = pos.king_sq(WHITE);
                Square bk = pos.king_sq(BLACK);
                std::cerr << "White king at: " << wk << " (piece: " << int(pos.piece_on(wk)) << ")\n";
                std::cerr << "Black king at: " << bk << " (piece: " << int(pos.piece_on(bk)) << ")\n";
            }

            // CRITICAL: Verify board state after each move
            if (verify_mode) {
                std::cout << "info string After " << move_str << ": " << pos.fen() << "\n";
                std::cout.flush();
            }
        }
    }

    // Debug: Log final FEN after all moves
    std::cerr << "Final FEN: " << pos.fen() << "\n";
    std::cerr << "=============================\n";
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

    // Run search
    Move best_move = search(pos, limits);

    // Send final info
    uci_info(pos, root_depth, root_score, nodes.load(), 0);

    // CRITICAL: Validate best_move is LEGAL before sending
    // This prevents illegal moves from being sent to the GUI
    if (best_move && !pos.legal(best_move)) {
        // Move is not legal - find first legal move instead
        std::cerr << "\n=== ILLEGAL BESTMOVE CAUGHT in handle_go ===\n";
        std::cerr << "Move: " << best_move << " (from " << best_move.from() << " to " << best_move.to() << ")\n";
        std::cerr << "FEN: " << pos.fen() << "\n";
        std::cerr << "Piece on source: " << int(pos.piece_on(best_move.from())) << "\n";
        std::cerr << "Piece on dest: " << int(pos.piece_on(best_move.to())) << "\n";
        std::cerr << "=================================================\n";
        ExtMove legal_moves[256];
        ExtMove* legal_end = generate_legals(pos, legal_moves);
        if (legal_end != legal_moves) {
            best_move = legal_moves[0].move;
        } else {
            best_move = MOVE_NONE;  // No legal moves - game over
        }
    }

    // ADDITIONAL CHECK: Validate piece on source square
    // This catches cases where board state is wrong but legal() passes
    if (best_move) {
        Square from = best_move.from();
        Piece pc = pos.piece_on(from);
        if (pc == NO_PIECE || color_of_piece(pc) != pos.side_to_move()) {
            std::cerr << "\n=== NO PIECE OR WRONG COLOR ON SOURCE ===\n";
            std::cerr << "Move: " << best_move << "\n";
            std::cerr << "Source: " << from << ", Piece: " << int(pc) << "\n";
            std::cerr << "FEN: " << pos.fen() << "\n";
            std::cerr << "Using first legal move instead.\n";
            std::cerr << "========================================\n";
            ExtMove legal_moves[256];
            ExtMove* legal_end = generate_legals(pos, legal_moves);
            if (legal_end != legal_moves) {
                best_move = legal_moves[0].move;
            } else {
                best_move = MOVE_NONE;
            }
        }
    }

    // DEBUG: Log board state before sending bestmove
    if (DEBUG_REPLAY && best_move) {
        std::cerr << "\n=== SENDING BESTMOVE ===\n";
        std::cerr << "Move: " << best_move << " (from " << best_move.from() << " to " << best_move.to() << ")\n";
        std::cerr << "Flags: 0x" << std::hex << best_move.flags() << std::dec << "\n";
        std::cerr << "FEN: " << pos.fen() << "\n";
        std::cerr << "Piece on " << best_move.from() << ": " << int(pos.piece_on(best_move.from())) << "\n";
        std::cerr << "Piece on " << best_move.to() << ": " << int(pos.piece_on(best_move.to())) << "\n";
        std::cerr << "========================\n";
    }

    // Send best move
    if (best_move) {
        std::cerr << "SENDING bestmove " << best_move << " (FEN: " << pos.fen() << ")\n";
        std::cout << "bestmove " << best_move << "\n";
    } else {
        std::cerr << "SENDING bestmove 0000 (no move)\n";
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
