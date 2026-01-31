#define _CRT_SECURE_NO_WARNINGS
#include "luminex.h"
#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>

namespace {
std::ofstream debug_log;
bool debug_initialized = false;

void init_debug() {
    if (!debug_initialized) {
        debug_log.open("C:\\Users\\chang\\Downloads\\Luminex\\luminex_debug.txt", std::ios::out | std::ios::trunc);
        debug_initialized = true;
    }
}
}

namespace luminex {

// Position for UCI
static Position pos;

void handle_uci() {
    init_debug();

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
            Color us = pos.side_to_move();

            // Check for promotion
            if (move_str.length() > 4) {
                Piece target = pos.piece_on(to);
                bool isCapture = (target != NO_PIECE && pos.color_of_piece(target) != us);
                switch (move_str[4]) {
                    case 'n': flags = isCapture ? MF_CAPTURE_PROMO_KNIGHT : MF_PROMO_KNIGHT; break;
                    case 'b': flags = isCapture ? MF_CAPTURE_PROMO_BISHOP : MF_PROMO_BISHOP; break;
                    case 'r': flags = isCapture ? MF_CAPTURE_PROMO_ROOK : MF_PROMO_ROOK; break;
                    case 'q': flags = isCapture ? MF_CAPTURE_PROMO_QUEEN : MF_PROMO_QUEEN; break;
                }
            } else {
                Piece target = pos.piece_on(to);
                if (target != NO_PIECE && pos.color_of_piece(target) != us) {
                    flags = MF_CAPTURE;
                }
            }

            // Check for castling - only if the piece on from is OUR king
            if (pos.piece_on(from) == make_piece(us, KING) && std::abs(file_of(to) - file_of(from)) > 1) {
                flags = (file_of(to) > file_of(from)) ? MF_CASTLING_KING : MF_CASTLING_QUEEN;
            }

            // Check for double pawn push (MUST be checked before en passant)
            if (flags == MF_QUIET && pos.piece_on(from) == make_piece(us, PAWN) &&
                file_of(from) == file_of(to)) {
                // Pawn moving straight - check if it's a double push
                Rank from_rank = relative_rank(us, from);
                Rank to_rank = relative_rank(us, to);
                if (from_rank == RANK_2 && to_rank == RANK_4) {
                    flags = MF_DOUBLE_PAWN;
                }
            }

            // Check for en passant (ONLY if not already flagged as double pawn push)
            if (flags == MF_QUIET && pos.piece_on(from) == make_piece(us, PAWN) &&
                file_of(from) != file_of(to)) {
                // Diagonal pawn move - check if target is empty (en passant)
                Piece target = pos.piece_on(to);
                if (target == NO_PIECE) {
                    flags = MF_EN_PASSANT;
                }
            }

            // DEBUG: Validate board state BEFORE move
            Piece from_piece_before = pos.piece_on(from);

            // Check if from_piece belongs to us
            if (from_piece_before != NO_PIECE) {
                Color piece_color = pos.color_of_piece(from_piece_before);
                if (piece_color != us) {
                    debug_log << "\n=== UCI CORRUPTION: Trying to move enemy piece ===\n";
                    debug_log << "Move: " << from << to << "\n";
                    debug_log << "From square has " << int(from_piece_before);
                    debug_log << " (color=" << piece_color << ", us=" << us << ")\n";
                    debug_log << "FEN before: " << pos.fen() << "\n";
                    debug_log << "===============================================\n";
                    debug_log.flush();
                }
            } else {
                debug_log << "\n=== UCI CORRUPTION: Moving from empty square ===\n";
                debug_log << "Move: " << from << to << "\n";
                debug_log << "FEN before: " << pos.fen() << "\n";
                debug_log << "===============================================\n";
                debug_log.flush();
            }

            // DEBUG: Log move application
            debug_log << "\n=== APPLYING UCI MOVE: " << move_str << " ===\n";
            debug_log << "From: " << from << " (" << int(from_piece_before);
            if (from_piece_before != NO_PIECE) {
                debug_log << " = " << (pos.color_of_piece(from_piece_before) == WHITE ? "W" : "B") << piece_type_of(from_piece_before);
            }
            debug_log << ")\n";
            debug_log << "To: " << to << "\n";
            debug_log << "Flags: " << std::hex << flags << std::dec << "\n";
            debug_log << "FEN before: " << pos.fen() << "\n";

            Move m(from, to, flags);
            pos.do_move(m);

            debug_log << "FEN after:  " << pos.fen() << "\n";
            debug_log << "=========================================\n";
            debug_log.flush();

            // DEBUG: Validate board state after each move application
            std::string check_fen = pos.fen();

            // Check for obvious corruption: piece counts
            int white_kings = 0, black_kings = 0;
            for (int s = 0; s < 64; ++s) {
                Piece p = pos.piece_on(Square(s));
                if (p != NO_PIECE) {
                    PieceType pt = piece_type_of(p);
                    if (pt == KING) {
                        if (p / 6 == WHITE) white_kings++;
                        else black_kings++;
                    }
                }
            }

            if (white_kings != 1 || black_kings != 1) {
                std::cerr << "\n=== CORRUPTION AFTER MOVE " << m << " ===\n";
                std::cerr << "White kings: " << white_kings << ", Black kings: " << black_kings << "\n";
                std::cerr << "FEN: " << check_fen << "\n";
            }
        }
    }
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

    // DEBUG: Log position state before search
    debug_log << "\n=== BEFORE SEARCH ===\n";
    debug_log << "FEN: " << pos.fen() << "\n";
    debug_log << "Legal moves: " << legal_count << "\n";
    for (ExtMove* it = move_list; it != move_end; ++it) {
        Move m = it->move;
        Square from = m.from();
        Square to = m.to();
        Piece from_pc = pos.piece_on(from);
        debug_log << "  " << m << " (from=" << from << " pc=" << int(from_pc);
        if (from_pc != NO_PIECE) {
            debug_log << " " << (pos.color_of_piece(from_pc) == WHITE ? "W" : "B") << piece_type_of(from_pc);
        }
        debug_log << " to=" << to << ")\n";
    }
    debug_log << "====================\n";
    debug_log.flush();

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

    // Send best move
    if (best_move) {
        // DEBUG: Validate move before sending
        Square from = best_move.from();
        Square to = best_move.to();
        Piece from_piece = pos.piece_on(from);
        Piece to_piece = pos.piece_on(to);

        std::cerr << "\n=== SENDING BESTMOVE ===\n";
        std::cerr << "Move: " << best_move << "\n";
        std::cerr << "From: " << from << " piece: " << int(from_piece);
        if (from_piece != NO_PIECE) {
            std::cerr << " (" << (pos.color_of_piece(from_piece) == WHITE ? "W" : "B") << piece_type_of(from_piece) << ")";
        }
        std::cerr << "\n";
        std::cerr << "To: " << to << " piece: " << int(to_piece);
        if (to_piece != NO_PIECE) {
            std::cerr << " (" << (pos.color_of_piece(to_piece) == WHITE ? "W" : "B") << piece_type_of(to_piece) << ")";
        }
        std::cerr << "\n";
        std::cerr << "FEN: " << pos.fen() << "\n";
        std::cerr << "========================\n";

        debug_log << "\n=== SENDING BESTMOVE ===\n";
        debug_log << "Move: " << best_move << "\n";
        debug_log << "From: " << from << " piece: " << int(from_piece);
        if (from_piece != NO_PIECE) {
            debug_log << " (" << (pos.color_of_piece(from_piece) == WHITE ? "W" : "B") << piece_type_of(from_piece) << ")";
        }
        debug_log << "\n";
        debug_log << "To: " << to << " piece: " << int(to_piece);
        if (to_piece != NO_PIECE) {
            debug_log << " (" << (pos.color_of_piece(to_piece) == WHITE ? "W" : "B") << piece_type_of(to_piece) << ")";
        }
        debug_log << "\n";
        debug_log << "FEN: " << pos.fen() << "\n";
        debug_log << "========================\n";
        debug_log.flush();

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

        // DEBUG: Log all incoming UCI commands
        debug_log << "\n=== RECEIVED: " << line << " ===\n";
        debug_log.flush();

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
