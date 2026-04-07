#pragma once

#include "board.h"
#include "types.h"
#include <atomic>
#include <cstdint>

namespace luminex {

// Search stack
class Stack {
public:
    Move pv[64];  // Reduced from MAX_PLY + 7 to save stack space
    int ply = 0;
    Move current_move = MOVE_NONE;
    Move excluded_move = MOVE_NONE;
    Piece moved_piece = NO_PIECE;  // Piece that moved at this ply
    Value static_eval = VALUE_ZERO;
    int move_count = 0;
    bool improving = false;
    Stack* previous = nullptr;
};

// Search limits
struct Limits {
    int time[2] = {};
    int inc[2] = {};
    int npmsec = 0;
    int movetime = 0;
    int depth = 0;  // Default 0 to allow time management
    int nodes = 0;
    int movestogo = 0;  // UCI movestogo parameter for time management
    uint64_t mate = 0;
    bool infinite = false;
    bool ponder = false;

    bool use_time_management() const { return !movetime && !depth && !nodes && !mate && !infinite; }
};

// Search parameters
struct SearchParams {
    int contempt = 0;
    int multi_pv = 1;
};

class Position;
class TranspositionTable;

// Root search
Move search(Position& pos, Limits& limits);

// Correction history - clear on new game
void clear_correction_history();

// UCI info output
void uci_info(const Position& pos, int depth, Value score, uint64_t nodes, int time);

// Search globals
extern Limits limits;
extern SearchParams params;
extern std::atomic<uint64_t> nodes;
extern std::atomic<bool> stop;
extern int root_depth;
extern Value root_score;
extern int num_threads; // Default 1, SMP only helps at longer TC
extern Move previous_root_best; // Previous iteration's best move for ordering

} // namespace luminex
